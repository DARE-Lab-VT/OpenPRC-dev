import numpy as np
import torch

from numpy.lib.stride_tricks import sliding_window_view


def compute_ipc_components(X, u_input, tau_s, n_s, washout, train_stop, test_duration, k_delay, 
                         chunk_size: int = 512, ridge: float = 1e-6, return_names: bool = True):
    """
    Computes the Information Processing Capacity (Capacity = 1 - NMSE) for the 
    cumulative multinomial basis of the input.
    """

    u = np.asarray(u_input, dtype=np.float32)
    X = np.asarray(X, dtype=np.float32)

    T = u.shape[0]
    if X.shape[0] != T:
        raise ValueError(f"X and u_input must have same length. Got X:{X.shape[0]} u:{T}")

    # [Validation checks omitted for brevity - same as before]
    max_lag = tau_s * k_delay
    t_start = max(washout, max_lag)
    t_test_end = train_stop + test_duration

    # --- 1. Construct Lagged Input Matrix ---
    U_full = sliding_window_view(u.flatten(), max_lag + 1)[:, ::-1]
    lag_indices = [j * k_delay for j in range(tau_s + 1)]
    U_sub = U_full[:, lag_indices]
    Lvars = U_sub.shape[1]

    # Indices
    r_train_start, r_train_end = t_start - max_lag, train_stop - max_lag
    r_test_start, r_test_end = train_stop - max_lag, t_test_end - max_lag

    U_train = U_sub[r_train_start:r_train_end]
    U_test  = U_sub[r_test_start:r_test_end]
    X_train = X[t_start:train_stop]
    X_test  = X[train_stop:t_test_end]
    
    N_train = U_train.shape[0]
    N_test  = U_test.shape[0]

    # --- 2. Prepare Reservoir States (Features) ---
    # Add bias column for affine readout
    ones_train = np.ones((N_train, 1), dtype=np.float32)
    ones_test  = np.ones((N_test,  1), dtype=np.float32)

    D_train = np.concatenate([ones_train, X_train], axis=1)
    D_test  = np.concatenate([ones_test,  X_test],  axis=1)
    
    # --- 3. Compute Readout Weights (Ridge Regression) ---
    # P = (X^T X + lambda I)^-1 X^T
    # This projects the reservoir states onto the target space
    p1 = D_train.shape[1]
    DtD = D_train.T @ D_train
    trace = np.trace(DtD)
    lam = ridge * (trace / p1) # Adaptive regularization
    I = np.eye(p1, dtype=np.float32)
    
    try:
        P = np.linalg.solve(DtD + lam * I, D_train.T)
    except np.linalg.LinAlgError:
        P = np.linalg.pinv(DtD + lam * I) @ D_train.T

    # --- 4. Generate Basis & Compute Capacity ---
    # Recursive exponent generation for polynomial basis
    exps = []
    vec = np.zeros(Lvars, dtype=np.int16)
    def rec(rem, i):
        if i == Lvars - 1:
            vec[i] = rem
            exps.append(vec.copy())
        else:
            for v in range(rem + 1):
                vec[i] = v
                rec(rem - v, i + 1)
    for d in range(0, n_s + 1):
        rec(d, 0)
    
    exps = np.asarray(exps, dtype=np.int16)
    K = exps.shape[0]
    exps_t = exps.astype(np.float32)

    # Capacity Storage
    # Dambre's C corresponds to R2_cols in your previous code
    capacities = np.full((K,), np.nan, dtype=np.float32)
    
    # Processing in chunks to save memory
    eps = 1e-12
    for start_idx in range(0, K, chunk_size):
        end_idx = min(start_idx + chunk_size, K)
        exps_chunk = exps_t[start_idx:end_idx]
        chunkK = exps_chunk.shape[0]

        # A. Construct Target Basis Functions Y (Ground Truth)
        # Y = Product of delayed inputs raised to powers
        Y_train = np.ones((N_train, chunkK), dtype=np.float32)
        Y_test  = np.ones((N_test,  chunkK), dtype=np.float32)

        for j in range(Lvars):
            u_tr_j = U_train[:, j].reshape(N_train, 1)
            u_te_j = U_test[:,  j].reshape(N_test,  1)
            e_j    = exps_chunk[:, j].reshape(1, chunkK)
            Y_train *= np.power(u_tr_j, e_j)
            Y_test  *= np.power(u_te_j, e_j)

        # B. Calculate Variance of Targets (SST)
        y_mean = Y_test.mean(axis=0, keepdims=True)
        SST = ((Y_test - y_mean) ** 2).sum(axis=0)

        # C. Predict using Reservoir (Y_pred)
        # W = P @ Y_train (Training weights)
        W = P @ Y_train
        Y_pred = D_test @ W
        
        # D. Calculate Error (SSE)
        SSE = ((Y_test - Y_pred) ** 2).sum(axis=0)

        # E. Calculate Dambre's Capacity: C = 1 - NMSE
        # NMSE = SSE / SST
        chunk_caps = np.full((chunkK,), np.nan, dtype=np.float32)
        valid = SST > eps
        
        chunk_caps[valid] = 1.0 - (SSE[valid] / SST[valid])
        
        # STRICT REQUIREMENT: Capacity must be in [0, 1]
        chunk_caps = np.clip(chunk_caps, 0.0, 1.0)
        
        capacities[start_idx:end_idx] = chunk_caps

    # Remove degree 0 (Constant term has 0 variance, capacity undefined or 0)
    if K > 0 and np.all(exps[0] == 0):
        capacities[0] = np.nan

    # (Optional) Generate Names
    basis_names = []
    if return_names:
        for e in exps:
            if np.all(e == 0):
                basis_names.append("1")
                continue
            parts = []
            lags = [j * k_delay for j in range(tau_s + 1)]
            for coeff, lag in zip(e, lags):
                if coeff == 0: continue
                nm = "u(t)" if lag == 0 else f"u(t-{lag})"
                parts.append(nm if coeff == 1 else f"{nm}^{int(coeff)}")
            basis_names.append(" ".join(parts))

    return basis_names, capacities, exps


def compute_ipc_components_gpu(X, u_input, tau_s, n_s, washout, train_stop, test_duration, k_delay, 
                         chunk_size: int = 512, ridge: float = 1e-6, device: str | None = None, return_names: bool = True
    ):
    """
    Fast + robust GPU computation of per-monomial test R^2 (Capacity) for the cumulative multinomial basis.
    
    Key Features:
    - Stable Solver: Uses standard Solve/Pseudo-Inverse instead of adaptive regularization loops.
      This ensures results match CPU precision and prevents capacity collapse.
    - Engineering Definition: Uses centered variance and bias term (matches Dambre's IPC correctly).
    - Returns: Basis names, Capacities (R2), and Exponents matrix.

    Parameters
    ----------
    X : (T, N_units) Reservoir states
    u_input : (T,) Input signal
    tau_s : Memory depth (max delays)
    n_s : Max polynomial degree
    k_delay : Stride for delays
    """

    # -------------------------
    # 1. Device & Input Setup
    # -------------------------
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    u = np.asarray(u_input, dtype=np.float32).flatten()
    X = np.asarray(X, dtype=np.float32)

    T = u.shape[0]
    if X.shape[0] != T:
        raise ValueError(f"X and u_input must have same length. Got X:{X.shape[0]} u:{T}")

    # Validation
    if tau_s < 0 or n_s < 0: raise ValueError("tau_s and n_s must be nonnegative.")
    if k_delay <= 0: raise ValueError("k_delay must be positive.")
    
    max_lag = tau_s * k_delay
    t_start = max(washout, max_lag)
    t_test_end = train_stop + test_duration

    if train_stop <= t_start: raise ValueError("train_stop must be > washout and max_lag.")
    if t_test_end > T: raise ValueError("Total duration exceeds signal length.")

    # Clean NaNs
    if not np.isfinite(u).all():
        u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(X).all():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # -------------------------
    # 2. Construct Lag Matrix (CPU -> GPU)
    # -------------------------
    # Sliding window on CPU first to save VRAM
    U_full = sliding_window_view(u, max_lag + 1)[:, ::-1]
    
    lag_indices = [j * k_delay for j in range(tau_s + 1)]
    U_sub = U_full[:, lag_indices]
    Lvars = U_sub.shape[1]

    # Map time indices
    r_train_start, r_train_end = t_start - max_lag, train_stop - max_lag
    r_test_start, r_test_end   = train_stop - max_lag, t_test_end - max_lag

    # Move to GPU
    U_train = torch.from_numpy(U_sub[r_train_start:r_train_end]).to(device=device, dtype=torch.float32)
    U_test  = torch.from_numpy(U_sub[r_test_start:r_test_end]).to(device=device, dtype=torch.float32)
    
    X_train = torch.from_numpy(X[t_start:train_stop]).to(device=device, dtype=torch.float32)
    X_test  = torch.from_numpy(X[train_stop:t_test_end]).to(device=device, dtype=torch.float32)

    N_train = U_train.shape[0]
    N_test  = U_test.shape[0]

    # -------------------------
    # 3. Reservoir Design Matrix (with Bias)
    # -------------------------
    ones_train = torch.ones((N_train, 1), device=device, dtype=torch.float32)
    ones_test  = torch.ones((N_test,  1), device=device, dtype=torch.float32)

    D_train = torch.cat([ones_train, X_train], dim=1)
    D_test  = torch.cat([ones_test,  X_test],  dim=1)
    p1 = D_train.shape[1]

    # -------------------------
    # 4. Readout Weights (STABLE SOLVER)
    # -------------------------
    DtD = D_train.T @ D_train
    trace = torch.trace(DtD)
    
    # Calculate Ridge Lambda ONCE (Strict adherence to input parameter)
    lam = ridge * (trace / p1).clamp(min=1e-12)
    I = torch.eye(p1, device=device, dtype=torch.float32)
    
    # Try fast solve, fallback to stable pseudo-inverse immediately if unstable
    # This prevents the "Adaptive Lambda" loop from artificially lowering scores
    try:
        # Cholesky/Solve is fast but can fail on singular matrices
        P = torch.linalg.solve(DtD + lam * I, D_train.T)
    except RuntimeError:
        # Fallback to SVD-based Pseudoinverse (Slower, but robust like CPU)
        P = torch.linalg.pinv(DtD + lam * I) @ D_train.T

    # -------------------------
    # 5. Exponent Enumeration (Stars & Bars)
    # -------------------------
    exps = []
    vec = np.zeros(Lvars, dtype=np.int16)
    def rec(rem, i):
        if i == Lvars - 1:
            vec[i] = rem
            exps.append(vec.copy())
        else:
            for v in range(rem + 1):
                vec[i] = v
                rec(rem - v, i + 1)
    for d in range(0, n_s + 1):
        rec(d, 0)
    
    exps = np.asarray(exps, dtype=np.int16)
    K = exps.shape[0]
    exps_t = torch.from_numpy(exps).to(device=device, dtype=torch.float32)

    # -------------------------
    # 6. Chunked Capacity Calculation
    # -------------------------
    R2_cols = torch.full((K,), float("nan"), device=device, dtype=torch.float32)
    eps = 1e-12

    for start_idx in range(0, K, chunk_size):
        end_idx = min(start_idx + chunk_size, K)
        exps_chunk = exps_t[start_idx:end_idx] # (chunkK, Lvars)
        chunkK = exps_chunk.shape[0]

        # A. Target Monomials Y (Broadcasting for speed)
        # U: (N, 1, Lvars) ^ Exps: (1, K, Lvars) -> (N, K, Lvars) -> Prod -> (N, K)
        u_tr_exp = U_train.unsqueeze(1)
        u_te_exp = U_test.unsqueeze(1)
        e_exp    = exps_chunk.unsqueeze(0)

        Y_train = torch.prod(torch.pow(u_tr_exp, e_exp), dim=2)
        Y_test  = torch.prod(torch.pow(u_te_exp, e_exp), dim=2)

        # B. Variance (SST) - Centered
        y_mean = Y_test.mean(dim=0, keepdim=True)
        SST = ((Y_test - y_mean) ** 2).sum(dim=0)

        # C. Prediction & Error
        W = P @ Y_train        # (p1, chunkK)
        Y_pred = D_test @ W    # (N_test, chunkK)
        SSE = ((Y_test - Y_pred) ** 2).sum(dim=0)

        # D. R^2 Capacity
        chunk_caps = torch.full((chunkK,), float("nan"), device=device, dtype=torch.float32)
        valid = SST > eps
        
        chunk_caps[valid] = 1.0 - (SSE[valid] / SST[valid])
        chunk_caps = torch.clamp(chunk_caps, 0.0, 1.0)
        
        R2_cols[start_idx:end_idx] = chunk_caps

    # Set degree 0 (constant) to NaN
    if K > 0 and np.all(exps[0] == 0):
        R2_cols[0] = float("nan")

    # -------------------------
    # 7. Generate Names
    # -------------------------
    basis_names = []
    if return_names:
        for e in exps:
            if np.all(e == 0):
                basis_names.append("1")
                continue
            parts = []
            for coeff, lag in zip(e, lag_indices):
                if coeff == 0: continue
                nm = "u(t)" if lag == 0 else f"u(t-{lag})"
                parts.append(nm if coeff == 1 else f"{nm}^{int(coeff)}")
            basis_names.append(" ".join(parts))

    return basis_names, R2_cols.detach().cpu().numpy(), exps