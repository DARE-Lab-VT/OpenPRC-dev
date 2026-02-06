import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def compute_basis_R2_numpy(X, u_input, tau_s, n_s, washout, train_stop, test_duration, k_delay, 
                         chunk_size: int = 512, ridge: float = 1e-6, return_names: bool = True):
    """
    Fast + robust CPU computation of per-monomial test R^2 for the cumulative multinomial basis
    at cell (n_s, tau_s), using reservoir response X as features.

    Targets (monomials) are built from lagged inputs:
        [u(t), u(t-k), u(t-2k), ..., u(t - tau_s*k)]
    and include all exponent vectors with total degree d = 0..n_s (cumulative).
    Degree-0 constant term is returned as NaN for R^2.

    Notes
    -----
    - Uses chunked monomial evaluation for speed.
    - Uses ridge projection P = (D^T D + λI)^-1 D^T with adaptive λ for stability.
    - eta is returned as η = 1 - R^2_lin, where R^2_lin is best ridge-linear fit from lagged inputs.

    Returns
    -------
    basis_names : list[str]
    R2_cols     : (K,) np.ndarray
    eta_cols    : (K,) np.ndarray
    exps        : (K, Lvars) np.ndarray
    """

    u = np.asarray(u_input, dtype=np.float32)
    X = np.asarray(X, dtype=np.float32)

    T = u.shape[0]
    if X.shape[0] != T:
        raise ValueError(f"X and u_input must have same length. Got X:{X.shape[0]} u:{T}")

    if tau_s < 0 or n_s < 0:
        raise ValueError("tau_s and n_s must be nonnegative.")
    if k_delay <= 0:
        raise ValueError("k_delay must be positive.")
    if train_stop <= washout:
        raise ValueError("train_stop must be > washout.")
    if train_stop + test_duration > T:
        raise ValueError("train_stop + test_duration exceeds signal length.")

    if not np.isfinite(u).all():
        u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if not np.isfinite(X).all():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    max_lag = tau_s * k_delay
    if max_lag >= T:
        raise ValueError("tau_s * k_delay must be < len(u_input).")

    t_start = max(washout, max_lag)
    t_test_end = train_stop + test_duration
    if t_start >= train_stop:
        raise ValueError("washout/max_lag too large -> no training samples left.")

    U_full = sliding_window_view(u.flatten(), max_lag + 1)[:, ::-1]

    lag_indices = [j * k_delay for j in range(tau_s + 1)]
    U_sub = U_full[:, lag_indices]
    Lvars = U_sub.shape[1]

    r_train_start = t_start - max_lag
    r_train_end   = train_stop - max_lag
    r_test_start  = train_stop - max_lag
    r_test_end    = t_test_end - max_lag

    U_train = U_sub[r_train_start:r_train_end]
    U_test  = U_sub[r_test_start:r_test_end]
    N_train = U_train.shape[0]
    N_test  = U_test.shape[0]

    X_train = X[t_start:train_stop]
    X_test  = X[train_stop:t_test_end]

    ones_train = np.ones((N_train, 1), dtype=np.float32)
    ones_test  = np.ones((N_test,  1), dtype=np.float32)

    D_train = np.concatenate([ones_train, X_train], axis=1)
    D_test  = np.concatenate([ones_test,  X_test],  axis=1)
    p1 = D_train.shape[1]

    if not np.isfinite(D_train).all():
        D_train = np.nan_to_num(D_train, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(D_test).all():
        D_test = np.nan_to_num(D_test, nan=0.0, posinf=0.0, neginf=0.0)

    L_train = np.concatenate([ones_train, U_train], axis=1)
    L_test  = np.concatenate([ones_test,  U_test],  axis=1)
    q1 = L_train.shape[1]
    
    if not np.isfinite(L_train).all():
        L_train = np.nan_to_num(L_train, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(L_test).all():
        L_test = np.nan_to_num(L_test, nan=0.0, posinf=0.0, neginf=0.0)

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

    DtD = D_train.T @ D_train
    trace = np.trace(DtD)
    base = np.clip(trace / p1, a_min=1e-12, a_max=None)
    lam = ridge * base
    I = np.eye(p1, dtype=np.float32)
    
    try:
        P = np.linalg.solve(DtD + lam * I, D_train.T)
    except np.linalg.LinAlgError:
        P = np.linalg.pinv(DtD + lam * I) @ D_train.T

    LtL = L_train.T @ L_train
    trace_lin = np.trace(LtL)
    base_lin = np.clip(trace_lin / q1, a_min=1e-12, a_max=None)
    lam_lin = ridge * base_lin
    I_lin = np.eye(q1, dtype=np.float32)
    
    try:
        P_lin = np.linalg.solve(LtL + lam_lin * I_lin, L_train.T)
    except np.linalg.LinAlgError:
        P_lin = np.linalg.pinv(LtL + lam_lin * I_lin) @ L_train.T

    R2_cols = np.full((K,), np.nan, dtype=np.float32)
    eta_cols = np.full((K,), np.nan, dtype=np.float32)
    eps = 1e-12

    for start_idx in range(0, K, chunk_size):
        end_idx = min(start_idx + chunk_size, K)
        exps_chunk = exps_t[start_idx:end_idx]
        chunkK = exps_chunk.shape[0]

        Y_train = np.ones((N_train, chunkK), dtype=np.float32)
        Y_test  = np.ones((N_test,  chunkK), dtype=np.float32)

        for j in range(Lvars):
            u_tr_j = U_train[:, j].reshape(N_train, 1)
            u_te_j = U_test[:,  j].reshape(N_test,  1)
            e_j    = exps_chunk[:, j].reshape(1, chunkK)
            Y_train *= np.power(u_tr_j, e_j)
            Y_test  *= np.power(u_te_j, e_j)

        y_mean = Y_test.mean(axis=0, keepdims=True)
        SST = ((Y_test - y_mean) ** 2).sum(axis=0)

        W = P @ Y_train
        Y_pred = D_test @ W
        SSE = ((Y_test - Y_pred) ** 2).sum(axis=0)

        R2_chunk = np.full((chunkK,), np.nan, dtype=np.float32)
        valid = SST > eps
        R2_chunk[valid] = 1.0 - SSE[valid] / SST[valid]
        R2_chunk = np.clip(R2_chunk, 0.0, 1.0)
        R2_cols[start_idx:end_idx] = R2_chunk

        W_lin = P_lin @ Y_train
        Y_pred_lin = L_test @ W_lin
        SSE_lin = ((Y_test - Y_pred_lin) ** 2).sum(axis=0)

        R2_lin_chunk = np.full((chunkK,), np.nan, dtype=np.float32)
        R2_lin_chunk[valid] = 1.0 - SSE_lin[valid] / SST[valid]
        R2_lin_chunk = np.clip(R2_lin_chunk, 0.0, 1.0)

        eta_chunk = 1.0 - R2_lin_chunk
        eta_chunk = np.clip(eta_chunk, 0.0, 1.0)
        eta_cols[start_idx:end_idx] = eta_chunk

    if K > 0 and np.all(exps[0] == 0):
        R2_cols[0] = np.nan
        eta_cols[0] = np.nan

    basis_names = []
    if return_names:
        for e in exps:
            if np.all(e == 0):
                basis_names.append("1")
                continue
            parts = []
            lag_indices = [j * k_delay for j in range(tau_s + 1)]
            for coeff, lag in zip(e, lag_indices):
                if coeff == 0:
                    continue
                nm = "u(t)" if lag == 0 else f"u(t-{lag})"
                parts.append(nm if coeff == 1 else f"{nm}^{int(coeff)}")
            basis_names.append(" ".join(parts))

    return basis_names, R2_cols, eta_cols, exps


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