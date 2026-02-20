import numpy as np
import torch

from numpy.lib.stride_tricks import sliding_window_view


def compute_ipc_components(X, u_iid_input, tau_s, n_s, washout, train_stop, test_duration, k_delay,
                          interp_factor: int = 1, 
                          epsilon: float = 1E-4,
                          ridge: float = 1e-6,
                          return_names: bool = True,): 
                          
    u = np.asarray(u_iid_input, dtype=np.float32).flatten()
    X = np.asarray(X, dtype=np.float32)

    if interp_factor > 1:
        X = X[::interp_factor]
        u = u[::interp_factor]
    
    min_len = min(len(X), len(u))
    X = X[:min_len]
    u = u[:min_len]

    t_test_end = train_stop + test_duration
    lag_indices = [j * k_delay for j in range(tau_s + 1)]
    Lvars = len(lag_indices)
    
    # --- FIX 1: Auto-detect bounds to strictly enforce [-1, 1] Legendre scaling ---
    u_min, u_max = np.min(u), np.max(u)
    if abs(u_max - u_min) < 1e-12:
        raise ValueError("Input signal u is constant. Cannot scale to [-1,1].")
    u_leg = (2.0 * (u - u_min) / (u_max - u_min) - 1.0).astype(np.float32)

    # 1. Generate Basis Exponents
    exps = []
    def rec(rem, i, vec):
        if i == Lvars - 1:
            vec[i] = rem
            exps.append(vec.copy())
        else:
            for v in range(rem + 1):
                vec[i] = v
                rec(rem - v, i + 1, vec)

    for d in range(1, n_s + 1):
        rec(d, 0, np.zeros(Lvars, dtype=np.int16))
    exps = np.asarray(exps)

    capacities = np.zeros(len(exps), dtype=np.float32)
    
    # Normalized Legendre
    def legendre_P_normalized(n, x):
        if n == 0: return np.ones_like(x)
        pm2, pm1 = np.ones_like(x), x
        for k in range(1, n):
            curr = ((2*k + 1) * x * pm1 - k * pm2) / (k + 1)
            pm2, pm1 = pm1, curr
        return np.sqrt(2*n + 1) * pm1

    for idx, exp_vec in enumerate(exps):
        active_lags = [lag_indices[j] for j, d in enumerate(exp_vec) if d > 0]
        max_lag_i = max(active_lags) if active_lags else 0
        
        t_start = max(washout, max_lag_i)
        if t_start >= train_stop: continue 

        X_tr = X[t_start:train_stop]
        X_te = X[train_stop:t_test_end]
        
        y_tr = np.ones(len(X_tr))
        y_te = np.ones(len(X_te))
        for j, deg in enumerate(exp_vec):
            if deg > 0:
                lag = lag_indices[j]
                y_tr *= legendre_P_normalized(deg, u_leg[t_start-lag : train_stop-lag])
                y_te *= legendre_P_normalized(deg, u_leg[train_stop-lag : t_test_end-lag])

        # --- Stable Ridge Regression with native Bias Term ---
        X_tr_b = np.concatenate([np.ones((len(X_tr), 1), dtype=np.float32), X_tr], axis=1)
        X_te_b = np.concatenate([np.ones((len(X_te), 1), dtype=np.float32), X_te], axis=1)
        
        p1 = X_tr_b.shape[1]
        DtD = X_tr_b.T @ X_tr_b
        trace = np.trace(DtD)
        lam = ridge * (trace / p1) 
        
        I = np.eye(p1, dtype=np.float32)
        I[0, 0] = 0.0  # Mathematically crucial: do not regularize the bias term
        
        try:
            w = np.linalg.solve(DtD + lam * I, X_tr_b.T @ y_tr)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(DtD + lam * I) @ (X_tr_b.T @ y_tr)
            
        y_pred = X_te_b @ w
        # ----------------------------------------------------
        
        # Capacity Evaluation
        sst = np.sum((y_te - y_te.mean())**2) + 1e-12
        sse = np.sum((y_te - y_pred)**2)
        
        raw_capacity = 1.0 - (sse / sst)
        capacities[idx] = raw_capacity if raw_capacity > epsilon else 0.0

    basis_names = []
    if return_names:
        for e in exps:
            parts = [f"P{d}(u(t-{l}))" for d, l in zip(e, lag_indices) if d > 0]
            basis_names.append(" * ".join(parts))

    return basis_names, capacities, exps


def compute_ipc_components_gpu(X, u_input, tau_s, n_s, washout, train_stop, test_duration, k_delay, 
                         chunk_size: int = 512, ridge: float = 1e-6, epsilon: float = 1e-4, 
                         device: str | None = None, return_names: bool = True):
    """
    High-fidelity GPU IPC computation using Normalized Legendre Polynomials.
    Matches CPU logic: Stable Ridge (unpenalized bias) + Orthogonal [-1, 1] scaling.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    u = np.asarray(u_input, dtype=np.float32).flatten()
    X = np.asarray(X, dtype=np.float32)
    
    # 1. Strict Domain Scaling [-1, 1] (Crucial for Legendre Orthogonality)
    u_min, u_max = np.min(u), np.max(u)
    u_leg = (2.0 * (u - u_min) / (u_max - u_min) - 1.0).astype(np.float32)

    max_lag = tau_s * k_delay
    t_start = max(washout, max_lag)
    t_test_end = train_stop + test_duration

    # 2. Construct Lag Matrix (CPU -> GPU)
    U_full = sliding_window_view(u_leg, max_lag + 1)[:, ::-1]
    lag_indices = [j * k_delay for j in range(tau_s + 1)]
    U_sub = U_full[:, lag_indices]
    Lvars = U_sub.shape[1]
    
    U_train = torch.from_numpy(U_sub[t_start - max_lag : train_stop - max_lag]).to(device, dtype=torch.float32)
    U_test  = torch.from_numpy(U_sub[train_stop - max_lag : t_test_end - max_lag]).to(device, dtype=torch.float32)
    X_train = torch.from_numpy(X[t_start:train_stop]).to(device, dtype=torch.float32)
    X_test  = torch.from_numpy(X[train_stop:t_test_end]).to(device, dtype=torch.float32)

    # 3. Stable Ridge Solver (Unpenalized Bias)
    D_train = torch.cat([torch.ones((U_train.shape[0], 1), device=device), X_train], dim=1)
    D_test  = torch.cat([torch.ones((U_test.shape[0], 1), device=device), X_test], dim=1)
    
    DtD = D_train.T @ D_train
    lam = ridge * (torch.trace(DtD) / D_train.shape[1]).clamp(min=1e-12)
    I = torch.eye(D_train.shape[1], device=device)
    I[0, 0] = 0.0 
    
    try:
        P = torch.linalg.solve(DtD + lam * I, D_train.T)
    except RuntimeError:
        P = torch.linalg.pinv(DtD + lam * I) @ D_train.T

    # 4. Normalized Legendre Generator for GPU
    def get_legendre_gpu(n, x_tensor):
        if n == 0: return torch.ones_like(x_tensor)
        pm2, pm1 = torch.ones_like(x_tensor), x_tensor
        for k in range(1, n):
            curr = ((2*k + 1) * x_tensor * pm1 - k * pm2) / (k + 1)
            pm2, pm1 = pm1, curr
        return np.sqrt(2*n + 1) * pm1

    # 5. Exponent Enumeration (Stars & Bars)
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
    for d in range(1, n_s + 1): # Start at 1 to skip constant degree
        rec(d, 0)
    
    exps = np.asarray(exps, dtype=np.int16)
    K = len(exps)
    capacities = torch.zeros(K, device=device)

    # 6. Chunked Execution
    for start in range(0, K, chunk_size):
        end = min(start + chunk_size, K)
        chunk_exps = exps[start:end]
        
        # Build Polynomial targets on GPU
        Y_train = torch.ones((U_train.shape[0], end-start), device=device)
        Y_test  = torch.ones((U_test.shape[0], end-start), device=device)
        
        for idx, exp_vec in enumerate(chunk_exps):
            for j, deg in enumerate(exp_vec):
                if deg > 0:
                    Y_train[:, idx] *= get_legendre_gpu(deg, U_train[:, j])
                    Y_test[:, idx]  *= get_legendre_gpu(deg, U_test[:, j])

        # Batch Prediction & R2 Evaluation
        SST = torch.sum((Y_test - Y_test.mean(dim=0))**2, dim=0) + 1e-12
        W = P @ Y_train
        SSE = torch.sum((Y_test - D_test @ W)**2, dim=0)
        
        raw_cap = 1.0 - (SSE / SST)
        capacities[start:end] = torch.where(raw_cap > epsilon, raw_cap, 0.0)

    # 7. Generate Basis Names (Post-Processing on CPU)
    basis_names = []
    if return_names:
        for e in exps:
            parts = [f"P{d}(u(t-{lag_indices[j]}))" for j, d in enumerate(e) if d > 0]
            basis_names.append(" * ".join(parts))

    return basis_names, capacities.cpu().numpy(), exps


def compute_R2(X, u_input, tau_s, n_s, washout, train_stop, test_duration, k_delay, 
                         ridge: float = 1e-6, return_names: bool = True):
    """
    Computes the Information Processing Capacity (Capacity = 1 - NMSE) for the 
    cumulative multinomial basis of the input, processed one target at a time.
    """

    u = np.asarray(u_input, dtype=np.float32)
    X = np.asarray(X, dtype=np.float32)

    T = u.shape[0]
    if X.shape[0] != T:
        raise ValueError(f"X and u_input must have same length. Got X:{X.shape[0]} u:{T}")

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
    ones_train = np.ones((N_train, 1), dtype=np.float32)
    ones_test  = np.ones((N_test,  1), dtype=np.float32)

    D_train = np.concatenate([ones_train, X_train], axis=1)
    D_test  = np.concatenate([ones_test,  X_test],  axis=1)
    
    # --- 3. Compute Readout Projection Matrix (P) ---
    # Since P only depends on the reservoir states, we still only compute it once.
    p1 = D_train.shape[1]
    DtD = D_train.T @ D_train
    trace = np.trace(DtD)
    lam = ridge * (trace / p1) 
    I = np.eye(p1, dtype=np.float32)
    
    try:
        P = np.linalg.solve(DtD + lam * I, D_train.T)
    except np.linalg.LinAlgError:
        P = np.linalg.pinv(DtD + lam * I) @ D_train.T

    # --- 4. Generate Basis ---
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
    
    # --- 5. Compute Capacity One Target at a Time ---
    capacities = np.full((K,), np.nan, dtype=np.float32)
    eps = 1e-12

    for idx in range(K):
        exp_vec = exps[idx]
        
        # A. Construct Single Target Basis Function (1D Vector)
        Y_train_i = np.ones(N_train, dtype=np.float32)
        Y_test_i  = np.ones(N_test,  dtype=np.float32)

        for j in range(Lvars):
            e_j = exp_vec[j]
            if e_j > 0:  # Only multiply if the exponent is non-zero
                Y_train_i *= np.power(U_train[:, j], e_j)
                Y_test_i  *= np.power(U_test[:, j], e_j)

        # B. Calculate Variance of Target (SST)
        y_mean = Y_test_i.mean()
        SST = np.sum((Y_test_i - y_mean) ** 2)

        # C. Predict using Reservoir
        W_i = P @ Y_train_i
        Y_pred_i = D_test @ W_i
        
        # D. Calculate Error (SSE)
        SSE = np.sum((Y_test_i - Y_pred_i) ** 2)

        # E. Calculate Capacity
        if SST > eps:
            cap = 1.0 - (SSE / SST)
            capacities[idx] = np.clip(cap, 0.0, 1.0)

    # Remove degree 0
    if K > 0 and np.all(exps[0] == 0):
        capacities[0] = np.nan

    # --- 6. Generate Names ---
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