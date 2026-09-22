"""Readout utilities. IPC expects original uniform IID symbols, one per state row."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def scale_iid_input(u, input_bounds=(-1., 1.)):
    """Map known *population* bounds to [-1,1], never observed extrema.

    Values must be the actual IID sequence used in acquisition, not an
    interpolated command or measured actuator trajectory. Bounds alone cannot
    establish independence or uniformity.
    """
    u = np.asarray(u, dtype=np.float64)
    if u.ndim == 2 and u.shape[1] == 1:
        u = u[:, 0]
    if u.ndim != 1 or len(u) < 2 or not np.isfinite(u).all():
        raise ValueError('Expected a finite scalar IID sequence.')
    bounds = np.asarray(input_bounds, dtype=float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[1] <= bounds[0]:
        raise ValueError('input_bounds must be finite increasing population bounds.')
    lo, hi = bounds
    if np.any(u < lo) or np.any(u > hi):
        raise ValueError('IID values fall outside input_bounds.')
    if np.ptp(u) == 0:
        raise ValueError('IID input is constant.')
    return 2*(u-lo)/(hi-lo)-1


def legendre_target(u_scaled, exponents, k_delay=1):
    """One orthonormal product target; unavailable history is NaN, never wrapped."""
    u = np.asarray(u_scaled, dtype=np.float64)
    y = np.ones(len(u), dtype=np.float64)
    for j, degree in enumerate(exponents):
        if degree:
            lag = j*k_delay
            values = np.polynomial.legendre.legval(u[:len(u)-lag], [0.]*int(degree)+[1.])
            y[lag:] *= np.sqrt(2*degree+1)*values
            y[:lag] = np.nan
    return y


def _ipc_setup(X, u, tau_s, n_s, washout, train_stop, test_duration,
               k_delay, ridge, input_bounds):
    for value, label, minimum in ((tau_s,'tau_s',0),(n_s,'n_s',1),
            (washout,'washout',0),(train_stop,'train_stop',1),
            (test_duration,'test_duration',2),(k_delay,'k_delay',1)):
        if isinstance(value, bool) or not isinstance(value, (int,np.integer)) or value < minimum:
            raise ValueError(f'{label} must be an integer >= {minimum}.')
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError('ridge must be finite and nonnegative.')
    u = scale_iid_input(u, input_bounds)
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] == 0 or len(X) != len(u) or not np.isfinite(X).all():
        raise ValueError('X must be finite 2D states with exactly one row per IID symbol.')
    first = max(washout, tau_s*k_delay)
    end = train_stop+test_duration
    if first >= train_stop or end > len(X):
        raise ValueError('Insufficient rows for common history/washout, train and test windows.')
    exps = []
    def rec(rem, vec):
        if len(vec) == tau_s:
            exps.append(vec+[rem]); return
        for degree in range(rem+1): rec(rem-degree, vec+[degree])
    for degree in range(1,n_s+1): rec(degree, [])
    exps = np.asarray(exps,dtype=np.int64)
    a = np.column_stack((np.ones(train_stop-first,dtype=np.float32),X[first:train_stop]))
    b = np.column_stack((np.ones(test_duration,dtype=np.float32),X[train_stop:end]))
    return u, exps, a, b, first, end


def _names(exps, k_delay, return_names):
    return [' * '.join(f'P{d}(u(t-{j*k_delay}))' for j,d in enumerate(e) if d)
            for e in exps] if return_names else []


def compute_ipc_components(X, u_iid_input, tau_s, n_s, washout, train_stop,
                           test_duration, k_delay, *, ridge=1e-6, return_names=True,
                           input_bounds=(-1.,1.),
                           return_raw=False):
    """Held-out IPC estimates using separate ridge fits for Legendre products.

    X is already prepared/standardized and aligned to the original IID symbols.
    All targets share [max(washout,tau_s*k_delay), train_stop) for training;
    testing follows immediately, with no gap. Indices count symbol rows.
    Trace-scaled ridge leaves the bias unpenalized. Negative R² is clipped
    to zero; no positive-capacity or rank-derived cutoff is applied.
    return_raw=True exposes unthresholded held-out R². Alignment and
    multiplexing must be performed before this function.
    Finite-data clipped sums are estimates, not guaranteed intrinsic capacities.
    """
    u,exps,a,b,first,end = _ipc_setup(X,u_iid_input,tau_s,n_s,washout,train_stop,
                                    test_duration,k_delay,ridge,input_bounds)
    gram = a.T@a
    penalty = np.eye(gram.shape[0],dtype=np.float32); penalty[0,0] = 0
    system = gram + ridge*np.trace(gram)/len(gram)*penalty
    scores = np.empty(len(exps),dtype=np.float64)
    for j,exp in enumerate(exps):
        target = legendre_target(u,exp,k_delay)
        ytr,yte = target[first:train_stop],target[train_stop:end]
        rhs = a.T@ytr
        try: weights = np.linalg.solve(system,rhs)
        except np.linalg.LinAlgError: weights = np.linalg.pinv(system)@rhs
        score = 1-np.sum((yte-b@weights)**2)/(np.sum((yte-yte.mean())**2)+1e-12)
        scores[j] = score if return_raw else max(score, 0.)
    return _names(exps,k_delay,return_names),scores,exps


def compute_ipc_components_gpu(X, u_input, tau_s, n_s, washout, train_stop,
                               test_duration, k_delay, *, chunk_size=512, ridge=1e-6,
                               device=None, return_names=True,
                               input_bounds=(-1.,1.), return_raw=False):
    """Same targets, windows and per-target solves as CPU, using Torch.

    chunk_size is retained for call compatibility; targets are streamed one at a
    time. Floating-point backend differences remain possible. Torch is optional
    for CPU-only users of this module.
    """
    import torch
    if not isinstance(chunk_size,int) or chunk_size < 1:
        raise ValueError('chunk_size must be a positive integer.')
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    u,exps,a,b,first,end = _ipc_setup(X,u_input,tau_s,n_s,washout,train_stop,
                                    test_duration,k_delay,ridge,input_bounds)
    # Match the float32 state Gram and float64 target arithmetic of the CPU path.
    a = torch.as_tensor(a,device=device); b = torch.as_tensor(b,device=device).double()
    gram = a.T@a
    penalty = torch.eye(gram.shape[0],device=device); penalty[0,0] = 0
    system = (gram+ridge*torch.trace(gram)/len(gram)*penalty).double()
    a = a.double()
    scores = np.empty(len(exps))
    for j,exp in enumerate(exps):
        target = legendre_target(u,exp,k_delay)
        ytr = torch.as_tensor(target[first:train_stop],device=device)
        yte = torch.as_tensor(target[train_stop:end],device=device)
        rhs = a.T@ytr
        try: weights = torch.linalg.solve(system,rhs)
        except RuntimeError: weights = torch.linalg.pinv(system)@rhs
        score = float((1-torch.sum((yte-b@weights)**2)/(torch.sum((yte-yte.mean())**2)+1e-12)).item())
        scores[j] = score if return_raw else max(score, 0.)
    return _names(exps,k_delay,return_names),scores,exps


def aggregate_ipc_heatmaps(capacities, exponents):
    """Exact degree/max-lag-index sums and cumulative target means.

    Columns index delays 0..tau_s (multiply by k_delay for symbol delays).
    Cumulative means include zero-scoring targets in their denominator.
    Neither map is the exponent-weighted-memory grid used in some papers.
    """
    exps = np.asarray(exponents)
    scores = np.asarray(capacities)
    if exps.ndim != 2 or len(exps) != len(scores) or len(exps) == 0:
        raise ValueError('Need matching nonempty capacity and exponent arrays.')
    if not np.isfinite(scores).all() or np.any(exps<0) or np.any(exps.sum(axis=1)<1):
        raise ValueError('Expected finite scores and nonconstant product exponents.')
    exact = np.zeros((int(exps.sum(axis=1).max()),exps.shape[1]))
    counts = np.zeros_like(exact,dtype=int)
    for score,exp in zip(scores,exps):
        degree = int(exp.sum()); lag = np.flatnonzero(exp)[-1]
        exact[degree-1,lag] += max(float(score),0.)
        counts[degree-1,lag] += 1
    totals = exact.cumsum(0).cumsum(1); counts = counts.cumsum(0).cumsum(1)
    return exact,np.divide(totals,counts,out=np.zeros_like(totals),where=counts>0)


def compute_R2(X, u_input, tau_s, n_s, washout, train_stop, test_duration, k_delay, 
                         ridge: float = 1e-6, return_names: bool = True):
    """
    Computes held-out reconstruction R² for monomial targets.
    This legacy calculation is separate from orthogonal Legendre IPC.
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
    variance_floor = 1e-12  # Numerical guard for constant targets, not an IPC cutoff.

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
        if SST > variance_floor:
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