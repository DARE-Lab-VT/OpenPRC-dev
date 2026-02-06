import numpy as np
from openprc.analysis.utils.training_utils import compute_ipc_components, compute_ipc_components_gpu


def NARMA_task(u_input, order=2, a=0.3, b=0.05, c=1.5, d=0.1):
    """
    Args:
        u_input (np.ndarray, optional): External input signal driving the reservoir.
                                        If None, generates random uniform [0, 0.5].
    """

    # If the input signal is multi-dimensional, compute its vector magnitude to make it 1D.
    if u_input.ndim > 1 and u_input.shape[1] > 1:
        u_input = np.linalg.norm(u_input, axis=1)

    length = len(u_input)
    # 1. Handle Input Signal
    # Ensure input matches requested length
    if len(u_input) != length:
        raise ValueError(f"Input u length ({len(u_input)}) does not match requested length ({length}).")
    u = u_input.flatten()
    
    y = np.zeros(length)
    N = order
    
    # 2. Logic matching target_narma from reference
    if N == 2:
        # N=2 Special Case (Cubic nonlinearity)
        for t in range(2, length):
            y[t] = 0.4 * y[t - 1] + \
                    0.4 * y[t - 1] * y[t - 2] + \
                    0.6 * u[t - 1] ** 3 + \
                    0.1
    else:
        # Standard NARMA Case
        for t in range(N, length):
            sum_y = np.sum(y[t - N:t])
            y[t] = a * y[t - 1] + \
                    b * y[t - 1] * sum_y + \
                    c * u[t - N] * u[t - 1] + \
                    d
                    
    return y.reshape(-1, 1)


def memory_task(X, u_input, washout: int, train_stop: int, test_duration: int,
                tau_s: int, n_s: int, k_delay: int = 1, ridge: float = 1e-6):
    """
    Computes Dambre's Information Processing Capacity (IPC) components.
    
    References:
        Dambre et al., "Information Processing Capacity of Dynamical Systems", Scientific Reports (2012).
        
    Parameters
    ----------
    k_delay : int, default=1
        Delay step size. For standard Memory Capacity (Jaeger 2001/Dambre 2012), this must be 1.
    """
    
    basis_names, capacities, exps = compute_ipc_components_gpu(
        X=X,
        u_input=u_input,
        tau_s=tau_s,
        n_s=n_s,
        washout=washout,
        train_stop=train_stop,
        test_duration=test_duration,
        k_delay=k_delay,
        ridge=ridge,
        return_names=True
    )
    
    # Summing capacities based on polynomial degree (d)
    degrees = np.sum(exps, axis=1)
    
    # --- Linear Memory Capacity (d=1) ---
    # Dambre (2012) identifies this as the capacity to reconstruct 
    # linear functions of past inputs (u(t-k)).
    linear_indices = np.where(degrees == 1)[0]
    linear_mc = np.nansum(capacities[linear_indices])
    
    # --- Nonlinear Capacity (d > 1) ---
    # Capacity to reconstruct nonlinear functions (products/powers) of past inputs.
    nonlinear_indices = np.where(degrees > 1)[0]
    nonlinear_mc = np.nansum(capacities[nonlinear_indices])
    
    # --- Total Information Processing Capacity (IPC) ---
    # Theoretical limit: Sum(C) <= N (number of linearly independent reservoir states)
    total_ipc = linear_mc + nonlinear_mc
    
    results = {
        'linear_memory_capacity': linear_mc,     # Matches Jaeger's MC if k_delay=1
        'nonlinear_memory_capacity': nonlinear_mc,
        'total_capacity': total_ipc,             # Dambre's Total Capacity
        'capacities': capacities,                # Individual C values for each basis function
        'basis_names': basis_names,
        'degrees': degrees,
        'exponents': exps
    }
    
    return results