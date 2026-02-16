"""
Cost Functions
===============

Pluggable cost functions for optimization / calibration.

All cost functions follow the signature:
    cost_fn(x_sim, x_ref, **kwargs) -> scalar

where:
    x_sim : (T, N, 3)  – simulated trajectory (positions at each saved step)
    x_ref : (T, N, 3)  – reference trajectory

They return a JAX scalar that can be differentiated.
"""

import jax.numpy as jnp


# ============================================================
# Built-in Cost Functions
# ============================================================

def position_mse(x_sim, x_ref, **kwargs):
    """
    Mean Squared Error over all nodes and timesteps.

    L = (1/T) * (1/N) * sum_t sum_i ||x_sim(t,i) - x_ref(t,i)||^2
    """
    diff = x_sim - x_ref
    return jnp.mean(jnp.sum(diff ** 2, axis=-1))


def position_rmse(x_sim, x_ref, **kwargs):
    """Root Mean Squared Error (sqrt of position_mse)."""
    return jnp.sqrt(position_mse(x_sim, x_ref))


def position_mse_weighted(x_sim, x_ref, node_weights=None, time_weights=None, **kwargs):
    """
    Weighted MSE.  Allows emphasizing specific nodes or time windows.

    Parameters
    ----------
    node_weights : (N,)   – per-node importance.  None = uniform.
    time_weights : (T,)   – per-timestep importance.  None = uniform.
    """
    diff_sq = jnp.sum((x_sim - x_ref) ** 2, axis=-1)  # (T, N)

    if node_weights is not None:
        diff_sq = diff_sq * node_weights[None, :]

    if time_weights is not None:
        diff_sq = diff_sq * time_weights[:, None]

    return jnp.mean(diff_sq)


def strain_mse(x_sim, x_ref, bar_indices=None, **kwargs):
    """
    MSE on bar strains instead of absolute positions.

    Strain = (L - L_ref) / L_ref  per bar per timestep.
    This is invariant to rigid-body translations/rotations of the reference,
    useful when the reference frame may differ.

    Parameters
    ----------
    bar_indices : (M, 2) int array – bar connectivity (required).
    """
    if bar_indices is None:
        raise ValueError("strain_mse requires bar_indices kwarg")

    i, j = bar_indices[:, 0], bar_indices[:, 1]

    # (T, M)
    L_sim = jnp.linalg.norm(x_sim[:, j] - x_sim[:, i], axis=-1)
    L_ref = jnp.linalg.norm(x_ref[:, j] - x_ref[:, i], axis=-1)

    strain_sim = (L_sim - L_ref) / (L_ref + 1e-12)

    return jnp.mean(strain_sim ** 2)


def final_position_mse(x_sim, x_ref, **kwargs):
    """MSE only on the final timestep.  Good for equilibrium matching."""
    diff = x_sim[-1] - x_ref[-1]
    return jnp.mean(jnp.sum(diff ** 2, axis=-1))


def trajectory_dtw_approx(x_sim, x_ref, **kwargs):
    """
    Soft approximation to Dynamic Time Warping cost.

    Uses a differentiable soft-min along the time axis.
    Useful when sim and ref may have slightly misaligned timing.
    """
    # (T_sim, T_ref, N)
    diff_sq = jnp.sum(
        (x_sim[:, None, :, :] - x_ref[None, :, :, :]) ** 2, axis=-1
    )  # (T_sim, T_ref, N)

    # Per-node soft-min across ref time for each sim time
    per_node = -jnp.log(jnp.mean(jnp.exp(-diff_sq), axis=1) + 1e-12)  # (T_sim, N)
    return jnp.mean(per_node)


# ============================================================
# Cost Function Registry
# ============================================================

COSTS = {
    'mse': position_mse,
    'rmse': position_rmse,
    'weighted_mse': position_mse_weighted,
    'strain_mse': strain_mse,
    'final_mse': final_position_mse,
    'dtw': trajectory_dtw_approx,
}


def get_cost(name_or_callable, **default_kwargs):
    """
    Resolve a cost function by name or pass through a callable.

    Parameters
    ----------
    name_or_callable : str or callable
        If str, looked up in COSTS registry.
        If callable, used directly (must match signature).
    default_kwargs : dict
        Extra kwargs baked into the returned function.

    Returns
    -------
    cost_fn : callable  (x_sim, x_ref) -> scalar
    """
    if callable(name_or_callable):
        fn = name_or_callable
    elif isinstance(name_or_callable, str):
        if name_or_callable not in COSTS:
            raise ValueError(
                f"Unknown cost '{name_or_callable}'. "
                f"Available: {list(COSTS.keys())}"
            )
        fn = COSTS[name_or_callable]
    else:
        raise TypeError(f"Expected str or callable, got {type(name_or_callable)}")

    if default_kwargs:
        def wrapped(x_sim, x_ref, **kw):
            merged = {**default_kwargs, **kw}
            return fn(x_sim, x_ref, **merged)
        return wrapped

    return fn