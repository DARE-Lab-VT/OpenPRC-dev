"""
Theta Management
=================

Utilities for working with the theta pytree:
  - Construction from geometry.h5
  - Masking (which params are optimizable)
  - Bounds enforcement
  - Gradient filtering
"""

import jax
import jax.numpy as jnp
import numpy as np


# ============================================================
# Optimizable parameter names
# ============================================================

ALL_PARAM_KEYS = [
    'bar_stiffness',
    'bar_damping',
    'bar_rest_length',
    'bar_prestress',
    'hinge_stiffness',
    'hinge_damping',
    'hinge_rest_angle',
    'global_damping',
    'gravity',
]


def make_mask(theta, optimize_params):
    """
    Build a boolean mask pytree matching theta structure.

    Parameters
    ----------
    theta : dict
        The parameter pytree.
    optimize_params : dict
        Mapping of param_key -> bool.  E.g.
        {'bar_stiffness': True, 'hinge_stiffness': True}
        Keys not present default to False (frozen).

    Returns
    -------
    mask : dict
        Same structure as theta, with True for optimizable leaves,
        False for frozen.
    """
    mask = {}
    for key, val in theta.items():
        mask[key] = jnp.ones_like(val, dtype=bool) if optimize_params.get(key, False) \
            else jnp.zeros_like(val, dtype=bool)
    return mask


def apply_mask(grads, mask):
    """Zero out gradients for frozen parameters."""
    return jax.tree.map(lambda g, m: jnp.where(m, g, 0.0), grads, mask)


# ============================================================
# Bounds
# ============================================================

def make_bounds(theta, bounds_config):
    """
    Build (lower, upper) bound pytrees.

    Parameters
    ----------
    theta : dict
        Parameter pytree (used for shape/structure).
    bounds_config : dict
        Mapping of param_key -> (lo, hi).  E.g.
        {'bar_stiffness': (0.1, 1000.0)}

    Returns
    -------
    lo, hi : dict, dict
        Lower and upper bound pytrees.
    """
    lo = {}
    hi = {}
    for key, val in theta.items():
        if key in bounds_config:
            lb, ub = bounds_config[key]
            lo[key] = jnp.full_like(val, lb)
            hi[key] = jnp.full_like(val, ub)
        else:
            lo[key] = jnp.full_like(val, -jnp.inf)
            hi[key] = jnp.full_like(val, jnp.inf)
    return lo, hi


def clip_to_bounds(theta, lo, hi):
    """Project theta onto the feasible box [lo, hi]."""
    return jax.tree.map(lambda t, l, h: jnp.clip(t, l, h), theta, lo, hi)


# ============================================================
# Gradient-based update
# ============================================================

def sgd_update(theta, grads, mask, lr):
    """Masked SGD step: theta_new = theta - lr * masked_grads."""
    masked = apply_mask(grads, mask)
    return jax.tree.map(lambda t, g: t - lr * g, theta, masked)


def adam_state_init(theta):
    """Initialize Adam optimizer state."""
    m = jax.tree.map(jnp.zeros_like, theta)
    v = jax.tree.map(jnp.zeros_like, theta)
    return {'m': m, 'v': v, 'step': 0}


def adam_update(theta, grads, mask, adam_state, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """Masked Adam step."""
    masked_grads = apply_mask(grads, mask)
    step = adam_state['step'] + 1

    m_new = jax.tree.map(lambda m, g: beta1 * m + (1 - beta1) * g, adam_state['m'], masked_grads)
    v_new = jax.tree.map(lambda v, g: beta2 * v + (1 - beta2) * g ** 2, adam_state['v'], masked_grads)

    # Bias correction
    bc1 = 1 - beta1 ** step
    bc2 = 1 - beta2 ** step

    theta_new = jax.tree.map(
        lambda t, m, v: t - lr * (m / bc1) / (jnp.sqrt(v / bc2) + eps),
        theta, m_new, v_new,
    )

    return theta_new, {'m': m_new, 'v': v_new, 'step': step}


# ============================================================
# Theta I/O  (to/from dict for HDF5 serialization)
# ============================================================

def theta_to_numpy(theta):
    """Convert theta pytree to numpy dict for serialization."""
    return {k: np.array(v) for k, v in theta.items()}


def theta_from_numpy(np_dict):
    """Reconstruct theta pytree from numpy dict."""
    return {k: jnp.array(v, dtype=jnp.float32) for k, v in np_dict.items()}