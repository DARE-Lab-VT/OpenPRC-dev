"""
Pure-function math core for PRC readout training.

Separated from train_readout.py so the math is unit-testable without
DEMLAT, h5py, or scipy. The only dependency is numpy.

The actual training script wraps these with I/O and CSV/JSON output.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Ridge regression (dual or primal, automatically picked)
# -----------------------------------------------------------------------------

def ridge_regression(
    X: np.ndarray,
    Y: np.ndarray,
    lam: float,
    fit_intercept: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve ridge regression min_W ||XW - Y||² + lam ||W||².

    Picks the dual form X^T (XX^T + λI)^-1 Y when n_samples < n_features,
    primal form (X^T X + λI)^-1 X^T Y otherwise. For wide matrices (e.g.
    1500 samples × 30000 strain features) the dual is ~100× faster.

    Args:
        X: (n_samples, n_features)
        Y: (n_samples, n_outputs) — must be 2D even for single output
        lam: regularization strength λ ≥ 0
        fit_intercept: if True, center X and Y before solving and recover
                       intercept b. Otherwise b is zero.

    Returns:
        W: (n_features, n_outputs) coefficient matrix
        b: (n_outputs,) intercept (zeros if fit_intercept=False)
    """
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError(f"X must be 2D (got {X.shape}), Y must be 2D (got {Y.shape})")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"X has {X.shape[0]} samples but Y has {Y.shape[0]}"
        )
    if lam < 0:
        raise ValueError(f"lambda must be ≥ 0, got {lam}")

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if fit_intercept:
        x_mean = X.mean(axis=0, keepdims=True)
        y_mean = Y.mean(axis=0, keepdims=True)
        Xc = X - x_mean
        Yc = Y - y_mean
    else:
        Xc, Yc = X, Y
        x_mean = np.zeros((1, X.shape[1]))
        y_mean = np.zeros((1, Y.shape[1]))

    n_samples, n_features = Xc.shape

    if n_samples < n_features:
        # Dual form: solve (XX^T + λI) α = Y, then W = X^T α
        K = Xc @ Xc.T  # (n, n)
        K[np.diag_indices_from(K)] += lam
        alpha = np.linalg.solve(K, Yc)
        W = Xc.T @ alpha
    else:
        # Primal form
        G = Xc.T @ Xc  # (p, p)
        G[np.diag_indices_from(G)] += lam
        W = np.linalg.solve(G, Xc.T @ Yc)

    if fit_intercept:
        b = y_mean.ravel() - x_mean @ W
        b = b.ravel()
    else:
        b = np.zeros(Y.shape[1])

    return W.astype(np.float32), b.astype(np.float32)


def predict(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Apply a trained ridge readout: Y_hat = X @ W + b."""
    return (X @ W + b).astype(np.float32)


def ridge_kernel_fit(
    X: np.ndarray,
    Y: np.ndarray,
    lam: float,
    fit_intercept: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Memory-efficient kernel-form ridge fit for wide matrices (p ≫ n).

    Returns (W, b) of shapes (n_features, n_outputs) and (n_outputs,).

    Computes α = (XX^T + λI)^{-1} Y, then W = X^T α. Avoids materializing
    a centered float64 X and instead applies centering correction to the
    Gram matrix arithmetically. For (20000, 30000) inputs this peaks at
    around 5 GB instead of 15+ GB.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    Y = np.ascontiguousarray(Y, dtype=np.float32)
    n = X.shape[0]

    if fit_intercept:
        x_mean = X.mean(axis=0, keepdims=True).astype(np.float32)
        y_mean = Y.mean(axis=0, keepdims=True).astype(np.float32)
    else:
        x_mean = np.zeros((1, X.shape[1]), dtype=np.float32)
        y_mean = np.zeros((1, Y.shape[1]), dtype=np.float32)

    # K = (X - x_mean) @ (X - x_mean)^T, computed without materializing Xc
    chunk_size = 2048
    K = np.zeros((n, n), dtype=np.float32)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        K[start:end] = X[start:end] @ X.T
    x_dot_xmean = (X @ x_mean.T).ravel()
    xmean_norm_sq = float((x_mean @ x_mean.T).ravel()[0])
    K -= x_dot_xmean[:, None]
    K -= x_dot_xmean[None, :]
    K += xmean_norm_sq

    K_reg = K.astype(np.float64)
    K_reg[np.diag_indices_from(K_reg)] += lam
    Yc = (Y - y_mean).astype(np.float64)
    alpha = np.linalg.solve(K_reg, Yc).astype(np.float32)

    # W = Xc^T α where Xc = X - x_mean.
    # Computed without materializing Xc:
    #   Xc^T α = X^T α - x_mean^T (1^T α) = X^T α - x_mean^T · sum(α)
    W = X.T @ alpha  # (p, n) @ (n, d) = (p, d)
    alpha_sum = alpha.sum(axis=0)  # (d,)
    W -= x_mean.ravel()[:, np.newaxis] * alpha_sum[np.newaxis, :]
    b = (y_mean - x_mean @ W).ravel().astype(np.float32)
    return W, b


# -----------------------------------------------------------------------------
# Cross-validation for lambda selection
# -----------------------------------------------------------------------------

def kfold_indices(n: int, k: int, shuffle: bool = False, seed: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate k-fold (train_idx, val_idx) pairs over n samples.

    If shuffle is True, samples are shuffled (deterministic given seed).
    Otherwise contiguous folds are used — better for time-series since
    contiguous folds preserve temporal structure within each split.

    For PRC training we typically want NON-shuffled folds because temporal
    correlations matter; shuffling leaks information across the train/val
    boundary.
    """
    if k < 2:
        raise ValueError(f"k must be ≥ 2, got {k}")
    if n < k:
        raise ValueError(f"need at least {k} samples for {k}-fold CV, got {n}")

    indices = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[: n % k] += 1   # first (n%k) folds get one extra

    folds = []
    start = 0
    for size in fold_sizes:
        val_idx = indices[start: start + size]
        train_idx = np.concatenate([indices[:start], indices[start + size:]])
        folds.append((train_idx, val_idx))
        start += size
    return folds


def grouped_kfold_indices(
    group_ids: np.ndarray, k: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Like kfold_indices but never splits a single group across the train/val
    boundary. Each fold's val set is a subset of distinct groups.

    Used for "leave-one-clip-out" CV: groups are clip indices, so the
    validation set always contains entire clips (no information leak from
    within-clip temporal correlation).
    """
    unique_groups = np.unique(group_ids)
    n_groups = len(unique_groups)
    if k > n_groups:
        # If we have fewer groups than requested folds, each group is its own fold
        k = n_groups
    if k < 2:
        raise ValueError(f"need at least 2 distinct groups, got {n_groups}")

    # Distribute groups across folds
    fold_sizes = np.full(k, n_groups // k, dtype=int)
    fold_sizes[: n_groups % k] += 1

    folds = []
    start = 0
    for size in fold_sizes:
        val_groups = set(unique_groups[start: start + size].tolist())
        val_mask = np.isin(group_ids, list(val_groups))
        val_idx = np.where(val_mask)[0]
        train_idx = np.where(~val_mask)[0]
        folds.append((train_idx, val_idx))
        start += size
    return folds


def cv_ridge_select_lambda(
    X: np.ndarray,
    Y: np.ndarray,
    lambdas: List[float],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    fit_intercept: bool = True,
) -> Tuple[float, np.ndarray]:
    """
    Cross-validate ridge regression over a list of lambdas.

    For each lambda, average the per-fold validation MSE; pick the lambda
    with the smallest mean MSE. Returns the chosen lambda and the full
    (n_lambdas, n_folds) MSE grid.
    """
    if not lambdas:
        raise ValueError("lambdas list is empty")
    if not folds:
        raise ValueError("folds list is empty")

    mse_grid = np.zeros((len(lambdas), len(folds)), dtype=np.float64)
    for i, lam in enumerate(lambdas):
        for j, (tr, va) in enumerate(folds):
            W, b = ridge_regression(X[tr], Y[tr], lam, fit_intercept)
            Y_hat = predict(X[va], W, b)
            mse_grid[i, j] = np.mean((Y_hat - Y[va]) ** 2)

    mean_mse = mse_grid.mean(axis=1)
    best_i = int(np.argmin(mean_mse))
    return lambdas[best_i], mse_grid


def cv_ridge_select_lambda_kernel(
    X: np.ndarray,
    Y: np.ndarray,
    lambdas: List[float],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    fit_intercept: bool = True,
    chunk_size: int = 2048,
) -> Tuple[float, np.ndarray]:
    """
    Fast CV for wide matrices (n_features ≫ n_samples).

    Speed-up: precompute the full Gram matrix K = X X^T once. Each fold
    then only needs to slice K, add λI, and solve a small (n_tr, n_tr)
    system. Compared to the naive `cv_ridge_select_lambda`, this turns
    a per-fold-per-lambda cost of O(n²p) into O(n²) plus a one-time O(n²p)
    precompute.

    Memory-efficient: keeps inputs in float32 (single precision is fine for
    ridge — the regularizer dominates numerical conditioning anyway), and
    builds the Gram matrix in row-chunks so we never materialize a full
    centered float64 copy of the (huge) feature matrix.

    For Go1 (n=20000, p=30000):
      naive:        prohibitive (>10 GB peak, ~1 hour)
      kernel f32:   ~5.6 GB peak (Xc + K), <1 minute
      kernel chunked f32: ~3 GB peak, similar speed
    """
    if not lambdas:
        raise ValueError("lambdas list is empty")
    if not folds:
        raise ValueError("folds list is empty")

    X = np.ascontiguousarray(X, dtype=np.float32)
    Y = np.ascontiguousarray(Y, dtype=np.float32)
    n = X.shape[0]

    if fit_intercept:
        x_mean = X.mean(axis=0, keepdims=True).astype(np.float32)
        y_mean = Y.mean(axis=0, keepdims=True).astype(np.float32)
    else:
        x_mean = np.zeros((1, X.shape[1]), dtype=np.float32)
        y_mean = np.zeros((1, Y.shape[1]), dtype=np.float32)

    # Build K = (X - x_mean) @ (X - x_mean)^T chunk-wise so we don't allocate
    # a full centered copy. For each row-chunk, compute its dot product with
    # the entire X (also in chunks if memory tight) and apply centering.
    # K[i, j] = (X[i] - x_mean) · (X[j] - x_mean)
    #        = X[i] · X[j] - x_mean · X[j] - X[i] · x_mean + x_mean · x_mean
    # We compute K_raw = X @ X^T chunk-wise, then apply the centering correction.
    K = np.zeros((n, n), dtype=np.float32)
    # x_dot_xmean: (n,) vector x_mean @ X[i] for every i
    x_dot_xmean = (X @ x_mean.T).ravel()  # shape (n,)
    xmean_norm_sq = float((x_mean @ x_mean.T).ravel()[0])

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        # Chunk @ full X^T → (chunk, n) raw inner products
        K[start:end] = X[start:end] @ X.T

    # Apply centering correction in place:
    # K[i, j] -= x_dot_xmean[i] + x_dot_xmean[j] - xmean_norm_sq
    K -= x_dot_xmean[:, None]
    K -= x_dot_xmean[None, :]
    K += xmean_norm_sq

    # Centered Y for the right-hand side
    Yc = Y - y_mean

    mse_grid = np.zeros((len(lambdas), len(folds)), dtype=np.float64)
    for i, lam in enumerate(lambdas):
        for j, (tr, va) in enumerate(folds):
            K_tr = K[np.ix_(tr, tr)].astype(np.float64, copy=True)
            K_tr[np.diag_indices_from(K_tr)] += lam
            alpha = np.linalg.solve(K_tr, Yc[tr].astype(np.float64))
            K_va = K[np.ix_(va, tr)].astype(np.float64)
            Y_pred_va = K_va @ alpha
            if fit_intercept:
                Y_pred_va = Y_pred_va + y_mean
            err = np.mean((Y_pred_va - Y[va]) ** 2)
            mse_grid[i, j] = err

    mean_mse = mse_grid.mean(axis=1)
    best_i = int(np.argmin(mean_mse))
    return lambdas[best_i], mse_grid


# -----------------------------------------------------------------------------
# Targets: derivatives via Savitzky-Golay
# -----------------------------------------------------------------------------

def savitzky_golay_derivative(
    signal: np.ndarray,
    dt: float,
    order: int = 1,
    window_length: int = 11,
    polyorder: int = 3,
) -> np.ndarray:
    """
    Compute the `order`-th time derivative of a signal via Savitzky-Golay
    filtering. Lower noise than plain finite differences while staying
    causal-friendly (centered window).

    Args:
        signal: (T,) or (T, D) — derivative is along axis 0
        dt: timestep between samples (seconds)
        order: 1 for velocity, 2 for acceleration
        window_length: SG window (must be odd, > polyorder)
        polyorder: polynomial order for fitting

    Returns:
        Same shape as signal, the derivative.
    """
    try:
        from scipy.signal import savgol_filter
    except ImportError as e:
        raise ImportError(
            "scipy is required for Savitzky-Golay smoothing. "
            "Install with: pip install scipy"
        ) from e

    if window_length < polyorder + 1:
        window_length = polyorder + 2
    if window_length % 2 == 0:
        window_length += 1
    if signal.shape[0] < window_length:
        # Fall back to finite differences for short sequences
        if order == 1:
            return _finite_diff_first(signal, dt)
        elif order == 2:
            return _finite_diff_second(signal, dt)

    return savgol_filter(
        signal, window_length=window_length, polyorder=polyorder,
        deriv=order, delta=dt, axis=0,
    ).astype(np.float32)


def _finite_diff_first(signal: np.ndarray, dt: float) -> np.ndarray:
    """Centered finite difference for first derivative; edges use one-sided."""
    return np.gradient(signal, dt, axis=0).astype(np.float32)


def _finite_diff_second(signal: np.ndarray, dt: float) -> np.ndarray:
    """Centered finite difference for second derivative."""
    first = np.gradient(signal, dt, axis=0)
    return np.gradient(first, dt, axis=0).astype(np.float32)


def quaternion_to_angular_velocity(
    quat_wxyz: np.ndarray, dt: float, smooth: bool = True,
) -> np.ndarray:
    """
    Convert a (T, 4) quaternion sequence to (T, 3) body-frame angular
    velocity in rad/s.

    Math: for unit quaternions q(t), the angular velocity ω satisfies
        q̇ = 0.5 · q ⊗ ω_quat,    where ω_quat = (0, ω_x, ω_y, ω_z)
    Solving for ω: ω = 2 · vec(q⁻¹ ⊗ q̇).

    We compute q̇ via Savitzky-Golay (or finite difference) then this product.
    Quaternion convention: w-x-y-z (scalar-first), matching our bundle schema.

    Hemisphere correction: adjacent quaternions q(t), q(t+1) on opposite
    hemispheres (q vs -q represent the same rotation but break differentiation).
    We flip signs to enforce continuity along the time axis before differentiating.
    """
    if quat_wxyz.shape[-1] != 4:
        raise ValueError(f"expected (..., 4) quaternion array, got {quat_wxyz.shape}")
    q = np.asarray(quat_wxyz, dtype=np.float64).copy()

    # Hemisphere fix: ensure dot(q[t-1], q[t]) >= 0 by negating where needed
    for t in range(1, len(q)):
        if np.dot(q[t - 1], q[t]) < 0:
            q[t] = -q[t]

    # Numerical derivative
    if smooth:
        q_dot = savitzky_golay_derivative(q, dt, order=1).astype(np.float64)
    else:
        q_dot = _finite_diff_first(q, dt).astype(np.float64)

    # ω = 2 · vec(q⁻¹ ⊗ q̇).  For a unit quaternion, q⁻¹ = conj(q) = (w, -x, -y, -z).
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    dw, dx, dy, dz = q_dot[:, 0], q_dot[:, 1], q_dot[:, 2], q_dot[:, 3]
    # q_inv = (qw, -qx, -qy, -qz). Compute q_inv ⊗ q_dot.
    # Hamilton product (a, x, y, z) ⊗ (b, X, Y, Z):
    #   w = aB - xX - yY - zZ
    #   x = aX + xB + yZ - zY
    #   y = aY - xZ + yB + zX
    #   z = aZ + xY - yX + zB
    a, x, y, z = qw, -qx, -qy, -qz
    b, X, Y, Z = dw, dx, dy, dz
    # vec part of q_inv ⊗ q_dot:
    omega_x = a * X + x * b + y * Z - z * Y
    omega_y = a * Y - x * Z + y * b + z * X
    omega_z = a * Z + x * Y - y * X + z * b
    omega = 2.0 * np.stack([omega_x, omega_y, omega_z], axis=1)
    return omega.astype(np.float32)


def rotate_world_to_body(
    vec_world: np.ndarray,
    quat_wxyz: np.ndarray,
) -> np.ndarray:
    """
    Rotate (T, 3) world-frame vectors into body frame using the inverse of
    the orientation quaternion (wxyz convention).

    Body frame = R(q)^T * world = R(q^{-1}) * world.
    Uses the Rodrigues formula with q_inv = (w, -x, -y, -z).
    """
    w  = quat_wxyz[:, 0].astype(np.float64)
    qx = -quat_wxyz[:, 1].astype(np.float64)
    qy = -quat_wxyz[:, 2].astype(np.float64)
    qz = -quat_wxyz[:, 3].astype(np.float64)
    vx = vec_world[:, 0].astype(np.float64)
    vy = vec_world[:, 1].astype(np.float64)
    vz = vec_world[:, 2].astype(np.float64)

    t0 = 2 * (qy * vz - qz * vy)
    t1 = 2 * (qz * vx - qx * vz)
    t2 = 2 * (qx * vy - qy * vx)

    bx = vx + w * t0 + (qy * t2 - qz * t1)
    by = vy + w * t1 + (qz * t0 - qx * t2)
    bz = vz + w * t2 + (qx * t1 - qy * t0)
    return np.stack([bx, by, bz], axis=1).astype(np.float32)


def base_pose_to_base_velocities(
    base_pose: np.ndarray, fps: float, smooth: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose a (T, 7) base-pose sequence (xyz + wxyz quat) into
    (T, 3) world-frame linear velocity and (T, 3) body-frame angular velocity.

    Returns (lin_vel_world, ang_vel_body).
    Use base_pose_to_body_velocity for body-frame linear velocity.
    """
    if base_pose.ndim != 2 or base_pose.shape[1] != 7:
        raise ValueError(
            f"expected (T, 7) base pose [x,y,z,qw,qx,qy,qz], got {base_pose.shape}"
        )
    dt = 1.0 / float(fps)
    if smooth:
        lin_vel = savitzky_golay_derivative(
            base_pose[:, :3].astype(np.float32), dt, order=1
        )
    else:
        lin_vel = _finite_diff_first(base_pose[:, :3], dt)
    ang_vel = quaternion_to_angular_velocity(base_pose[:, 3:7], dt, smooth=smooth)
    return lin_vel, ang_vel


def base_pose_to_body_velocity(
    base_pose: np.ndarray, fps: float, smooth: bool = True,
) -> np.ndarray:
    """
    Compute (T, 6) body-frame 6D velocity [vx, vy, vz, wx, wy, wz] from a
    (T, 7) base-pose sequence (xyz + wxyz quaternion).

    Linear velocity is rotated from world frame into body frame.
    Angular velocity is already body-frame (from quaternion differentiation).

    This is the correct target for physical reservoir computing on legged
    robots: body-frame vx has 5-10x more variance than world-frame for a
    walking robot (world-frame oscillates as the robot turns; body-frame
    forward speed is the actual locomotion signal).
    """
    lin_world, ang_body = base_pose_to_base_velocities(base_pose, fps, smooth)
    lin_body = rotate_world_to_body(lin_world, base_pose[:, 3:7])
    return np.concatenate([lin_body, ang_body], axis=1)


def base_pose_to_base_accelerations(
    base_pose: np.ndarray, fps: float, smooth: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute base linear and angular accelerations (both world-frame for lin,
    body-frame derivative-of-omega for ang).
    """
    if base_pose.ndim != 2 or base_pose.shape[1] != 7:
        raise ValueError(
            f"expected (T, 7) base pose, got {base_pose.shape}"
        )
    dt = 1.0 / float(fps)
    if smooth:
        lin_acc = savitzky_golay_derivative(
            base_pose[:, :3].astype(np.float32), dt, order=2
        )
    else:
        lin_acc = _finite_diff_second(base_pose[:, :3], dt)

    # Angular acceleration = derivative of body-frame angular velocity.
    ang_vel = quaternion_to_angular_velocity(base_pose[:, 3:7], dt, smooth=smooth)
    if smooth:
        ang_acc = savitzky_golay_derivative(ang_vel, dt, order=1)
    else:
        ang_acc = _finite_diff_first(ang_vel, dt)
    return lin_acc, ang_acc


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def mse(y_true: np.ndarray, y_pred: np.ndarray, axis=None) -> np.ndarray:
    """Mean squared error. axis=None gives scalar; axis=0 gives per-output."""
    return np.mean((y_pred - y_true) ** 2, axis=axis)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray, axis=None) -> np.ndarray:
    """
    Coefficient of determination R² = 1 - SS_res / SS_tot.

    Per-output R² when axis=0; scalar when axis=None.
    """
    if axis == 0:
        ss_res = np.sum((y_pred - y_true) ** 2, axis=0)
        y_mean = y_true.mean(axis=0, keepdims=True)
        ss_tot = np.sum((y_true - y_mean) ** 2, axis=0)
    else:
        ss_res = np.sum((y_pred - y_true) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    # Guard against zero-variance targets
    return 1.0 - ss_res / np.where(ss_tot < 1e-12, 1.0, ss_tot)


# -----------------------------------------------------------------------------
# Multi-step rollout
# -----------------------------------------------------------------------------

def autoregressive_rollout(
    initial_state: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    """
    Run the readout autoregressively: feed each prediction back as the next
    input. Only meaningful when W maps state to next-state of the same shape.

    Args:
        initial_state: (D,) seed state vector
        W: (D, D) coefficient matrix
        b: (D,) intercept
        n_steps: number of forward steps

    Returns:
        (n_steps + 1, D) trajectory including the initial state.
    """
    if W.shape[0] != W.shape[1]:
        raise ValueError(
            f"autoregressive rollout needs square W, got {W.shape}"
        )
    if initial_state.shape[0] != W.shape[0]:
        raise ValueError(
            f"initial_state has {initial_state.shape[0]} dims, W expects {W.shape[0]}"
        )
    states = np.empty((n_steps + 1, W.shape[0]), dtype=np.float32)
    states[0] = initial_state
    for t in range(n_steps):
        states[t + 1] = states[t] @ W + b
    return states
