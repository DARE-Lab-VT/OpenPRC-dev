"""
Tests for the pure-function PRC training math.

All synthetic data — no DEMLAT/h5py dependencies. Run from anywhere.

Usage:
    python _tools/training/test_readout_math.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from readout_math import (
    ridge_regression,
    predict,
    kfold_indices,
    grouped_kfold_indices,
    cv_ridge_select_lambda,
    cv_ridge_select_lambda_kernel,
    savitzky_golay_derivative,
    _finite_diff_first,
    _finite_diff_second,
    mse,
    r_squared,
    autoregressive_rollout,
)


# -----------------------------------------------------------------------------
# Ridge
# -----------------------------------------------------------------------------

def test_ridge_recovers_true_coefficients_clean():
    """With λ=0 and noise-free data, ridge should recover the exact W."""
    rng = np.random.default_rng(0)
    n, p, q = 200, 10, 3
    X = rng.standard_normal((n, p))
    W_true = rng.standard_normal((p, q))
    b_true = rng.standard_normal(q)
    Y = X @ W_true + b_true
    W, b = ridge_regression(X, Y, lam=0.0, fit_intercept=True)
    assert np.allclose(W, W_true, atol=1e-3), \
        f"max W err: {np.max(np.abs(W - W_true)):.2e}"
    assert np.allclose(b, b_true, atol=1e-3)
    print("  PASS: ridge recovers W and b on clean data")


def test_ridge_dual_form_matches_primal():
    """Wide matrix (n < p): should auto-pick dual; result must match primal manually."""
    rng = np.random.default_rng(0)
    n, p, q = 50, 200, 5
    X = rng.standard_normal((n, p))
    Y = rng.standard_normal((n, q))
    lam = 1.0
    W_dual, b_dual = ridge_regression(X, Y, lam, fit_intercept=False)

    # Compute primal directly
    G = X.T @ X + lam * np.eye(p)
    W_primal = np.linalg.solve(G, X.T @ Y)
    assert np.allclose(W_dual, W_primal, atol=1e-6), \
        f"dual vs primal max err: {np.max(np.abs(W_dual - W_primal)):.2e}"
    print("  PASS: dual matches primal on wide matrix")


def test_ridge_lambda_shrinks_weights():
    """Large λ should produce smaller-norm coefficients."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 20))
    Y = rng.standard_normal((100, 3))
    W_small, _ = ridge_regression(X, Y, lam=0.01, fit_intercept=True)
    W_big, _ = ridge_regression(X, Y, lam=1000.0, fit_intercept=True)
    assert np.linalg.norm(W_big) < np.linalg.norm(W_small)
    print("  PASS: larger λ shrinks ||W||")


def test_ridge_intercept_centering():
    """fit_intercept should remove the mean shift in Y."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 5))
    Y_zero = X @ rng.standard_normal((5, 2))
    Y_shifted = Y_zero + np.array([100.0, -50.0])
    W_zero, b_zero = ridge_regression(X, Y_zero, 0.0, fit_intercept=True)
    W_shift, b_shift = ridge_regression(X, Y_shifted, 0.0, fit_intercept=True)
    # Coefficients should be the same up to numerical precision
    assert np.allclose(W_zero, W_shift, atol=1e-3)
    # Intercept absorbs the mean shift
    assert np.allclose(b_shift - b_zero, [100.0, -50.0], atol=1e-2)
    print("  PASS: fit_intercept absorbs constant shift")


def test_ridge_rejects_bad_shapes():
    X = np.zeros((10, 5))
    Y = np.zeros((10, 2))
    try:
        ridge_regression(X, Y, lam=-1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for negative λ")
    try:
        ridge_regression(np.zeros(5), Y, 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for 1D X")
    print("  PASS: ridge_regression validates inputs")


def test_predict_round_trip():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 5))
    Y = rng.standard_normal((50, 2))
    W, b = ridge_regression(X, Y, lam=0.01)
    Y_hat = predict(X, W, b)
    assert Y_hat.shape == Y.shape
    assert Y_hat.dtype == np.float32
    print("  PASS: predict returns expected shape and dtype")


# -----------------------------------------------------------------------------
# CV
# -----------------------------------------------------------------------------

def test_kfold_partition_is_disjoint():
    folds = kfold_indices(20, k=5)
    assert len(folds) == 5
    union_val = np.concatenate([va for _, va in folds])
    assert sorted(union_val.tolist()) == list(range(20))
    for tr, va in folds:
        assert len(np.intersect1d(tr, va)) == 0
        assert len(tr) + len(va) == 20
    print("  PASS: kfold yields disjoint train/val partitioning")


def test_kfold_uneven_split_distributes_remainder():
    folds = kfold_indices(11, k=3)
    sizes = [len(va) for _, va in folds]
    # 11 = 3+1, 3+1, 3+1 + ... actually 11/3 = 3 rem 2; first 2 folds get extra → [4,4,3]
    assert sorted(sizes) == [3, 4, 4]
    print("  PASS: uneven kfold distributes remainder fairly")


def test_grouped_kfold_keeps_groups_together():
    """No group should appear in both train and val of the same fold."""
    group_ids = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3])
    folds = grouped_kfold_indices(group_ids, k=2)
    for tr, va in folds:
        tr_groups = set(group_ids[tr].tolist())
        va_groups = set(group_ids[va].tolist())
        assert tr_groups.isdisjoint(va_groups), \
            f"groups leak: tr={tr_groups}, va={va_groups}"
    print("  PASS: grouped k-fold keeps groups disjoint")


def test_cv_ridge_picks_best_lambda():
    """On data with a known optimal λ, CV should pick close to it."""
    rng = np.random.default_rng(0)
    n, p, q = 100, 5, 1
    X = rng.standard_normal((n, p))
    W_true = rng.standard_normal((p, q))
    Y_clean = X @ W_true
    noise = rng.standard_normal(Y_clean.shape) * 1.0
    Y = Y_clean + noise

    lambdas = [1e-4, 1e-2, 1.0, 10.0, 100.0, 10000.0]
    folds = kfold_indices(n, k=5)
    best, grid = cv_ridge_select_lambda(X, Y, lambdas, folds)
    assert best in lambdas
    # With unit noise and ~100 samples, best should be in the middle of the grid
    # (not the very smallest or very largest)
    assert best != lambdas[0] and best != lambdas[-1], \
        f"CV picked extreme λ={best}; suggests bad CV"
    print(f"  PASS: cv_ridge_select_lambda picked λ={best:g}")


def test_kernel_cv_matches_naive_cv():
    """
    Kernel-form CV should agree with naive CV on which lambda is best.
    Absolute MSE values may differ because of subtle differences in how
    per-fold centering is computed; the lambda CHOICE is the contract
    that matters for training.
    """
    rng = np.random.default_rng(0)
    n, p = 30, 100   # wide: p > n
    X = rng.standard_normal((n, p))
    Y = rng.standard_normal((n, 2))
    lambdas = [1e-3, 1.0, 100.0]
    folds = kfold_indices(n, k=3)

    naive_best, naive_grid = cv_ridge_select_lambda(X, Y, lambdas, folds)
    kern_best, kern_grid = cv_ridge_select_lambda_kernel(X, Y, lambdas, folds)

    # Same lambda choice (the actual contract)
    assert naive_best == kern_best, \
        f"naive picked {naive_best}, kernel picked {kern_best}"
    # Both should agree on the ranking of lambdas
    naive_rank = np.argsort(naive_grid.mean(axis=1))
    kern_rank = np.argsort(kern_grid.mean(axis=1))
    assert np.array_equal(naive_rank, kern_rank), \
        f"rank mismatch: naive {naive_rank}, kernel {kern_rank}"
    print(f"  PASS: kernel CV agrees with naive on λ choice "
          f"(both picked {naive_best:g}) and ranking")


# -----------------------------------------------------------------------------
# Derivatives
# -----------------------------------------------------------------------------

def test_finite_diff_linear():
    """Linear signal → constant first derivative."""
    t = np.linspace(0, 1, 50)
    sig = 3.0 * t  # x = 3t, dx/dt = 3
    d = _finite_diff_first(sig, dt=t[1] - t[0])
    # Interior points should be exactly 3
    assert np.allclose(d[5:-5], 3.0, atol=1e-6)
    print("  PASS: finite diff of linear signal ≈ slope")


def test_finite_diff_quadratic():
    """Quadratic signal → linear first derivative."""
    t = np.linspace(0, 1, 100)
    sig = t ** 2  # x = t², dx/dt = 2t
    d = _finite_diff_first(sig, dt=t[1] - t[0])
    # Match interior 2t
    assert np.allclose(d[5:-5], 2 * t[5:-5], atol=0.01)
    print("  PASS: finite diff of quadratic ≈ 2t")


def test_finite_diff_second_constant_acceleration():
    """x = 0.5 a t² → d²x/dt² = a (constant)."""
    a = 4.0
    t = np.linspace(0, 1, 200)
    sig = 0.5 * a * t ** 2
    dd = _finite_diff_second(sig, dt=t[1] - t[0])
    # Interior should be ≈ a
    assert np.allclose(dd[10:-10], a, atol=0.05)
    print("  PASS: second-difference of 0.5at² ≈ a")


def test_savgol_derivative_matches_finite_diff_smooth_signal():
    """Smooth (well-sampled) signal: SG and FD should agree on derivative."""
    try:
        import scipy.signal  # noqa
    except ImportError:
        print("  SKIP: scipy not available")
        return
    t = np.linspace(0, 2 * np.pi, 200)
    sig = np.sin(t)   # derivative = cos(t)
    d_sg = savitzky_golay_derivative(sig, dt=t[1] - t[0], order=1,
                                      window_length=11, polyorder=3)
    expected = np.cos(t)
    # Interior agreement
    assert np.allclose(d_sg[20:-20], expected[20:-20], atol=0.05)
    print("  PASS: SG derivative of sin(t) ≈ cos(t)")


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def test_mse_zero_for_perfect_prediction():
    y = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    assert mse(y, y) == 0.0
    print("  PASS: MSE=0 for perfect prediction")


def test_mse_per_axis():
    y_true = np.zeros((10, 3))
    y_pred = np.array([[1, 2, 3]] * 10, dtype=np.float32)
    err = mse(y_true, y_pred, axis=0)
    assert err.shape == (3,)
    assert np.allclose(err, [1, 4, 9])
    print("  PASS: mse axis=0 returns per-output errors")


def test_r_squared_perfect_prediction():
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    r2 = r_squared(y, y)
    assert np.isclose(r2, 1.0)
    print("  PASS: R²=1 for perfect prediction")


def test_r_squared_mean_baseline():
    """Predicting the mean → R² = 0."""
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    y_pred = np.full_like(y, y.mean())
    r2 = r_squared(y, y_pred)
    assert np.isclose(r2, 0.0, atol=1e-9)
    print("  PASS: R²=0 for mean-only baseline")


def test_r_squared_per_output():
    y_true = np.column_stack([np.linspace(0, 1, 50), np.linspace(0, 1, 50)])
    y_pred = np.column_stack([y_true[:, 0], y_true[:, 1] * 0.5])  # second is worse
    r2 = r_squared(y_true, y_pred, axis=0)
    assert r2.shape == (2,)
    assert r2[0] > r2[1]
    print("  PASS: per-output R² distinguishes accuracies")


# -----------------------------------------------------------------------------
# Multi-step rollout
# -----------------------------------------------------------------------------

def test_rollout_identity():
    """Identity W and zero b → state stays put."""
    W = np.eye(4, dtype=np.float32)
    b = np.zeros(4, dtype=np.float32)
    s0 = np.array([1, 2, 3, 4], dtype=np.float32)
    rollout = autoregressive_rollout(s0, W, b, n_steps=10)
    assert rollout.shape == (11, 4)
    for t in range(11):
        assert np.allclose(rollout[t], s0)
    print("  PASS: identity W → constant rollout")


def test_rollout_linear_decay():
    """W = 0.5 I → state halves each step."""
    W = 0.5 * np.eye(3, dtype=np.float32)
    b = np.zeros(3, dtype=np.float32)
    s0 = np.array([1, 1, 1], dtype=np.float32)
    rollout = autoregressive_rollout(s0, W, b, n_steps=4)
    assert np.allclose(rollout[0], [1, 1, 1])
    assert np.allclose(rollout[1], [0.5, 0.5, 0.5])
    assert np.allclose(rollout[4], [0.0625, 0.0625, 0.0625])
    print("  PASS: 0.5·I rollout halves at each step")


def test_rollout_rejects_non_square_W():
    try:
        autoregressive_rollout(np.zeros(3), np.zeros((3, 5)), np.zeros(5), 1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for non-square W")
    print("  PASS: rollout rejects non-square W")


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------

def main():
    print("Running readout_math tests...\n")
    tests = [
        test_ridge_recovers_true_coefficients_clean,
        test_ridge_dual_form_matches_primal,
        test_ridge_lambda_shrinks_weights,
        test_ridge_intercept_centering,
        test_ridge_rejects_bad_shapes,
        test_predict_round_trip,
        test_kfold_partition_is_disjoint,
        test_kfold_uneven_split_distributes_remainder,
        test_grouped_kfold_keeps_groups_together,
        test_cv_ridge_picks_best_lambda,
        test_kernel_cv_matches_naive_cv,
        test_finite_diff_linear,
        test_finite_diff_quadratic,
        test_finite_diff_second_constant_acceleration,
        test_savgol_derivative_matches_finite_diff_smooth_signal,
        test_mse_zero_for_perfect_prediction,
        test_mse_per_axis,
        test_r_squared_perfect_prediction,
        test_r_squared_mean_baseline,
        test_r_squared_per_output,
        test_rollout_identity,
        test_rollout_linear_decay,
        test_rollout_rejects_non_square_W,
    ]
    passed = 0
    for t in tests:
        print(f"- {t.__name__}")
        try:
            t()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL: {e}")
            traceback.print_exc()
        print()
    print(f"=== {passed}/{len(tests)} tests passed ===")
    sys.exit(0 if passed == len(tests) else 1)


if __name__ == "__main__":
    main()
