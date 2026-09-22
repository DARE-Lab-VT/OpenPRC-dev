"""
Nonparametric correlation analysis for Physical Reservoir Computing.

This module provides the ``Nonparametric`` class, which quantifies
monotonic, nonlinear, and kernel-based dependencies between an input
signal *x* and a reservoir state matrix *y* — relationships that
linear metrics like Pearson completely miss.

Classes
-------
Nonparametric
    Spearman, Kendall, Distance Correlation (dCor), and HSIC.

Example
-------
>>> from openprc.analysis import correlation as corr
>>> nr = corr.Nonparametric(x, y)
>>> nr.spearman         # monotonic rank correlation
>>> nr.kendall          # ordinal association
>>> nr.dcor             # global nonlinear dependency (0 ⟺ independent)
>>> nr.hsic()           # kernel-based independence test
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
from scipy import stats

from .result import Result


# ═══════════════════════════════════════════════════════════════════════════
#  Input validation (shared with linear — duplicated here to keep modules
#  independently importable; a future refactor can extract to _utils.py)
# ═══════════════════════════════════════════════════════════════════════════

def _validate_inputs(x: np.ndarray, y: np.ndarray):
    """Validate and coerce x, y into canonical shapes.

    Returns
    -------
    x : np.ndarray, shape (T,)
    y : np.ndarray, shape (T, N)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim == 2:
        if x.shape[1] > 1:
            warnings.warn(
                f"Multi-dimensional input x with shape {x.shape} detected. "
                f"Using x[:, 0] (first column). Multi-variate x support is "
                f"planned for a future release.",
                UserWarning,
                stacklevel=3,
            )
        x = x[:, 0]
    elif x.ndim != 1:
        raise ValueError(f"x must be 1-D or 2-D, got ndim={x.ndim}.")

    if y.ndim == 1:
        y = y[:, np.newaxis]
    elif y.ndim != 2:
        raise ValueError(f"y must be 1-D or 2-D, got ndim={y.ndim}.")

    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"Time dimension mismatch: x has {x.shape[0]} samples, "
            f"y has {y.shape[0]} samples."
        )

    if x.shape[0] < 3:
        raise ValueError(
            f"Need at least 3 time samples, got {x.shape[0]}."
        )

    return x, y


# ═══════════════════════════════════════════════════════════════════════════
#  Computation helpers — Spearman / Kendall (delegate to scipy)
# ═══════════════════════════════════════════════════════════════════════════

def _spearman_per_channel(x: np.ndarray, y: np.ndarray):
    """Spearman rho and p-value for x vs each column of y.

    Returns
    -------
    rho : (N,)
    p   : (N,)
    """
    N = y.shape[1]
    rho = np.empty(N)
    p = np.empty(N)
    for i in range(N):
        rho[i], p[i] = stats.spearmanr(x, y[:, i])
    return rho, p


def _kendall_per_channel(x: np.ndarray, y: np.ndarray):
    """Kendall tau-b and p-value for x vs each column of y.

    Returns
    -------
    tau : (N,)
    p   : (N,)
    """
    N = y.shape[1]
    tau = np.empty(N)
    p = np.empty(N)
    for i in range(N):
        tau[i], p[i] = stats.kendalltau(x, y[:, i])
    return tau, p


# ═══════════════════════════════════════════════════════════════════════════
#  Distance Correlation  (self-contained, no external dependency)
#
#  References
#  ----------
#  Székely, G. J., Rizzo, M. L., & Bakirov, N. K. (2007).
#  "Measuring and testing dependence by correlation of distances."
#  Annals of Statistics, 35(6), 2769–2794.
# ═══════════════════════════════════════════════════════════════════════════

def _dcov_components(x_col: np.ndarray, y_col: np.ndarray):
    """Compute distance covariance components for two 1-D vectors.

    Uses the O(n²) doubly-centred distance matrix formulation.

    Returns
    -------
    dcov2_xy : float   squared distance covariance
    dcov2_xx : float   squared distance variance of x
    dcov2_yy : float   squared distance variance of y
    """
    n = len(x_col)
    # pairwise distance matrices
    a = np.abs(x_col[:, None] - x_col[None, :])
    b = np.abs(y_col[:, None] - y_col[None, :])

    # double centring
    a_row = a.mean(axis=1, keepdims=True)
    a_col = a.mean(axis=0, keepdims=True)
    a_grand = a.mean()
    A = a - a_row - a_col + a_grand

    b_row = b.mean(axis=1, keepdims=True)
    b_col = b.mean(axis=0, keepdims=True)
    b_grand = b.mean()
    B = b - b_row - b_col + b_grand

    dcov2_xy = (A * B).mean()
    dcov2_xx = (A * A).mean()
    dcov2_yy = (B * B).mean()

    return dcov2_xy, dcov2_xx, dcov2_yy


def _dcor_single(x_col: np.ndarray, y_col: np.ndarray):
    """Distance correlation between two 1-D vectors.

    Returns
    -------
    dcor : float   in [0, 1]; 0 iff independent.
    """
    dcov2_xy, dcov2_xx, dcov2_yy = _dcov_components(x_col, y_col)
    denom = np.sqrt(dcov2_xx * dcov2_yy)
    if denom < 1e-15:
        return 0.0
    dcov_xy = np.sqrt(max(dcov2_xy, 0.0))
    return dcov_xy / np.sqrt(denom)


def _dcor_per_channel(x: np.ndarray, y: np.ndarray):
    """Distance correlation for x vs each column of y.

    Returns
    -------
    dcor_vals : (N,)
    """
    N = y.shape[1]
    dcor_vals = np.empty(N)
    for i in range(N):
        dcor_vals[i] = _dcor_single(x, y[:, i])
    return dcor_vals


def _dcor_permutation_test(
        x: np.ndarray,
        y_col: np.ndarray,
        n_permutations: int = 499,
        rng: Optional[np.random.Generator] = None,
):
    """Permutation p-value for distance correlation.

    Parameters
    ----------
    n_permutations : int
        Number of random shuffles (default 499 → resolution 1/500).
    rng : np.random.Generator, optional

    Returns
    -------
    p_value : float
    """
    if rng is None:
        rng = np.random.default_rng()
    observed = _dcor_single(x, y_col)
    count = 0
    for _ in range(n_permutations):
        x_perm = rng.permutation(x)
        if _dcor_single(x_perm, y_col) >= observed:
            count += 1
    return (count + 1) / (n_permutations + 1)


# ═══════════════════════════════════════════════════════════════════════════
#  HSIC — Hilbert-Schmidt Independence Criterion
#
#  References
#  ----------
#  Gretton, A., Bousquet, O., Smola, A., & Schölkopf, B. (2005).
#  "Measuring Statistical Dependence with Hilbert-Schmidt Norms."
#  ALT 2005, LNAI 3734, pp. 63–77.
# ═══════════════════════════════════════════════════════════════════════════

def _rbf_kernel_matrix(v: np.ndarray, sigma: Optional[float] = None):
    """RBF (Gaussian) kernel matrix for a 1-D vector.

    Parameters
    ----------
    v : (n,)
    sigma : float, optional
        Bandwidth.  Defaults to the median heuristic.

    Returns
    -------
    K : (n, n)
    """
    dists = (v[:, None] - v[None, :]) ** 2
    if sigma is None:
        # median heuristic
        med = np.median(np.sqrt(dists[np.triu_indices_from(dists, k=1)]))
        sigma = med if med > 1e-10 else 1.0
    return np.exp(-dists / (2.0 * sigma ** 2))


def _hsic_single(
        x_col: np.ndarray,
        y_col: np.ndarray,
        sigma_x: Optional[float] = None,
        sigma_y: Optional[float] = None,
):
    """Biased HSIC estimator between two 1-D vectors.

    Returns
    -------
    hsic : float
    """
    n = len(x_col)
    K = _rbf_kernel_matrix(x_col, sigma_x)
    L = _rbf_kernel_matrix(y_col, sigma_y)

    # centring matrix H = I - (1/n) 11^T
    H = np.eye(n) - np.ones((n, n)) / n
    # HSIC = (1/n²) trace(KHLH)
    KH = K @ H
    LH = L @ H
    hsic = np.trace(KH @ LH) / (n ** 2)
    return float(hsic)


def _hsic_per_channel(
        x: np.ndarray,
        y: np.ndarray,
        sigma_x: Optional[float] = None,
        sigma_y: Optional[float] = None,
):
    """HSIC for x vs each column of y.

    Returns
    -------
    hsic_vals : (N,)
    """
    N = y.shape[1]
    hsic_vals = np.empty(N)
    for i in range(N):
        hsic_vals[i] = _hsic_single(x, y[:, i], sigma_x, sigma_y)
    return hsic_vals


def _hsic_permutation_test(
        x: np.ndarray,
        y_col: np.ndarray,
        sigma_x: Optional[float] = None,
        sigma_y: Optional[float] = None,
        n_permutations: int = 499,
        rng: Optional[np.random.Generator] = None,
):
    """Permutation p-value for HSIC.

    Returns
    -------
    p_value : float
    """
    if rng is None:
        rng = np.random.default_rng()
    observed = _hsic_single(x, y_col, sigma_x, sigma_y)
    count = 0
    for _ in range(n_permutations):
        x_perm = rng.permutation(x)
        if _hsic_single(x_perm, y_col, sigma_x, sigma_y) >= observed:
            count += 1
    return (count + 1) / (n_permutations + 1)


# ═══════════════════════════════════════════════════════════════════════════
#  Nonparametric class
# ═══════════════════════════════════════════════════════════════════════════

class Nonparametric:
    """Nonparametric correlation diagnostics between input *x* and reservoir *y*.

    Parameters
    ----------
    x : array-like, shape (T,) or (T, d)
        Input signal.  If *d > 1*, only ``x[:, 0]`` is used (with a
        warning).
    y : array-like, shape (T, N)
        Reservoir state matrix (*N* channels / probes).
    channel_names : list[str], optional
        Human-readable labels for the *N* channels.

    Attributes (eagerly / lazily computed)
    --------------------------------------
    spearman : Result  — *eager*
        Spearman rank correlation per channel.
    kendall : Result  — *eager*
        Kendall tau-b per channel.
    dcor : Result  — *eager*
        Distance correlation per channel (no p-values by default;
        call ``.dcor_test()`` for permutation p-values).

    Methods (lazy / explicit)
    -------------------------
    dcor_test(n_permutations=499) -> Result
        dCor with permutation-based p-values.
    hsic(sigma_x=None, sigma_y=None) -> Result
        HSIC with median-heuristic bandwidth (lazy).
    hsic_test(n_permutations=499, ...) -> Result
        HSIC with permutation p-values.
    """

    def __init__(
            self,
            x,
            y,
            channel_names: Optional[Sequence[str]] = None,
    ):
        self._x, self._y = _validate_inputs(x, y)
        self._T, self._N = self._y.shape
        self._channel_names = (
            list(channel_names) if channel_names is not None else None
        )

        # lazy caches
        self._spearman: Optional[Result] = None
        self._kendall: Optional[Result] = None
        self._dcor: Optional[Result] = None

    # ── eager properties ───────────────────────────────────────────────

    @property
    def spearman(self) -> Result:
        """Spearman rank correlation per channel."""
        if self._spearman is None:
            rho, p = _spearman_per_channel(self._x, self._y)
            self._spearman = Result(
                metric="spearman",
                values=rho,
                p_values=p,
                channel_names=self._channel_names,
            )
        return self._spearman

    @property
    def kendall(self) -> Result:
        """Kendall tau-b per channel."""
        if self._kendall is None:
            tau, p = _kendall_per_channel(self._x, self._y)
            self._kendall = Result(
                metric="kendall",
                values=tau,
                p_values=p,
                channel_names=self._channel_names,
            )
        return self._kendall

    @property
    def dcor(self) -> Result:
        """Distance correlation per channel (no p-values).

        dCor is zero if and only if x and y_i are statistically
        independent.  Detects arbitrary nonlinear dependencies.

        For p-values, use ``.dcor_test()``.

        Notes
        -----
        Complexity is O(T² · N).  For T > 10 000 this can be slow;
        consider sub-sampling.
        """
        if self._dcor is None:
            vals = _dcor_per_channel(self._x, self._y)
            self._dcor = Result(
                metric="dcor",
                values=vals,
                channel_names=self._channel_names,
            )
        return self._dcor

    # ── explicit methods (expensive / parameterised) ───────────────────

    def dcor_test(
            self,
            n_permutations: int = 499,
            seed: Optional[int] = None,
    ) -> Result:
        """Distance correlation with permutation-based p-values.

        Parameters
        ----------
        n_permutations : int
            Number of shuffles per channel (default 499).
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Result
            ``.values`` = dCor, ``.p_values`` = permutation p-values.
        """
        rng = np.random.default_rng(seed)
        vals = self.dcor.values.copy()
        p = np.empty(self._N)
        for i in range(self._N):
            p[i] = _dcor_permutation_test(
                self._x, self._y[:, i],
                n_permutations=n_permutations, rng=rng,
            )
        return Result(
            metric="dcor",
            values=vals,
            p_values=p,
            channel_names=self._channel_names,
        )

    def hsic(
            self,
            sigma_x: Optional[float] = None,
            sigma_y: Optional[float] = None,
    ) -> Result:
        """HSIC per channel (no p-values).

        Uses RBF kernels with median-heuristic bandwidth by default.

        Parameters
        ----------
        sigma_x, sigma_y : float, optional
            Kernel bandwidth overrides.  ``None`` → median heuristic.

        Returns
        -------
        Result
        """
        vals = _hsic_per_channel(self._x, self._y, sigma_x, sigma_y)
        return Result(
            metric="hsic",
            values=vals,
            channel_names=self._channel_names,
        )

    def hsic_test(
            self,
            sigma_x: Optional[float] = None,
            sigma_y: Optional[float] = None,
            n_permutations: int = 499,
            seed: Optional[int] = None,
    ) -> Result:
        """HSIC with permutation-based p-values.

        Parameters
        ----------
        sigma_x, sigma_y : float, optional
            Kernel bandwidth overrides.
        n_permutations : int
            Number of shuffles per channel (default 499).
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Result
            ``.values`` = HSIC scores, ``.p_values`` = permutation p-values.
        """
        rng = np.random.default_rng(seed)
        vals = _hsic_per_channel(self._x, self._y, sigma_x, sigma_y)
        p = np.empty(self._N)
        for i in range(self._N):
            p[i] = _hsic_permutation_test(
                self._x, self._y[:, i],
                sigma_x=sigma_x, sigma_y=sigma_y,
                n_permutations=n_permutations, rng=rng,
            )
        return Result(
            metric="hsic",
            values=vals,
            p_values=p,
            channel_names=self._channel_names,
        )

    # ── summary ────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Quick diagnostic string."""
        sp = self.spearman
        dc = self.dcor
        lines = [
            f"Nonparametric Analysis  |  T={self._T}, N={self._N}",
            f"  Spearman  mean|ρ|={np.mean(np.abs(sp.values)):.4f}  "
            f"max|ρ|={np.max(np.abs(sp.values)):.4f}  "
            f"significant={int(np.sum(sp.p_values < 0.05))}/{self._N}",
            f"  dCor     mean={np.mean(dc.values):.4f}  "
            f"max={np.max(dc.values):.4f}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Nonparametric(T={self._T}, N={self._N})"
