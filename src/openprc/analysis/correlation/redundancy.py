"""
Redundancy analysis for Physical Reservoir Computing.

This module provides the ``Redundancy`` class, which quantifies the
internal correlation structure of a reservoir state matrix *y* — without
reference to any input signal.  It answers: "how many truly independent
degrees of freedom does this reservoir have?"

Classes
-------
Redundancy
    Pairwise correlation, partial correlation, ACF, effective rank,
    condition number, and channel clustering.

Example
-------
>>> from openprc.analysis import correlation as corr
>>> red = corr.Redundancy(y, channel_names=names)
>>> red.correlation        # N×N Pearson matrix
>>> red.partial            # N×N partial correlation
>>> red.acf                # per-channel autocorrelation
>>> red.rank               # effective rank
>>> red.condition          # condition number
>>> red.groups(threshold=0.9)  # cluster redundant channels
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
from scipy import stats

from .result import Result

# ═══════════════════════════════════════════════════════════════════════════
#  Input validation
# ═══════════════════════════════════════════════════════════════════════════

def _validate_y(y: np.ndarray):
    """Validate and coerce y into canonical (T, N) shape.

    Returns
    -------
    y : np.ndarray, shape (T, N)
    """
    y = np.asarray(y, dtype=np.float64)

    if y.ndim == 1:
        y = y[:, np.newaxis]
    elif y.ndim != 2:
        raise ValueError(f"y must be 1-D or 2-D, got ndim={y.ndim}.")

    T, N = y.shape
    if T < 3:
        raise ValueError(f"Need at least 3 time samples, got {T}.")
    if N < 2:
        raise ValueError(
            f"Redundancy analysis requires at least 2 channels, got {N}."
        )

    return y


# ═══════════════════════════════════════════════════════════════════════════
#  Computation helpers
# ═══════════════════════════════════════════════════════════════════════════

def _pairwise_pearson(y: np.ndarray):
    """Full N×N Pearson correlation matrix with p-values.

    Returns
    -------
    r_matrix : (N, N)
    p_matrix : (N, N)
    """
    N = y.shape[1]
    r_matrix = np.corrcoef(y, rowvar=False)  # fast path
    p_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(i + 1, N):
            _, p = stats.pearsonr(y[:, i], y[:, j])
            p_matrix[i, j] = p
            p_matrix[j, i] = p
    # diagonal p = 0 (perfect self-correlation)
    return r_matrix, p_matrix


def _partial_correlation_matrix(y: np.ndarray):
    r"""Partial correlation via precision matrix.

    .. math::

        \rho_{ij \cdot \text{rest}} =
        -\frac{P_{ij}}{\sqrt{P_{ii}\,P_{jj}}}

    Returns
    -------
    pcorr : (N, N)
    """
    C = np.cov(y, rowvar=False)
    try:
        P = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        warnings.warn(
            "Covariance matrix is singular; using pseudo-inverse for "
            "partial correlation. Results may be unreliable.",
            UserWarning,
            stacklevel=3,
        )
        P = np.linalg.pinv(C)

    diag = np.sqrt(np.abs(np.diag(P)))
    outer = np.outer(diag, diag)
    outer[outer == 0] = 1.0
    pcorr = -P / outer
    np.fill_diagonal(pcorr, 1.0)
    return pcorr


def _acf_single(y_col: np.ndarray, n_lags: int):
    """FFT-based auto-correlation of a single channel.

    Returns
    -------
    acf : (n_lags + 1,)
    """
    ym = y_col - y_col.mean()
    T = len(ym)
    fft_len = 2 ** int(np.ceil(np.log2(2 * T - 1)))
    F = np.fft.rfft(ym, n=fft_len)
    acf_full = np.fft.irfft(F * np.conj(F), n=fft_len)[:T]
    if acf_full[0] > 0:
        acf_full /= acf_full[0]
    return acf_full[: n_lags + 1]


def _effective_rank(corr_matrix: np.ndarray):
    """Effective rank via Shannon entropy of normalised eigenvalues.

    Defined as:

    .. math::

        r_{\\text{eff}} = \\exp\\left(
            -\\sum_i \\hat{\\lambda}_i \\ln \\hat{\\lambda}_i
        \\right)

    where :math:`\\hat{\\lambda}_i = \\lambda_i / \\sum_j \\lambda_j`.

    This is a continuous measure: equals N when all eigenvalues are
    equal (maximally diverse reservoir), equals 1 when a single
    eigenvalue dominates (fully redundant).

    Returns
    -------
    eff_rank : float
    eigenvalues : (N,) sorted descending
    """
    eigvals = np.linalg.eigvalsh(corr_matrix)
    # numerical cleanup — clamp small negatives to zero
    eigvals = np.maximum(eigvals, 0.0)
    eigvals = np.sort(eigvals)[::-1]

    total = eigvals.sum()
    if total < 1e-15:
        return 1.0, eigvals

    p = eigvals / total
    # avoid log(0)
    p_safe = p[p > 1e-15]
    entropy = -np.sum(p_safe * np.log(p_safe))
    eff_rank = np.exp(entropy)

    return float(eff_rank), eigvals


def _channel_groups(corr_matrix: np.ndarray, threshold: float):
    """Cluster channels by greedy agglomeration on |correlation|.

    Two channels are in the same group if there exists a path of
    pairwise |r| >= threshold between them (single-linkage).

    Returns
    -------
    groups : list[list[int]]
        Each inner list contains channel indices in one cluster.
    """
    N = corr_matrix.shape[0]
    visited = [False] * N
    groups = []

    for i in range(N):
        if visited[i]:
            continue
        # BFS from channel i
        group = []
        queue = [i]
        while queue:
            node = queue.pop(0)
            if visited[node]:
                continue
            visited[node] = True
            group.append(node)
            for j in range(N):
                if not visited[j] and abs(corr_matrix[node, j]) >= threshold:
                    queue.append(j)
        groups.append(sorted(group))

    return groups


# ═══════════════════════════════════════════════════════════════════════════
#  Redundancy class
# ═══════════════════════════════════════════════════════════════════════════

class Redundancy:
    """Correlation-based redundancy diagnostics for reservoir state matrix *y*.

    Parameters
    ----------
    y : array-like, shape (T, N)
        Reservoir state matrix (*N* channels / probes, *T* time steps).
    max_lag : int, optional
        Maximum lag for ACF computation.  Defaults to ``T // 4``.
    channel_names : list[str], optional
        Human-readable labels for the *N* channels.

    Attributes (all lazily computed on first access)
    -------------------------------------------------
    correlation : Result
        N×N Pearson correlation matrix with p-values.
    partial : Result
        N×N partial correlation matrix (direct connections only).
    acf : Result
        Per-channel auto-correlation, shape ``(N, n_lags+1)``.
    rank : float
        Effective rank (Shannon entropy of eigenvalue spectrum).
        Equals N for maximally diverse reservoir, 1 for fully redundant.
    condition : float
        Condition number of the correlation matrix.
        High values indicate near-singular (redundant) reservoirs.
    eigenvalues : np.ndarray
        Eigenvalues of the correlation matrix, sorted descending.

    Methods
    -------
    groups(threshold=0.9) -> list[list[int]]
        Cluster channels with |r| >= threshold (single-linkage).

    Example
    -------
    >>> red = corr.Redundancy(y, channel_names=names)
    >>> print(f"Effective rank: {red.rank:.1f} / {N}")
    >>> print(f"Condition number: {red.condition:.1f}")
    >>> for g in red.groups(threshold=0.85):
    ...     print([names[i] for i in g])
    """

    def __init__(
            self,
            y,
            max_lag: Optional[int] = None,
            channel_names: Optional[Sequence[str]] = None,
    ):
        self._y = _validate_y(y)
        self._T, self._N = self._y.shape
        self._max_lag = max_lag if max_lag is not None else self._T // 4
        self._channel_names = (
            list(channel_names) if channel_names is not None else None
        )

        # lazy caches
        self._correlation: Optional[Result] = None
        self._partial: Optional[Result] = None
        self._acf: Optional[Result] = None
        self._rank: Optional[float] = None
        self._condition: Optional[float] = None
        self._eigenvalues: Optional[np.ndarray] = None

    # ── properties (lazy) ──────────────────────────────────────────────

    @property
    def correlation(self) -> Result:
        """N×N Pearson correlation matrix of reservoir channels."""
        if self._correlation is None:
            r_mat, p_mat = _pairwise_pearson(self._y)
            self._correlation = Result(
                metric="correlation",
                values=r_mat,
                p_values=p_mat,
                channel_names=self._channel_names,
            )
        return self._correlation

    @property
    def partial(self) -> Result:
        r"""N×N partial correlation matrix.

        Quantifies the direct linear relationship between channels
        *i* and *j* after removing the influence of all other channels.
        Low partial correlation + high pairwise correlation means the
        connection is mediated through other channels (indirect redundancy).
        """
        if self._partial is None:
            pcorr = _partial_correlation_matrix(self._y)
            self._partial = Result(
                metric="partial_corr",
                values=pcorr,
                channel_names=self._channel_names,
            )
        return self._partial

    @property
    def acf(self) -> Result:
        """Per-channel auto-correlation, shape ``(N, n_lags+1)``."""
        if self._acf is None:
            n_lags = self._max_lag
            acf_mat = np.empty((self._N, n_lags + 1))
            for i in range(self._N):
                acf_mat[i] = _acf_single(self._y[:, i], n_lags)
            self._acf = Result(
                metric="acf",
                values=acf_mat,
                channel_names=self._channel_names,
                extra={"lag_axis": np.arange(n_lags + 1)},
            )
        return self._acf

    def _compute_eigen(self):
        """Compute eigenvalues, effective rank, and condition number."""
        if self._eigenvalues is None:
            r_mat = self.correlation.values
            self._rank, self._eigenvalues = _effective_rank(r_mat)
            # condition number from eigenvalues
            pos = self._eigenvalues[self._eigenvalues > 1e-15]
            if len(pos) > 0:
                self._condition = float(pos[0] / pos[-1])
            else:
                self._condition = float("inf")

    @property
    def rank(self) -> float:
        """Effective rank (Shannon entropy of eigenvalue spectrum).

        Equals N for maximally diverse reservoir, 1 for fully redundant.
        """
        self._compute_eigen()
        return self._rank

    @property
    def condition(self) -> float:
        """Condition number of the correlation matrix.

        High values indicate near-singular (ill-conditioned) reservoirs
        where some channels are nearly linear combinations of others.
        """
        self._compute_eigen()
        return self._condition

    @property
    def eigenvalues(self) -> np.ndarray:
        """Eigenvalues of the correlation matrix, sorted descending."""
        self._compute_eigen()
        return self._eigenvalues

    # ── methods ────────────────────────────────────────────────────────

    def groups(self, threshold: float = 0.9) -> list[list[int]]:
        """Cluster channels with |r| >= threshold.

        Uses single-linkage: channels are grouped if there exists a
        path of pairwise |r| >= threshold between them.

        Parameters
        ----------
        threshold : float
            Minimum |correlation| to consider two channels redundant.

        Returns
        -------
        list[list[int]]
            Each inner list contains channel indices.  Singletons
            indicate unique, non-redundant channels.
        """
        return _channel_groups(self.correlation.values, threshold)

    def groups_named(self, threshold: float = 0.9) -> list[list[str]]:
        """Like ``groups()`` but returns channel names instead of indices.

        Requires ``channel_names`` to have been provided at init.
        """
        names = self._channel_names
        if names is None:
            names = [f"ch_{i}" for i in range(self._N)]
        idx_groups = self.groups(threshold)
        return [[names[i] for i in g] for g in idx_groups]

    # ── summary ────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Quick diagnostic string."""
        r_mat = self.correlation.values
        # off-diagonal mean |r|
        mask = ~np.eye(self._N, dtype=bool)
        mean_abs_r = np.mean(np.abs(r_mat[mask]))
        max_abs_r = np.max(np.abs(r_mat[mask]))

        lines = [
            f"Redundancy Analysis  |  T={self._T}, N={self._N}",
            f"  Pairwise |r|   mean={mean_abs_r:.4f}  max={max_abs_r:.4f}",
            f"  Effective rank  {self.rank:.2f} / {self._N}  "
            f"({self.rank / self._N * 100:.1f}% utilisation)",
            f"  Condition #     {self.condition:.1f}",
        ]

        grp = self.groups(threshold=0.9)
        n_groups = len(grp)
        n_singletons = sum(1 for g in grp if len(g) == 1)
        lines.append(
            f"  Groups (|r|≥0.9)  {n_groups} clusters, "
            f"{n_singletons} unique channels"
        )

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Redundancy(T={self._T}, N={self._N})"