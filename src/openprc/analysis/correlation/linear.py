"""
Linear correlation analysis for Physical Reservoir Computing.

This module provides the ``Linear`` class, which quantifies straight-line
and time-lagged linear relationships between an input signal *x* and a
reservoir state matrix *y*.

Classes
-------
Linear
    Pearson, CCF (with optional lag sweep), partial correlation, ACF, and CCA.

Example
-------
>>> from openprc.analysis import correlation as corr
>>> res = corr.Linear(x, y, lag_sweep=True)
>>> res.pearson          # zero-lag Pearson r per channel
>>> res.peak             # best-lag Pearson r per channel
>>> res.ccf              # full cross-correlation profiles
>>> res.partial          # partial correlation matrix (y–y)
>>> res.acf              # auto-correlation of each y channel
>>> res.cca(n_components=3)  # canonical correlation analysis
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
from scipy import signal, stats

from .result import CCAResult, Result

# ═══════════════════════════════════════════════════════════════════════════
#  Input validation
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

    # Handle multi-dimensional x
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
#  Computation helpers (pure functions, no state)
# ═══════════════════════════════════════════════════════════════════════════

def _pearson_per_channel(x: np.ndarray, y: np.ndarray):
    """Pearson r and p-value for x vs each column of y (zero-lag).

    Returns
    -------
    r_values : (N,)
    p_values : (N,)
    """
    N = y.shape[1]
    r_values = np.empty(N)
    p_values = np.empty(N)
    for i in range(N):
        r_values[i], p_values[i] = stats.pearsonr(x, y[:, i])
    return r_values, p_values


def _ccf_single(x: np.ndarray, y_col: np.ndarray, max_lag: int):
    """Normalised cross-correlation between x and a single y channel.

    Uses ``scipy.signal.correlate`` with ``"full"`` mode, then normalises
    by the geometric mean of the energies (so the output is bounded by
    [-1, 1] at zero shift).

    Returns
    -------
    ccf : (2*max_lag + 1,)  cross-correlation values
    lags : (2*max_lag + 1,) corresponding lag indices
    """
    # zero-mean
    xm = x - x.mean()
    ym = y_col - y_col.mean()

    full = signal.correlate(xm, ym, mode="full")
    norm = np.sqrt(np.sum(xm ** 2) * np.sum(ym ** 2))
    if norm > 0:
        full /= norm

    T = len(x)
    mid = T - 1  # index of zero-lag in the full output
    lo = max(mid - max_lag, 0)
    hi = min(mid + max_lag + 1, len(full))
    ccf = full[lo:hi]
    lags = np.arange(lo - mid, hi - mid)
    return ccf, lags


def _ccf_all_channels(x: np.ndarray, y: np.ndarray, max_lag: int):
    """CCF for every channel, stacked.

    Returns
    -------
    ccf_matrix : (N, 2*max_lag+1)
    lag_axis   : (2*max_lag+1,)
    peak_vals  : (N,)   peak |CCF| value per channel
    peak_lags  : (N,)   lag at peak |CCF| per channel
    """
    N = y.shape[1]
    # compute first to get actual lag length (edge handling)
    ccf0, lag_axis = _ccf_single(x, y[:, 0], max_lag)
    L = len(lag_axis)

    ccf_matrix = np.empty((N, L))
    ccf_matrix[0] = ccf0

    for i in range(1, N):
        ccf_matrix[i], _ = _ccf_single(x, y[:, i], max_lag)

    # peak detection (by absolute value)
    peak_idx = np.argmax(np.abs(ccf_matrix), axis=1)
    peak_lags = lag_axis[peak_idx]
    peak_vals = ccf_matrix[np.arange(N), peak_idx]

    return ccf_matrix, lag_axis, peak_vals, peak_lags


def _pearson_at_lags(
        x: np.ndarray, y: np.ndarray, lags: np.ndarray
):
    """Pearson r between x(t) and y_i(t + lag_i) for per-channel lags.

    Returns
    -------
    r_values : (N,)
    p_values : (N,)
    """
    T = x.shape[0]
    N = y.shape[1]
    r_values = np.empty(N)
    p_values = np.empty(N)

    for i in range(N):
        lag = int(lags[i])
        if lag >= 0:
            x_seg = x[:T - lag] if lag > 0 else x
            y_seg = y[lag:, i]
        else:
            alag = -lag
            x_seg = x[alag:]
            y_seg = y[:T - alag, i]

        if len(x_seg) < 3:
            r_values[i] = np.nan
            p_values[i] = np.nan
        else:
            r_values[i], p_values[i] = stats.pearsonr(x_seg, y_seg)

    return r_values, p_values


def _partial_correlation_matrix(y: np.ndarray):
    r"""Partial correlation matrix of reservoir channels.

    Uses the precision-matrix (inverse covariance) method:

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
        # singular covariance — fall back to pseudo-inverse
        warnings.warn(
            "Covariance matrix is singular; using pseudo-inverse for "
            "partial correlation. Results may be unreliable.",
            UserWarning,
            stacklevel=3,
        )
        P = np.linalg.pinv(C)

    diag = np.sqrt(np.abs(np.diag(P)))
    outer = np.outer(diag, diag)
    # avoid division by zero
    outer[outer == 0] = 1.0
    pcorr = -P / outer
    np.fill_diagonal(pcorr, 1.0)
    return pcorr


def _acf_single(y_col: np.ndarray, n_lags: int):
    """Auto-correlation of a single channel via FFT.

    Returns
    -------
    acf : (n_lags + 1,)   normalised ACF from lag 0 to n_lags.
    """
    ym = y_col - y_col.mean()
    T = len(ym)
    # FFT-based auto-correlation (efficient for large T)
    fft_len = 2 ** int(np.ceil(np.log2(2 * T - 1)))
    F = np.fft.rfft(ym, n=fft_len)
    acf_full = np.fft.irfft(F * np.conj(F), n=fft_len)[:T]
    if acf_full[0] > 0:
        acf_full /= acf_full[0]
    return acf_full[: n_lags + 1]


# ═══════════════════════════════════════════════════════════════════════════
#  Linear class
# ═══════════════════════════════════════════════════════════════════════════

class Linear:
    """Linear correlation diagnostics between input *x* and reservoir *y*.

    Parameters
    ----------
    x : array-like, shape (T,) or (T, d)
        Input signal.  If *d > 1*, only ``x[:, 0]`` is used (with a
        warning).
    y : array-like, shape (T, N)
        Reservoir state matrix (*N* channels / probes).
    max_lag : int, optional
        Maximum lag (in samples) for CCF and lag-sweep analyses.  Defaults
        to ``T // 4``.
    lag_sweep : bool, optional
        If ``True``, ``pearson`` is automatically computed at the
        lag that maximises |CCF| per channel (accessible via ``.peak``).
        Default is ``False``.
    channel_names : list[str], optional
        Human-readable labels for the *N* channels.  Propagated to every
        ``Result`` object.

    Attributes (all lazily computed on first access)
    -------------------------------------------------
    pearson : Result
        Zero-lag Pearson *r* per channel, shape ``(N,)``.
    ccf : Result
        Full cross-correlation profiles with lag axis.
    peak : Result
        Peak-|CCF| Pearson *r* per channel (requires ``lag_sweep=True``
        or calls CCF internally).
    partial : Result
        Partial correlation matrix of *y*, shape ``(N, N)``.
    acf : Result
        Auto-correlation per channel, shape ``(N, n_lags+1)``.

    Methods
    -------
    cca(n_components=None) -> CCAResult
        Canonical Correlation Analysis between *x* and *y*.
    """

    def __init__(
            self,
            x,
            y,
            max_lag: Optional[int] = None,
            lag_sweep: bool = False,
            channel_names: Optional[Sequence[str]] = None,
    ):
        self._x, self._y = _validate_inputs(x, y)
        self._T, self._N = self._y.shape
        self._max_lag = max_lag if max_lag is not None else self._T // 4
        self._lag_sweep = lag_sweep
        self._channel_names = (
            list(channel_names) if channel_names is not None else None
        )

        # lazy caches
        self._pearson: Optional[Result] = None
        self._ccf: Optional[Result] = None
        self._peak: Optional[Result] = None
        self._partial: Optional[Result] = None
        self._acf: Optional[Result] = None

    # ── properties (lazy) ──────────────────────────────────────────────

    @property
    def pearson(self) -> Result:
        """Zero-lag Pearson *r* per channel."""
        if self._pearson is None:
            r, p = _pearson_per_channel(self._x, self._y)
            self._pearson = Result(
                metric="pearson",
                values=r,
                p_values=p,
                channel_names=self._channel_names,
            )
        return self._pearson

    @property
    def ccf(self) -> Result:
        """Full cross-correlation profiles (lag sweep)."""
        if self._ccf is None:
            ccf_mat, lag_axis, peak_vals, peak_lags = _ccf_all_channels(
                self._x, self._y, self._max_lag
            )
            self._ccf = Result(
                metric="ccf",
                values=peak_vals,
                lags=peak_lags,
                channel_names=self._channel_names,
                extra={
                    "ccf_full": ccf_mat,
                    "lag_axis": lag_axis,
                },
            )
        return self._ccf

    @property
    def peak(self) -> Result:
        """Pearson *r* evaluated at the lag that maximises |CCF| per channel.

        This is the metric that answers: "what is the best linear
        correlation this channel can achieve if we allow time delay?"
        """
        if self._peak is None:
            ccf_res = self.ccf  # ensure CCF has been computed
            optimal_lags = ccf_res.lags
            r, p = _pearson_at_lags(self._x, self._y, optimal_lags)
            self._peak = Result(
                metric="pearson_peak",
                values=r,
                p_values=p,
                lags=optimal_lags,
                channel_names=self._channel_names,
            )
        return self._peak

    @property
    def partial(self) -> Result:
        r"""Partial correlation matrix of reservoir channels.

        Quantifies the direct linear relationship between channel *i*
        and channel *j* after removing the influence of all other
        channels.  Useful for detecting redundancy in the reservoir.
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
        """Auto-correlation per channel (reservoir-only diagnostic).

        Shape of ``values`` is ``(N, n_lags+1)`` where the lag axis
        runs from 0 to ``max_lag``.
        """
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

    # ── CCA (method, not property — takes parameters) ─────────────────

    def cca(self, n_components: Optional[int] = None) -> CCAResult:
        """Canonical Correlation Analysis between *x* and *y*.

        Parameters
        ----------
        n_components : int, optional
            Number of canonical components.  Defaults to
            ``min(x_dim, N)``.  Since *x* is 1-D here, the practical
            max is 1 unless *x* was multi-dimensional before truncation.

        Returns
        -------
        CCAResult
        """
        try:
            from sklearn.cross_decomposition import CCA
        except ImportError:
            raise ImportError(
                "scikit-learn is required for CCA. "
                "Install it with: pip install scikit-learn"
            )

        x_2d = self._x[:, np.newaxis]  # (T, 1) for sklearn
        d = x_2d.shape[1]

        if n_components is None:
            n_components = min(d, self._N)
        n_components = min(n_components, d, self._N)

        model = CCA(n_components=n_components)
        x_scores, y_scores = model.fit_transform(x_2d, self._y)

        # canonical correlations = pearson r between paired score columns
        correlations = np.array([
            stats.pearsonr(x_scores[:, k], y_scores[:, k])[0]
            for k in range(n_components)
        ])

        return CCAResult(
            correlations=correlations,
            x_weights=model.x_weights_,
            y_weights=model.y_weights_,
            x_scores=x_scores,
            y_scores=y_scores,
        )

    # ── summary ────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Quick one-line diagnostic string."""
        p = self.pearson
        lines = [
            f"Linear Analysis  |  T={self._T}, N={self._N}, max_lag={self._max_lag}",
            f"  Pearson  mean|r|={np.mean(np.abs(p.values)):.4f}  "
            f"max|r|={np.max(np.abs(p.values)):.4f}  "
            f"significant={int(np.sum(p.p_values < 0.05))}/{self._N}",
        ]
        if self._peak is not None:
            pk = self._peak
            lines.append(
                f"  Peak     mean|r|={np.mean(np.abs(pk.values)):.4f}  "
                f"max|r|={np.max(np.abs(pk.values)):.4f}  "
                f"lag_range=[{int(pk.lags.min())}, {int(pk.lags.max())}]"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"Linear(T={self._T}, N={self._N}, "
            f"max_lag={self._max_lag}, lag_sweep={self._lag_sweep})"
        )