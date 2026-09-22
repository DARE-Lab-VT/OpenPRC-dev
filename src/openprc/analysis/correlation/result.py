"""
Result container for correlation analyses in openprc.

Every metric in the correlation module returns a ``Result`` object rather than
a bare NumPy array.  This gives the researcher a uniform interface for
inspection, visualization, export, and statistical filtering—without
sacrificing direct array access for downstream pipelines.

Classes
-------
Result
    General-purpose container (Pearson, Spearman, dCor, …).
CCAResult
    Specialized container for Canonical Correlation Analysis.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Lazy optional imports – we don't force pandas / matplotlib at import time
# ---------------------------------------------------------------------------

def _require_pandas():
    try:
        import pandas as pd
        return pd
    except ImportError:
        raise ImportError(
            "pandas is required for .to_dataframe(). "
            "Install it with: pip install pandas"
        )


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for .plot(). "
            "Install it with: pip install matplotlib"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Multiple-comparison correction helpers
# ═══════════════════════════════════════════════════════════════════════════

def _bonferroni(p: np.ndarray) -> np.ndarray:
    """Bonferroni correction: multiply p-values by the number of tests."""
    return np.minimum(p * p.size, 1.0)


def _fdr_bh(p: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR correction."""
    n = p.size
    flat = p.ravel()
    order = np.argsort(flat)
    ranked = np.empty_like(flat)
    ranked[order] = np.arange(1, n + 1)
    corrected = np.minimum(flat * n / ranked, 1.0)
    # enforce monotonicity (step-up)
    rev = order[::-1]
    corrected[rev] = np.minimum.accumulate(corrected[rev])
    return corrected.reshape(p.shape)


_CORRECTIONS = {
    "bonferroni": _bonferroni,
    "fdr_bh": _fdr_bh,
    "none": lambda p: p,
}


# ═══════════════════════════════════════════════════════════════════════════
#  P-value significance stars (for LaTeX / repr)
# ═══════════════════════════════════════════════════════════════════════════

def _stars(p: float) -> str:
    if p <= 0.001:
        return "***"
    elif p <= 0.01:
        return "**"
    elif p <= 0.05:
        return "*"
    return ""


# ═══════════════════════════════════════════════════════════════════════════
#  Result
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Result:
    """Container returned by every correlation metric.

    Parameters
    ----------
    metric : str
        Human-readable name of the metric (e.g. ``"pearson"``, ``"dcor"``).
    values : np.ndarray
        The primary result array.  Shape is typically ``(N,)`` for per-channel
        scores or ``(N, N)`` for pairwise matrices.
    p_values : np.ndarray or None
        Raw (uncorrected) p-values, same shape as *values*.
    lags : np.ndarray or None
        Optimal lag per channel (only populated when ``lag_sweep=True``).
    channel_names : list[str] or None
        Optional labels inherited from the simulation.
    extra : dict
        Arbitrary additional data a metric wants to stash (e.g. full CCF
        profiles, confidence bands, etc.).

    Examples
    --------
    >>> res = corr.Linear(x, y).pearson
    >>> res.values          # raw (N,) array
    >>> res.significant()   # boolean mask at alpha=0.05
    >>> res.plot()          # publication-quality bar chart
    >>> res.to_latex()      # NeurIPS-ready table
    """

    metric: str
    values: np.ndarray
    p_values: Optional[np.ndarray] = None
    lags: Optional[np.ndarray] = None
    channel_names: Optional[Sequence[str]] = None
    extra: dict = field(default_factory=dict)

    # ── array-like access ──────────────────────────────────────────────

    @property
    def matrix(self) -> np.ndarray:
        """Alias for ``values`` (convenient for pairwise results)."""
        return self.values

    @property
    def shape(self) -> tuple:
        return self.values.shape

    def __array__(self, dtype=None):
        """Allow ``np.asarray(result)``."""
        return np.asarray(self.values, dtype=dtype)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        return self.values[idx]

    # ── statistical helpers ────────────────────────────────────────────

    def p_values_corrected(self, method: str = "fdr_bh") -> np.ndarray:
        """Return p-values after multiple-comparison correction.

        Parameters
        ----------
        method : ``"bonferroni"`` | ``"fdr_bh"`` | ``"none"``
        """
        if self.p_values is None:
            raise ValueError(f"No p-values available for metric '{self.metric}'.")
        if method not in _CORRECTIONS:
            raise ValueError(
                f"Unknown correction method '{method}'. "
                f"Choose from {list(_CORRECTIONS.keys())}."
            )
        return _CORRECTIONS[method](self.p_values)

    def significant(
            self,
            alpha: float = 0.05,
            correction: str = "fdr_bh",
    ) -> np.ndarray:
        """Boolean mask of channels surviving significance threshold.

        Parameters
        ----------
        alpha : float
            Significance level (default 0.05).
        correction : str
            Multiple-comparison correction method (default ``"fdr_bh"``).
        """
        return self.p_values_corrected(correction) < alpha

    # ── repr ───────────────────────────────────────────────────────────

    def _summary_stats(self) -> dict:
        v = self.values
        stats = {"shape": v.shape}
        if v.ndim <= 1 or (v.ndim == 2 and min(v.shape) > 1):
            stats["mean"] = float(np.nanmean(v))
            stats["max"] = float(np.nanmax(v))
            stats["min"] = float(np.nanmin(v))
        if self.p_values is not None:
            n_sig = int(np.sum(self.p_values < 0.05))
            stats["significant_005"] = f"{n_sig}/{self.p_values.size}"
        return stats

    def __repr__(self) -> str:
        stats = self._summary_stats()
        parts = [f"Result(metric='{self.metric}'"]
        for k, v in stats.items():
            if isinstance(v, float):
                parts.append(f"  {k}={v:.4f}")
            else:
                parts.append(f"  {k}={v}")
        if self.lags is not None:
            parts.append(f"  lags_range=[{int(self.lags.min())}, {int(self.lags.max())}]")
        return ",\n".join(parts) + "\n)"

    # ── export: DataFrame ──────────────────────────────────────────────

    def to_dataframe(self, channel_names: Optional[Sequence[str]] = None):
        """Export to a ``pandas.DataFrame``.

        Parameters
        ----------
        channel_names : list[str], optional
            Override channel labels.  Falls back to ``self.channel_names``
            or auto-generated ``ch_0, ch_1, …``.
        """
        pd = _require_pandas()

        names = channel_names or self.channel_names
        v = self.values

        # --- 1-D per-channel vector ---
        if v.ndim == 1:
            n = v.shape[0]
            if names is None:
                names = [f"ch_{i}" for i in range(n)]
            data = {"channel": names, self.metric: v}
            if self.p_values is not None:
                data["p_value"] = self.p_values
                data["p_corrected_fdr"] = self.p_values_corrected("fdr_bh")
            if self.lags is not None:
                data["lag"] = self.lags
            return pd.DataFrame(data)

        # --- 2-D pairwise matrix ---
        if v.ndim == 2:
            n = v.shape[0]
            if names is None:
                names = [f"ch_{i}" for i in range(n)]
            return pd.DataFrame(v, index=names, columns=names)

        # fallback
        return pd.DataFrame(v)

    # ── export: LaTeX ──────────────────────────────────────────────────

    def to_latex(
            self,
            channel_names: Optional[Sequence[str]] = None,
            stars: bool = True,
            caption: Optional[str] = None,
            label: Optional[str] = None,
    ) -> str:
        r"""Generate a LaTeX table string.

        Parameters
        ----------
        channel_names : list[str], optional
            Override channel labels.
        stars : bool
            Append significance stars (\*\*\* / \*\* / \*).
        caption, label : str, optional
            LaTeX ``\caption{}`` and ``\label{}`` values.

        Returns
        -------
        str
            Ready-to-paste LaTeX code.
        """
        names = channel_names or self.channel_names
        v = self.values

        if v.ndim == 1:
            return self._latex_vector(v, names, stars, caption, label)
        elif v.ndim == 2:
            return self._latex_matrix(v, names, caption, label)
        else:
            raise ValueError("LaTeX export supports 1-D and 2-D results only.")

    def _latex_vector(self, v, names, show_stars, caption, label) -> str:
        n = v.shape[0]
        if names is None:
            names = [f"ch$_{{{i}}}$" for i in range(n)]

        header_cols = ["Channel", self.metric.replace("_", r"\_")]
        if self.p_values is not None:
            header_cols.append("$p$-value")
        if self.lags is not None:
            header_cols.append(r"$\tau^*$")

        col_spec = "l" + "r" * (len(header_cols) - 1)
        lines = [
            r"\begin{table}[ht]",
            r"\centering",
        ]
        if caption:
            lines.append(rf"\caption{{{caption}}}")
        if label:
            lines.append(rf"\label{{{label}}}")
        lines += [
            rf"\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            " & ".join(header_cols) + r" \\",
            r"\midrule",
            ]

        for i in range(n):
            row = [names[i], f"{v[i]:.4f}"]
            if self.p_values is not None:
                p = self.p_values[i]
                star_str = _stars(p) if show_stars else ""
                row.append(f"{p:.2e}{star_str}")
            if self.lags is not None:
                row.append(str(int(self.lags[i])))
            lines.append(" & ".join(row) + r" \\")

        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        return "\n".join(lines)

    def _latex_matrix(self, v, names, caption, label) -> str:
        n = v.shape[0]
        if names is None:
            names = [f"ch$_{{{i}}}$" for i in range(n)]

        col_spec = "l" + "r" * n
        lines = [
            r"\begin{table}[ht]",
            r"\centering",
        ]
        if caption:
            lines.append(rf"\caption{{{caption}}}")
        if label:
            lines.append(rf"\label{{{label}}}")
        lines += [
            rf"\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            " & ".join([""] + names) + r" \\",
            r"\midrule",
            ]
        for i in range(n):
            row = [names[i]] + [f"{v[i, j]:.4f}" for j in range(n)]
            lines.append(" & ".join(row) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        return "\n".join(lines)

    # ── plotting ───────────────────────────────────────────────────────

    def plot(self, kind: Optional[str] = None, ax=None, **kwargs):
        """Generate a publication-ready figure.

        Parameters
        ----------
        kind : str, optional
            ``"bar"`` (default for 1-D), ``"heatmap"`` (default for 2-D),
            ``"lag_profile"`` (requires ``extra["ccf_profiles"]``).
        ax : matplotlib Axes, optional
            Draw on an existing axes.
        **kwargs
            Forwarded to the underlying matplotlib call.

        Returns
        -------
        matplotlib.axes.Axes
        """
        plt = _require_matplotlib()

        v = self.values

        # auto-select kind
        if kind is None:
            if "ccf_full" in self.extra:
                kind = "lag_profile"
            elif v.ndim == 2:
                kind = "heatmap"
            else:
                kind = "bar"

        dispatch = {
            "bar": self._plot_bar,
            "heatmap": self._plot_heatmap,
            "lag_profile": self._plot_lag_profile,
        }
        if kind not in dispatch:
            raise ValueError(
                f"Unknown plot kind '{kind}'. Choose from {list(dispatch.keys())}."
            )
        return dispatch[kind](plt, ax=ax, **kwargs)

    def _channel_labels(self) -> list[str]:
        if self.channel_names is not None:
            return list(self.channel_names)
        return [f"ch_{i}" for i in range(self.values.shape[0])]

    def _plot_bar(self, plt, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (10, 4)))
        labels = self._channel_labels()
        x_pos = np.arange(len(labels))
        bars = ax.bar(x_pos, self.values, **kwargs)

        # colour significant channels differently
        if self.p_values is not None:
            sig = self.p_values < 0.05
            for i, bar in enumerate(bars):
                bar.set_color("steelblue" if sig[i] else "lightgrey")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(self.metric)
        ax.set_title(f"{self.metric} per channel")
        ax.axhline(0, color="k", linewidth=0.5)
        plt.tight_layout()
        return ax

    def _plot_heatmap(self, plt, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(
                figsize=kwargs.pop("figsize", (8, 7))
            )
        labels = self._channel_labels()
        cmap = kwargs.pop("cmap", "RdBu_r")
        im = ax.imshow(self.values, cmap=cmap, aspect="auto", **kwargs)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(f"{self.metric} matrix")
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        return ax

    def _plot_lag_profile(self, plt, ax=None, **kwargs):
        ccf_full = self.extra.get("ccf_full")
        lag_axis = self.extra.get("lag_axis")
        if ccf_full is None or lag_axis is None:
            raise ValueError(
                "Lag profile plot requires 'ccf_full' and 'lag_axis' in "
                "extra. Use the CCF result from Linear."
            )
        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (10, 5)))

        n_channels = ccf_full.shape[0]
        cmap = plt.cm.viridis
        for i in range(n_channels):
            color = cmap(i / max(n_channels - 1, 1))
            ax.plot(lag_axis, ccf_full[i], color=color, alpha=0.6, linewidth=0.8)

        # mark peaks
        if self.lags is not None:
            for i in range(n_channels):
                peak_lag = int(self.lags[i])
                idx = np.searchsorted(lag_axis, peak_lag)
                idx = np.clip(idx, 0, len(lag_axis) - 1)
                ax.plot(peak_lag, ccf_full[i, idx], "o", color="red",
                        markersize=3, alpha=0.7)

        ax.set_xlabel("Lag (samples)")
        ax.set_ylabel("Cross-correlation")
        ax.set_title("CCF Lag Profiles (peaks in red)")
        ax.axhline(0, color="k", linewidth=0.5)
        plt.tight_layout()
        return ax


# ═══════════════════════════════════════════════════════════════════════════
#  CCAResult
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CCAResult:
    """Container for Canonical Correlation Analysis output.

    Parameters
    ----------
    correlations : np.ndarray
        Canonical correlations, shape ``(n_components,)``.
    x_weights : np.ndarray
        Projection matrix for the input space, shape ``(d, n_components)``.
    y_weights : np.ndarray
        Projection matrix for the reservoir space, shape ``(N, n_components)``.
    x_scores : np.ndarray
        Projected input, shape ``(T, n_components)``.
    y_scores : np.ndarray
        Projected reservoir, shape ``(T, n_components)``.
    """

    correlations: np.ndarray
    x_weights: np.ndarray
    y_weights: np.ndarray
    x_scores: np.ndarray
    y_scores: np.ndarray

    @property
    def n_components(self) -> int:
        return len(self.correlations)

    def __repr__(self) -> str:
        corrs = ", ".join(f"{c:.4f}" for c in self.correlations[:5])
        suffix = ", …" if self.n_components > 5 else ""
        return (
            f"CCAResult(n_components={self.n_components}, "
            f"correlations=[{corrs}{suffix}])"
        )

    def to_dataframe(self):
        pd = _require_pandas()
        return pd.DataFrame({
            "component": np.arange(1, self.n_components + 1),
            "canonical_correlation": self.correlations,
        })

    def plot(self, ax=None, **kwargs):
        """Bar chart of canonical correlations."""
        plt = _require_matplotlib()
        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (6, 4)))
        ax.bar(range(1, self.n_components + 1), self.correlations, **kwargs)
        ax.set_xlabel("Component")
        ax.set_ylabel("Canonical Correlation")
        ax.set_title("CCA — Canonical Correlations")
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        return ax