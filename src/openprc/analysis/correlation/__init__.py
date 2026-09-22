"""
``openprc.analysis.correlation`` — Correlation diagnostics for Physical Reservoir Computing.

This is **Layer 1** of the ``openprc.analysis`` suite.  It bridges the gap
between raw physics simulation data and quantified statistical relationships.

Quick-start
-----------
>>> from openprc.analysis import correlation as corr
>>>
>>> # Linear diagnostics
>>> lin = corr.Linear(x, y, lag_sweep=True)
>>> lin.pearson          # zero-lag Pearson r per channel
>>> lin.peak             # best-lag Pearson r per channel
>>> lin.ccf.plot()       # cross-correlation lag profiles
>>>
>>> # Nonparametric diagnostics
>>> nr = corr.Nonparametric(x, y)
>>> nr.dcor              # distance correlation (detects nonlinear deps)
>>> nr.hsic()            # kernel-based independence criterion

Classes
-------
Linear
    Pearson, CCF (with lag sweep), partial correlation, ACF, CCA.
Nonparametric
    Spearman, Kendall, distance correlation, HSIC.
Redundancy
    y-only internal structure: pairwise correlation, partial correlation,
    ACF, effective rank, condition number, channel clustering.
Result
    Uniform container returned by every metric — ``.values``, ``.plot()``,
    ``.to_dataframe()``, ``.to_latex()``, ``.significant()``.
CCAResult
    Specialised container for Canonical Correlation Analysis.
"""

from .linear import Linear
from .nonparametric import Nonparametric
from .redundancy import Redundancy
from .result import CCAResult, Result

__all__ = [
    "Linear",
    "Nonparametric",
    "Redundancy",
    "Result",
    "CCAResult",
]