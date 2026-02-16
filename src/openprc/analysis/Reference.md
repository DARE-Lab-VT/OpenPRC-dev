# `openprc.analysis`

**Layer 1 `openprc.analysis.correlatiom`**

The correlation module is the first diagnostic stop after a physics simulation. It bridges the gap between raw reservoir dynamics and quantified statistical relationships — answering three fundamental questions:

1. **How well does each channel represent the input?** → `Linear`, `Nonparametric`
2. **What nonlinear structure is the reservoir exploiting?** → `Nonparametric`
3. **How much of the reservoir is actually useful?** → `Redundancy`

---

## Table of Contents

- [Installation & Dependencies](#installation--dependencies)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
    - [Linear](#linear)
    - [Nonparametric](#nonparametric)
    - [Redundancy](#redundancy)
    - [Result](#result)
    - [CCAResult](#ccaresult)
- [Mathematical Definitions](#mathematical-definitions)
    - [Linear Metrics](#linear-metrics)
    - [Nonparametric Metrics](#nonparametric-metrics)
    - [Redundancy Metrics](#redundancy-metrics)
- [Interpretation Guide](#interpretation-guide)
    - [Metric Selection Flowchart](#metric-selection-flowchart)
    - [Per-Metric Interpretation](#per-metric-interpretation)
    - [Common Diagnostic Patterns](#common-diagnostic-patterns)
- [Examples](#examples)

---

## Installation & Dependencies

**Required** (always available):
- `numpy`
- `scipy`

**Optional** (lazy-imported only when needed):
- `pandas` — for `.to_dataframe()`
- `matplotlib` — for `.plot()`
- `scikit-learn` — for `.cca()`

---

## Quick Start

```python
import numpy as np
from openprc.analysis import correlation as corr

# x: input signal (T,), y: reservoir state (T, N)

# ── Input–Reservoir relationship ──
lin = corr.Linear(x, y, lag_sweep=True)
lin.pearson                 # zero-lag linear correlation
lin.peak                    # best-lag linear correlation
lin.ccf.plot()              # cross-correlation lag profiles

# ── Nonlinear dependencies ──
nr = corr.Nonparametric(x, y)
nr.dcor                     # catches what Pearson misses
nr.hsic()                   # kernel-based independence

# ── Internal reservoir structure ──
red = corr.Redundancy(y)
print(f"Effective rank: {red.rank:.1f} / {N}")
print(f"Condition number: {red.condition:.1f}")
red.correlation.plot()      # N×N heatmap
```

---

## API Reference

### `Linear`

Quantifies straight-line and time-lagged linear relationships between input $x$ and reservoir $y$.

```python
lin = corr.Linear(x, y, max_lag=None, lag_sweep=False, channel_names=None)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | array `(T,)` or `(T,d)` | *required* | Input signal. If `d > 1`, uses `x[:, 0]` with a warning. |
| `y` | array `(T, N)` | *required* | Reservoir state matrix. |
| `max_lag` | `int` | `T // 4` | Maximum lag (samples) for CCF and lag-sweep. |
| `lag_sweep` | `bool` | `False` | Reserved for future automatic peak computation. |
| `channel_names` | `list[str]` | `None` | Labels propagated to all Result objects. |

**Properties** (all lazily computed, cached after first access):

| Property | Returns | Shape | Description |
|----------|---------|-------|-------------|
| `.pearson` | `Result` | `(N,)` | Zero-lag Pearson $r$ per channel, with p-values. |
| `.ccf` | `Result` | `(N,)` | Peak cross-correlation per channel. Full profiles in `.extra["ccf_full"]`. |
| `.peak` | `Result` | `(N,)` | Pearson $r$ evaluated at the optimal lag per channel (from CCF). |
| `.partial` | `Result` | `(N, N)` | Partial correlation matrix of $y$ (precision-matrix method). |
| `.acf` | `Result` | `(N, max_lag+1)` | Auto-correlation per channel (FFT-based). |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `.cca(n_components=None)` | `CCAResult` | Canonical Correlation Analysis between $x$ and $y$. |
| `.summary()` | `str` | One-line diagnostic string. |

---

### `Nonparametric`

Quantifies monotonic, nonlinear, and kernel-based dependencies that linear metrics miss.

```python
nr = corr.Nonparametric(x, y, channel_names=None)
```

**Parameters:** Same as `Linear` (without `max_lag` and `lag_sweep`).

**Properties** (lazily computed):

| Property | Returns | Shape | Description |
|----------|---------|-------|-------------|
| `.spearman` | `Result` | `(N,)` | Spearman rank correlation with p-values. |
| `.kendall` | `Result` | `(N,)` | Kendall $\tau_b$ with p-values. |
| `.dcor` | `Result` | `(N,)` | Distance correlation (no p-values; use `.dcor_test()` for those). |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `.dcor_test(n_permutations=499, seed=None)` | `Result` | dCor with permutation-based p-values. |
| `.hsic(sigma_x=None, sigma_y=None)` | `Result` | HSIC with RBF kernel (median heuristic bandwidth). |
| `.hsic_test(sigma_x=None, sigma_y=None, n_permutations=499, seed=None)` | `Result` | HSIC with permutation p-values. |
| `.summary()` | `str` | One-line diagnostic string. |

---

### `Redundancy`

Quantifies the internal correlation structure of $y$ — no input signal required.

```python
red = corr.Redundancy(y, max_lag=None, channel_names=None)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `y` | array `(T, N)` | *required* | Reservoir state matrix (≥ 2 channels). |
| `max_lag` | `int` | `T // 4` | Maximum lag for ACF. |
| `channel_names` | `list[str]` | `None` | Labels for channels. |

**Properties** (lazily computed):

| Property | Returns | Shape | Description |
|----------|---------|-------|-------------|
| `.correlation` | `Result` | `(N, N)` | Pairwise Pearson matrix with p-values. |
| `.partial` | `Result` | `(N, N)` | Partial correlation matrix. |
| `.acf` | `Result` | `(N, max_lag+1)` | Per-channel auto-correlation. |
| `.rank` | `float` | scalar | Effective rank (Shannon entropy of eigenspectrum). |
| `.condition` | `float` | scalar | Condition number $\lambda_{\max} / \lambda_{\min}$. |
| `.eigenvalues` | `ndarray` | `(N,)` | Eigenvalues of correlation matrix, sorted descending. |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `.groups(threshold=0.9)` | `list[list[int]]` | Channel index clusters (single-linkage on $|r|$). |
| `.groups_named(threshold=0.9)` | `list[list[str]]` | Same, but with channel names. |
| `.summary()` | `str` | Diagnostic string with rank, condition, and group counts. |

---

### `Result`

Uniform container returned by every metric. Wraps raw arrays with statistical tools, export, and visualisation.

**Core access:**

| Attribute / Method | Returns | Description |
|-------------------|---------|-------------|
| `.values` | `ndarray` | Primary result array. |
| `.matrix` | `ndarray` | Alias for `.values` (convenient for 2-D results). |
| `.p_values` | `ndarray` or `None` | Raw (uncorrected) p-values. |
| `.lags` | `ndarray` or `None` | Optimal lag per channel (when applicable). |
| `.extra` | `dict` | Additional data (e.g. full CCF profiles). |

**Statistical methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `.p_values_corrected(method)` | `ndarray` | Corrected p-values. `method`: `"fdr_bh"`, `"bonferroni"`, `"none"`. |
| `.significant(alpha=0.05, correction="fdr_bh")` | `bool ndarray` | Mask of channels surviving threshold. |

**Export methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `.to_dataframe(channel_names=None)` | `DataFrame` | Labelled table with values, p-values, corrected p-values, lags. |
| `.to_latex(stars=True, caption=None, label=None)` | `str` | Publication-ready LaTeX table with significance stars. |

**Plotting:**

| Method | Description |
|--------|-------------|
| `.plot(kind=None, ax=None, **kwargs)` | Auto-dispatches: `"bar"` for 1-D, `"heatmap"` for 2-D, `"lag_profile"` when CCF data is present. Returns `matplotlib.axes.Axes`. |

**NumPy interop:**

```python
arr = np.asarray(result)     # __array__ protocol
val = result[0]              # indexing
sub = result[:3]             # slicing
mean = np.mean(np.abs(result))  # direct math
```

---

### `CCAResult`

Specialised container for Canonical Correlation Analysis.

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `.correlations` | `(n_components,)` | Canonical correlations. |
| `.x_weights` | `(d, n_components)` | Input space projection. |
| `.y_weights` | `(N, n_components)` | Reservoir space projection. |
| `.x_scores` | `(T, n_components)` | Projected input. |
| `.y_scores` | `(T, n_components)` | Projected reservoir. |
| `.n_components` | `int` | Number of components. |

**Methods:** `.plot()`, `.to_dataframe()`

---

## Mathematical Definitions

### Linear Metrics

**Pearson Correlation Coefficient**

$$r_{xy} = \frac{\sum_{t=1}^{T}(x_t - \bar{x})(y_t - \bar{y})}{\sqrt{\sum_{t=1}^{T}(x_t - \bar{x})^2 \sum_{t=1}^{T}(y_t - \bar{y})^2}}$$

Measures the strength and direction of the linear relationship. Bounded in $[-1, 1]$. The p-value tests $H_0: r = 0$ using a $t$-distribution with $T - 2$ degrees of freedom.

**Cross-Correlation Function (CCF)**

$$\text{CCF}_{xy}(\tau) = \frac{\sum_{t}(x_t - \bar{x})(y_{t+\tau} - \bar{y})}{\sqrt{\sum_t(x_t - \bar{x})^2 \sum_t(y_t - \bar{y})^2}}$$

The normalised correlation between $x(t)$ and $y(t + \tau)$ as a function of lag $\tau$. The peak of $|\text{CCF}(\tau)|$ identifies the time delay at which the channel is most responsive to the input. Computed via `scipy.signal.correlate` (FFT-accelerated).

**Partial Correlation**

$$\rho_{ij \cdot \text{rest}} = -\frac{P_{ij}}{\sqrt{P_{ii} P_{jj}}}$$

where $P = \Sigma^{-1}$ is the precision matrix (inverse of the covariance matrix). This gives the linear relationship between channels $i$ and $j$ after removing the influence of all other channels. If the covariance matrix is singular, a pseudo-inverse is used.

**Auto-Correlation Function (ACF)**

$$\text{ACF}_y(\tau) = \frac{\sum_{t=1}^{T-\tau}(y_t - \bar{y})(y_{t+\tau} - \bar{y})}{\sum_{t=1}^{T}(y_t - \bar{y})^2}$$

Measures how correlated a channel is with its own past. High ACF at large lags indicates the channel has long memory (slow dynamics). Computed via FFT for efficiency.

**Canonical Correlation Analysis (CCA)**

Finds weight vectors $\mathbf{a}$ and $\mathbf{b}$ that maximise:

$$\rho = \max_{\mathbf{a}, \mathbf{b}} \text{corr}(\mathbf{a}^\top X, \mathbf{b}^\top Y)$$

The canonical correlation $\rho$ represents the best achievable linear mapping from the reservoir to the input. With 1-D input, CCA finds the single best linear combination of all $N$ channels.

---

### Nonparametric Metrics

**Spearman Rank Correlation**

$$r_s = 1 - \frac{6 \sum d_i^2}{T(T^2 - 1)}$$

where $d_i$ is the difference between the ranks of $x_i$ and $y_i$. Equivalent to Pearson $r$ computed on the rank-transformed data. Captures monotonic relationships (not just linear).

**Kendall's $\tau_b$**

$$\tau_b = \frac{C - D}{\sqrt{(C + D + T_x)(C + D + T_y)}}$$

where $C$ = concordant pairs, $D$ = discordant pairs, $T_x$ and $T_y$ = ties. More robust than Spearman for small samples or many ties.

**Distance Correlation (dCor)**

Given pairwise distance matrices $A$ (from $x$) and $B$ (from $y$), each doubly centred:

$$\text{dCov}^2(x, y) = \frac{1}{T^2}\sum_{k,l} A_{kl} B_{kl}$$

$$\text{dCor}(x, y) = \frac{\text{dCov}(x, y)}{\sqrt{\text{dCov}(x, x) \cdot \text{dCov}(y, y)}}$$

The key property: $\text{dCor}(x, y) = 0 \iff x \perp y$ (statistical independence). This is strictly stronger than Pearson or Spearman — it detects *any* dependency, including quadratic, periodic, and other nonlinear structures. Complexity is $O(T^2)$ per channel.

Reference: Székely, Rizzo & Bakirov (2007), *Annals of Statistics*.

**Hilbert-Schmidt Independence Criterion (HSIC)**

$$\text{HSIC}(x, y) = \frac{1}{T^2} \text{tr}(K H L H)$$

where $K$ and $L$ are kernel matrices (RBF with median-heuristic bandwidth by default), and $H = I - \frac{1}{T}\mathbf{1}\mathbf{1}^\top$ is the centring matrix.

HSIC measures statistical dependence in a reproducing kernel Hilbert space. Like dCor, $\text{HSIC} = 0 \iff$ independence (for characteristic kernels like RBF). The advantage over dCor is sensitivity to specific frequency structures via kernel choice; the disadvantage is the additional bandwidth hyperparameter.

Reference: Gretton, Bousquet, Smola & Schölkopf (2005), *ALT*.

---

### Redundancy Metrics

**Effective Rank**

$$r_{\text{eff}} = \exp\left(-\sum_{i=1}^{N} \hat{\lambda}_i \ln \hat{\lambda}_i\right)$$

where $\hat{\lambda}_i = \lambda_i / \sum_j \lambda_j$ are the normalised eigenvalues of the correlation matrix. This is the exponential of the Shannon entropy of the eigenvalue distribution.

- $r_{\text{eff}} = N$: all eigenvalues are equal → maximally diverse reservoir (every channel contributes unique information).
- $r_{\text{eff}} = 1$: one eigenvalue dominates → fully redundant reservoir (all channels carry the same information).

**Condition Number**

$$\kappa = \frac{\lambda_{\max}}{\lambda_{\min}}$$

The ratio of the largest to smallest eigenvalue. In PRC context, this directly measures readout stability: when you train a linear readout $\hat{x} = W y$ via least squares, the error amplification is bounded by $\kappa$. High $\kappa$ means small noise in $y$ causes large swings in $W$.

**Channel Groups**

Single-linkage clustering on $|r_{ij}| \geq \theta$: channels $i$ and $j$ are in the same group if there exists a path of pairwise correlations all exceeding the threshold $\theta$. Singletons indicate unique, non-redundant channels.

---

## Interpretation Guide

### Metric Selection Flowchart

```
Start
  │
  ├─ "How does x relate to each y channel?"
  │   ├─ Expect linear response?  →  Linear.pearson
  │   ├─ Reservoir has latency?   →  Linear.ccf / Linear.peak
  │   ├─ Expect monotonic?        →  Nonparametric.spearman
  │   └─ Unknown / nonlinear?     →  Nonparametric.dcor
  │
  ├─ "Is the reservoir redundant?"
  │   ├─ Quick check              →  Redundancy.rank, .condition
  │   ├─ Which channels overlap?  →  Redundancy.partial (heatmap)
  │   └─ Grouping for pruning?    →  Redundancy.groups(threshold)
  │
  └─ "What is the best linear readout?"
      └─  Linear.cca()
```

### Per-Metric Interpretation

| Metric | Range | Good Reservoir                                         | Bad Reservoir                                          |
|--------|-------|--------------------------------------------------------|--------------------------------------------------------|
| **Pearson** $r$ | $[-1, 1]$ | High $\|r\|$ on many channels → strong linear encoding | All $\|r\| \approx 0$ → input not linearly represented |
| **Peak** $r$ | $[-1, 1]$ | High $\|r\|$ at varied lags → distributed temporal encoding | All peaks at lag 0 → no memory |
| **CCF lags** | integer | Spread across multiple lags → rich temporal basis      | All identical → channels are copies                    |
| **Partial corr** | $[-1, 1]$ | Off-diagonal near 0 → channels carry independent info  | Large off-diagonal blocks → redundant clusters         |
| **ACF** | $[-1, 1]$ | Slow decay → long memory; varied across channels → diversity | Identical ACF profiles → homogeneous dynamics          |
| **CCA** $\rho$ | $[0, 1]$ | Close to 1 → input is recoverable from $y$             | Close to 0 → reservoir fails to encode input           |
| **Spearman** $\rho_s$ | $[-1, 1]$ | High where Pearson is high → relationship is monotonic | High Spearman, low Pearson → monotonic but curved      |
| **Kendall** $\tau$ | $[-1, 1]$ | Consistent with Spearman → robust monotonic signal     | Large Spearman-Kendall gap → outlier sensitivity       |
| **dCor** | $[0, 1]$ | High where Pearson is low → **nonlinear encoding detected** | $\approx 0$ → channel is genuinely independent of input |
| **HSIC** | $[0, \infty)$ | Relative ranking matters; high = strong kernel-space dependency | Near 0 → independent (confirm with permutation test)   |
| **Effective rank** | $[1, N]$ | Close to $N$ → full utilisation of reservoir dimensions | Close to 1 → wasting channels on redundant info        |
| **Condition** $\kappa$ | $[1, \infty)$ | $< 100$ → stable readout                               | $> 10^4$ → regularisation essential, channels nearly collinear |

### Common Diagnostic Patterns

**Pattern 1: "Pearson-blind" channels**
```
Pearson ≈ 0, dCor ≈ 0.5  →  Nonlinear dependency detected
```
The channel encodes the input through a nonlinear transformation (e.g. $x^2$, $|x|$, $\sin(x)$). A linear readout will ignore this channel, but a nonlinear readout (or feature expansion) can exploit it.

**Pattern 2: High Pearson, zero-lag only**
```
Pearson(lag=0) = 0.95, Peak improvement < 1%
```
The channel responds instantaneously — no memory contribution. Useful for feedforward mapping but not for temporal tasks.

**Pattern 3: Low rank, high condition number**
```
rank = 5 / 100, κ = 10^6
```
The reservoir has 100 channels but only 5 independent directions. Ridge regression is mandatory. Consider pruning to the top 5 channels from `Redundancy.groups()`.

**Pattern 4: Spearman ≫ Pearson**
```
Spearman = 0.85, Pearson = 0.40
```
The channel has a strong monotonic but nonlinear response (e.g. saturation via `tanh`). A rank-based readout or simple nonlinear feature (e.g. polynomial) will capture this.

**Pattern 5: Partial correlation reveals mediated connections**
```
Pairwise r(A,B) = 0.90
Partial  r(A,B) = 0.05
```
Channels A and B appear correlated, but the relationship is entirely mediated through other channels. Removing one of them loses almost no information.

---

## Examples

### Minimal: Check if the reservoir encodes the input

```python
from openprc.analysis import correlation as corr

lin = corr.Linear(x, y)
print(lin.summary())
# Linear Analysis  |  T=500, N=8, max_lag=125
#   Pearson  mean|r|=0.43  max|r|=0.99  significant=5/8
```

### Full diagnostic with plots

```python
lin = corr.Linear(x, y, channel_names=names)
nr  = corr.Nonparametric(x, y, channel_names=names)
red = corr.Redundancy(y, channel_names=names)

# Linear
lin.pearson.plot()           # bar chart, significant channels in blue
lin.ccf.plot()               # lag profiles with red peak markers
lin.partial.plot()           # N×N heatmap

# Nonparametric
nr.dcor.plot()               # dCor bar chart (compare with Pearson)

# Redundancy
red.correlation.plot()       # N×N heatmap
print(red.summary())
```

### Export to paper

```python
# LaTeX table with significance stars
print(lin.peak.to_latex(
    caption="Peak Pearson correlation at optimal lag",
    label="tab:peak-pearson",
))

# DataFrame for further analysis
df = lin.peak.to_dataframe()
significant = df[df["p_corrected_fdr"] < 0.05]
```

### Multi-panel figure

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
lin.pearson.plot(ax=axes[0, 0])
lin.ccf.plot(kind="lag_profile", ax=axes[0, 1])
nr.dcor.plot(ax=axes[1, 0])
red.correlation.plot(ax=axes[1, 1])
plt.tight_layout()
plt.savefig("reservoir_diagnostics.png", dpi=300)
```

### Readout optimisation pipeline

```python
red = corr.Redundancy(y, channel_names=names)

# 1. Check overall health
print(f"Effective rank: {red.rank:.1f} / {N}")
print(f"Condition number: {red.condition:.1f}")

# 2. Identify redundant clusters
for group in red.groups_named(threshold=0.85):
    if len(group) > 1:
        print(f"Redundant cluster: {group}")

# 3. Select representative channels
groups = red.groups(threshold=0.85)
selected = [g[0] for g in groups]  # keep one per cluster
y_pruned = y[:, selected]
print(f"Pruned: {N} → {len(selected)} channels")
```

---
