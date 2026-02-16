"""
===============================================================================
openprc.analysis.correlation — Complete Walkthrough
===============================================================================

This example simulates a minimal Physical Reservoir Computing scenario:

    x(t)  →  [physical reservoir with 8 channels]  →  y(t)

Each channel responds to x with a different lag, gain, and nonlinearity.
We then run every diagnostic in the correlation module and inspect the
outputs exactly as a researcher would after a DEMLAT simulation.

Run:
    python example_correlation.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from openprc.demlat import SimulationData

DEMO_DIR = Path("experiments/yoshimura_test")
data = SimulationData(DEMO_DIR / "output" / "simulation.h5")
positions, _ = data.get_dataset("time_series/nodes/positions")
print(positions.shape)

y = positions[:, :, 0]
x = y[:, 0]
N = y.shape[1]
channel_names = [str(i) for i in range(N)]
# np.random.seed(2025)
#
# T = 500  # time steps
# N = 8  # reservoir channels
#
# t = np.arange(T)
# x = np.sin(2 * np.pi * 0.02 * t) + 0.5 * np.sin(2 * np.pi * 0.07 * t)
# x += 0.15 * np.random.randn(T)
#
# channel_names = [
#     "linear_lag0",  # direct linear copy
#     "linear_lag5",  # linear, delayed by 5 steps
#     "linear_lag20",  # linear, delayed by 20 steps
#     "quadratic",  # x² — invisible to Pearson
#     "saturated",  # tanh(x) — monotonic but nonlinear
#     "rectified",  # |x| — V-shape, non-monotonic
#     "noisy_linear",  # weak linear + heavy noise
#     "independent",  # pure noise, no relationship to x
# ]
#
# y = np.zeros((T, N))
#
# y[:, 0] = 0.9 * x + 0.1 * np.random.randn(T)
# y[5:, 1] = 0.85 * x[:-5] + 0.15 * np.random.randn(T - 5)
# y[20:, 2] = 0.8 * x[:-20] + 0.2 * np.random.randn(T - 20)
# y[:, 3] = x ** 2 + 0.2 * np.random.randn(T)
# y[:, 4] = np.tanh(2 * x) + 0.1 * np.random.randn(T)
# y[:, 5] = np.abs(x) + 0.15 * np.random.randn(T)
# y[:, 6] = 0.3 * x + 1.0 * np.random.randn(T)
# y[:, 7] = np.random.randn(T)
#
print(f"  Input x:     shape {x.shape}")
print(f"  Reservoir y: shape {y.shape}  ({N} channels)")

from openprc.analysis import correlation as corr

lin = corr.Linear(x, y, channel_names=channel_names, lag_sweep=True)

# ── 2a. Pearson (zero-lag) ─────────────────────────────────────────────
print(repr(lin.pearson))
print(lin.pearson.to_dataframe().to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 4))
lin.pearson.plot(kind="bar", ax=ax)
plt.show()

# # ── 2b. Cross-Correlation (CCF) ───────────────────────────────────────

ccf = lin.ccf
for i, name in enumerate(channel_names):
    print(f"    {name:20s}  lag = {int(ccf.lags[i]):+4d}  |CCF| = {abs(ccf.values[i]):.4f}")

fig, ax = plt.subplots(figsize=(10, 5))
ccf.plot(kind="lag_profile", ax=ax)
plt.show()

# ── 2c. Peak Pearson (at optimal lag) ─────────────────────────────────
peak = lin.peak
print(repr(peak))
print(peak.to_dataframe().to_string(index=False))
fig, ax = plt.subplots(figsize=(10, 4))
peak.plot(kind="bar", ax=ax)
plt.show()

# ── 2d. Partial Correlation Matrix ────────────────────────────────────
partial = lin.partial
mat = partial.values
print("  Upper triangle (|partial r| > 0.3):")
for i in range(N):
    for j in range(i + 1, N):
        if abs(mat[i, j]) > 0.3:
            print(f"    {channel_names[i]:20s} ↔ {channel_names[j]:20s}  r = {mat[i, j]:+.4f}")

fig, ax = plt.subplots(figsize=(8, 7))
partial.plot(kind="heatmap", ax=ax)
plt.show()

# ── 2e. Auto-Correlation (ACF) ────────────────────────────────────────
acf = lin.acf
for i, name in enumerate(channel_names):
    print(f"    {name:20s}  ACF(1) = {acf.values[i, 1]:.4f}")
fig, ax = plt.subplots(figsize=(10, 5))
acf.plot(kind="heatmap", ax=ax)
plt.show()


# ── 2f. Canonical Correlation Analysis (CCA) ──────────────────────────
cca = lin.cca()
order = np.argsort(np.abs(cca.y_weights[:, 0]))[::-1]
for rank, idx in enumerate(order[:3]):
    print(f"    #{rank + 1}  {channel_names[idx]:20s}  weight = {cca.y_weights[idx, 0]:+.4f}")
fig, ax = plt.subplots(figsize=(6, 4))
cca.plot(ax=ax)
plt.show()


# ── 2g. Summary ───────────────────────────────────────────────────────
print(lin.summary())

nr = corr.Nonparametric(x, y, channel_names=channel_names)

# ── 3a. Spearman ──────────────────────────────────────────────────────
print(repr(nr.spearman))
print(nr.spearman.to_dataframe().to_string(index=False))
fig, ax = plt.subplots(figsize=(10, 4))
nr.spearman.plot(kind="bar", ax=ax)
plt.show()


# ── 3b. Kendall ───────────────────────────────────────────────────────
print(repr(nr.kendall))
for i, name in enumerate(channel_names):
    print(f"    {name:20s}  Spearman = {nr.spearman.values[i]:+.4f}  "
          f"Kendall = {nr.kendall.values[i]:+.4f}")

# ── 3c. Distance Correlation ─────────────────────────────────────────
dc = nr.dcor
pearson_vals = lin.pearson.values
for i, name in enumerate(channel_names):
    flag = ""
    if abs(pearson_vals[i]) < 0.2 and dc.values[i] > 0.3:
        flag = "  ← NONLINEAR DEPENDENCY DETECTED"
    print(f"    {name:20s}  Pearson = {pearson_vals[i]:+.4f}  "
          f"dCor = {dc.values[i]:.4f}{flag}")
fig, ax = plt.subplots(figsize=(10, 4))
dc.plot(kind="bar", ax=ax)
plt.show()


# # ── 3d. dCor with permutation test ───────────────────────────────────
# dc_test = nr.dcor_test(n_permutations=199, seed=42)
# print(dc_test.to_dataframe().to_string(index=False))
# sig = dc_test.significant(alpha=0.05)
# for i, name in enumerate(channel_names):
#     status = "✓ significant" if sig[i] else "✗ not significant"
#     print(f"    {name:20s}  p = {dc_test.p_values[i]:.3f}  {status}")
# fig, ax = plt.subplots(figsize=(10, 4))
# dc_test.plot(kind="bar", ax=ax)
# plt.show()


# # ── 3e. HSIC ──────────────────────────────────────────────────────────
# hs = nr.hsic()
# print(repr(hs))
# for i, name in enumerate(channel_names):
#     print(f"    {name:20s}  HSIC = {hs.values[i]:.6f}")
# fig, ax = plt.subplots(figsize=(10, 4))
# hs.plot(kind="bar", ax=ax)
# plt.show()



# # ── 3f. HSIC with permutation test ───────────────────────────────────
# hs_test = nr.hsic_test(n_permutations=199, seed=42)
# print(hs_test.to_dataframe().to_string(index=False))
# fig, ax = plt.subplots(figsize=(10, 4))
# hs_test.plot(kind="bar", ax=ax)
# plt.show()

# ── 3g. Nonparametric Summary ────────────────────────────────────────
print(nr.summary())

# ── 4a. LaTeX table ───────────────────────────────────────────────────
print(peak.to_latex(
    caption="Peak Pearson correlation at optimal lag",
    label="tab:peak-pearson",
))

# # ── 4b. LaTeX table — dCor with p-values ─────────────────────────────
# print(dc_test.to_latex(
#     caption="Distance correlation with permutation $p$-values",
#     label="tab:dcor",
# ))

# ── 4c. DataFrame round-trip ──────────────────────────────────────────
df = peak.to_dataframe()
print(df.dtypes)
# Example: filter to significant channels with |r| > 0.5
strong = df[(df["p_corrected_fdr"] < 0.05) & (df["pearson_peak"].abs() > 0.5)]
print(strong[["channel", "pearson_peak", "lag"]].to_string(index=False))


for i, name in enumerate(channel_names):
    raw = lin.pearson.p_values[i]
    fdr = lin.pearson.p_values_corrected("fdr_bh")[i]
    bonf = lin.pearson.p_values_corrected("bonferroni")[i]
    print(f"    {name:20s}  raw = {raw:.2e}  FDR = {fdr:.2e}  Bonf = {bonf:.2e}")
print()

sig_raw = np.sum(lin.pearson.p_values < 0.05)
sig_fdr = np.sum(lin.pearson.significant(alpha=0.05, correction="fdr_bh"))
sig_bonf = np.sum(lin.pearson.significant(alpha=0.05, correction="bonferroni"))
print(f"  Significant at α=0.05:  raw={sig_raw}/{N}  FDR={sig_fdr}/{N}  Bonferroni={sig_bonf}/{N}")

# Result works like an array
arr = np.asarray(lin.pearson)
print(f"  np.asarray(lin.pearson) → shape {arr.shape}, dtype {arr.dtype}")

# Indexing
print(f"  lin.pearson[0] = {lin.pearson[0]:.4f}")
print(f"  lin.pearson[:3] = {lin.pearson[:3]}")

# Math on results
mean_abs_r = np.mean(np.abs(lin.pearson))
print(f"  np.mean(np.abs(lin.pearson)) = {mean_abs_r:.4f}")


red = corr.Redundancy(y, channel_names=channel_names)

print(red.correlation)          # N×N Pearson matrix (Result with p-values)
print(red.partial)              # N×N partial correlation (direct connections)
print(red.acf)                  # per-channel autocorrelation

print(red.rank)                 # effective rank via Shannon entropy of eigenspectrum
print(red.condition)            # condition number (high = redundant)
print(red.eigenvalues)          # raw eigenvalues, sorted descending

print(red.groups(threshold=0.9))        # [[0,1,4], [2], [3,5], ...] index clusters
print(red.groups_named(threshold=0.9))  # [["ch_0","ch_1","ch_4"], ...] name clusters

print(red.summary())            # one-shot diagnostic string
