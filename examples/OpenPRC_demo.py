"""
===============================================================================
OpenPRC — Comprehensive Demo
===============================================================================

Single-script walkthrough of the full OpenPRC pipeline:

    Stage 1 · Simulation       demlat  →  simulation.h5
    Stage 2 · Reservoir        reservoir → feature matrix, trainer
    Stage 3 · Diagnostics      analysis.correlation → Linear / Nonparametric / Redundancy
    Stage 4 · Benchmarking     analysis → NARMA2, IPC memory benchmark
    Stage 5 · Summary figure   multi-panel publication-style plot

Run:
    python openprc_demo.py

Dependencies:
    openprc, numpy, matplotlib, scipy, sklearn
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler

# ── OpenPRC imports ──────────────────────────────────────────────────────────
from openprc import demlat
from openprc.demlat import SimulationSetup, BarHingeModel
from openprc.reservoir.io.state_loader import StateLoader
from openprc.reservoir.features.node_features import NodePositions
from openprc.reservoir.readout.ridge import Ridge
from openprc.reservoir.training.trainer import Trainer
from openprc.analysis import correlation as corr
from openprc.analysis.benchmarks.narma_benchmark import NARMABenchmark
from openprc.analysis.benchmarks.memory_benchmark import MemoryBenchmark

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        9,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.linewidth":   0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "figure.dpi":       150,
})

C = {                          # module colour palette
    "demlat":    "#2E86AB",
    "vision":    "#5C9E6B",
    "reservoir": "#E07B39",
    "analysis":  "#8B5E9E",
    "optimize":  "#C0392B",
    "neutral":   "#444444",
    "grid":      "#EEEEEE",
}

EXPERIMENT_DIR = Path("./experiments/demo_miura_4x8")

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 · SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def stage1_simulate():
    print("\n── Stage 1: demlat simulation ──────────────────────────────────")

    setup = SimulationSetup(str(EXPERIMENT_DIR), overwrite=True)
    setup.set_simulation_params(duration=10.0, dt=0.001, save_interval=0.01)
    setup.set_physics(gravity=0, damping=0.8)

    # geometry via user geometry helper (replace with your own)
    from openprc.demlat.examples.miura import create_miura_ori_geometry
    faces, meta = create_miura_ori_geometry(
        setup=setup, xn=4, yn=8,
        a=0.053, b=0.053, gamma_deg=45.0, theta_deg=25.0,
        k_axial=222.15, k_fold=0.01, k_facet=0.01,
        c_axial=0.01, c_hinge=0.01, mass=0.01,
    )

    # band-limited random base excitation
    from scipy.interpolate import CubicSpline
    t        = np.arange(0.0, 10.0, 0.0005)
    t_coarse = np.arange(0.0, 10.033, 0.033)
    u_coarse = np.random.default_rng(42).uniform(-1.0, 1.0, size=len(t_coarse))
    disp     = 0.001 * CubicSpline(t_coarse, u_coarse)(t)

    base_idx = [meta["get_idx"](i, 0) for i in range(meta["i_max"])]
    for k, idx in enumerate(base_idx):
        p0  = np.array(setup.nodes["positions"][idx], dtype=float)
        sig = np.tile(p0, (len(t), 1));  sig[:, 2] += disp
        setup.add_signal(f"sig_base_{k}", sig, dt=0.0005)
        setup.add_actuator(idx, f"sig_base_{k}", type="position")

    setup.save()
    demlat.Engine(BarHingeModel, backend="cuda").run(
        demlat.Simulation(str(EXPERIMENT_DIR)))
    print("  simulation.h5 written")


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 · LOAD RESERVOIR STATE
# ═══════════════════════════════════════════════════════════════════════════════

def stage2_load():
    print("\n── Stage 2: reservoir state ─────────────────────────────────────")
    sim_path = EXPERIMENT_DIR / "output" / "simulation.h5"
    loader   = StateLoader(sim_path)
    Y        = NodePositions(dims=[2]).transform(loader)          # (T, N)
    u_raw    = loader.get_actuation_signal(actuator_idx=0, dof=2)
    u_dyn    = u_raw - u_raw[0]
    x        = 0.5 * (u_dyn - u_dyn.min()) / (u_dyn.max() - u_dyn.min())
    print(f"  state matrix Y: {Y.shape}   input x: {x.shape}")
    return loader, Y, x


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 · CORRELATION DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════

def stage3_diagnostics(Y, x):
    print("\n── Stage 3: correlation diagnostics ────────────────────────────")
    lin = corr.Linear(x, Y, lag_sweep=True)
    nr  = corr.Nonparametric(x, Y)
    red = corr.Redundancy(Y)

    print(f"  Pearson  mean|r| = {np.mean(np.abs(lin.pearson.values)):.3f}")
    print(f"  dCor     mean    = {np.mean(nr.dcor.values):.3f}")
    print(f"  Eff. rank        = {red.rank:.1f} / {Y.shape[1]}")
    print(f"  Condition number = {red.condition:.2e}")
    return lin, nr, red


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4 · BENCHMARKING
# ═══════════════════════════════════════════════════════════════════════════════

def stage4_benchmark(loader, Y, x):
    print("\n── Stage 4: benchmarking ────────────────────────────────────────")
    features = NodePositions(dims=[2])
    trainer  = Trainer(
        loader=loader, features=features, readout=Ridge(1e-5),
        experiment_dir=EXPERIMENT_DIR,
        washout=5.0, train_duration=10.0, test_duration=10.0,
    )

    # NARMA2
    narma_score = NARMABenchmark(group_name="narma_demo").run(trainer, x, order=2)
    narma_score.save()
    _, target, prediction = narma_score.trainer_result.cache["test"]
    nrmse = np.sqrt(np.mean((target - prediction)**2)) / np.std(target)
    print(f"  NARMA2 NRMSE = {nrmse:.4f}")

    # IPC / memory benchmark
    X_std = StandardScaler().fit_transform(features.transform(loader))
    s     = np.linalg.svd(X_std, compute_uv=False)
    N_eff = float(np.exp(-np.sum((s/s.sum()) * np.log(s/s.sum() + 1e-12))))
    eps   = float(2 * chi2.isf(1e-4, df=N_eff) / int(10.0 / loader.dt))

    mem_score = MemoryBenchmark(group_name="ipc_demo").run(
        trainer, x, tau_s=30, n_s=2, k_delay=1, eps=eps, ridge=1e-6)
    mem_score.save()

    mc_lin    = mem_score.metrics.get("MC_linear",    0.0)
    mc_nonlin = mem_score.metrics.get("MC_nonlinear", 0.0)
    print(f"  MC_linear = {mc_lin:.3f}   MC_nonlinear = {mc_nonlin:.3f}")

    return target, prediction, nrmse, mem_score, mc_lin, mc_nonlin


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 5 · SUMMARY FIGURE
# ═══════════════════════════════════════════════════════════════════════════════

def stage5_figure(Y, x, lin, nr, red,
                  target, prediction, nrmse,
                  mem_score, mc_lin, mc_nonlin):
    print("\n── Stage 5: summary figure ──────────────────────────────────────")

    N = Y.shape[1]
    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor("white")

    # ── layout ────────────────────────────────────────────────────────────────
    outer = gridspec.GridSpec(3, 4, figure=fig,
                              hspace=0.52, wspace=0.42,
                              top=0.91, bottom=0.07, left=0.06, right=0.97)

    def ax(row, col, rowspan=1, colspan=1):
        return fig.add_subplot(outer[row:row+rowspan, col:col+colspan])

    # ── title bar ─────────────────────────────────────────────────────────────
    fig.text(0.5, 0.965, "OpenPRC  —  Miura-ori Reservoir  ·  Demo Summary",
             ha="center", va="top", fontsize=13, fontweight="bold",
             color=C["neutral"])
    fig.text(0.5, 0.942,
             f"N = {N} channels  ·  T = {Y.shape[0]} frames  ·  "
             f"NARMA2 NRMSE = {nrmse:.4f}  ·  "
             f"MC_lin = {mc_lin:.2f}  ·  MC_nonlin = {mc_nonlin:.2f}",
             ha="center", va="top", fontsize=8.5, color="#666666")

    # ── helper ────────────────────────────────────────────────────────────────
    def style(a, title, xlabel="", ylabel="", color=C["neutral"]):
        a.set_title(title, fontsize=8.5, fontweight="bold",
                    color=color, pad=4, loc="left")
        if xlabel: a.set_xlabel(xlabel, fontsize=7.5)
        if ylabel: a.set_ylabel(ylabel, fontsize=7.5)
        a.tick_params(labelsize=7)
        a.set_facecolor("white")
        a.grid(color=C["grid"], linewidth=0.5, zorder=0)

    # ══ Row 0 ══════════════════════════════════════════════════════════════════

    # [0,0] Input signal
    a00 = ax(0, 0)
    t_ax = np.linspace(0, 10, len(x))
    a00.plot(t_ax, x, color=C["demlat"], lw=0.8, alpha=0.9)
    a00.fill_between(t_ax, x, alpha=0.12, color=C["demlat"])
    style(a00, "Input signal  u(t)", "Time (s)", "Amplitude", C["demlat"])

    # [0,1] Node trajectories (first 8 channels)
    a01 = ax(0, 1)
    show = min(8, N)
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, show))
    for i in range(show):
        a01.plot(t_ax, Y[:, i], lw=0.6, color=cmap[i], alpha=0.8)
    style(a01, "Reservoir trajectories (8 ch.)", "Time (s)", "z-position", C["reservoir"])

    # [0,2] Pearson r bar chart
    a02 = ax(0, 2)
    r_vals = lin.pearson.values
    colors = [C["analysis"] if abs(v) > 0.3 else "#BBBBBB" for v in r_vals]
    a02.bar(np.arange(N), r_vals, color=colors, width=0.7, zorder=3)
    a02.axhline(0, color=C["neutral"], lw=0.6)
    a02.axhline( 0.3, color=C["analysis"], lw=0.6, ls="--", alpha=0.5)
    a02.axhline(-0.3, color=C["analysis"], lw=0.6, ls="--", alpha=0.5)
    style(a02, "Pearson r  (zero-lag)", "Channel", "r", C["analysis"])

    # [0,3] dCor vs |Pearson|  scatter
    a03 = ax(0, 3)
    dc_vals = nr.dcor.values
    scatter_c = [C["optimize"] if (abs(r_vals[i]) < 0.2 and dc_vals[i] > 0.3)
                 else C["neutral"] for i in range(N)]
    a03.scatter(np.abs(r_vals), dc_vals, c=scatter_c,
                s=28, alpha=0.85, zorder=3, edgecolors="none")
    lim = max(np.max(np.abs(r_vals)), np.max(dc_vals)) * 1.05
    a03.plot([0, lim], [0, lim], "--", color="#AAAAAA", lw=0.8)
    a03.text(0.03, 0.92, "● nonlinear channel", transform=a03.transAxes,
             fontsize=7, color=C["optimize"])
    style(a03, "dCor vs |Pearson|", "|Pearson r|", "dCor", C["analysis"])

    # ══ Row 1 ══════════════════════════════════════════════════════════════════

    # [1,0-1] CCF lag profiles (first 6 channels)
    a10 = ax(1, 0, colspan=2)
    show_ccf  = min(6, N)
    ccf_full  = lin.ccf.extra.get("ccf_full", None)   # (N, 2*max_lag+1)
    if ccf_full is not None:
        max_lag = (ccf_full.shape[1] - 1) // 2
        lags    = np.arange(-max_lag, max_lag + 1)
        cmap2   = plt.cm.plasma(np.linspace(0.1, 0.85, show_ccf))
        for i in range(show_ccf):
            a10.plot(lags, ccf_full[i], lw=0.7, color=cmap2[i],
                     alpha=0.8, label=f"ch{i}")
            opt_lag = int(lin.ccf.lags[i])
            a10.axvline(opt_lag, color=cmap2[i], lw=0.5, ls=":", alpha=0.6)
        a10.legend(fontsize=6.5, ncol=3, loc="upper right",
                   framealpha=0.7, edgecolor="none")
    else:
        a10.bar(np.arange(N), np.abs(lin.ccf.values),
                color=C["analysis"], alpha=0.7, zorder=3)
    style(a10, "CCF lag profiles  (6 channels)", "Lag (frames)", "CCF", C["analysis"])

    # [1,2] Partial correlation heatmap
    a12 = ax(1, 2)
    pmat  = lin.partial.values
    lim_p = np.nanpercentile(np.abs(pmat), 95) or 1.0
    norm  = TwoSlopeNorm(vmin=-lim_p, vcenter=0, vmax=lim_p)
    im    = a12.imshow(pmat, cmap="RdBu_r", norm=norm, aspect="auto")
    plt.colorbar(im, ax=a12, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)
    style(a12, "Partial correlation", "Channel", "Channel", C["analysis"])

    # [1,3] Redundancy: eigenvalue spectrum
    a13 = ax(1, 3)
    eigs = red.eigenvalues
    a13.bar(np.arange(len(eigs)), eigs / eigs.sum(),
            color=C["analysis"], alpha=0.8, zorder=3)
    a13.axhline(1 / N, color=C["optimize"], lw=0.9, ls="--",
                label=f"uniform (1/N)")
    a13.text(0.97, 0.93, f"eff. rank = {red.rank:.1f}",
             transform=a13.transAxes, ha="right", fontsize=7.5,
             color=C["analysis"], fontweight="bold")
    a13.text(0.97, 0.83, f"κ = {red.condition:.1e}",
             transform=a13.transAxes, ha="right", fontsize=7.5,
             color=C["neutral"])
    a13.legend(fontsize=7, framealpha=0.6, edgecolor="none")
    style(a13, "Eigenvalue spectrum", "Mode", "Norm. eigenvalue", C["analysis"])

    # ══ Row 2 ══════════════════════════════════════════════════════════════════

    # [2,0-1] NARMA2 prediction vs target
    a20 = ax(2, 0, colspan=2)
    t_test = np.arange(len(target))
    a20.plot(t_test, target,     color=C["neutral"],   lw=0.9,
             label="Target",     alpha=0.85)
    a20.plot(t_test, prediction, color=C["reservoir"], lw=0.9,
             label="Prediction", alpha=0.9, ls="--")
    a20.legend(fontsize=7.5, framealpha=0.7, edgecolor="none")
    a20.text(0.97, 0.92, f"NRMSE = {nrmse:.4f}",
             transform=a20.transAxes, ha="right", fontsize=8,
             color=C["reservoir"], fontweight="bold")
    style(a20, "NARMA2 benchmark — target vs. readout",
          "Test frame", "Signal value", C["reservoir"])

    # [2,2] Linear MC bar chart
    a22 = ax(2, 2)
    if hasattr(mem_score, "capacities"):
        caps  = mem_score.capacities            # dict lag → C_raw
        lags  = sorted(caps.keys())
        c_lin = [caps[l].get("deg1", 0.0) for l in lags]
        c_nl  = [caps[l].get("deg2", 0.0) for l in lags]
        a22.bar(lags, c_lin, color=C["reservoir"],  label="Linear",    alpha=0.85)
        a22.bar(lags, c_nl,  bottom=c_lin,
                color=C["optimize"],  label="Nonlinear", alpha=0.75)
    else:
        a22.bar([0, 1], [mc_lin, mc_nonlin],
                color=[C["reservoir"], C["optimize"]], alpha=0.85, zorder=3)
        a22.set_xticks([0, 1])
        a22.set_xticklabels(["MC_lin", "MC_nonlin"], fontsize=8)
    a22.legend(fontsize=7, framealpha=0.6, edgecolor="none")
    style(a22, "IPC decomposition", "Delay τ", "Capacity", C["optimize"])

    # [2,3] Pipeline summary text panel
    a23 = ax(2, 3)
    a23.axis("off")
    summary = (
        f"{'OpenPRC pipeline summary':^28}\n"
        f"{'─'*28}\n\n"
        f"  Substrate   Miura-ori 4×8\n"
        f"  Channels    {N}\n"
        f"  Frames      {Y.shape[0]}\n\n"
        f"  Pearson sig.  "
        f"{np.sum(lin.pearson.p_values < 0.05)}/{N}\n"
        f"  Eff. rank     {red.rank:.1f} / {N}\n"
        f"  Cond. no.     {red.condition:.1e}\n\n"
        f"  NARMA2 NRMSE  {nrmse:.4f}\n"
        f"  MC linear     {mc_lin:.3f}\n"
        f"  MC nonlinear  {mc_nonlin:.3f}\n"
        f"  IPC total     {mc_lin+mc_nonlin:.3f}\n"
    )
    a23.text(0.05, 0.97, summary, transform=a23.transAxes,
             va="top", ha="left", fontsize=7.8,
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.6", fc="#F7F7F7",
                       ec="#CCCCCC", lw=0.8))

    # ── module colour legend ───────────────────────────────────────────────────
    handles = [plt.Rectangle((0,0),1,1, fc=v, ec="none")
               for v in [C["demlat"], C["reservoir"],
                         C["analysis"], C["optimize"]]]
    labels  = ["demlat", "reservoir", "analysis", "optimize"]
    fig.legend(handles, labels, loc="lower center", ncol=4,
               fontsize=8, framealpha=0.0, edgecolor="none",
               bbox_to_anchor=(0.5, 0.005))

    out = EXPERIMENT_DIR / "openprc_demo_summary.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"),
                bbox_inches="tight", dpi=180)
    print(f"  saved → {out}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-sim", action="store_true",
                        help="Skip Stage 1 if simulation.h5 already exists")
    args = parser.parse_args()

    sim_h5 = EXPERIMENT_DIR / "output" / "simulation.h5"

    if not args.skip_sim or not sim_h5.exists():
        stage1_simulate()
    else:
        print("\n── Stage 1: using existing simulation.h5 (--skip-sim) ──────")

    loader, Y, x     = stage2_load()
    lin, nr, red      = stage3_diagnostics(Y, x)
    target, pred, nrmse, mem, mc_lin, mc_nonlin = stage4_benchmark(loader, Y, x)
    stage5_figure(Y, x, lin, nr, red,
                  target, pred, nrmse,
                  mem, mc_lin, mc_nonlin)