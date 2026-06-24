#!/usr/bin/env python3
"""
compute_homogenized_stiffness.py
=================================
Load an optimised spring-mass topology from experiment.json (produced by
run_taichi_global_stiffness.py) and compute several homogenised / effective
stiffness measures together with the structural natural frequencies.

Usage
-----
    python compute_homogenized_stiffness.py [EXPERIMENT_JSON] [OPTIONS]

    EXPERIMENT_JSON   Path to experiment.json
                      (default: experiments/Taichi_subspring_global_stiffness/M4_run0/experiment.json)
    --rows  INT       Grid rows   (auto-inferred from square matrix if omitted)
    --cols  INT       Grid cols   (auto-inferred from square matrix if omitted)
    --spacing FLOAT   Node spacing in metres (default: 0.053)
    --mass  FLOAT     Logical mass per main node in kg (default: 0.01)
    --nmodes INT      Number of modes to report (default: 8)
    --plot            Display topology + stiffness colour map
    --save-plot PATH  Save topology plot to this file path

Physical background
-------------------
The experiment.json stores two N×N symmetric adjacency matrices
(N = rows × cols main nodes):

  k_mat_opt[i,j]  spring stiffness between nodes i and j  [N/m]
  c_mat_opt[i,j]  viscous damping coefficient             [N·s/m]

Zero entries mean "no spring" (void material).

Homogenised stiffness measures computed
----------------------------------------
1.  Spring statistics
    k_mean     : arithmetic mean of active spring stiffnesses
    k_harmonic : harmonic mean — relevant when springs are in series (weakest
                 path bottleneck)

2.  Modal analysis  ← most meaningful for resonant frequency
    Assemble the 2-DOF-per-node structural stiffness matrix K (graph
    Laplacian of the spring network, projected along each bond direction).
    Apply fixed-corner boundary conditions. Solve the generalised eigenvalue
    problem K_free φ = ω² M_free φ.

    Natural frequency of the n-th mode:
        f_n  =  (1 / 2π) × √(ω_n²)          [Hz]

    Modal stiffness and mass:
        k_modal,n  =  ω_n² × m_modal,n
        m_modal,n  =  φ_n^T  M  φ_n

    The FIRST non-trivial natural frequency f₁ is the FUNDAMENTAL RESONANT
    FREQUENCY of the structure under the given boundary conditions.

    To target a resonant frequency f_target during topology optimisation:
        k_required  ≈  (2π f_target)²  ×  m_modal,1

3.  Static input stiffness
    Schur-complement condensation gives the DC stiffness felt at the
    actuator node (node 0) in the excitation direction (+x), with all
    internal free nodes condensed out and fixed corners held.

    Rough single-DOF resonant-frequency estimate:
        f₀  ≈  (1/2π) × √(K_in,x / m_modal,1)
    This is exact only if the mode shape concentrates all motion at the
    actuator; use the modal analysis result for better accuracy.

4.  Voigt homogenised stiffness (continuum limit)
    Energy-weighted average over all active bonds:
        C_eff[α,β]  =  (1/A) × Σ_bonds  k_ij × L_ij × n_α × n_β   [N/m]
    where A is the total lattice area and n is the bond unit vector.
    Gives an effective 2-D stiffness modulus per unit width.
    E_eff,x and E_eff,y are the longitudinal moduli in x and y.
    Note: meaningful only when many unit cells are present; treat as
    qualitative for small grids.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

# ── Defaults matching run_taichi_global_stiffness.py ─────────────────────────
_DEFAULT_JSON = (
    Path(__file__).parent.parent
    / "experiments/Taichi_subspring_global_stiffness/M3_run0/experiment.json"
)
DEFAULT_SPACING   = 0.053   # metres
DEFAULT_NODE_MASS = 0.01    # kg per main node
DEFAULT_NMODES    = 8


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_topology(json_path: Path):
    with open(json_path) as f:
        data = json.load(f)
    k_mat = np.array(data["k_mat_opt"])
    c_mat = np.array(data["c_mat_opt"])
    assert k_mat.ndim == 2 and k_mat.shape[0] == k_mat.shape[1], \
        "k_mat_opt must be a square 2-D array"
    return k_mat, c_mat


def infer_grid(N: int, rows=None, cols=None):
    if rows is not None and cols is not None:
        if rows * cols != N:
            raise ValueError(f"rows×cols = {rows*cols} ≠ matrix size {N}")
        return rows, cols
    side = int(round(N ** 0.5))
    if side * side != N:
        raise ValueError(
            f"Cannot auto-infer a square grid for N={N}.  "
            "Pass --rows and --cols explicitly."
        )
    return side, side


# ── Geometry ──────────────────────────────────────────────────────────────────

def build_node_positions(rows: int, cols: int, spacing: float) -> np.ndarray:
    """Node i is at row r = i // cols, col c = i % cols."""
    pos = np.zeros((rows * cols, 2))
    for r in range(rows):
        for c in range(cols):
            pos[r * cols + c] = [c * spacing, -r * spacing]
    return pos


def fixed_corner_nodes(rows: int, cols: int):
    return [cols - 1, (rows - 1) * cols, rows * cols - 1]


# ── Structural matrices ───────────────────────────────────────────────────────

def build_stiffness_matrix(k_mat: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """
    Assemble the 2N×2N structural stiffness matrix from axial springs.
    DOF layout: node i → DOFs [2i, 2i+1] = [x_i, y_i].
    Each spring (i,j) with stiffness k contributes the rank-1 block
        k × n⊗n  (n = unit bond vector)
    to the 2×2 diagonal blocks and its negative to the off-diagonal blocks.
    """
    N = k_mat.shape[0]
    K = np.zeros((2 * N, 2 * N))
    for i in range(N):
        for j in range(i + 1, N):
            k_ij = k_mat[i, j]
            if k_ij == 0.0:
                continue
            d = positions[j] - positions[i]
            L = np.linalg.norm(d)
            if L < 1e-14:
                continue
            n  = d / L
            Ks = k_ij * np.outer(n, n)      # 2×2 axial stiffness block
            K[2*i:2*i+2, 2*i:2*i+2] += Ks
            K[2*j:2*j+2, 2*j:2*j+2] += Ks
            K[2*i:2*i+2, 2*j:2*j+2] -= Ks
            K[2*j:2*j+2, 2*i:2*i+2] -= Ks
    return K


def build_mass_matrix(N: int, node_mass: float) -> np.ndarray:
    return np.diag(np.full(2 * N, node_mass))


# ── Modal analysis ────────────────────────────────────────────────────────────

def compute_natural_frequencies(
    K: np.ndarray,
    M: np.ndarray,
    fixed_nodes: list,
    N: int,
    n_modes: int = 8,
):
    """
    Solve the generalised eigenvalue problem on the free DOFs:
        K_free φ = ω² M_free φ

    Returns
    -------
    freqs_hz    : first n_modes natural frequencies [Hz]
    omega_sq    : corresponding ω² values [rad²/s²]
    modes       : eigenvectors (columns), shape (n_free_dofs, n_modes)
    free_dofs   : DOF indices that are free (used to map modes back to full system)
    """
    fixed_dofs = [dof for n in fixed_nodes for dof in (2 * n, 2 * n + 1)]
    free_dofs  = np.setdiff1d(np.arange(2 * N), fixed_dofs)

    K_free = K[np.ix_(free_dofs, free_dofs)]
    M_free = M[np.ix_(free_dofs, free_dofs)]

    eigvals, eigvecs = eigh(K_free, M_free)   # ascending order

    # Discard numerically-zero (rigid-body) modes
    tol  = max(eigvals[-1] * 1e-8, 1e-6)
    mask = eigvals > tol
    omega_sq = eigvals[mask]
    modes    = eigvecs[:, mask]

    freqs_hz = np.sqrt(omega_sq) / (2.0 * np.pi)
    n_show   = min(n_modes, len(freqs_hz))
    return freqs_hz[:n_show], omega_sq[:n_show], modes[:, :n_show], free_dofs


def compute_modal_quantities(omega_sq, M_free, modes):
    """
    Renormalise each mode shape to unit maximum displacement, then compute:
        m_modal,n  =  φ_n^T M φ_n   [kg]   — mass participating in mode n
        k_modal,n  =  ω_n² × m_modal,n   [N/m]

    scipy.linalg.eigh returns mass-normalised vectors (φ^T M φ = I kg⁻¹),
    so we rescale to unit-max-displacement before reporting physical units.
    """
    modes_phys = modes.copy()
    for col in range(modes_phys.shape[1]):
        scale = np.max(np.abs(modes_phys[:, col]))
        if scale > 1e-14:
            modes_phys[:, col] /= scale
    m_modal = np.einsum("ij,ij->j", modes_phys, M_free @ modes_phys)
    k_modal = omega_sq * m_modal
    return k_modal, m_modal


# ── Static input stiffness ────────────────────────────────────────────────────

def compute_input_stiffness(
    K: np.ndarray,
    actuator_node: int,
    fixed_nodes: list,
    N: int,
):
    """
    Static input stiffness at the actuator via Schur-complement condensation.

    The actuator DOFs are retained; all other free DOFs are condensed out;
    fixed-corner DOFs are removed.

    Returns (k_in_x, k_in_y, K_eff_2x2)
        k_in_x  : stiffness in excitation direction (+x) [N/m]
        k_in_y  : transverse stiffness (+y) [N/m]
        K_eff   : 2×2 effective stiffness matrix at the actuator
    """
    fixed_dofs  = {dof for n in fixed_nodes for dof in (2 * n, 2 * n + 1)}
    act_dofs    = [2 * actuator_node, 2 * actuator_node + 1]
    other_free  = [d for d in range(2 * N) if d not in fixed_dofs and d not in act_dofs]

    if len(other_free) == 0:
        return float(K[act_dofs[0], act_dofs[0]]), float(K[act_dofs[1], act_dofs[1]]), None

    K_aa = K[np.ix_(act_dofs,   act_dofs)]
    K_af = K[np.ix_(act_dofs,   other_free)]
    K_ff = K[np.ix_(other_free, other_free)]
    K_fa = K[np.ix_(other_free, act_dofs)]

    try:
        # Use least-squares / pseudoinverse to handle floating DOFs gracefully.
        # A DOF with no stiffness connections (e.g. x-DOF of a node whose only
        # horizontal spring is void) contributes zero reaction force; pinv
        # correctly returns zero for those rows, leaving K_eff well-defined.
        K_ff_inv_Kfa = np.linalg.lstsq(K_ff, K_fa, rcond=None)[0]
        K_eff = K_aa - K_af @ K_ff_inv_Kfa
        return float(K_eff[0, 0]), float(K_eff[1, 1]), K_eff
    except np.linalg.LinAlgError:
        return float("nan"), float("nan"), None


# ── Spring statistics ─────────────────────────────────────────────────────────

def spring_statistics(k_mat: np.ndarray, rows: int, cols: int):
    N = k_mat.shape[0]
    active = np.array(
        [k_mat[i, j] for i in range(N) for j in range(i + 1, N) if k_mat[i, j] > 0]
    )
    n_possible = rows * (cols - 1) + (rows - 1) * cols   # nearest-neighbour bonds only
    if active.size == 0:
        return {"n_active": 0, "n_possible": n_possible}
    return {
        "n_active":    int(active.size),
        "n_possible":  n_possible,
        "k_mean":      float(np.mean(active)),
        "k_harmonic":  float(active.size / np.sum(1.0 / active)),
        "k_std":       float(np.std(active)),
        "k_min":       float(np.min(active)),
        "k_max":       float(np.max(active)),
        "k_parallel":  float(np.sum(active)),
    }


# ── Voigt homogenised stiffness ───────────────────────────────────────────────

def voigt_homogenized_stiffness(k_mat: np.ndarray, positions: np.ndarray, spacing: float):
    """
    Energy-based Voigt homogenisation for a 2-D lattice.

        C_eff[α,β]  =  (1/A) × Σ_bonds  k_ij × L_ij × n_α × n_β   [N/m]

    A = total lattice area.  L_ij = equilibrium bond length.
    n = unit vector along bond.

    Gives the effective 2-D stiffness modulus tensor (units: N/m).
    """
    N   = k_mat.shape[0]
    A   = np.prod(positions.max(axis=0) - positions.min(axis=0))   # bounding-box area
    if A < 1e-20:
        return np.zeros((2, 2))

    C = np.zeros((2, 2))
    for i in range(N):
        for j in range(i + 1, N):
            k_ij = k_mat[i, j]
            if k_ij == 0.0:
                continue
            d = positions[j] - positions[i]
            L = np.linalg.norm(d)
            if L < 1e-14:
                continue
            n  = d / L
            C += (k_ij * L / A) * np.outer(n, n)
    return C


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_topology(
    k_mat, positions, fixed_nodes, actuator_node,
    freqs_hz=None, title="Optimised Topology", save_path=None
):
    from matplotlib.lines import Line2D

    N      = k_mat.shape[0]
    active = [k_mat[i, j] for i in range(N) for j in range(i + 1, N) if k_mat[i, j] > 0]
    if not active:
        print("[Warning] No active springs — nothing to plot.")
        return

    k_max = max(active)
    cmap  = plt.colormaps["plasma"]
    norm  = mcolors.Normalize(vmin=0, vmax=k_max)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

    for i in range(N):
        for j in range(i + 1, N):
            k_ij = k_mat[i, j]
            if k_ij <= 0:
                continue
            xi, yi = positions[i]
            xj, yj = positions[j]
            lw     = 1.0 + 5.0 * (k_ij / k_max)
            ax.plot([xi, xj], [yi, yj], c=cmap(norm(k_ij)), lw=lw,
                    solid_capstyle="round")

    node_colors = []
    for i in range(N):
        if i in fixed_nodes:
            node_colors.append("royalblue")
        elif i == actuator_node:
            node_colors.append("limegreen")
        else:
            node_colors.append("tomato")

    ax.scatter(positions[:, 0], positions[:, 1],
               c=node_colors, s=100, zorder=5,
               edgecolors="k", linewidths=0.6)

    for i in range(N):
        ax.annotate(str(i), positions[i],
                    textcoords="offset points", xytext=(4, 4),
                    fontsize=6, color="k")

    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="limegreen",
               markersize=9, label="Actuator (node 0, input)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="royalblue",
               markersize=9, label="Fixed boundary"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tomato",
               markersize=9, label="Free node"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=7)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Spring stiffness  k  (N/m)",
                 fraction=0.03, pad=0.04)

    freq_str = f"   f₁ = {freqs_hz[0]:.2f} Hz" if freqs_hz is not None else ""
    ax.set_title(title + freq_str, fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[Saved] Topology plot → {save_path}")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compute homogenised stiffness of an optimised spring-mass topology.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "json_path", nargs="?", default=str(_DEFAULT_JSON),
        help="Path to experiment.json  (default: M4_run0/experiment.json)"
    )
    parser.add_argument("--rows",     type=int,   default=None,
                        help="Grid rows  (auto-inferred if matrix is square)")
    parser.add_argument("--cols",     type=int,   default=None,
                        help="Grid cols  (auto-inferred if matrix is square)")
    parser.add_argument("--spacing",  type=float, default=DEFAULT_SPACING,
                        help=f"Node spacing in metres  (default: {DEFAULT_SPACING})")
    parser.add_argument("--mass",     type=float, default=DEFAULT_NODE_MASS,
                        help=f"Logical mass per main node in kg  (default: {DEFAULT_NODE_MASS})")
    parser.add_argument("--nmodes",   type=int,   default=DEFAULT_NMODES,
                        help=f"Number of modes to report  (default: {DEFAULT_NMODES})")
    parser.add_argument("--plot",     action="store_true",
                        help="Display the topology + stiffness colour map")
    parser.add_argument("--save-plot", dest="save_plot", type=str, default=None,
                        help="Save topology plot to this file path")
    args = parser.parse_args(argv)

    # ── Load ─────────────────────────────────────────────────────────────────
    json_path = Path(args.json_path)
    if not json_path.exists():
        print(f"[Error] File not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\n[Load]  {json_path.resolve()}")
    k_mat, c_mat = load_topology(json_path)
    N = k_mat.shape[0]

    rows, cols   = infer_grid(N, args.rows, args.cols)
    positions    = build_node_positions(rows, cols, args.spacing)
    fixed_nodes  = fixed_corner_nodes(rows, cols)
    actuator_node = 0

    m_total = N * args.mass

    print(f"        Grid    : {rows}×{cols}  ({N} main nodes)")
    print(f"        Spacing : {args.spacing} m  |  Mass/node : {args.mass} kg  |  m_total : {m_total:.4f} kg")
    print(f"        Fixed   : nodes {fixed_nodes}  |  Actuator : node {actuator_node}")

    # ── 1. Spring statistics ──────────────────────────────────────────────────
    stats = spring_statistics(k_mat, rows, cols)
    print(f"\n{'═'*62}")
    print(f"  1 · Spring Statistics")
    print(f"{'═'*62}")
    if stats.get("n_active", 0) == 0:
        print("  No active springs found — topology is void.")
        return
    print(f"  Active springs  : {stats['n_active']} / {stats['n_possible']}")
    print(f"  k_mean          : {stats['k_mean']:.3f} N/m")
    print(f"  k_harmonic      : {stats['k_harmonic']:.3f} N/m  (series-path bottleneck)")
    print(f"  k_std           : {stats['k_std']:.3f} N/m")
    print(f"  k_min / k_max   : {stats['k_min']:.3f} / {stats['k_max']:.3f} N/m")
    print(f"  k_parallel      : {stats['k_parallel']:.3f} N/m  (all springs in parallel)")

    # ── Build structural matrices ─────────────────────────────────────────────
    K = build_stiffness_matrix(k_mat, positions)
    M = build_mass_matrix(N, args.mass)

    # ── 2. Modal analysis ─────────────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print(f"  2 · Modal Analysis  (fixed nodes: {fixed_nodes}, actuator free)")
    print(f"{'═'*62}")

    freqs_hz = None
    try:
        freqs_hz, omega_sq, modes, free_dofs = compute_natural_frequencies(
            K, M, fixed_nodes, N, n_modes=args.nmodes
        )
        M_free             = M[np.ix_(free_dofs, free_dofs)]
        k_modal, m_modal   = compute_modal_quantities(omega_sq, M_free, modes)

        print(f"  {'Mode':>4}  {'f (Hz)':>10}  {'ω (rad/s)':>10}  "
              f"{'k_modal (N/m)':>14}  {'m_modal (kg)':>13}")
        print(f"  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*14}  {'─'*13}")
        for idx in range(len(freqs_hz)):
            print(
                f"  {idx+1:>4}  {freqs_hz[idx]:>10.3f}  "
                f"{np.sqrt(omega_sq[idx]):>10.3f}  "
                f"{k_modal[idx]:>14.2f}  "
                f"{m_modal[idx]:>13.6f}"
            )
        f1, k1, m1 = freqs_hz[0], k_modal[0], m_modal[0]
        print(f"\n  Fundamental resonant frequency  : f₁  = {f1:.4f} Hz")
        print(f"  Corresponding modal stiffness   : k₁  = {k1:.2f} N/m")
        print(f"  Corresponding modal mass        : m₁  = {m1:.6f} kg")
        print(f"  Modal mass fraction             : m₁/m_total = {m1/m_total:.4f}")
    except Exception as exc:
        print(f"  [Error] Eigenvalue analysis failed: {exc}")

    # ── 3. Static input stiffness ─────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print(f"  3 · Static Input Stiffness at Actuator (node {actuator_node})")
    print(f"{'═'*62}")
    try:
        k_in_x, k_in_y, _ = compute_input_stiffness(
            K, actuator_node, fixed_nodes, N
        )
        print(f"  K_in,x  (excitation direction)  : {k_in_x:.3f} N/m")
        print(f"  K_in,y  (transverse)            : {k_in_y:.3f} N/m")
        if freqs_hz is not None:
            f_approx = (1.0 / (2.0 * np.pi)) * np.sqrt(k_in_x / m1)
            print(f"  Single-DOF estimate f₀ ≈ (1/2π)√(K_in,x / m₁)  = {f_approx:.4f} Hz")
            print(f"  (cf. modal f₁ = {freqs_hz[0]:.4f} Hz)")
    except Exception as exc:
        print(f"  [Error] Input stiffness failed: {exc}")

    # ── 4. Voigt homogenised stiffness ────────────────────────────────────────
    print(f"\n{'═'*62}")
    print(f"  4 · Voigt Homogenised Stiffness Tensor")
    print(f"{'═'*62}")
    C = voigt_homogenized_stiffness(k_mat, positions, args.spacing)
    print(f"  C_eff  =  [[{C[0,0]:8.2f},  {C[0,1]:8.2f}]")
    print(f"             [{C[1,0]:8.2f},  {C[1,1]:8.2f}]]  N/m")
    print(f"  E_eff,x  (longitudinal x) : {C[0,0]:.2f} N/m")
    print(f"  E_eff,y  (longitudinal y) : {C[1,1]:.2f} N/m")
    print(f"  G_eff    (shear coupling) : {C[0,1]:.2f} N/m")
    rho_eff = m_total / np.prod(positions.max(0) - positions.min(0))
    if C[0, 0] > 0 and rho_eff > 0:
        c_x = np.sqrt(C[0, 0] / rho_eff)
        f_wave_x = c_x / (2.0 * (cols - 1) * args.spacing)
        print(f"  Longitudinal wave speed c_x  : {c_x:.4f} m/s")
        print(f"  Wave-based f estimate (x)    : {f_wave_x:.4f} Hz  "
              f"(f = c_x / 2L_x)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print(f"  Summary — Correlating topology to resonant frequency")
    print(f"{'═'*62}")
    print("""
  ┌──────────────────────────────────────────────────────────┐
  │  Measure              │  What it tells you               │
  ├──────────────────────────────────────────────────────────┤
  │  f₁  (modal, §2)      │  ACTUAL fundamental resonance    │
  │                       │  under experimental BCs — best   │
  │                       │  predictor of measured peak freq │
  ├──────────────────────────────────────────────────────────┤
  │  k₁, m₁  (modal)      │  Effective spring/mass for the   │
  │                       │  dominant mode; use to set f₁ by │
  │                       │  topology design                  │
  ├──────────────────────────────────────────────────────────┤
  │  K_in,x  (Schur, §3)  │  DC stiffness at actuator;       │
  │                       │  quick proxy for f₀ when mode    │
  │                       │  shape is localised at actuator  │
  ├──────────────────────────────────────────────────────────┤
  │  k_harmonic  (§1)     │  Weakest-link estimate; relevant │
  │                       │  when signal must propagate along │
  │                       │  a series path                   │
  ├──────────────────────────────────────────────────────────┤
  │  E_eff,x  (Voigt, §4) │  Continuum modulus; valid for    │
  │                       │  large periodic lattices; sets    │
  │                       │  longitudinal wave speed         │
  └──────────────────────────────────────────────────────────┘

  To target a resonant frequency f_target:
      k_required  =  (2π · f_target)²  ×  m₁
      e.g. f_target = 5 Hz, m₁ = 0.005 kg  →  k_req ≈ 4.93 N/m
""")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if args.plot or args.save_plot:
        title = f"{json_path.parent.name}"
        save_path = args.save_plot
        if args.plot and save_path is None:
            # show interactively
            plot_topology(k_mat, positions, fixed_nodes, actuator_node,
                          freqs_hz=freqs_hz, title=title, save_path=None)
        elif save_path:
            plot_topology(k_mat, positions, fixed_nodes, actuator_node,
                          freqs_hz=freqs_hz, title=title, save_path=save_path)
            if args.plot:
                plot_topology(k_mat, positions, fixed_nodes, actuator_node,
                              freqs_hz=freqs_hz, title=title, save_path=None)


if __name__ == "__main__":
    main()
