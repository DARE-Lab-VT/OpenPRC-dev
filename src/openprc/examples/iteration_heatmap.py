"""openprc-faithful per-iteration R2 memory-capacity heatmap.

This module replicates, in differentiable torch, the exact basis generation and R2
(Information Processing Capacity) algorithm that openprc uses for its final heatmap, so a
heatmap can be produced *every optimization epoch* from the reservoir states the loop
already simulates -- with no extra physics and no redundant re-simulation.

Authoritative references (all under ``openprc/``):
  * R2 engine        : ``analysis/utils/training_utils.py::compute_ipc_components`` /
                       ``compute_ipc_components_gpu``
  * heatmap assembly : ``examples/plot_heatmap_sim2real_bending.py::
                       run_heatmap_pipeline_for_topology`` (cell = ``nanmean(capacities)``)
  * plotting         : ``examples/plot_heatmap_sim2real_bending.py::plot_heatmap``
  * feature build    : ``reservoir/features/node_features.py::NodeDisplacements``
                       (reference_node=0, dims=[0,1]) + ``StandardScaler``

The caller (the optimization loop) hands this module raw waveforms; everything
openprc-specific lives here.
"""

import os

import numpy as np
import torch
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Basis generation (normalized Legendre over [-1, 1]-scaled, lagged input)
# ---------------------------------------------------------------------------
def legendre_P_normalized(n, x):
    """Normalized Legendre polynomial P_n(x), verbatim from
    ``training_utils.py::compute_ipc_components.legendre_P_normalized`` so the basis is
    bit-identical to openprc's."""
    if n == 0:
        return np.ones_like(x)
    pm2, pm1 = np.ones_like(x), x
    for k in range(1, n):
        curr = ((2 * k + 1) * x * pm1 - k * pm2) / (k + 1)
        pm2, pm1 = pm1, curr
    return np.sqrt(2 * n + 1) * pm1


def _enumerate_exponents(Lvars, n):
    """Stars-and-bars enumeration of exponent vectors of total degree 1..n over ``Lvars``
    lag variables -- identical ordering to ``generate_benchmark_target_matrix`` and
    ``compute_ipc_components``."""
    exps = []
    vec = np.zeros(Lvars, dtype=np.int16)

    def rec(rem, idx):
        if idx == Lvars - 1:
            vec[idx] = rem
            exps.append(vec.copy())
        else:
            for v in range(rem + 1):
                vec[idx] = v
                rec(rem - v, idx + 1)

    for d in range(1, n + 1):
        rec(d, 0)
    return np.asarray(exps, dtype=np.int16)


def build_legendre_targets(u_input, tau_d=29, n=4, k=1):
    """Build the openprc-exact target matrix: products of normalized Legendre polynomials
    of the [-1, 1]-scaled, lagged input.

    Returns
    -------
    Y_full : np.ndarray, shape (T, K)
        Raw (un-normalized) Legendre target signals; row ``t`` of target ``idx`` is
        ``prod_j P_normalized(exps[idx, j], u_leg[t - j*k])``. The first ``max_lag`` rows
        are zero-padded (no valid history), matching ``generate_benchmark_target_matrix``.
    exps : np.ndarray, shape (K, Lvars)
        Exponent vector per target (degrees 1..n).
    """
    u = np.asarray(u_input, dtype=np.float64).flatten()
    T = u.shape[0]
    max_lag = tau_d * k

    # FIX 1: strict [-1, 1] scaling required for Legendre orthogonality (matches
    # compute_ipc_components: u_leg = 2*(u-min)/(max-min) - 1).
    u_min, u_max = np.min(u), np.max(u)
    if abs(u_max - u_min) < 1e-12:
        raise ValueError("Input signal u is constant; cannot scale to [-1, 1].")
    u_leg = 2.0 * (u - u_min) / (u_max - u_min) - 1.0

    # Lagged input matrix: U_sub[i, j] = u_leg[i + max_lag - j*k]  (lag j*k at row i+max_lag)
    from numpy.lib.stride_tricks import sliding_window_view
    lag_cols = [j * k for j in range(tau_d + 1)]
    U_sub = sliding_window_view(u_leg, max_lag + 1)[:, ::-1][:, lag_cols]
    Lvars = U_sub.shape[1]

    exps = _enumerate_exponents(Lvars, n)

    # Precompute normalized Legendre evaluations per (column, degree) and multiply.
    max_deg = int(exps.max()) if exps.size else 0
    leg_cache = {0: np.ones_like(U_sub)}
    for d in range(1, max_deg + 1):
        leg_cache[d] = np.empty_like(U_sub)
        for j in range(Lvars):
            leg_cache[d][:, j] = legendre_P_normalized(d, U_sub[:, j])

    Y_valid = np.ones((U_sub.shape[0], exps.shape[0]), dtype=np.float64)
    for j in range(Lvars):
        col_degs = exps[:, j]
        if not np.any(col_degs):
            continue
        for idx in range(exps.shape[0]):
            d = int(col_degs[idx])
            if d > 0:
                Y_valid[:, idx] *= leg_cache[d][:, j]

    Y_full = np.zeros((T, exps.shape[0]), dtype=np.float64)
    Y_full[max_lag:, :] = Y_valid
    return Y_full, exps


# ---------------------------------------------------------------------------
# Feature construction (openprc NodeDisplacements + StandardScaler), differentiable
# ---------------------------------------------------------------------------
def build_features(X_30Hz, n_main_nodes, reference_node=0, eps=1e-12):
    """Reconstruct openprc's reservoir feature matrix from the flat per-node displacement
    waveform, as a differentiable torch tensor.

    Mirrors ``NodeDisplacements(reference_node=0, dims=[0,1], node_ids=all)`` followed by
    ``StandardScaler`` (FIX 3 + FIX 4), then prepends a bias-ones column (matching the
    ``X_tr_b``/``X_te_b`` augmentation inside ``compute_ipc_components``).

    Parameters
    ----------
    X_30Hz : torch.Tensor, shape (T, n_main_nodes * 2)
        Flattened XY displacements of every main node (node order preserved).
    """
    T = X_30Hz.shape[0]
    X = X_30Hz.reshape(T, n_main_nodes, 2)

    # FIX 3: displacements relative to the (moving) reference/actuator node, then drop it.
    X = X - X[:, reference_node:reference_node + 1, :]
    keep = [i for i in range(n_main_nodes) if i != reference_node]
    X = X[:, keep, :].reshape(T, -1)

    # FIX 4: StandardScaler uses the population std (ddof=0); torch defaults to ddof=1.
    mean = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, unbiased=False, keepdim=True) + eps
    X_norm = (X - mean) / std

    ones = torch.ones((T, 1), dtype=X_norm.dtype, device=X_norm.device)
    return torch.cat([ones, X_norm], dim=1)


# ---------------------------------------------------------------------------
# Ridge + R2 evaluation and heatmap assembly
# ---------------------------------------------------------------------------
def evaluate(X_full, y_target_torch, exps, washout, train_stop, test_stop,
             n_list, tau_d_list, k_delay=1, ridge=1e-6, eps=0.0):
    """openprc-exact ridge readout + per-target R2 (capacity), assembled into the
    ``(n_list x tau_d_list)`` heatmap grid.

    ``X_full`` must already contain the leading bias column (see :func:`build_features`).

    Returns
    -------
    mse_per_task : torch.Tensor, shape (K,)
        Differentiable test MSE per target -- the loop sums/weights this into the loss.
    capacities : np.ndarray, shape (K,)
        Thresholded R2 (capacity) per target, detached for plotting/logging.
    heatmap : np.ndarray, shape (len(n_list), len(tau_d_list))
        ``nanmean`` of capacities over each cumulative ``(degree<=n_s, maxlag<=tau_s)`` cell.
    """
    X_train = X_full[washout:train_stop]
    X_test = X_full[train_stop:test_stop]
    y_train = y_target_torch[washout:train_stop]
    y_test = y_target_torch[train_stop:test_stop]

    # FIX 5: trace-scaled ridge with an UNPENALIZED bias term (matches compute_ipc_components).
    p1 = X_train.shape[1]
    DtD = X_train.T @ X_train
    lam = ridge * (torch.trace(DtD) / p1)
    reg = torch.eye(p1, dtype=X_train.dtype, device=X_train.device) * lam
    reg[0, 0] = 0.0
    W = torch.linalg.solve(DtD + reg, X_train.T @ y_train)

    pred = X_test @ W
    resid = pred - y_test
    mse_per_task = torch.mean(resid ** 2, dim=0)

    # FIX 6: R2 = 1 - SSE/SST with SST measured on the test target (not approximated by N).
    sse = torch.sum(resid ** 2, dim=0)
    sst = torch.sum((y_test - y_test.mean(dim=0, keepdim=True)) ** 2, dim=0) + 1e-12
    cap = 1.0 - sse / sst

    # FIX 7: threshold at the Dambre epsilon only (cap if cap > eps else 0).
    cap = torch.where(cap > eps, cap, torch.zeros_like(cap))
    capacities = cap.detach().cpu().numpy()

    heatmap = _assemble_heatmap(capacities, exps, n_list, tau_d_list, k_delay)
    return mse_per_task, capacities, heatmap


def _assemble_heatmap(capacities, exps, n_list, tau_d_list, k_delay=1):
    """Map per-target capacities onto the cumulative ``(n_s, tau_s)`` grid exactly as
    ``run_heatmap_pipeline_for_topology`` does (each cell = mean capacity over all basis
    functions with ``1 <= degree <= n_s`` and ``maxlag <= tau_s``)."""
    degrees = exps.sum(axis=1)
    lag_per_col = np.arange(exps.shape[1]) * k_delay
    maxlag = np.where(exps > 0, lag_per_col[None, :], -1).max(axis=1)

    heatmap = np.full((len(n_list), len(tau_d_list)), np.nan, dtype=float)
    for i, n_s in enumerate(n_list):
        for j, tau_s in enumerate(tau_d_list):
            mask = (degrees >= 1) & (degrees <= n_s) & (maxlag <= tau_s * k_delay)
            if np.any(mask):
                heatmap[i, j] = np.nanmean(capacities[mask])
    return heatmap


# ---------------------------------------------------------------------------
# Plotting -- verbatim mirror of plot_heatmap_sim2real_bending.plot_heatmap
# ---------------------------------------------------------------------------
def plot_iteration_heatmap(heatmap, n_list, tau_d_list, k_delay, amp, n_mass,
                           title_prefix, epoch, save_dir,
                           save_name=None, vmin=None, vmax=None,
                           save_svg=False, save_png=True, dpi=200):
    """Render and save the per-epoch heatmap. Visual settings are copied verbatim from
    ``plot_heatmap_sim2real_bending.plot_heatmap`` so the output is format-identical to
    openprc's final ``heatmap_optimized.png``; only the title/filename carry the epoch."""
    fig, ax = plt.subplots(figsize=(10, 8))
    heatmap = heatmap.T

    full_title = (f"{title_prefix} " + rf"$R^2$ (upper)" + "  " +
                  rf"num_mass={n_mass}" + " " + rf"k={k_delay}, A={amp}")

    im = ax.imshow(
        heatmap, aspect='auto', origin='lower',
        cmap='RdYlBu_r', vmin=vmin or 0.0, vmax=vmax or 1.0
    )

    n_rows, n_cols = heatmap.shape
    for y in range(n_rows):
        for x in range(n_cols):
            r2_val = heatmap[y, x]
            if not np.isnan(r2_val):
                ax.text(x, y, f'{r2_val:.2f}',
                        ha='center', va='center', color='black', fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('$R^2$ Mean')
    ax.set_xlabel(r'$n$ (monomial degree)')
    ax.set_ylabel(r'$\tau$ (time delay)')
    ax.set_title(full_title, fontsize=8)

    ax.set_xticks(np.arange(len(n_list)))
    ax.set_yticks(np.arange(len(tau_d_list)))
    ax.set_xticklabels(n_list, fontsize=6)
    ax.set_yticklabels((np.array(tau_d_list) * k_delay), fontsize=6)

    fig.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        if save_name is None:
            save_name = f"heatmap_epoch_{epoch:03d}"
        if save_svg:
            fig.savefig(os.path.join(str(save_dir), f"{save_name}.svg"),
                        format="svg", bbox_inches="tight")
        if save_png:
            fig.savefig(os.path.join(str(save_dir), f"{save_name}.png"),
                        dpi=dpi, bbox_inches="tight")
    plt.close(fig)
