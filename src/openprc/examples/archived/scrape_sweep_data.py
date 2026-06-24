import numpy as np
from pathlib import Path
from demlat.utils.plot_timeseries import SimulationPlotter
import json

# Configuration
BASE_DIR = Path("experiments/yoshimura_beta_d_sweep/")
OUTPUT_FILE = BASE_DIR / "hysteresis_data.npz"
n = 3
betas = np.linspace(30, 45, 500)
n_bins = 100

# Storage
all_data = {
    'betas': [],
    'bin_centers': None,
    'loading_mean': [],
    'loading_min': [],
    'loading_max': [],
    'unloading_mean': [],
    'unloading_min': [],
    'unloading_max': [],
}

# Global displacement range (compute first pass or set manually)
global_disp_min = np.inf
global_disp_max = -np.inf

print("First pass: determining global displacement range...")
for beta in betas:
    dir_name = f"exp_beta_{beta}_n_{n}"
    sim_file = BASE_DIR / dir_name / "output" / "simulation.h5"

    if not sim_file.exists():
        continue

    try:
        plotter = SimulationPlotter(sim_file)
        time, _ = plotter.get_dataset("time")
        positions, _ = plotter.get_dataset("nodes/positions")

        time = np.asarray(time).flatten()
        mask = time >= 5.0
        positions = positions[mask]

        driven_node_idx = 4 * n

        # === FIX: Use absolute z position, not relative to z0 ===
        z_positions = positions[:, driven_node_idx, 2]

        global_disp_min = min(global_disp_min, z_positions.min())
        global_disp_max = max(global_disp_max, z_positions.max())
    except Exception as e:
        print(f"  Skipping {dir_name}: {e}")

print(f"Global z range: [{global_disp_min:.6f}, {global_disp_max:.6f}]")

# Common bin edges for all experiments
# We'll compute displacement as: d = z_max - z (so fully extended = 0, compressed = positive)
bin_edges = np.linspace(0, global_disp_max - global_disp_min, n_bins + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
all_data['bin_centers'] = bin_centers

print(f"Displacement range: [0, {global_disp_max - global_disp_min:.6f}]")


def compute_envelope_stats(disp, energy, bin_edges):
    """Compute mean, min, max per bin."""
    n_bins = len(bin_edges) - 1
    means = np.full(n_bins, np.nan)
    mins = np.full(n_bins, np.nan)
    maxs = np.full(n_bins, np.nan)

    if len(disp) == 0:
        return means, mins, maxs

    bin_idx = np.digitize(disp, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    for i in range(n_bins):
        mask = bin_idx == i
        if np.sum(mask) > 0:
            vals = energy[mask]
            means[i] = np.mean(vals)
            mins[i] = np.min(vals)
            maxs[i] = np.max(vals)

    return means, mins, maxs


print("\nSecond pass: extracting hysteresis data...")
for i, beta in enumerate(betas):
    dir_name = f"exp_beta_{beta}_n_{n}"
    sim_file = BASE_DIR / dir_name / "output" / "simulation.h5"

    if not sim_file.exists():
        print(f"  [{i + 1}/{len(betas)}] Missing: {dir_name}")
        continue

    try:
        plotter = SimulationPlotter(sim_file)

        time, _ = plotter.get_dataset("time")
        positions, _ = plotter.get_dataset("nodes/positions")
        potential_energy, _ = plotter.get_dataset("system/potential_energy")

        time = np.asarray(time).flatten()
        potential_energy = np.asarray(potential_energy).flatten()

        # Skip transient
        mask = time >= 5.0
        time = time[mask]
        positions = positions[mask]
        potential_energy = potential_energy[mask]

        # === FIX: Displacement = z_max - z ===
        # This way: z = z_max (fully extended) -> displacement = 0
        #           z = z_min (compressed)     -> displacement = max
        driven_node_idx = 4 * n
        z_positions = positions[:, driven_node_idx, 2]
        displacement = global_disp_max - z_positions
        displacement = np.asarray(displacement).flatten()

        # Loading/unloading based on displacement rate
        # Loading = displacement increasing (compressing)
        # Unloading = displacement decreasing (extending)
        d_disp = np.gradient(displacement, time)
        loading = d_disp > 0
        unloading = d_disp <= 0

        # Compute stats
        load_mean, load_min, load_max = compute_envelope_stats(
            displacement[loading], potential_energy[loading], bin_edges
        )
        unload_mean, unload_min, unload_max = compute_envelope_stats(
            displacement[unloading], potential_energy[unloading], bin_edges
        )

        # Store
        all_data['betas'].append(beta)
        all_data['loading_mean'].append(load_mean)
        all_data['loading_min'].append(load_min)
        all_data['loading_max'].append(load_max)
        all_data['unloading_mean'].append(unload_mean)
        all_data['unloading_min'].append(unload_min)
        all_data['unloading_max'].append(unload_max)

        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(betas)}] Processed {dir_name}")

    except Exception as e:
        print(f"  [{i + 1}/{len(betas)}] Error in {dir_name}: {e}")

# Convert to arrays and save
all_data['betas'] = np.array(all_data['betas'])
all_data['loading_mean'] = np.array(all_data['loading_mean'])
all_data['loading_min'] = np.array(all_data['loading_min'])
all_data['loading_max'] = np.array(all_data['loading_max'])
all_data['unloading_mean'] = np.array(all_data['unloading_mean'])
all_data['unloading_min'] = np.array(all_data['unloading_min'])
all_data['unloading_max'] = np.array(all_data['unloading_max'])

np.savez(OUTPUT_FILE, **all_data)
print(f"\nSaved data to {OUTPUT_FILE}")
print(f"  Shape: {len(all_data['betas'])} betas x {n_bins} displacement bins")
