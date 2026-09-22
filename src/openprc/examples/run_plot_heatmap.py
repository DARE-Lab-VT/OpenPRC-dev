"""Plot clean-IID capacity heatmaps for a saved experiment or simulation.

Edit USER SETTINGS below, then run:
    python3 -m openprc.examples.run_plot_heatmap

Optional command-line arguments override these settings; see --help.
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

# --- Core Library Imports ---
from openprc.analysis.utils.training_utils import (
    aggregate_ipc_heatmaps,
)

from openprc.reservoir.io.state_loader import StateLoader
from openprc.reservoir.features.node_features import NodePositions
from openprc.reservoir.training.trainer import Trainer
from openprc.reservoir.readout.ridge import Ridge
from openprc.analysis.benchmarks.memory_benchmark import MemoryBenchmark

# =============================================================================
# USER SETTINGS — edit this section for your recording
# =============================================================================

# --- Experiment Paths ---
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1] / "experiments" / "spring_mass_fully_connected"
TOPOLOGY = "sample_0"
EXPERIMENT_DIR = EXPERIMENT_ROOT / TOPOLOGY
STATE_FILE = EXPERIMENT_DIR / "output" / "experiment.h5"  # Or experiment.h5

# --- Original IID Symbols ---
# None generates symbols from the settings below; a .npy/H5 file overrides generation.
# Do not use the interpolated command or measured actuator displacement.
IID_FILE = None
IID_DATASET = "iid"              # Used only when IID_FILE is an H5 file
IID_GENERATOR = "numpy_randomstate"  # Must match acquisition; not interchangeable with PCG64
IID_SEED = 42
IID_HZ = 30.0
IID_DURATION = 120.0             # Seconds; generation produces duration * rate symbols
IID_START_SECONDS = 0.0          # First symbol onset on the saved state clock
INPUT_BOUNDS = (-1.0, 1.0)      # Known uniform population bounds, not sample extrema

# --- Reservoir Features ---
NODE_IDS = None                  # None selects all nodes; otherwise use e.g. [1, 2, 4]
DIMS = (0, 1)                    # xy coordinates; exclude any tracking-status channel
SNAPSHOT_PHASES = (0.0, 0.25, 0.5, 0.75)    # Four snapshots within each IID-symbol interval

# --- Training and Capacity ---
WASHOUT_SECONDS = 5.0
TRAIN_FRACTION = 0.5             # Fraction after washout; testing follows immediately
MAX_DEGREE = 4
MAX_DELAY = 30                   # In IID symbols, not camera frames
RIDGE = 1e-6

# --- Plot Output ---
SAVE_SVG = True
SAVE_PNG = True
DPI = 300
SHOW_PLOTS = True


def load_or_generate_iid(iid_file, dataset, generator, seed, hz, duration, bounds):
    """Read original symbols or reproduce them with an explicitly chosen generator."""
    if iid_file is not None:
        iid_path = Path(iid_file)
        if iid_path.suffix == ".npy":
            values = np.load(iid_path, allow_pickle=False)
        else:
            with h5py.File(iid_path) as f:
                values = f[dataset][:]
        print(f"-> Loading IID symbols: {iid_path}")
    else:
        if not np.isfinite([hz, duration]).all() or hz <= 0 or duration <= 0:
            raise ValueError("IID_HZ and IID_DURATION must be positive and finite.")
        symbol_count = hz * duration
        nearest = round(symbol_count)
        count = nearest if np.isclose(symbol_count, nearest, rtol=0, atol=1e-8) else int(np.ceil(symbol_count))
        if count < 2:
            raise ValueError("IID duration must include at least two symbol onsets.")
        limits = np.asarray(bounds, dtype=float)
        if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] >= limits[1]:
            raise ValueError("INPUT_BOUNDS must contain two finite increasing values.")
        if generator == "numpy_randomstate":
            rng = np.random.RandomState(seed)
        elif generator == "numpy_pcg64":
            rng = np.random.Generator(np.random.PCG64(seed))
        else:
            raise ValueError(f"Unknown IID generator: {generator}")
        values = rng.uniform(limits[0], limits[1], count)
        print(f"-> Generating {count} IID symbols: {generator}, seed {seed}, {hz:g} Hz, {duration:g} s.")
        print("Generator, seed, distribution and symbol onset must match acquisition; the seed alone is insufficient.")

    values = np.asarray(values)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("IID symbols must be a finite scalar sequence with at least two values.")
    return values


def multiplex_states(time, states, count, iid_hz, start, phases):
    """Interpolate states at fixed symbol phases without extrapolation.

    Noninteger camera/IID rate ratios are supported. Interpolation does not add
    independent observations; choose the phases for the measurement bandwidth.
    """
    timestamp_dtype = np.asarray(time).dtype
    time = np.asarray(time, dtype=float)
    states = np.asarray(states, dtype=float)
    phases = np.asarray(phases, dtype=float)

    if (
        time.ndim != 1 or len(time) < 2
        or states.ndim != 2 or len(time) != len(states)
        or not np.isfinite(time).all() or not np.isfinite(states).all()
        or np.any(np.diff(time) <= 0)
    ):
        raise ValueError("Need finite states and strictly increasing matching timestamps.")

    if (
        len(phases) == 0 or not np.isfinite(phases).all()
        or np.any(phases < 0) or np.any(phases >= 1)
        or np.any(np.diff(phases) <= 0)
    ):
        raise ValueError("Phases must be strictly increasing in [0, 1).")

    if not np.isfinite([iid_hz, start]).all() or iid_hz <= 0 or count < 1:
        raise ValueError("Positive IID rate/count and finite start required.")

    # Allow only the rounding error of stored timestamps, not missing frames.
    timestamp_eps = np.finfo(timestamp_dtype).eps if np.issubdtype(timestamp_dtype, np.floating) else 0.0
    time_tolerance = max(1e-9, 2 * timestamp_eps * max(1.0, np.max(np.abs(time))))
    snapshot_times = start + (np.arange(count)[:, None] + phases[None, :]) / iid_hz
    if snapshot_times.min() < time[0] - time_tolerance or snapshot_times.max() > time[-1] + time_tolerance:
        raise ValueError(
            "States do not cover requested symbol phases. Correct IID start/count/phases; "
            "no extrapolation is performed."
        )
    if np.max(np.diff(time)) > 1 / iid_hz + time_tolerance:
        raise ValueError("A state sampling gap exceeds one IID interval; inspect missing frames.")

    snapshot_times = np.clip(snapshot_times, time[0], time[-1])
    snapshots = np.column_stack([
        np.interp(snapshot_times.ravel(), time, states[:, j])
        for j in range(states.shape[1])
    ])
    return snapshots.reshape(count, -1)


def plot_heatmap(
    heatmap, n_list=None, tau_d_list=None, k_delay=1,
    vmin=0.0, vmax=None,
    save_dir=None,
    save_name=None,
    save_svg=True,
    save_png=False,
    dpi=300,
    show=True,
    cumulative=False,
):
    """Plot exact capacity sums or cumulative target means with two-decimal labels."""
    fig, ax = plt.subplots(figsize=(10, 8))
    display_values = heatmap.T

    if vmax is None:
        vmax = max(1.0, float(np.max(heatmap)))

    im = ax.imshow(
        display_values, aspect="auto", origin="lower",
        cmap="RdYlBu_r", vmin=vmin, vmax=vmax,
    )

    n_rows, n_cols = display_values.shape
    for y in range(n_rows):
        for x in range(n_cols):
            ax.text(
                x, y, f"{display_values[y, x]:.2f}",
                ha="center", va="center", color="black", fontsize=7,
            )

    cbar = fig.colorbar(im, ax=ax)
    if cumulative:
        cbar.set_label("Mean capacity over included targets")
        ax.set_xlabel("Maximum total degree")
        ax.set_title("Clean-IID cumulative mean")
    else:
        cbar.set_label("Summed capacity")
        ax.set_xlabel("Exact total degree")
        ax.set_title("Clean-IID exact degree / maximum delay")
    ax.set_ylabel("Maximum delay (IID symbols)")

    if n_list is None:
        n_list = np.arange(1, len(heatmap) + 1)
    if tau_d_list is None:
        tau_d_list = np.arange(heatmap.shape[1])

    ax.set_xticks(np.arange(len(n_list)), n_list)
    ax.set_yticks(np.arange(len(tau_d_list)), np.asarray(tau_d_list) * k_delay)
    fig.tight_layout()

    # Save before displaying the figure.
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        if save_name is None:
            save_name = "heatmap_clean_iid_cumulative" if cumulative else "heatmap_clean_iid"

        if save_svg:
            svg_path = save_dir / f"{save_name}.svg"
            fig.savefig(svg_path, dpi=dpi)
            print(f"[Saved] Heatmap SVG -> {svg_path}")
        if save_png:
            png_path = save_dir / f"{save_name}.png"
            fig.savefig(png_path, dpi=dpi)
            print(f"[Saved] Heatmap PNG -> {png_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def parse_arguments():
    """Optional overrides; normal interactive use only needs USER SETTINGS above."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state_file", nargs="?", type=Path, default=STATE_FILE)
    parser.add_argument("--iid-file", type=Path, default=IID_FILE)
    parser.add_argument("--iid-dataset", default=IID_DATASET)
    parser.add_argument("--iid-generator", choices=("numpy_randomstate", "numpy_pcg64"), default=IID_GENERATOR)
    parser.add_argument("--iid-seed", type=int, default=IID_SEED)
    parser.add_argument("--iid-duration", type=float, default=IID_DURATION)
    parser.add_argument("--iid-hz", type=float, default=IID_HZ)
    parser.add_argument("--iid-start", type=float, default=IID_START_SECONDS)
    parser.add_argument("--input-bounds", nargs=2, type=float, default=INPUT_BOUNDS)
    parser.add_argument("--phases", nargs="+", type=float, default=SNAPSHOT_PHASES)
    parser.add_argument("--nodes", nargs="+", type=int, default=NODE_IDS)
    parser.add_argument("--dims", nargs="+", type=int, default=DIMS)
    parser.add_argument("--washout", type=float, default=WASHOUT_SECONDS)
    parser.add_argument("--train-fraction", type=float, default=TRAIN_FRACTION)
    parser.add_argument("--max-degree", type=int, default=MAX_DEGREE)
    parser.add_argument("--max-delay", type=int, default=MAX_DELAY)
    parser.add_argument("--ridge", type=float, default=RIDGE)
    parser.add_argument("--no-show", action="store_true", default=not SHOW_PLOTS)
    return parser.parse_args()


def main():
    """Load recording → prepare states → calculate IPC → plot both heatmaps."""

    # 1. Define the Experiment Paths
    settings = parse_arguments()
    h5_path = Path(settings.state_file)
    save_path = h5_path.parent / "plots"

    if not h5_path.exists():
        raise FileNotFoundError(f"Experiment file not found: {h5_path}. Set STATE_FILE above.")
    # 2. Load or Generate IID Symbols and Load Recorded Node Positions
    print(f"-> Loading Experiment: {h5_path}")
    u_input = load_or_generate_iid(
        iid_file=settings.iid_file,
        dataset=settings.iid_dataset,
        generator=settings.iid_generator,
        seed=settings.iid_seed,
        hz=settings.iid_hz,
        duration=settings.iid_duration,
        bounds=settings.input_bounds,
    )

    loader = StateLoader(h5_path)
    print(f"Loaded {loader.total_frames} frames and {len(u_input)} original IID symbols.")

    # 3. Extract Positions, Then Explicitly Convert Them to Displacements
    features = NodePositions(
        node_ids="all" if settings.nodes is None else settings.nodes,
        dims=DIMS,
    )
    positions = features.transform(loader)
    # Select dimensions in memory: H5 cannot fancy-index both nodes and dims.
    feature_info = features.get_feature_info(loader)
    selected = [i for i, info in enumerate(feature_info) if info["dim"] in settings.dims]
    if not selected:
        raise ValueError("No position features match the requested dimensions.")
    positions = positions[:, selected]
    displacements = positions - positions[0:1]

    # Align to the IID clock; one snapshot per already-aligned frame leaves
    # the row structure unchanged. Multiple phases stack snapshots into columns.
    states = multiplex_states(
        loader.time, displacements, len(u_input), settings.iid_hz,
        settings.iid_start, settings.phases,
    )

    # 4. Define Washout and Contiguous Training / Testing Windows
    if (
        not np.isfinite(settings.washout) or settings.washout < 0
        or not 0 < settings.train_fraction < 1
    ):
        raise ValueError("Use nonnegative washout and train fraction strictly between 0 and 1.")

    if not np.isfinite(settings.iid_hz) or settings.iid_hz <= 0:
        raise ValueError("IID_HZ must be positive and finite.")

    train_start = max(round(settings.washout * settings.iid_hz), settings.max_delay)
    train_stop = train_start + int((len(u_input) - train_start) * settings.train_fraction)
    test_duration = len(u_input) - train_stop

    # 5. Configure Trainer and Run the Existing MemoryBenchmark Once
    # Duration arguments are seconds. The feature rows represent IID symbols,
    # so tell the benchmark their interval instead of using the camera interval.
    sample_dt = 1.0 / settings.iid_hz
    trainer = Trainer(
        loader=loader,
        features=features,
        readout=Ridge(settings.ridge),
        experiment_dir=h5_path.parent.parent,
        washout=train_start * sample_dt,
        train_duration=(train_stop - train_start) * sample_dt,
        test_duration=test_duration * sample_dt,
    )
    benchmark = MemoryBenchmark(group_name="memory_benchmark")
    score = benchmark.run(
        trainer,
        u_input,
        tau_s=settings.max_delay,
        n_s=settings.max_degree,
        k_delay=1,
        ridge=settings.ridge,
        input_bounds=settings.input_bounds,
        sample_dt=sample_dt,
        prepared_states=states,
    )
    score.save()

    # Standardization and per-target regression are handled by MemoryBenchmark.
    capacities = score.metrics["capacities"]
    exponents = score.metrics["exponents"]
    exact_heatmap, cumulative_heatmap = aggregate_ipc_heatmaps(capacities, exponents)

    print(
        f"{len(u_input)} IID rows; train [{train_start}, {train_stop}), "
        f"test [{train_stop}, {len(u_input)});"
    )
    print(
        f"Positive targets: {np.count_nonzero(capacities > 0)}/{len(capacities)}; "
        f"linear={score.metrics['linear_memory_capacity']:.6g}; "
        f"nonlinear={score.metrics['nonlinear_memory_capacity']:.6g}"
    )

    # 6. Save and Display Exact and Cumulative Heatmaps
    n_list = list(range(1, settings.max_degree + 1))
    tau_d_list = list(range(settings.max_delay + 1))

    for heatmap, cumulative in [(exact_heatmap, False), (cumulative_heatmap, True)]:
        plot_heatmap(
            heatmap, n_list, tau_d_list,
            save_dir=save_path,
            save_svg=SAVE_SVG,
            save_png=SAVE_PNG,
            dpi=DPI,
            show=not settings.no_show,
            cumulative=cumulative,
        )


if __name__ == "__main__":
    main()
