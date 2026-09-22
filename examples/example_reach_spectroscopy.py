"""
Reach Spectroscopy — Example
============================
Device-agnostic machine identification on a driven reservoir.

The whole idea, in five lines::

    loader = StateLoader("output/simulation.h5")                    # a saved run
    states = feat.NodeDisplacements(reference_node=0).transform(loader)  # readouts (T, N)
    drive  = loader.get_actuation_signal(actuator_idx=0, dof=1)     # actuation (T,)
    M = reach_spectroscopy(states, t=loader.time, drive=drive)      # <-- the analysis
    print(M.D)                                                      # {0.01: 6, 0.001: 4}

`reach_spectroscopy` takes *parsed timeseries*, never a path — so the same call
runs on a simulation, a control, or digitized experimental data.

The pipeline is deliberately split into the two stages the analysis is
designed around:

  Stage 1 — PARSE a saved experiment
    Load a ``simulation.h5`` with openprc's ``StateLoader`` and turn it into
    plain readout arrays with a feature transform (``NodeDisplacements``) and
    the actuation drive (``get_actuation_signal``).

  Stage 2 — ANALYSE the readouts
    Hand those arrays to ``reach_spectroscopy`` — which is loader-independent
    and never touches the experiment directory.  It returns a
    ``MachineAbstraction`` with the reach ``D``, the harmonic phases ``phi_n``,
    quadrature quality, capacities, and the six-panel scientific figure.

To keep the example self-contained it first synthesizes a single-tone-driven
lattice and writes it in the ``simulation.h5`` schema, then parses that back
through the *real* openprc loader.  Point ``--sim`` at your own experiment to
run on real data instead.

Usage
-----
    python example_reach_spectroscopy.py                    # synthesize + analyse
    python example_reach_spectroscopy.py --no-plot          # skip figures
    python example_reach_spectroscopy.py --lean             # mask heavy panels
    python example_reach_spectroscopy.py --sim path/to/simulation.h5
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ═══════════════════════════════════════════════════════════════════════════
#  Build a driven-reservoir dataset and save it in the simulation.h5 schema
# ═══════════════════════════════════════════════════════════════════════════

def synthesize_probe_experiment(exp_dir: Path, f0=1.0, dt=0.005, duration=40.0,
                                 n_nodes=16, reach=6, seed=0) -> Path:
    """Write a single-tone-driven lattice to ``<exp_dir>/output/simulation.h5``.

    Node 0 is the driven node (pure tone on its y-DOF); every interior node
    responds with a harmonic ladder up to ``reach`` (a stand-in for the
    substrate's nonlinearity), each channel at its own phase so a linear
    readout can recover both quadratures.  Small white noise sets the floor,
    so the reach saturates near ``reach``.
    """
    import h5py

    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration, dt)
    T = len(t)
    w0 = 2 * np.pi * f0

    # 4x4 grid rest positions (planar, z=0)
    gx, gy = np.meshgrid(np.arange(4), np.arange(4))
    rest = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(16)]).astype(float)
    rest = rest[:n_nodes]

    # driven node 0: a clean single tone on the y-DOF
    drive_amp = 0.4
    drive_disp = drive_amp * np.sin(w0 * t)

    # interior nodes: harmonic ladder with decaying amplitude and random phases
    amps = drive_amp * 0.6 ** np.arange(reach)          # a_1 > a_2 > ...
    pos = np.tile(rest, (T, 1, 1))                       # (T, N, 3)
    pos[:, 0, 1] += drive_disp                           # node 0 = the drive
    for n in range(1, n_nodes):
        for dof in (0, 1):                               # x and y respond
            phases = rng.uniform(0, 2 * np.pi, reach)
            ladder = sum(a * np.sin(k * w0 * t + ph)
                         for k, (a, ph) in enumerate(zip(amps, phases), start=1))
            pos[:, n, dof] += 0.5 * ladder
    pos += 0.002 * rng.standard_normal(pos.shape)        # measurement noise

    out_dir = exp_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    sim_path = out_dir / "simulation.h5"
    with h5py.File(sim_path, "w") as f:
        f.attrs["schema_version"] = "2.1.0"
        f.attrs["dt"] = dt
        f.attrs["total_frames"] = T
        ts = f.create_group("time_series")
        ts.create_dataset("time", data=t)
        ts.create_group("nodes").create_dataset("positions", data=pos)
        # actuation signal for node 0 (3-DOF; only y is driven)
        act = ts.create_group("actuation_signals")
        sig = np.zeros((T, 3))
        sig[:, 1] = drive_disp
        act.create_dataset("0", data=sig)

    print(f"  synthesized probe experiment -> {sim_path}")
    print(f"  (f0={f0} Hz, dt={dt} s, {T} frames, {n_nodes} nodes, true reach≈{reach})")
    return sim_path


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 1 — parse the saved experiment into readout arrays
# ═══════════════════════════════════════════════════════════════════════════

def parse_experiment(sim_path: Path):
    """Load ``simulation.h5`` and return (states, time, drive) plain arrays."""
    from openprc.reservoir import StateLoader
    from openprc.reservoir import features as feat

    loader = StateLoader(sim_path)

    # readouts: displacement of every node relative to the driven node 0
    states = feat.NodeDisplacements(reference_node=0).transform(loader)  # (T, N)

    # drive: the y-DOF actuation of node 0 (the single tone we probed with)
    drive = loader.get_actuation_signal(actuator_idx=0, dof=1)           # (T,)

    print(f"  parsed readouts: states {states.shape}, drive {drive.shape}, "
          f"dt = {loader.dt:.5f} s")
    return states, loader.time, drive


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim", type=Path, default=None,
                    help="path to an existing simulation.h5 (default: synthesize one)")
    ap.add_argument("--no-plot", action="store_true", help="skip figures")
    ap.add_argument("--lean", action="store_true",
                    help="mask heavier measurements (noise floor, rank, spectrum)")
    ap.add_argument("--f0", type=float, default=None,
                    help="fundamental drive frequency [Hz] (default: auto-estimate)")
    args = ap.parse_args()

    from openprc.analysis import reach_spectroscopy

    print("=" * 64)
    print("Reach Spectroscopy — device-agnostic machine identification")
    print("=" * 64)

    # ── locate / build the experiment ────────────────────────────────────
    if args.sim is not None:
        sim_path = args.sim
        if not sim_path.exists():
            raise FileNotFoundError(sim_path)
        print(f"\n[Stage 1] Parsing experiment: {sim_path}")
    else:
        exp_dir = Path("experiments/reach_spectroscopy_demo")
        print(f"\n[Stage 0] No --sim given; synthesizing a probe experiment.")
        sim_path = synthesize_probe_experiment(exp_dir)
        print(f"\n[Stage 1] Parsing experiment: {sim_path}")

    # ── Stage 1: parse to plain arrays ────────────────────────────────────
    states, time, drive = parse_experiment(sim_path)

    # ── Stage 2: analyse the readouts (loader-independent) ────────────────
    print("\n[Stage 2] Reach spectroscopy on the parsed readouts")
    mask = dict(noise_floor=False, rank=False, input_spectrum=False) if args.lean else {}
    if args.lean:
        print("  (--lean: masking noise floor, effective rank, input spectrum)")

    M = reach_spectroscopy(
        states, t=time, drive=drive, f0=args.f0,
        title=f"Reach spectroscopy — {sim_path.parent.parent.name}",
        **mask,
    )

    print("\n[Result]")
    print(f"  {M!r}")
    for e, d in M.D.items():
        print(f"  D(eps={e:g}) = {d}   (reach: harmonics faithfully reconstructed)")

    # ── save the frozen abstraction alongside the experiment ──────────────
    out_dir = M.save(sim_path.parent)
    print(f"  saved machine_abstraction.json + reach_spectroscopy.npz -> {out_dir}")

    # ── figures ───────────────────────────────────────────────────────────
    if not args.no_plot:
        import matplotlib.pyplot as plt
        M.plot(save_to=sim_path.parent / "reach_spectroscopy.png")
        M.plot_harmonic_fits(save_to=sim_path.parent / "harmonic_fits.png")
        plt.show()

    print("\nDone.")


if __name__ == "__main__":
    main()