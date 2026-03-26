"""
miura_ori_base_excitation.py
============================

DEMLAT-only Miura-ori example:
- upright Miura-ori sheet
- base excitation only (all bottom-edge nodes move together up/down)
- optional payload mass near upper-left corner
- simulation + visualizer

No reservoir / analysis code included.
"""

import sys
import h5py
from pathlib import Path
import numpy as np

from scipy.interpolate import CubicSpline

# Match spring_mass_2D.py import style
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.insert(0, str(src_dir))

from openprc import demlat
from openprc.demlat.models.barhinge import BarHingeModel
from openprc.demlat.io.simulation_setup import SimulationSetup
from openprc.demlat.utils.animator import ShowSimulation


def create_miura_ori_geometry(
    setup: SimulationSetup,
    xn: int = 4,
    yn: int = 8,
    a: float = 0.053,
    b: float = 0.053,
    gamma_deg: float = 45.0,
    theta_deg: float = 25.0,
    k_axial: float = 222.15,
    k_fold: float = 5.0,
    k_facet: float = 10.0,
    c_axial: float = 5.0,
    c_hinge: float = 0.2,
    mass: float = 0.01,
):
    """
    Build an upright Miura-ori sheet.

    Coordinates used here:
    - x: sheet width (left-right in the picture)
    - y: sheet height (up-down in the picture)
    - z: corrugation depth (out-of-plane folds)

    Returns:
        faces, node_meta
    """
    gamma = np.deg2rad(gamma_deg)
    theta = np.deg2rad(theta_deg)

    # Miura geometry, following the relevant repo example
    ht = a * np.sin(gamma) * np.sin(theta)
    l = b * np.tan(gamma) * np.cos(theta) / np.sqrt(
        1.0 + (np.cos(theta) ** 2) * (np.tan(gamma) ** 2)
    )
    w = a * np.sqrt(1.0 - (np.sin(theta) ** 2) * (np.sin(gamma) ** 2))
    v = b / np.sqrt(1.0 + (np.cos(theta) ** 2) * (np.tan(gamma) ** 2))
    fa = np.sqrt(a**2 + b**2 - 2.0 * a * b * np.cos(np.pi - gamma))

    i_max = 2 * xn + 1
    j_max = 2 * yn + 1

    def get_idx(i, j):
        return i * j_max + j

    faces = []

    # Nodes
    # Rotate the usual Miura indexing into an upright sheet:
    # x <- horizontal span
    # y <- height
    # z <- corrugation depth
    for i in range(i_max):
        for j in range(j_max):
            # Standing Miura wall:
            # x = horizontal span
            # y = corrugation depth
            # z = vertical height
            px = j * w + v * (i % 2)
            py = ht * (j % 2)
            pz = i * l

            setup.add_node([px, py, pz], mass=mass, fixed=False)

    # Boundary bars
    for i in range(i_max - 1):
        setup.add_bar(get_idx(i, 0), get_idx(i + 1, 0),
                      stiffness=k_axial, rest_length=b, damping=c_axial)
        setup.add_bar(get_idx(i, j_max - 1), get_idx(i + 1, j_max - 1),
                      stiffness=k_axial, rest_length=b, damping=c_axial)

    for j in range(j_max - 1):
        setup.add_bar(get_idx(0, j), get_idx(0, j + 1),
                      stiffness=k_axial, rest_length=a, damping=c_axial)
        setup.add_bar(get_idx(i_max - 1, j), get_idx(i_max - 1, j + 1),
                      stiffness=k_axial, rest_length=a, damping=c_axial)

    # Facet diagonals + facet hinges
    for i in range(i_max - 1):
        for j in range(j_max - 1):
            p00, p10 = get_idx(i, j), get_idx(i + 1, j)
            p01, p11 = get_idx(i, j + 1), get_idx(i + 1, j + 1)

            if i % 2 == 0:
                setup.add_bar(p00, p11,
                              stiffness=k_axial, rest_length=fa, damping=c_axial)
                setup.add_hinge([p00, p11, p01, p10],
                                stiffness=k_facet, damping=c_hinge,
                                rest_angle=np.pi)
                faces.append([p00, p11, p01])
                faces.append([p00, p10, p11])
            else:
                setup.add_bar(p10, p01,
                              stiffness=k_axial, rest_length=fa, damping=c_axial)
                setup.add_hinge([p10, p01, p11, p00],
                                stiffness=k_facet, damping=c_hinge,
                                rest_angle=np.pi)
                faces.append([p10, p01, p11])
                faces.append([p10, p00, p01])

    # Horizontal fold lines
    th_h = 2.0 * theta
    for j in range(1, j_max - 1):
        for i in range(i_max - 1):
            idx1, idx2 = get_idx(i, j), get_idx(i + 1, j)
            setup.add_bar(idx1, idx2,
                          stiffness=k_axial, rest_length=b, damping=c_axial)

            sign = -1 + 2 * (j % 2)
            phi = np.pi + th_h * sign

            if i % 2 == 0:
                wing_i, wing_l = get_idx(i + 1, j + 1), get_idx(i, j - 1)
            else:
                wing_i, wing_l = get_idx(i, j + 1), get_idx(i + 1, j - 1)

            setup.add_hinge([idx1, idx2, wing_i, wing_l],
                            stiffness=k_fold, damping=c_hinge,
                            rest_angle=phi)

    # Vertical fold lines
    numerator = -(np.sin(gamma) ** 2 * np.sin(theta) ** 2 - 2 * np.sin(theta) ** 2 + 1)
    denominator = (np.sin(gamma) ** 2 * np.sin(theta) ** 2 - 1)
    th_v = np.arccos(numerator / denominator)

    for i in range(1, i_max - 1):
        for j in range(j_max - 1):
            idx1, idx2 = get_idx(i, j), get_idx(i, j + 1)
            setup.add_bar(idx1, idx2,
                          stiffness=k_axial, rest_length=a, damping=c_axial)

            sign = (1 - 2 * (j % 2)) * (1 - 2 * (i % 2))
            phi = np.pi + th_v * sign

            if i % 2 == 0:
                wing_i, wing_l = get_idx(i - 1, j + 1), get_idx(i + 1, j + 1)
            else:
                wing_i, wing_l = get_idx(i - 1, j), get_idx(i + 1, j)

            setup.add_hinge([idx1, idx2, wing_i, wing_l],
                            stiffness=k_fold, damping=c_hinge,
                            rest_angle=phi)

    node_meta = {
        "i_max": i_max,
        "j_max": j_max,
        "get_idx": get_idx,
    }
    return faces, node_meta


def add_payload_mass(setup: SimulationSetup, get_idx, j_max: int, extra_mass: float = 0.05):
    """
    Approximate a payload by increasing the mass of a node near the upper-left corner.
    """
    payload_idx = get_idx(0, j_max - 1)
    setup.nodes["masses"][payload_idx] += extra_mass
    return payload_idx


from scipy.interpolate import CubicSpline
import numpy as np


def generate_base_drive(
    duration: float,
    dt_sig: float,
    input_mode: str = "iid",
    physical_amp: float = 0.001,
    coarse_step: float = 0.033,
    seed: int = 42,
    sine_freqs=(2.11, 3.73, 4.33),
    sine_phases=(0.0, 0.0, 0.0),
):
    """
    Generate a 1D base-driving signal.

    Modes
    -----
    iid:
        coarse iid uniform samples + cubic spline interpolation
        (same style as spring_mass_2D)

    triple_sine:
        deterministic product-of-three-sines signal
        suitable when you want a smooth non-random drive for NARMA tests
    """
    t_fine = np.arange(0.0, duration, dt_sig)

    if input_mode == "iid":
        rng = np.random.default_rng(seed)
        t_coarse = np.arange(0.0, duration + coarse_step, coarse_step)
        u_coarse = rng.uniform(-1.0, 1.0, size=len(t_coarse))
        u_fine = CubicSpline(t_coarse, u_coarse)(t_fine)

    elif input_mode == "triple_sine":
        f1, f2, f3 = sine_freqs
        p1, p2, p3 = sine_phases

        s1 = np.sin(2.0 * np.pi * f1 * t_fine + p1)
        s2 = np.sin(2.0 * np.pi * f2 * t_fine + p2)
        s3 = np.sin(2.0 * np.pi * f3 * t_fine + p3)

        u_fine = s1 * s2 * s3

        # normalize to roughly [-1, 1]
        max_abs = np.max(np.abs(u_fine))
        if max_abs > 1e-12:
            u_fine = u_fine / max_abs

    else:
        raise ValueError(
            f"Unknown input_mode='{input_mode}'. "
            f"Choose from ['iid', 'triple_sine']."
        )

    disp = physical_amp * u_fine
    return t_fine, u_fine, disp


def add_base_excitation(
    setup: SimulationSetup,
    j_max: int,
    get_idx,
    duration: float,
    dt_sig: float,
    input_mode: str = "iid",
    physical_amp: float = 0.001,
    coarse_step: float = 0.033,
    seed: int = 42,
    sine_freqs=(0.23, 0.41, 0.73),
    sine_phases=(0.0, 0.7, 1.3),
):
    """
    Shared base excitation for the standing Miura wall.

    Base = i = 0 edge
    Motion = vertical z direction

    input_mode:
        "iid"         -> coarse iid + spline
        "triple_sine" -> product of three sine waves
    """
    t_fine, u_fine, disp = generate_base_drive(
        duration=duration,
        dt_sig=dt_sig,
        input_mode=input_mode,
        physical_amp=physical_amp,
        coarse_step=coarse_step,
        seed=seed,
        sine_freqs=sine_freqs,
        sine_phases=sine_phases,
    )

    base_indices = [get_idx(0, j) for j in range(j_max)]

    for k, idx in enumerate(base_indices):
        p0 = np.array(setup.nodes["positions"][idx], dtype=float)
        sig = np.tile(p0, (len(t_fine), 1))
        sig[:, 2] += disp

        setup.add_signal(f"sig_base_{k}", sig, dt=dt_sig)
        setup.add_actuator(idx, f"sig_base_{k}", type="position")

    return base_indices, t_fine, u_fine


def run_pipeline(
    xn: int = 4,
    yn: int = 8,
    gamma_deg: float = 45.0,
    theta_deg: float = 25.0,
    ga_generation: int = 0,
    add_payload: bool = True,
    show_animation: bool = True,
    input_mode: str = "triple_sine",   # "iid" or "triple_sine"
):
    output_dir = (
        src_dir
        / "experiments"
        / f"miura_ori_{xn}x{yn}"
        / f"generation_{ga_generation}"
    )

    print(f"[Step 1] Setting up Miura-ori sheet in {output_dir}")
    setup = SimulationSetup(output_dir, overwrite=True)

    # Conservative settings for stability
    setup.set_simulation_params(duration=30.0, dt=0.001, save_interval=0.01)
    setup.set_physics(gravity=0, damping=0.7, enable_collision=True)

    faces, meta = create_miura_ori_geometry(
        setup=setup,
        xn=xn,
        yn=yn,
        a=0.05,
        b=0.05,
        gamma_deg=gamma_deg,
        theta_deg=theta_deg,
        k_axial=22.15,
        k_fold=0.05,
        k_facet=0.05,
        c_axial=0.05,
        c_hinge=0.5,
        mass=0.001,
    )

    print(f"Added {len(setup.nodes['positions'])} nodes.")
    print(f"Added {len(setup.bars['indices'])} bars.")
    print(f"Added {len(setup.hinges['indices'])} hinges.")

    if add_payload:
        payload_idx = add_payload_mass(
            setup,
            meta["get_idx"],
            meta["j_max"],
            extra_mass=0.05,
        )
        print(f"Added payload mass at node {payload_idx}")

    sim_params = setup.config["simulation"]
    base_indices, t_input, u_input = add_base_excitation(
        setup=setup,
        j_max=meta["j_max"],
        get_idx=meta["get_idx"],
        duration=sim_params["duration"],
        dt_sig=sim_params["dt_base"],
        input_mode=input_mode,
        physical_amp=0.02,
        coarse_step=0.033,
        seed=42,
        sine_freqs=(2.11, 3.73, 4.33),
        sine_phases=(0.0, 0.0, 0.0),
    )
    print(f"Added base excitation to nodes: {base_indices}")
    print(f"Input mode: {input_mode}")

    print("\n[Step 2] Saving experiment files...")
    setup.save()

    viz_path = output_dir / "input" / "visualization.h5"
    with h5py.File(viz_path, "w") as f:
        f.create_dataset("faces", data=np.array(faces, dtype=np.int32))

    print("\n[Step 3] Running simulation...")
    sim = demlat.Simulation(output_dir)

    try:
        import pycuda.driver
        pycuda.driver.init()
        if pycuda.driver.Device.count() > 0:
            backend = "cuda"
            print("Using CUDA backend.")
        else:
            raise ImportError("No CUDA devices found")
    except Exception:
        backend = "cpu"
        print("CUDA not available. Using CPU backend.")

    eng = demlat.Engine(BarHingeModel, backend=backend)
    eng.run(sim)

    print(f"\nDone. Output saved to: {output_dir}")
    print(f"Visualizer command: python src/demlat/utils/viz_player.py {output_dir}")

    if show_animation:
        try:
            ShowSimulation(str(output_dir))
        except Exception as e:
            print(f"Visualizer failed: {e}")

    res_path = output_dir / "output" / "simulation.h5"
    data = None
    if res_path.exists():
        with h5py.File(res_path, "r") as f:
            data = f["time_series/nodes/positions"][:]

    return data, output_dir


if __name__ == "__main__":
    data, output_dir = run_pipeline(
        xn=4,
        yn=8,
        gamma_deg=45.0,
        theta_deg=75.0,
        ga_generation=3,
        add_payload=False,
        show_animation=True,
        input_mode="iid",
    )
    if data is not None:
        print("Trajectory shape:", data.shape)