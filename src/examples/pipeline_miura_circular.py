"""
Miura-Ori Circular Actuation Test
=================================
Demonstrates coordinated multi-node actuation using the ExperimentSetup API.
The four corners move in synchronized circles on the XY plane
while the structure hangs under gravity.
"""
import numpy as np
from pathlib import Path
import sys
import shutil

# Ensure import
sys.path.insert(0, str(Path(__file__).parent.parent))

import demlat
from demlat.models.barhinge import BarHingeModel
from demlat.io.experiment_setup import ExperimentSetup

DEMO_DIR = Path("experiments/miura_circular_test")


def create_miura_ori_geometry(setup: ExperimentSetup, xn=4, yn=4, a=1.0, b=1.0,
                              gamma=np.deg2rad(45), theta=np.deg2rad(30),
                              k_axial=1000.0, k_fold=10.0, k_facet=200.0, mass=0.01):
    """
    Generates Miura-Ori geometry and adds it to the ExperimentSetup.
    """
    # 1. Derived Dimensions
    ht = a * np.sin(gamma) * np.sin(theta)
    l = b * np.tan(gamma) * np.cos(theta) / np.sqrt(1 + np.cos(theta) ** 2 * np.tan(gamma) ** 2)
    w = a * np.sqrt(1 - np.sin(theta) ** 2 * np.sin(gamma) ** 2)
    v = b / np.sqrt(1 + np.cos(theta) ** 2 * np.tan(gamma) ** 2)
    fa = np.sqrt(a ** 2 + b ** 2 - 2 * a * b * np.cos(np.pi - gamma))

    i_max = 2 * xn + 1
    j_max = 2 * yn + 1

    def get_idx(i, j):
        return i * j_max + j

    # 2. Add Nodes
    for i in range(i_max):
        for j in range(j_max):
            px = i * l
            py = j * w + v * (i % 2)
            pz = ht * (j % 2)
            setup.add_node([px, py, pz], mass=mass, fixed=False)

    # 3. Add Edges (Bars)
    # Damping for bars: typically proportional to stiffness or mass.
    # Let's use a small fraction of stiffness for stability.
    # c_axial = k_axial * 0.05
    c_axial = 5.0

    for i in range(i_max - 1):
        setup.add_bar(get_idx(i, 0), get_idx(i + 1, 0), stiffness=k_axial, rest_length=b, damping=c_axial)
        setup.add_bar(get_idx(i, j_max - 1), get_idx(i + 1, j_max - 1), stiffness=k_axial, rest_length=b,
                      damping=c_axial)
    for j in range(j_max - 1):
        setup.add_bar(get_idx(0, j), get_idx(0, j + 1), stiffness=k_axial, rest_length=a, damping=c_axial)
        setup.add_bar(get_idx(i_max - 1, j), get_idx(i_max - 1, j + 1), stiffness=k_axial, rest_length=a,
                      damping=c_axial)

    # 4. Facets & Faces (Bars + Hinges)
    faces = []
    for i in range(i_max - 1):
        for j in range(j_max - 1):
            p00, p10 = get_idx(i, j), get_idx(i + 1, j)
            p01, p11 = get_idx(i, j + 1), get_idx(i + 1, j + 1)

            if i % 2 == 0:
                setup.add_bar(p00, p11, stiffness=k_axial, rest_length=fa, damping=c_axial)
                setup.add_hinge([p00, p11, p01, p10], stiffness=k_facet, rest_angle=np.pi)
                faces.append([p00, p11, p01])
                faces.append([p00, p10, p11])
            else:
                setup.add_bar(p10, p01, stiffness=k_axial, rest_length=fa, damping=c_axial)
                setup.add_hinge([p10, p01, p11, p00], stiffness=k_facet, rest_angle=np.pi)
                faces.append([p10, p01, p11])
                faces.append([p10, p00, p01])

    # 5. Folds (Hinges)
    th_h = 2 * theta
    for j in range(1, j_max - 1):
        for i in range(i_max - 1):
            idx1, idx2 = get_idx(i, j), get_idx(i + 1, j)
            setup.add_bar(idx1, idx2, stiffness=k_axial, rest_length=b, damping=c_axial)
            sign = -1 + 2 * (j % 2)
            phi = np.pi + th_h * sign

            if i % 2 == 0:
                wing_i, wing_l = get_idx(i + 1, j + 1), get_idx(i, j - 1)
            else:
                wing_i, wing_l = get_idx(i, j + 1), get_idx(i + 1, j - 1)
            setup.add_hinge([idx1, idx2, wing_i, wing_l], stiffness=k_fold, rest_angle=phi)

    numerator = -(np.sin(gamma) ** 2 * np.sin(theta) ** 2 - 2 * np.sin(theta) ** 2 + 1)
    denominator = (np.sin(gamma) ** 2 * np.sin(theta) ** 2 - 1)
    th_v = np.arccos(numerator / denominator)

    for i in range(1, i_max - 1):
        for j in range(j_max - 1):
            idx1, idx2 = get_idx(i, j), get_idx(i, j + 1)
            setup.add_bar(idx1, idx2, stiffness=k_axial, rest_length=a, damping=c_axial)
            sign = (1 - 2 * (j % 2)) * (1 - 2 * (i % 2))
            phi = np.pi + th_v * sign

            if i % 2 == 0:
                wing_i, wing_l = get_idx(i - 1, j + 1), get_idx(i + 1, j + 1)
            else:
                wing_i, wing_l = get_idx(i - 1, j), get_idx(i + 1, j)
            setup.add_hinge([idx1, idx2, wing_i, wing_l], stiffness=k_fold, rest_angle=phi)

    # Save visualization faces manually for now (Setup doesn't have a dedicated viz buffer yet)
    # We can append it after save, or extend Setup later.
    # For this example, we'll just return it to be saved manually.
    return faces


def step_1_setup_experiment():
    print("\n[Step 1] Setting up Experiment...")

    # Initialize Setup
    setup = ExperimentSetup(DEMO_DIR, overwrite=True)

    # 1. Configure Simulation
    setup.set_simulation_params(duration=10.0, dt=0.0005, save_interval=0.01)
    setup.set_physics(gravity=-9.8, damping=0.2)

    # 2. Build Geometry
    xn, yn = 8, 8
    faces = create_miura_ori_geometry(setup, xn=xn, yn=yn, k_fold=5.0, k_facet=10.0)

    # 3. Define Actuation (Circular Motion)
    i_max = 2 * xn + 1
    j_max = 2 * yn + 1

    corners = {
        "BL": 0,  # Bottom-Left
        "TL": j_max - 1,  # Top-Left
        "BR": (i_max - 1) * j_max,  # Bottom-Right
        "TR": (i_max * j_max) - 1  # Top-Right
    }

    # Signal Parameters
    dt_sig = 0.001
    t = np.arange(0, 10.0, dt_sig)
    radius = 1.5
    omega = 2 * np.pi * 0.5  # 0.5 Hz
    ramp = np.clip(t, 0, 1.0)  # Smooth start over 1s

    # Create signals and link actuators
    positions = setup.nodes['positions']

    for name, idx in corners.items():
        p0 = positions[idx]

        # Generate Signal: Target = P0 + Circle - Offset
        sig = np.zeros((len(t), 3), dtype=np.float32)
        sig[:, 0] = p0[0] + (ramp * radius * np.cos(omega * t)) - (ramp * radius)
        sig[:, 1] = p0[1] + (ramp * radius * np.sin(omega * t))
        sig[:, 2] = p0[2]

        sig_name = f"sig_{name}"
        setup.add_signal(sig_name, sig, dt=dt_sig)
        setup.add_actuator(idx, sig_name, type='position')

    # 4. Save Everything
    setup.save()

    # 5. Save Visualization Faces (Manual step for now)
    import h5py
    with h5py.File(DEMO_DIR / "input" / "visualization.h5", 'w') as f:
        f.create_dataset("faces", data=np.array(faces, dtype=np.int32))


def step_2_run_simulation():
    print("\n[Step 2] Running Simulation...")
    exp = demlat.Experiment(DEMO_DIR)
    eng = demlat.Engine(BarHingeModel, backend='cuda')
    eng.run(exp)


def check_results():
    sim_path = DEMO_DIR / "output" / "simulation.h5"
    if not sim_path.exists():
        print("Error: simulation.h5 was not created.")
        return

    import h5py
    with h5py.File(sim_path, 'r') as f:
        print("\n[Check] simulation.h5 Attributes:")
        for k, v in f.attrs.items():
            print(f"  - {k}: {v}")

        if 'time_series/nodes/positions' in f:
            print(f"  - Frames: {f['time_series/nodes/positions'].shape[0]}")


if __name__ == "__main__":
    step_1_setup_experiment()
    step_2_run_simulation()
    check_results()
