"""
Yoshimura-Ori Geometry Test
===========================
Generates Yoshimura origami geometry in folded configuration.
"""
import numpy as np
from pathlib import Path
import h5py
from demlat.io.experiment_setup import ExperimentSetup
from demlat.utils.viz_player import visualize_experiment

DEMO_DIR = Path("experiments/yoshimura_test")


def general_transform_matrix(psi, gamma, d):
    """Generate transformation matrix for Yoshimura unit."""
    return np.array([
        [np.cos(psi) ** 2 + np.cos(gamma) * np.sin(psi) ** 2,
         (1 - np.cos(gamma)) * np.sin(psi) * np.cos(psi),
         np.sin(gamma) * np.sin(psi),
         d * np.sin(psi) * np.sin(gamma / 2)],
        [(1 - np.cos(gamma)) * np.sin(psi) * np.cos(psi),
         np.sin(psi) ** 2 + np.cos(gamma) * np.cos(psi) ** 2,
         -np.sin(gamma) * np.cos(psi),
         -d * np.cos(psi) * np.sin(gamma / 2)],
        [-np.sin(gamma) * np.sin(psi),
         np.sin(gamma) * np.cos(psi),
         np.cos(gamma),
         d * np.cos(gamma / 2)],
        [0, 0, 0, 1]
    ])


def find_circumcenter(A, B, C, D):
    """Find circumcenter of quadrilateral ABCD."""
    AC = (A + C) / 2
    BD = (B + D) / 2

    a = np.linalg.norm(A - B)
    b = np.linalg.norm(B - D)
    c = np.linalg.norm(D - C)
    d = np.linalg.norm(C - A)

    s = (a + b + c + d) / 2
    R = 0.25 * ((a * b + c * d) * (a * c + b * d) * (a * d + b * c) /
                ((s - a) * (s - b) * (s - c) * (s - d))) ** 0.5

    m = (BD - AC) / np.linalg.norm(BD - AC)
    circ_cent = AC + m * (R ** 2 - (d / 2) ** 2) ** 0.5

    return circ_cent


def generate_yoshimura_geometry(n, beta, d=None, gamma=0.0, psi=0.0):
    """
    Generate Yoshimura origami unit geometry in folded configuration.

    Parameters
    ----------
    n : int
        Number of sides in the polygon base
    beta : float
        Sector angle parameter
    psi : float
        Rotation angle (default: 0.0)
    T0 : np.ndarray, optional
        Initial transformation matrix (4x4)

    Returns
    -------
    nodes : np.ndarray
        Array of node positions, shape (N, 3)
    bars : list of tuples
        List of (node_i, node_j, length) for each bar
    faces : list of tuples
        List of (node_i, node_j, node_k) for each triangular face
    """

    # Calculate derived parameters for folded configuration
    r = 1 / (2 * np.sin(np.pi / n))
    w = 0.5 * np.tan(beta)

    # Folded configuration: gamma=0, d calculated from beta
    if d is None:
        d = (np.tan(beta) ** 2 - np.tan(np.pi / (2 * n)) ** 2) ** 0.5

    # Generate base polygon vertices
    base = np.array([
        [r * np.sin(2 * np.pi / n * i),
         -r * np.cos(2 * np.pi / n * i),
         0, 1]
        for i in range(n)
    ]).T

    # Transform to get top polygon
    T = general_transform_matrix(psi, gamma, d)
    top = T @ base

    params = [n, beta, d, gamma, psi]

    # Calculate midpoint positions
    mid = np.zeros((4, 2 * n))
    centers = np.zeros((2 * n, 3))

    for i in range(n):
        A = base[:3, i]
        B = base[:3, (i + 1) % n]
        C = top[:3, i]
        D = top[:3, (i + 1) % n]

        p = np.cross(C - B, D - A)
        s = np.linalg.norm((A + B) / 2 - (C + D) / 2)
        a = 1 / (2 * np.cos(beta))

        x_ = w ** 2 - (s / 2) ** 2 + 1e-20
        x = np.abs(x_) ** 0.5 if x_ >= -1e-2 else 0.0

        if np.linalg.norm(p) < 1e-10:
            # Degenerate case - planar quadrilateral
            A_ = base[:3, (i + n // 2) % n]
            B_ = base[:3, (i + 1 + n // 2) % n]
            C_ = top[:3, (i + n // 2) % n]
            D_ = top[:3, (i + 1 + n // 2) % n]

            ct = (A + B + C + D) / 4

            if n % 2 == 0:
                q_hat = ct - (A_ + B_ + C_ + D_) / 4
            else:
                q_hat = ct - (B_ + D_) / 2

            centers[i] = ct + x * q_hat / np.linalg.norm(q_hat)
        else:
            # Non-planar quadrilateral
            ct = find_circumcenter(A, B, C, D)
            centers[i] = ct - p / np.linalg.norm(p) * np.abs(a ** 2 - np.linalg.norm(A - ct) ** 2) ** 0.5

    # Assign mid-edge vertices
    for i in range(n):
        B = base[:3, (i + 1) % n]
        D = top[:3, (i + 1) % n]

        mid[:3, 2 * i] = centers[i]

        if np.linalg.norm(B - D) < 1.95 * w:
            mid[:3, 2 * i + 1] = (centers[i] + centers[(i + 1) % n]) / 2
        else:
            mid[:3, 2 * i + 1] = (B + D) / 2

        mid[3, :] = 1.0

    # Create three sets of 2n nodes each: base_nodes, mid_nodes, top_nodes
    # Each set contains: [vertex, edge_midpoint, vertex, edge_midpoint, ...]
    base_nodes = np.zeros((2 * n, 3))
    mid_nodes = np.zeros((2 * n, 3))
    top_nodes = np.zeros((2 * n, 3))

    for i in range(n):
        # Base layer: vertex at 2*i, edge midpoint at 2*i+1
        base_nodes[2 * i] = base[:3, i]
        base_nodes[2 * i + 1] = (base[:3, i] + base[:3, (i + 1) % n]) / 2

        # Mid layer: interior vertices (from centers and edge mids)
        mid_nodes[2 * i] = mid[:3, 2 * i - 1]
        mid_nodes[2 * i + 1] = mid[:3, 2 * i]

        # Top layer: vertex at 2*i, edge midpoint at 2*i+1
        top_nodes[2 * i] = top[:3, i]
        top_nodes[2 * i + 1] = (top[:3, i] + top[:3, (i + 1) % n]) / 2

    # Assemble node list: [base_nodes, mid_nodes, top_nodes]
    nodes = []
    nodes.extend(base_nodes)  # indices 0 to 2n-1
    nodes.extend(mid_nodes)  # indices 2n to 4n-1
    nodes.extend(top_nodes)  # indices 4n to 6n-1
    nodes = np.array(nodes)

    # Index helper functions for the new structure
    def base_idx(i):
        """Get index in base layer (0 to 2n-1)"""
        return i % (2 * n)

    def mid_idx(i):
        """Get index in mid layer (2n to 4n-1)"""
        return 2 * n + i % (2 * n)

    def top_idx(i):
        """Get index in top layer (4n to 6n-1)"""
        return 4 * n + i % (2 * n)

    # Generate bars and faces
    bars = []
    faces = []

    for i in range(2 * n):
        j = i + 1
        k = i - 1

        # edges
        i1, i2 = base_idx(i), base_idx(j)
        length = np.linalg.norm(nodes[i1] - nodes[i2])
        bars.append((i1, i2, length))

        i1, i2 = mid_idx(i), mid_idx(j)
        length = np.linalg.norm(nodes[i1] - nodes[i2])
        bars.append((i1, i2, length))

        i1, i2 = top_idx(i), top_idx(j)
        length = np.linalg.norm(nodes[i1] - nodes[i2])
        bars.append((i1, i2, length))

        # base to mid
        i1, i2 = base_idx(i), mid_idx(i)
        length = np.linalg.norm(nodes[i1] - nodes[i2])
        bars.append((i1, i2, length))

        if i % 2 == 0:
            i1, i2 = base_idx(i), mid_idx(j)
            length = np.linalg.norm(nodes[i1] - nodes[i2])
            bars.append((i1, i2, length))

            i1, i2 = base_idx(i), mid_idx(k)
            length = np.linalg.norm(nodes[i1] - nodes[i2])
            bars.append((i1, i2, length))

            # faces
            faces.append((base_idx(i), mid_idx(i), mid_idx(j)))
            faces.append((base_idx(i), mid_idx(i), mid_idx(k)))
            faces.append((base_idx(i), base_idx(j), mid_idx(j)))
            faces.append((base_idx(i), base_idx(k), mid_idx(k)))

        # mid to top
        i1, i2 = mid_idx(i), top_idx(i)
        length = np.linalg.norm(nodes[i1] - nodes[i2])
        bars.append((i1, i2, length))

        if i % 2 == 0:
            i1, i2 = top_idx(i), mid_idx(j)
            length = np.linalg.norm(nodes[i1] - nodes[i2])
            bars.append((i1, i2, length))

            i1, i2 = top_idx(i), mid_idx(k)
            length = np.linalg.norm(nodes[i1] - nodes[i2])
            bars.append((i1, i2, length))

            # faces
            faces.append((top_idx(i), mid_idx(i), mid_idx(j)))
            faces.append((top_idx(i), mid_idx(i), mid_idx(k)))
            faces.append((top_idx(i), top_idx(j), mid_idx(j)))
            faces.append((top_idx(i), top_idx(k), mid_idx(k)))

    return nodes, bars, faces, params


def create_yoshimura_geometry(setup: ExperimentSetup, n=4, beta=np.pi / 6, d=None, gamma=0.0, psi=0.0,
                              k_axial=1000.0, k_fold=10.0, k_facet=200.0,
                              mass=0.01, damping=5.0):
    """
    Generates Yoshimura-Ori geometry and adds it to the ExperimentSetup.

    Parameters
    ----------
    setup : ExperimentSetup
        The experiment setup object
    n : int
        Number of polygon sides
    beta : float
        Sector angle in radians
    k_axial : float
        Axial stiffness for bars
    k_fold : float
        Fold stiffness for hinges (not yet implemented)
    k_facet : float
        Facet stiffness for hinges (not yet implemented)
    mass : float
        Mass per node
    damping : float
        Damping coefficient for bars

    Returns
    -------
    faces : list
        List of triangular faces for visualization
    node_info : dict
        Dictionary containing node indices for actuation setup
    """

    print(f"\nYoshimura Parameters:")
    print(f"  n={n}, beta={np.rad2deg(beta):.2f}°")

    # Generate geometry
    nodes, bars, faces, params = generate_yoshimura_geometry(n, beta, d=d, gamma=gamma, psi=psi)

    print(f"\nGenerated Geometry:")
    print(f"  Nodes: {len(nodes)}")
    print(f"  Bars: {len(bars)}")
    print(f"  Faces: {len(faces)}")

    node_info = {
        'base_corners': [2 * i for i in range(n)],
        'top_corners': [4 * n + 2 * i for i in range(n)],
        'n': n,
        'total_nodes': len(nodes)
    }

    # Add nodes to setup
    for i, node_pos in enumerate(nodes):
        # if node is in base or top corner fix it
        if i in node_info['base_corners'] or i in node_info['top_corners']:
            setup.add_node(node_pos, mass=mass, fixed=True)
        else:
            setup.add_node(node_pos, mass=mass, fixed=False)

    # Add bars to setup
    for i, j, length in bars:
        setup.add_bar(i, j, stiffness=k_axial, rest_length=length, damping=damping)

    # TODO: Add hinges here
    # This is where we'll add hinge identification and creation logic

    # Store node indices for actuation
    # Base corners (vertices only, even indices): 0, 2, 4, ..., 2*(n-1)
    # Top corners (vertices only, even indices): 4*n, 4*n+2, 4*n+4, ..., 4*n+2*(n-1)

    return faces, node_info, params


def setup_actuation(setup: ExperimentSetup, node_info: dict,
                    amplitude=0.5, min_pos=0.0, max_pos=1.0, frequency=0.5, duration=10.0):
    """
    Setup sinusoidal actuation for top corner nodes.

    Parameters
    ----------
    setup : ExperimentSetup
        The experiment setup object
    node_info : dict
        Dictionary containing node indices
    amplitude : float
        Amplitude of vertical oscillation
    frequency : float
        Frequency of oscillation in Hz
    duration : float
        Duration of simulation
    """

    # Get initial positions
    positions = setup.nodes['positions']

    # Signal parameters
    dt_sig = 0.001
    t = np.arange(0, duration, dt_sig)
    omega = 2 * np.pi * frequency

    # Create signals for each top corner
    base_corners = node_info['base_corners']
    top_corners = node_info['top_corners']

    print(f"\nSetting up actuation:")
    print(f"  Fixed base corners: {base_corners}")
    print(f"  Actuated top corners: {top_corners}")

    # Create sinusoidal signals for top corners
    for i, idx in enumerate(top_corners):
        p0 = positions[idx]

        #
        # sig = np.zeros((len(t), 3), dtype=np.float32)
        # sig[:, 0] = p0[0]  # X stays constant
        # sig[:, 1] = p0[1]  # Y stays constant
        # sig[:, 2] = p0[2] - ramp * amplitude * np.sin(omega * t)  # Z oscillates

        # generate a signal to go from min_pos to max_pos from p0[2] with given frequency
        sig = np.zeros((len(t), 3), dtype=np.float32)
        sig[:, 0] = p0[0]
        sig[:, 1] = p0[1]
        sig[:, 2] = p0[2] - (max_pos - min_pos) * (1 + np.sin(omega * t)) / 2

        sig_name = f"sig_top_corner_{i}"
        setup.add_signal(sig_name, sig, dt=dt_sig)
        setup.add_actuator(idx, sig_name, type='position')


def main():
    """Setup the Yoshimura experiment"""
    print("\n[Setup] Creating Yoshimura Experiment...")

    # Initialize Setup
    setup = ExperimentSetup(DEMO_DIR, overwrite=True)

    # Simulation parameters
    duration = 10.0
    dt = 0.0005
    save_interval = 0.01

    # Configure Simulation
    setup.set_simulation_params(duration=duration, dt=dt, save_interval=save_interval)
    setup.set_physics(gravity=0.0, damping=0.2)

    # Build Geometry
    faces, node_info, params = create_yoshimura_geometry(
        setup,
        n=4,
        beta=np.deg2rad(40),
        # d=np.tan(np.deg2rad(35)),
        k_axial=1000.0,
        mass=0.01,
        damping=5.0
    )

    n, beta, d, gamma, psi = params

    # Setup Actuation
    setup_actuation(
        setup,
        node_info,
        amplitude=0.0,
        min_pos=0.0,
        max_pos=d,
        frequency=0.1,
        duration=duration
    )

    # Save Everything
    setup.save()

    # Save Visualization Faces
    with h5py.File(DEMO_DIR / "input" / "visualization.h5", 'w') as f:
        f.create_dataset("faces", data=np.array(faces, dtype=np.int32))

    print(f"\nSaved to: {DEMO_DIR}")


def run_simulation():
    """Run the simulation"""
    print("\n[Step 2] Running Simulation...")
    import demlat
    from demlat.models.barhinge import BarHingeModel

    exp = demlat.Experiment(DEMO_DIR)
    eng = demlat.Engine(BarHingeModel, backend='cuda')
    eng.run(exp)

    print("\nSimulation complete!")


if __name__ == "__main__":
    main()
    run_simulation()
    visualize_experiment(DEMO_DIR)
