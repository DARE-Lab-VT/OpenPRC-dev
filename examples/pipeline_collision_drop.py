"""
Newton's Cradle
===============
5 pendulum balls, each suspended from two fixed anchors (front and back rail)
by rigid rods.  The V-shaped suspension constrains each ball to swing cleanly
in the X-Z plane — just like a real cradle.

Ball 0 is pulled back 45° and released; the elastic collision chain propagates
through balls 1–3 and ball 4 flies out.

Node layout
-----------
  Nodes  0–4  : front anchors  (fixed, y = +Y_SPREAD/2, z = PENDULUM_L)
  Nodes  5–9  : back  anchors  (fixed, y = -Y_SPREAD/2, z = PENDULUM_L)
  Nodes 10–14 : ball nodes     (free, collidable, y = 0)

Visual sphere radius
--------------------
After setup.save(), nodes/radius is patched into geometry.h5 so the
visualiser renders balls at the correct physical size — no library changes.
"""

import numpy as np
import h5py
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from openprc.demlat import BarHingeModel, Simulation, Engine, ShowSimulation, SimulationSetup

# ── experiment location ───────────────────────────────────────────────────────
DEMO_DIR = Path("experiments/newton_cradle")

# ── parameters ────────────────────────────────────────────────────────────────
N_BALLS = 5  # number of pendulum balls
RADIUS = 0.05  # m  — collision sphere radius (contact at 2 * RADIUS)
VISUAL_RADIUS = 0.05  # m  — rendered sphere radius (tweak freely)
Y_SPREAD = 0.15  # m  — front/back anchor separation (V-shape width)
PENDULUM_L = 0.50  # m  — effective pendulum length (vertical component)
RAISE_ANGLE = 45.0  # °  — how far ball 0 is pulled back
BALL_MASS = 1.0  # kg
ROD_STIFF = -1  # N/m — high stiffness → nearly rigid rod
RESTITUTION = 0.99  # near-elastic so the chain propagates cleanly
DURATION = 5.0  # s
BACKEND = 'cuda'  # 'cpu' or 'cuda'


def step_1_setup():
    print("\n[Step 1] Building Newton's Cradle (two-rod V suspension)...")
    setup = SimulationSetup(DEMO_DIR, overwrite=True)

    setup.set_simulation_params(duration=DURATION, dt=0.001, save_interval=0.01)

    setup.set_physics(
        gravity=-9.81,
        damping=0.005,  # light air damping — pendulum swings freely
        enable_collision=True,
        collision_radius=RADIUS,
        collision_restitution=RESTITUTION,
        collision_iterations=8,  # enough PBD passes for the chain to propagate
    )

    spacing = 2.0 * RADIUS  # 0.10 m — balls just touching at rest
    theta = np.deg2rad(RAISE_ANGLE)
    y_half = Y_SPREAD / 2.0

    # Ball rest x-positions, centred at x = 0
    ball_x = [(i - (N_BALLS - 1) / 2.0) * spacing for i in range(N_BALLS)]
    # → [-0.20, -0.10, 0.00, 0.10, 0.20] m

    # ── front anchor nodes (0 .. N-1): fixed, y = +y_half ────────────────────
    for i in range(N_BALLS):
        setup.add_node([ball_x[i], +y_half, PENDULUM_L],
                       mass=1.0, fixed=True, collidable=False)

    # ── back anchor nodes (N .. 2N-1): fixed, y = -y_half ────────────────────
    for i in range(N_BALLS):
        setup.add_node([ball_x[i], -y_half, PENDULUM_L],
                       mass=1.0, fixed=True, collidable=False)

    # ── ball nodes (2N .. 3N-1): free, collidable, y = 0 ─────────────────────
    for i in range(N_BALLS):
        if i == 0:
            # ball 0: pulled back in X-Z plane — arc preserves rod length
            bx = float(ball_x[0] - PENDULUM_L * np.sin(theta))
            bz = float(PENDULUM_L * (1.0 - np.cos(theta)))
        else:
            bx, bz = ball_x[i], 0.0
        setup.add_node([bx, 0.0, bz], mass=BALL_MASS,
                       fixed=False, collidable=True)

    # ── rods: front + back anchor → ball[i]  (two per ball) ──────────────────
    for i in range(N_BALLS):
        ball_idx = 2 * N_BALLS + i
        # rest_length auto-computed = sqrt(PENDULUM_L² + y_half²)
        setup.add_bar(i, ball_idx, stiffness=ROD_STIFF, damping=0.5)
        setup.add_bar(N_BALLS + i, ball_idx, stiffness=ROD_STIFF, damping=0.5)

    setup.save()

    # ── patch visual sphere radius into geometry.h5 ───────────────────────────
    # The visualiser reads nodes/radius (animator.py line 143) and takes the
    # median as the global sphere draw radius — no library file needs changing.
    geom_path = DEMO_DIR / "input" / "geometry.h5"
    with h5py.File(geom_path, 'a') as f:
        n_nodes = len(f['nodes/positions'])
        radii = np.full(n_nodes, VISUAL_RADIUS, dtype=np.float32)
        if 'radius' in f['nodes']:
            del f['nodes/radius']
        f['nodes'].create_dataset('radius', data=radii)

    rod_len = float(np.sqrt(PENDULUM_L ** 2 + y_half ** 2))
    t_impact = (np.pi / 2.0) * np.sqrt(PENDULUM_L / 9.81)
    print(f"  {N_BALLS} balls, R = {RADIUS} m, L = {PENDULUM_L} m, Y-spread = {Y_SPREAD} m")
    print(f"  Rod length = {rod_len:.4f} m  (2 rods per ball, V-shape)")
    print(f"  Visual radius = {VISUAL_RADIUS} m  (patched into geometry.h5)")
    print(f"  Ball 0 raised {RAISE_ANGLE:.0f}° — first impact ≈ {t_impact:.2f} s")


def step_2_run():
    print("\n[Step 2] Running simulation...")
    exp = Simulation(DEMO_DIR)
    eng = Engine(BarHingeModel, backend=BACKEND)
    eng.run(exp)
    print("  Done.")


def step_3_visualize():
    print("\n[Step 3] Launching visualizer...")
    ShowSimulation(str(DEMO_DIR))


if __name__ == "__main__":
    step_1_setup()
    step_2_run()
    step_3_visualize()
