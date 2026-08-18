"""
DEMLAT Module — Comprehensive Example
======================================
Demonstrates the full SimulationSetup → Engine → ShowSimulation workflow:

  Geometry     : 10×10 planar node grid (100 nodes)
  Boundary     : Top row fixed; rest free under gravity
  Bars         : Axial spring-dampers on all grid edges + diagonals
  Hinges       : Bending stiffness at every shared-edge quad
  Actuators    :
      - Corner BL: full-DOF position actuator (circular orbit)
      - Corner BR: partial-DOF force actuator (z-axis push only)
  Collision    : Bottom floor is collidable — free nodes bounce off each other
  Visualization: Triangular mesh faces for the animator
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from openprc.demlat import SimulationSetup, Simulation, Engine, ShowSimulation

# ── Experiment location ───────────────────────────────────────────────────────
EXP = Path("experiments/example_demlat")

# ── Geometry parameters ───────────────────────────────────────────────────────
ROWS, COLS  = 10, 10          # 3×3 grid → 100 nodes
SPACING     = 1.0           # metres between nodes
MASS        = 0.5           # kg per free node
K_AXIAL     = 800.0         # bar stiffness  [N/m]
C_AXIAL     = 2.0           # bar damping    [N·s/m]
K_HINGE     = 30.0          # hinge stiffness [N·m/rad]
DURATION    = 8.0           # simulation seconds
BACKEND     = "cuda"        # "cpu", "cuda", or "jax"


def _node(row, col):
    """Linear index for node at (row, col)."""
    return row * COLS + col


# ─────────────────────────────────────────────────────────────────────────────
def step_1_build():
    print("\n[Step 1] Building geometry...")

    setup = SimulationSetup(EXP, overwrite=True)
    setup.set_simulation_params(duration=DURATION, dt=0.001, save_interval=0.01)
    setup.set_physics(
        gravity=-9.81,
        damping=0.05,
        enable_collision=True,
        collision_radius=0.08,
        collision_restitution=0.4,
        collision_iterations=3,
    )

    # ── Nodes ─────────────────────────────────────────────────────────────────
    # Top row (row=2) is fixed; remaining rows are free
    # Bottom row (row=0) is additionally collidable so they interact on contact
    for row in range(ROWS):
        for col in range(COLS):
            x = col * SPACING
            z = row * SPACING
            is_fixed     = (row == ROWS - 1)            # top row fixed
            is_collidable = (row == 0)                  # bottom row collidable
            setup.add_node([x, 0.0, z], mass=MASS,
                           fixed=is_fixed, collidable=is_collidable)

    n = len(setup.nodes['positions'])
    print(f"  Nodes   : {n}")

    # ── Bars: horizontal + vertical + diagonal ────────────────────────────────
    bar_count = 0
    for row in range(ROWS):
        for col in range(COLS):
            a = _node(row, col)
            # horizontal
            if col + 1 < COLS:
                b = _node(row, col + 1)
                setup.add_bar(a, b, stiffness=K_AXIAL, damping=C_AXIAL)
                bar_count += 1
            # vertical
            if row + 1 < ROWS:
                b = _node(row + 1, col)
                setup.add_bar(a, b, stiffness=K_AXIAL, damping=C_AXIAL)
                bar_count += 1
            # diagonal (NE)
            if row + 1 < ROWS and col + 1 < COLS:
                b = _node(row + 1, col + 1)
                setup.add_bar(a, b,
                              stiffness=K_AXIAL * 0.5,
                              damping=C_AXIAL,
                              rest_length=SPACING * np.sqrt(2))
                bar_count += 1

    print(f"  Bars    : {bar_count}")

    # ── Hinges: one per quad (4 quads in a 3×3 grid) ─────────────────────────
    hinge_count = 0
    for row in range(ROWS - 1):
        for col in range(COLS - 1):
            j = _node(row,     col)
            k = _node(row,     col + 1)
            i = _node(row + 1, col)
            l = _node(row + 1, col + 1)
            # axis = j–k; left wing = i; right wing = l
            setup.add_hinge([j, k, i, l], stiffness=K_HINGE)
            hinge_count += 1

    print(f"  Hinges  : {hinge_count}")

    # ── Visualization faces (triangulate each quad) ───────────────────────────
    face_count = 0
    for row in range(ROWS - 1):
        for col in range(COLS - 1):
            bl = _node(row,     col)
            br = _node(row,     col + 1)
            tl = _node(row + 1, col)
            tr = _node(row + 1, col + 1)
            setup.add_face([bl, br, tr], color=[100, 180, 255, 200])
            setup.add_face([bl, tr, tl], color=[100, 180, 255, 200])
            face_count += 2

    print(f"  Faces   : {face_count}")

    # ── Signal 1: circular position orbit for bottom-left corner ─────────────
    dt_sig = 0.001
    t      = np.arange(0, DURATION, dt_sig)
    omega  = 2 * np.pi * 0.4     # 0.4 Hz orbit
    ramp   = np.clip(t / 1.5, 0.0, 1.0)  # 1.5-second ramp-up
    radius = 0.4

    bl_node  = _node(0, 0)
    p0       = np.array(setup.nodes['positions'][bl_node])
    pos_sig  = np.zeros((len(t), 3), dtype=np.float32)
    pos_sig[:, 0] = p0[0] + ramp * radius * (np.cos(omega * t) - 1.0)
    pos_sig[:, 1] = 0.0
    pos_sig[:, 2] = p0[2] + ramp * radius * np.sin(omega * t)

    setup.add_signal("circle_drive", pos_sig, dt=dt_sig)
    setup.add_actuator(bl_node, "circle_drive", type="position")   # full 3-DOF
    print(f"  Actuator: node {bl_node} — full-DOF circular position")

    # ── Signal 2: z-only force burst on bottom-right corner ──────────────────
    br_node    = _node(0, COLS - 1)
    force_sig  = np.zeros((len(t), 3), dtype=np.float32)
    force_sig[:, 2] = 8.0 * np.sin(2 * np.pi * 1.2 * t) * ramp   # 1.2 Hz z-force

    setup.add_signal("z_force", force_sig, dt=dt_sig)
    setup.add_actuator(br_node, "z_force", type="force", dof=[0, 0, 1])  # z-axis only
    print(f"  Actuator: node {br_node} — partial-DOF (z-only) force")

    setup.save()
    print(f"  Saved → {EXP}")


# ─────────────────────────────────────────────────────────────────────────────
def step_2_run():
    print(f"\n[Step 2] Running simulation (backend={BACKEND})...")
    eng = Engine(backend=BACKEND)   # BarHingeModel is the default
    eng.run(Simulation(EXP))
    print("  Done.")


# ─────────────────────────────────────────────────────────────────────────────
def step_3_animate():
    print("\n[Step 3] Launching animator...")
    ShowSimulation(str(EXP))


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    step_1_build()
    step_2_run()
    step_3_animate()
