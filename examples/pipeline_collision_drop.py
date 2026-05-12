"""
Collision Drop Test
===================
Two-node minimal collision test:
  - Node 0: fixed at origin, collidable  (the "floor node")
  - Node 1: free, collidable, dropped from height under gravity

Node 1 falls, hits Node 0, and bounces. No bars — pure collision.

Collision parameters:
  collision_radius    : each node acts as a sphere of this radius;
                        contact triggers when centre-to-centre dist < 2 * radius
  collision_restitution: 0.0 = fully inelastic, 1.0 = perfectly elastic
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import openprc.demlat
from openprc.demlat.models.barhinge import BarHingeModel
from openprc.demlat.io.simulation_setup import SimulationSetup
from openprc.demlat.utils.animator import ShowSimulation

# ── experiment location ───────────────────────────────────────────────────────
DEMO_DIR = Path("experiments/collision_drop_test")

# ── tunable parameters ────────────────────────────────────────────────────────
DROP_HEIGHT   = 1.0   # metres — starting height of the free node
NODE_MASS     = 1.0   # kg
RADIUS        = 0.05  # collision sphere radius per node (contact at 2 * 0.025 = 0.16 m)
RESTITUTION   = 0.3  # bounciness: 0 = dead stop, 1 = perfect bounce
DURATION      = 4.0   # simulation seconds
BACKEND       = 'cuda' # 'cpu' or 'cuda'


def step_1_setup():
    print("\n[Step 1] Building experiment...")
    setup = SimulationSetup(DEMO_DIR, overwrite=True)

    # ── simulation time ───────────────────────────────────────────────────────
    setup.set_simulation_params(duration=DURATION, dt=0.001, save_interval=0.01)

    # ── physics: gravity + collision enabled ──────────────────────────────────
    setup.set_physics(
        gravity=-9.81,
        damping=0.02,               # light air damping so bounces gradually decay
        enable_collision=True,
        collision_radius=RADIUS,
        collision_restitution=RESTITUTION,
        collision_iterations=3,     # extra PBD passes for clean separation
    )

    # ── node 0: fixed at origin, collidable (floor node) ─────────────────────
    setup.add_node([0.0, 0.0, 0.0], mass=NODE_MASS, fixed=True,  collidable=True)

    # ── node 1: free, collidable, starting above node 0 ──────────────────────
    setup.add_node([0.0, 0.0, DROP_HEIGHT], mass=NODE_MASS, fixed=False, collidable=True)

    # no bars — the only interaction is via collision

    setup.save()

    t_contact = np.sqrt(2 * (DROP_HEIGHT - 2 * RADIUS) / 9.81)
    print(f"  Node 0  : fixed at (0, 0, 0)")
    print(f"  Node 1  : free   at (0, 0, {DROP_HEIGHT}) — falling under gravity")
    print(f"  Contact : when centre-to-centre dist < {2*RADIUS:.3f} m")
    print(f"  Expected first contact at ≈ {t_contact:.3f} s")


def step_2_run():
    print("\n[Step 2] Running simulation...")
    exp = openprc.demlat.Simulation(DEMO_DIR)
    eng = openprc.demlat.Engine(BarHingeModel, backend=BACKEND)
    eng.run(exp)
    print("  Done.")


def step_3_visualize():
    print("\n[Step 3] Launching visualizer...")
    ShowSimulation(str(DEMO_DIR))


if __name__ == "__main__":
    step_1_setup()
    step_2_run()
    step_3_visualize()