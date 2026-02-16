"""
openprc.optimize
=================

Gradient-based optimization tools for DEMLAT physics models.

Uses the JAX differentiable solver to compute exact gradients of
simulation outputs w.r.t. model parameters (stiffness, damping, etc.),
enabling calibration, inverse design, and sensitivity analysis.

Modules
-------
calibration : Calibrate model parameters against reference trajectories
costs       : Pluggable cost/loss functions
theta       : Parameter pytree utilities (masking, bounds, updates)
reference   : Reference data loaders

Quick Start
-----------
::

    from openprc.optimize import Calibration
    from openprc.demlat.models.barhinge import BarHingeModel

    cal = Calibration(BarHingeModel, backend='jax')
    cal.load_geometry("experiments/my_test")
    cal.load_reference("experiments/my_test/output/simulation.h5")

    cal.optimize_params(bar_stiffness=True, hinge_stiffness=True)
    cal.set_bounds(bar_stiffness=(0.1, 1000.0))

    result = cal.run(max_iterations=500, lr=0.01, cost='mse')
    cal.save("experiments/my_test/optimized_geometry.h5")
"""

from .calibration import Calibration, CalibrationResult
from .costs import get_cost, COSTS
from .reference import ReferenceData

__all__ = [
    'Calibration',
    'CalibrationResult',
    'get_cost',
    'COSTS',
    'ReferenceData',
]