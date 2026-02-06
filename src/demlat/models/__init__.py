"""
DEMLAT Physics Models
=====================

This package contains the physics model implementations for the DEMLAT engine.

Available Models:
-----------------
- BarHingeModel: A discrete element model simulating structures using bars (axial springs)
                 and hinges (torsional springs). Supports both soft (elastic) and rigid
                 constraints via a hybrid RK4 / Position-Based Dynamics solver.

Usage:
------
    from demlat.models import BarHingeModel
    
    # Initialize model with an experiment configuration
    model = BarHingeModel(experiment, backend='cuda')
"""

from .barhinge import BarHingeModel

__all__ = ['BarHingeModel']
