"""
demlat (Discrete Element Method Lattice)
========================================

Core physics engine for simulating discrete element lattices.
"""

# Expose the main user-facing classes
from .core.simulation import Simulation
from .core.engine import Engine
from .models.barhinge import BarHingeModel
from .io.simulation_setup import SimulationSetup
from .utils.animator import ShowSimulation
from .utils.data_parser import SimulationData

# Define package version
__version__ = "0.0.1"

# Define what happens on 'from demlat import *'
__all__ = ["Simulation",
           "Engine",
           "BarHingeModel",
           "SimulationSetup",
           "ShowSimulation",
           "SimulationData",
           ]
