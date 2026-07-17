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
from .utils.data_parser import SimulationData


def ShowSimulation(*args, **kwargs):
    """Launch the interactive 3-D viewer for a finished experiment.

    The viewer stack (``piviz-3d`` + ``imgui``) is imported lazily, so headless
    simulation and training never pay for a GUI import. If the stack is missing
    you get a clear message naming the correct PyPI packages — note the package
    is ``piviz-3d`` (not ``piviz``, which is the import name).
    """
    try:
        from .utils.animator import ShowSimulation as _ShowSimulation
    except ImportError as e:
        raise ImportError(
            "ShowSimulation needs the 3-D viewer stack, which is not installed.\n"
            "    pip install piviz-3d imgui\n"
            "(the PyPI package is 'piviz-3d', even though you import it as 'piviz')"
        ) from e
    return _ShowSimulation(*args, **kwargs)

# Define package version
__version__ = "0.1.0"

# Define what happens on 'from demlat import *'
__all__ = ["Simulation",
           "Engine",
           "BarHingeModel",
           "SimulationSetup",
           "ShowSimulation",
           "SimulationData",
           ]
