"""
demlat (Discrete Element Method Lattice)
========================================

Core physics engine for simulating discrete element lattices.
"""

# Expose the main user-facing classes
from .core.experiment import Experiment
from .core.engine import Engine

# Define package version
__version__ = "0.0.1"

# Define what happens on 'from demlat import *'
__all__ = ["Experiment", "Engine"]
