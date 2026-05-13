"""
openprc.reservoir — Readout layer for Physical Reservoir Computing
==================================================================

Modules
-------
io.StateLoader          : Load node/bar time-series from a simulation.h5
features.NodePositions  : Node position state features
features.NodeVelocities : Node velocity state features
features.BarStrains     : Bar strain state features
readout.Ridge           : Regularised linear readout (sklearn Ridge)
training.Trainer        : Fit + evaluate a readout on loaded states
"""

from .io.state_loader import StateLoader
from .readout.ridge import Ridge
from .training.trainer import Trainer, TrainingResult
from . import features, readout, io, training

__all__ = [
    "StateLoader",
    "Ridge",
    "Trainer",
    "TrainingResult",
    "features",
    "readout",
    "io",
    "training",
]
