"""
demlat I/O Module
=================

Handles input/output operations, experiment setup, and validation.
"""

from .simulation_setup import SimulationSetup

__all__ = ["SimulationSetup", "DemlatSimValidator"]


def __getattr__(name):
    # Lazy re-export: importing DemlatSimValidator here at module load would
    # create a circular import (schemas.__init__ -> demlat_sim_validator ->
    # demlat package init -> this module -> demlat_sim_validator).
    if name == "DemlatSimValidator":
        from openprc.schemas.demlat_sim_validator import DemlatSimValidator
        return DemlatSimValidator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
