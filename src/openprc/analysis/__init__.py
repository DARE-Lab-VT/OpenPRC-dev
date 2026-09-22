"""
demlat.analysis — Multistability Analysis
==========================================

Tools for finding and classifying equilibria of nonlinear bar-hinge systems.

Modules
-------
equilibria : Deflated Newton solver for finding multiple equilibria

Example
-------
    from demlat.analysis.equilibria import EquilibriumFinder

    finder = EquilibriumFinder.from_experiment("experiments/yoshimura_test")
    results = finder.find_all(num_random=50)
    results.summary()

    # Save results
    finder.save_results(results, "equilibria.h5")
"""

from .multistability.equilibria import EquilibriumFinder, Equilibrium, FinderResults
from .spectroscopy import reach_spectroscopy, spectroscopy_core, MachineAbstraction

__all__ = [
    "EquilibriumFinder",
    "Equilibrium",
    "FinderResults",
    "reach_spectroscopy",
    "spectroscopy_core",
    "MachineAbstraction",
]
