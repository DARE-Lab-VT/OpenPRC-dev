"""
demlat.analysis — Multistability Analysis
==========================================

Tools for finding and classifying equilibria of nonlinear bar-hinge systems.

Modules
-------
equilibria : Deflated Newton solver for finding multiple equilibria
stability  : Eigenvalue-based stability classification (included in equilibria)
landscape  : Energy landscape sampling and visualization (included in equilibria)

Example
-------
    from demlat.analysis.equilibria import EquilibriumAnalyzer

    exp = demlat.Experiment("my_experiment/")
    analyzer = EquilibriumAnalyzer(exp)
    results = analyzer.find_equilibria(n_attempts=200)
    print(results.summary())

    # Get only stable states
    for eq in results.stable:
        print(f"Stable state: E={eq.energy:.4f}, min_eig={eq.min_eigenvalue:.2e}")
"""

from .equilibria import EquilibriumAnalyzer, EquilibriumSet, Equilibrium, StabilityType

__all__ = [
    "EquilibriumAnalyzer",
    "EquilibriumSet",
    "Equilibrium",
    "StabilityType",
]