"""
``openprc.analysis.spectroscopy`` — Reach spectroscopy for Physical Reservoir Computing.

Device-agnostic machine identification: measure the *machine abstraction*
``M = (D, {phi_n})`` of any reservoir substrate from a single-tone probe run.
The analysis operates on **parsed time-series arrays** (drive + multi-channel
readout), so the same code path runs on a simulation, a control, or digitized
experimental data.

Quick-start
-----------
>>> from openprc.analysis import reach_spectroscopy
>>> M = reach_spectroscopy(states, t=time, drive=drive)   # arrays, not a path
>>> M.D          # reach per tolerance
>>> M.summary()
>>> M.plot()

See :func:`reach_spectroscopy` for the masking flags that skip individual
measurements, and :func:`spectroscopy_core` for the loader-independent core.
"""

from .reach import (
    reach_spectroscopy,
    spectroscopy_core,
    estimate_f0,
    MachineAbstraction,
    plot_spectroscopy,
    plot_harmonic_fits,
)

__all__ = [
    "reach_spectroscopy",
    "spectroscopy_core",
    "estimate_f0",
    "MachineAbstraction",
    "plot_spectroscopy",
    "plot_harmonic_fits",
]