"""
demlat Core Module
==================

This module exposes the core components of the DEMLat physics engine.
It includes the main Engine, Simulation management, Base Model definitions,
Scaling utilities, and custom Exceptions.
"""

import logging
from ..utils.logging import get_logger

# Initialize module-level logger
logger = get_logger("demlat.core")

try:
    from ..core.exceptions import (
        DemlatError,
        ExperimentNotFoundError,
        SchemaValidationError,
        ConfigurationError,
        DataIntegrityError
    )
    from ..core.base_model import BaseModel
    from ..core.engine import Engine
    from ..core.simulation import Simulation
    from ..core.scaler import SimulationScaler

    logger.debug("Successfully imported core components.")

except ImportError as e:
    logger.critical(f"Failed to import core components: {e}", exc_info=True)
    raise
