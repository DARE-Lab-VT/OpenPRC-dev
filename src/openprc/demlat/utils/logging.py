"""
demlat Logging System
=====================
Provides consistent, scoped loggers for the package.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, List

# UPDATED: Now includes the logger name (e.g. [demlat.engine])
CONSOLE_FMT = "[%(name)s] %(levelname)s: %(message)s"

# File format remains detailed with timestamps
FILE_FMT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

# Global state to manage shared file logging across modules
_KNOWN_LOGGERS: List[logging.Logger] = []
_SHARED_FILE_HANDLER: Optional[logging.FileHandler] = None


def get_logger(name: str, log_dir: Optional[Path] = None, file_level: int = logging.DEBUG,
               console_level: int = logging.INFO) -> logging.Logger:
    """
    Creates or retrieves a logger with specific formatting and handlers.
    
    If 'log_dir' is provided, it initializes (or resets) the shared file handler
    for ALL loggers created by this function. This ensures that loggers created
    before the experiment path was known (e.g. Engine) will start writing to the
    correct log file once the Experiment initializes.

    Args:
        name: Dot-separated module name (e.g., 'demlat.io').
        log_dir: If provided, a log file will be created in this directory, and
                 attached to ALL known demlat loggers.
        file_level: Logging level for the file handler.
        console_level: Logging level for the console handler.
    """
    global _SHARED_FILE_HANDLER

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture everything, handlers filter it down
    logger.propagate = False  # Don't double-log to root

    # Track this logger so we can update it later if the file handler changes
    if logger not in _KNOWN_LOGGERS:
        _KNOWN_LOGGERS.append(logger)

    # 1. Console Handler (Ensure one exists)
    # Check if we already have a StreamHandler (that isn't a FileHandler)
    has_console = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers)

    if not has_console:
        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setLevel(console_level)
        c_handler.setFormatter(logging.Formatter(CONSOLE_FMT))
        logger.addHandler(c_handler)

    # 2. File Handler Logic
    if log_dir:
        # We are initializing a NEW file logging session (e.g. Experiment start or reset)
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "demlat_execution.log"

        # Create new handler
        new_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        new_handler.setLevel(file_level)
        new_handler.setFormatter(logging.Formatter(FILE_FMT))

        # Replace the shared handler
        old_handler = _SHARED_FILE_HANDLER
        _SHARED_FILE_HANDLER = new_handler

        # Update ALL known loggers (retroactively fix Engine, etc.)
        for l in _KNOWN_LOGGERS:
            # Remove old shared handler if present
            if old_handler and old_handler in l.handlers:
                l.removeHandler(old_handler)

            # Add new handler if not present
            if new_handler not in l.handlers:
                l.addHandler(new_handler)

        # Close the old handler to release the file lock
        if old_handler:
            old_handler.close()

    else:
        # Just retrieving a logger. Attach the shared handler if it exists.
        if _SHARED_FILE_HANDLER and _SHARED_FILE_HANDLER not in logger.handlers:
            logger.addHandler(_SHARED_FILE_HANDLER)

    return logger
