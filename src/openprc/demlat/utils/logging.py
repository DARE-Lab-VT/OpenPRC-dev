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


def get_logger(name: str) -> logging.Logger:
    """
    Creates or retrieves a logger with specific formatting and handlers.
    
    This function ensures that all loggers created through it will share
    the same file handler once it's set up by `setup_file_logging`.

    Args:
        name: Dot-separated module name (e.g., 'demlat.io').
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
        c_handler.setLevel(logging.INFO) # Default console level
        c_handler.setFormatter(logging.Formatter(CONSOLE_FMT))
        logger.addHandler(c_handler)

    # 2. Attach shared file handler if it already exists
    if _SHARED_FILE_HANDLER and _SHARED_FILE_HANDLER not in logger.handlers:
        logger.addHandler(_SHARED_FILE_HANDLER)

    return logger


def setup_file_logging(log_dir: Path, file_level: int = logging.DEBUG):
    """
    Initializes the shared file logger for the entire application.
    This should be called once at the start of an execution script.
    
    Args:
        log_dir: The directory where the log file will be created.
        file_level: The logging level for the file.
    """
    global _SHARED_FILE_HANDLER
    
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "demlat_execution.log"

    # Create the new shared handler
    new_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8') # 'w' to overwrite old logs
    new_handler.setLevel(file_level)
    new_handler.setFormatter(logging.Formatter(FILE_FMT))

    # If there was an old handler, close it
    if _SHARED_FILE_HANDLER:
        _SHARED_FILE_HANDLER.close()
    
    _SHARED_FILE_HANDLER = new_handler

    # Update ALL known loggers to use the new handler
    for l in _KNOWN_LOGGERS:
        # Remove any old file handlers
        old_handlers = [h for h in l.handlers if isinstance(h, logging.FileHandler)]
        for h in old_handlers:
            l.removeHandler(h)
        
        # Add the new one
        l.addHandler(new_handler)
        
    # Log the initialization
    root_logger = get_logger("demlat")
    root_logger.info(f"File logging initialized at: {log_file}")
