"""
demlat Exceptions
=================
Centralized exception hierarchy for the demlat package.
"""


class DemlatError(Exception):
    """Base class for all demlat exceptions."""
    pass


class ExperimentNotFoundError(DemlatError):
    """Raised when the experiment directory or required files are missing."""
    pass


class SchemaValidationError(DemlatError):
    """Raised when data formats (HDF5/JSON) do not match the expected specification."""
    pass


class ConfigurationError(DemlatError):
    """Raised when the configuration logic (wiring) is invalid (e.g., connecting a signal that doesn't exist)."""
    pass


class DataIntegrityError(DemlatError):
    """Raised when loaded data arrays have mismatched shapes (e.g., 100 nodes but 90 masses)."""
    pass


class InvalidArgumentError(DemlatError):
    """Raised when a function receives an argument of an invalid type or value."""
    pass


class MissingDependencyError(DemlatError):
    """Raised when a required external dependency (e.g., a specific library) is not found."""
    pass


class ResourceNotFoundError(DemlatError):
    """Raised when a required resource (e.g., a file, directory, or database entry) is not found."""
    pass


class CalculationError(DemlatError):
    """Raised when a numerical calculation fails or produces an invalid result."""
    pass


class ProcessingError(DemlatError):
    """Raised when an error occurs during data processing (e.g., during a pipeline stage)."""
    pass


class VisualizationError(DemlatError):
    """Raised when an error occurs during data visualization."""
    pass


class ExportError(DemlatError):
    """Raised when an error occurs during data export."""
    pass


class ImportError(DemlatError):
    """Raised when an error occurs during data import."""
    pass


class DatabaseError(DemlatError):
    """Raised when an error occurs during database operations."""
