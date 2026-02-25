"""Custom exception classes for Ldpj_backend.

Each exception type maps to a distinct failure domain so that the health
monitoring layer can classify and report faults accurately.
"""


class LdpjBackendError(Exception):
    """Base exception for all Ldpj_backend errors."""


class PLCConnectionError(LdpjBackendError):
    """Raised when the PLC connection cannot be established or is lost."""


class PLCReadError(LdpjBackendError):
    """Raised when a PLC read operation fails."""


class PLCWriteError(LdpjBackendError):
    """Raised when a PLC write-back operation fails."""


class ModelLoadError(LdpjBackendError):
    """Raised when the XGBoost model or scaler cannot be loaded."""


class ModelPredictError(LdpjBackendError):
    """Raised when model inference fails unexpectedly."""


class DataValidationError(LdpjBackendError):
    """Raised when incoming sensor data fails validation checks."""


class StorageError(LdpjBackendError):
    """Raised when a database write or query operation fails."""


class HealthCheckError(LdpjBackendError):
    """Raised when a health-check item reports a failure."""


class ConfigError(LdpjBackendError):
    """Raised when a configuration file is missing or malformed."""
