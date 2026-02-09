"""
RVC Worker Custom Exceptions

Defines all possible exceptions that can occur during the RVC inference pipeline.
Each exception includes:
- error_stage: Which stage of the pipeline failed
- recoverable: Whether the worker can continue processing other requests
- http_status: Suggested HTTP status code for API responses
"""

from typing import Optional


class RVCWorkerException(Exception):
    """
    Base exception for all RVC Worker errors.
    """
    error_stage: str = "unknown"
    recoverable: bool = True
    http_status: int = 500
    
    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(message)
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message}: {self.details}"
        return self.message


# =============================================================================
# Input Validation Exceptions
# =============================================================================

class ValidationException(RVCWorkerException):
    """Base class for input validation errors."""
    error_stage = "input_validation"
    recoverable = True
    http_status = 400


class MissingRequiredFieldException(ValidationException):
    """Required field is missing from input."""
    
    def __init__(self, field_name: str):
        super().__init__(
            message=f"Missing required field: {field_name}",
            details=f"The field '{field_name}' is required but was not provided"
        )
        self.field_name = field_name


class InvalidFieldTypeException(ValidationException):
    """Field has wrong type."""
    
    def __init__(self, field_name: str, expected_type: str, actual_type: str):
        super().__init__(
            message=f"Invalid type for field '{field_name}'",
            details=f"Expected {expected_type}, got {actual_type}"
        )
        self.field_name = field_name
        self.expected_type = expected_type
        self.actual_type = actual_type


class FieldConstraintException(ValidationException):
    """Field value violates constraints."""
    
    def __init__(self, field_name: str, constraint: str, value: any):
        super().__init__(
            message=f"Constraint violation for field '{field_name}'",
            details=f"Value {value} violates constraint: {constraint}"
        )
        self.field_name = field_name
        self.constraint = constraint
        self.value = value


class InvalidURLException(ValidationException):
    """URL format is invalid."""
    
    def __init__(self, field_name: str, url: str):
        super().__init__(
            message=f"Invalid URL format for '{field_name}'",
            details=f"URL: {url[:100]}..."
        )
        self.field_name = field_name
        self.url = url


# =============================================================================
# Download Exceptions
# =============================================================================

class DownloadException(RVCWorkerException):
    """Base class for download errors."""
    error_stage = "download"
    recoverable = True
    http_status = 502


class URLNotReachableException(DownloadException):
    """Cannot reach the URL (network error)."""
    
    def __init__(self, url: str, reason: str = "Network error"):
        super().__init__(
            message="Cannot reach URL",
            details=f"{reason}: {url[:100]}..."
        )
        self.url = url
        self.reason = reason


class URLNotFoundException(DownloadException):
    """URL returned 404."""
    http_status = 404
    
    def __init__(self, url: str, resource_type: str = "file"):
        super().__init__(
            message=f"{resource_type.capitalize()} not found",
            details=f"URL returned 404: {url[:100]}..."
        )
        self.url = url
        self.resource_type = resource_type


class URLForbiddenException(DownloadException):
    """URL returned 403."""
    http_status = 403
    
    def __init__(self, url: str):
        super().__init__(
            message="Access forbidden",
            details=f"URL returned 403: {url[:100]}..."
        )
        self.url = url


class DownloadTimeoutException(DownloadException):
    """Download timed out."""
    http_status = 504
    
    def __init__(self, url: str, timeout_seconds: float):
        super().__init__(
            message="Download timeout",
            details=f"Exceeded {timeout_seconds}s for {url[:100]}..."
        )
        self.url = url
        self.timeout_seconds = timeout_seconds


class DownloadedFileCorruptException(DownloadException):
    """Downloaded file is corrupted or invalid."""
    
    def __init__(self, url: str, file_type: str, reason: str):
        super().__init__(
            message=f"Downloaded {file_type} file is corrupted",
            details=f"{reason}: {url[:100]}..."
        )
        self.url = url
        self.file_type = file_type
        self.reason = reason


# =============================================================================
# Audio Processing Exceptions
# =============================================================================

class AudioException(RVCWorkerException):
    """Base class for audio processing errors."""
    error_stage = "audio_processing"
    recoverable = True
    http_status = 422


class InvalidAudioFormatException(AudioException):
    """Audio file format is not supported or invalid."""
    
    def __init__(self, file_path: str, detected_format: Optional[str] = None):
        super().__init__(
            message="Invalid or unsupported audio format",
            details=f"Detected format: {detected_format or 'unknown'}"
        )
        self.file_path = file_path
        self.detected_format = detected_format


class AudioCorruptedException(AudioException):
    """Audio file is corrupted and cannot be read."""
    
    def __init__(self, file_path: str, reason: str):
        super().__init__(
            message="Audio file is corrupted",
            details=reason
        )
        self.file_path = file_path
        self.reason = reason


class AudioTooShortException(AudioException):
    """Audio is too short for processing."""
    
    def __init__(self, duration_seconds: float, min_duration: float = 0.5):
        super().__init__(
            message="Audio is too short",
            details=f"Duration {duration_seconds:.2f}s is less than minimum {min_duration}s"
        )
        self.duration_seconds = duration_seconds
        self.min_duration = min_duration


class AudioTooLongException(AudioException):
    """Audio is too long for processing."""
    
    def __init__(self, duration_seconds: float, max_duration: float):
        super().__init__(
            message="Audio is too long",
            details=f"Duration {duration_seconds:.2f}s exceeds maximum {max_duration}s"
        )
        self.duration_seconds = duration_seconds
        self.max_duration = max_duration


class AudioLoadException(AudioException):
    """Failed to load audio file."""
    
    def __init__(self, file_path: str, reason: str):
        super().__init__(
            message="Failed to load audio",
            details=reason
        )
        self.file_path = file_path
        self.reason = reason


# =============================================================================
# Model Exceptions
# =============================================================================

class ModelException(RVCWorkerException):
    """Base class for model-related errors."""
    error_stage = "model_loading"
    recoverable = True
    http_status = 422


class InvalidModelFileException(ModelException):
    """Model file is not a valid RVC model."""
    
    def __init__(self, file_path: str, reason: str):
        super().__init__(
            message="Invalid RVC model file",
            details=reason
        )
        self.file_path = file_path
        self.reason = reason


class ModelCorruptedException(ModelException):
    """Model file is corrupted."""
    
    def __init__(self, file_path: str, reason: str):
        super().__init__(
            message="Model file is corrupted",
            details=reason
        )
        self.file_path = file_path
        self.reason = reason


class ModelArchitectureMismatchException(ModelException):
    """Model architecture doesn't match expected structure."""
    
    def __init__(self, expected: str, actual: str):
        super().__init__(
            message="Model architecture mismatch",
            details=f"Expected {expected}, got {actual}"
        )
        self.expected = expected
        self.actual = actual


class InvalidIndexFileException(ModelException):
    """Index file is invalid or corrupted."""
    
    def __init__(self, file_path: str, reason: str):
        super().__init__(
            message="Invalid index file",
            details=reason
        )
        self.file_path = file_path
        self.reason = reason


class ModelLoadException(ModelException):
    """Failed to load model into memory."""
    
    def __init__(self, file_path: str, reason: str):
        super().__init__(
            message="Failed to load model",
            details=reason
        )
        self.file_path = file_path
        self.reason = reason


# =============================================================================
# Inference Exceptions
# =============================================================================

class InferenceException(RVCWorkerException):
    """Base class for inference errors."""
    error_stage = "inference"
    recoverable = False  # Usually need worker restart
    http_status = 500


class CUDAOutOfMemoryException(InferenceException):
    """GPU ran out of memory."""
    recoverable = False
    
    def __init__(self, required_mb: Optional[float] = None, available_mb: Optional[float] = None):
        details = "GPU memory exhausted"
        if required_mb and available_mb:
            details = f"Requires ~{required_mb:.0f}MB, only {available_mb:.0f}MB available"
        super().__init__(
            message="CUDA out of memory",
            details=details
        )
        self.required_mb = required_mb
        self.available_mb = available_mb


class PitchExtractionException(InferenceException):
    """Pitch extraction (F0) failed."""
    recoverable = True
    
    def __init__(self, method: str, reason: str):
        super().__init__(
            message=f"Pitch extraction failed ({method})",
            details=reason
        )
        self.method = method
        self.reason = reason


class FeatureExtractionException(InferenceException):
    """Feature extraction (HuBERT) failed."""
    recoverable = True
    
    def __init__(self, reason: str):
        super().__init__(
            message="Feature extraction failed",
            details=reason
        )
        self.reason = reason


class VoiceConversionException(InferenceException):
    """Voice conversion inference failed."""
    recoverable = True
    
    def __init__(self, reason: str):
        super().__init__(
            message="Voice conversion failed",
            details=reason
        )
        self.reason = reason


class InferenceTensorException(InferenceException):
    """Tensor operation failed during inference."""
    recoverable = False
    
    def __init__(self, operation: str, reason: str):
        super().__init__(
            message=f"Tensor operation failed: {operation}",
            details=reason
        )
        self.operation = operation
        self.reason = reason


# =============================================================================
# Encoding Exceptions
# =============================================================================

class EncodingException(RVCWorkerException):
    """Base class for audio encoding errors."""
    error_stage = "encoding"
    recoverable = True
    http_status = 500


class FFmpegNotAvailableException(EncodingException):
    """FFmpeg is not installed or not accessible."""
    recoverable = False
    
    def __init__(self):
        super().__init__(
            message="FFmpeg not available",
            details="FFmpeg is required for audio encoding but was not found"
        )


class FFmpegEncodingException(EncodingException):
    """FFmpeg encoding failed."""
    
    def __init__(self, reason: str, stderr: Optional[str] = None):
        super().__init__(
            message="Audio encoding failed",
            details=reason
        )
        self.reason = reason
        self.stderr = stderr


class OutputWriteException(EncodingException):
    """Failed to write output file."""
    
    def __init__(self, file_path: str, reason: str):
        super().__init__(
            message="Failed to write output file",
            details=reason
        )
        self.file_path = file_path
        self.reason = reason


# =============================================================================
# Upload Exceptions
# =============================================================================

class UploadException(RVCWorkerException):
    """Base class for upload errors."""
    error_stage = "upload"
    recoverable = True
    http_status = 502


class S3NotConfiguredException(UploadException):
    """S3 is required but not configured."""
    
    def __init__(self, missing_vars: list):
        super().__init__(
            message="S3 storage not configured",
            details=f"Missing environment variables: {', '.join(missing_vars)}"
        )
        self.missing_vars = missing_vars


class S3CredentialsException(UploadException):
    """S3 credentials are invalid."""
    http_status = 403
    
    def __init__(self, reason: str):
        super().__init__(
            message="S3 authentication failed",
            details=reason
        )
        self.reason = reason


class S3BucketException(UploadException):
    """S3 bucket access error."""
    
    def __init__(self, bucket: str, reason: str):
        super().__init__(
            message=f"S3 bucket error: {bucket}",
            details=reason
        )
        self.bucket = bucket
        self.reason = reason


class S3UploadException(UploadException):
    """S3 upload failed."""
    
    def __init__(self, key: str, reason: str):
        super().__init__(
            message="S3 upload failed",
            details=f"Failed to upload {key}: {reason}"
        )
        self.key = key
        self.reason = reason


class UploadTimeoutException(UploadException):
    """Upload timed out."""
    http_status = 504
    
    def __init__(self, timeout_seconds: float):
        super().__init__(
            message="Upload timeout",
            details=f"Upload exceeded {timeout_seconds}s timeout"
        )
        self.timeout_seconds = timeout_seconds


# =============================================================================
# Exception Mapping Helper
# =============================================================================

def classify_exception(exc: Exception) -> RVCWorkerException:
    """
    Convert a standard exception to an RVCWorkerException.
    Useful for wrapping unexpected exceptions.
    """
    error_message = str(exc)
    error_type = type(exc).__name__
    
    # CUDA OOM
    if "CUDA out of memory" in error_message or "OutOfMemoryError" in error_type:
        return CUDAOutOfMemoryException()
    
    # Network errors
    if "URLError" in error_type or "ConnectionError" in error_type:
        return URLNotReachableException("unknown", error_message)
    
    if "TimeoutError" in error_type or "timeout" in error_message.lower():
        return DownloadTimeoutException("unknown", 30.0)
    
    # File errors
    if "FileNotFoundError" in error_type:
        return DownloadedFileCorruptException("unknown", "file", "File not found")
    
    # Audio errors
    if "soundfile" in error_message.lower() or "LibsndfileError" in error_type:
        return AudioCorruptedException("unknown", error_message)
    
    # Model errors
    if "RuntimeError" in error_type and "size mismatch" in error_message.lower():
        return ModelArchitectureMismatchException("expected", "actual")
    
    # Default: wrap as generic inference exception
    return VoiceConversionException(f"{error_type}: {error_message}")
