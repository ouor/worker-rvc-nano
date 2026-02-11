"""
RVC Worker Exceptions (v2)

Structured exception hierarchy with error codes.
Each exception maps to a response `code` field for client consumption.
"""

from typing import Optional, List


# =============================================================================
# Base Exception
# =============================================================================

class RVCError(Exception):
    """Base exception for all RVC worker errors."""

    code: str = "INTERNAL_ERROR"
    http_status: int = 500
    recoverable: bool = True

    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} — {self.details}"
        return self.message


# =============================================================================
# Input Validation
# =============================================================================

class ValidationError(RVCError):
    """Base for input validation errors."""
    code = "VALIDATION_ERROR"
    http_status = 400


class MissingRequiredFieldError(ValidationError):
    code = "MISSING_REQUIRED_FIELD"

    def __init__(self, field_name: str):
        super().__init__(
            f"Missing required field: {field_name}",
            f"The field '{field_name}' is required but was not provided",
        )
        self.field_name = field_name


class EmptyInputURLError(ValidationError):
    code = "EMPTY_INPUT_URL"

    def __init__(self):
        super().__init__(
            "At least 1 valid url is required",
            "input_urls must be a non-empty list of URLs or data URLs",
        )


class FormatNotSupportedError(ValidationError):
    code = "FORMAT_NOT_SUPPORTED"

    def __init__(self, fmt: str, supported: List[str]):
        super().__init__(
            f"Format '{fmt}' is not supported",
            f"Supported formats: {', '.join(supported)}",
        )
        self.fmt = fmt


class InvalidBitrateError(ValidationError):
    code = "INVALID_BITRATE"

    def __init__(self, bitrate: str, supported: List[str]):
        super().__init__(
            f"Bitrate '{bitrate}' is not supported",
            f"Supported bitrates: {', '.join(supported)}",
        )
        self.bitrate = bitrate


class InvalidSampleRateError(ValidationError):
    code = "INVALID_SAMPLE_RATE"

    def __init__(self, sample_rate: int, supported: List[int]):
        super().__init__(
            f"Sample rate {sample_rate} is not supported",
            f"Supported sample rates: {', '.join(map(str, supported))}",
        )
        self.sample_rate = sample_rate


class InvalidF0UpKeyError(ValidationError):
    code = "INVALID_F0_UP_KEY"

    def __init__(self, value):
        super().__init__(
            f"f0_up_key must be an integer between -24 and 24",
            f"Got: {value}",
        )


class InvalidIndexRateError(ValidationError):
    code = "INVALID_INDEX_RATE"

    def __init__(self, value):
        super().__init__(
            f"index_rate must be a float between 0.0 and 1.0",
            f"Got: {value}",
        )


class InvalidDataURLError(ValidationError):
    code = "INVALID_DATA_URL"

    def __init__(self, reason: str):
        super().__init__(
            "Invalid data URL",
            reason,
        )


# =============================================================================
# Download
# =============================================================================

class DownloadError(RVCError):
    """Base for download errors."""
    code = "DOWNLOAD_FAILED"
    http_status = 502


class ModelDownloadError(DownloadError):
    code = "MODEL_DOWNLOAD_FAILED"

    def __init__(self, url: str, reason: str):
        super().__init__(
            "Failed to download model",
            f"{reason} | url={url[:120]}",
        )
        self.url = url


class IndexDownloadError(DownloadError):
    code = "INDEX_DOWNLOAD_FAILED"

    def __init__(self, url: str, reason: str):
        super().__init__(
            "Failed to download index file",
            f"{reason} | url={url[:120]}",
        )
        self.url = url


class InputDownloadError(DownloadError):
    code = "INPUT_DOWNLOAD_FAILED"

    def __init__(self, url_or_key: str, reason: str):
        super().__init__(
            "Failed to download input audio",
            f"{reason} | key={url_or_key[:120]}",
        )


# =============================================================================
# Audio
# =============================================================================

class AudioError(RVCError):
    """Base for audio processing errors."""
    code = "AUDIO_ERROR"
    http_status = 422


class InvalidAudioError(AudioError):
    code = "INVALID_AUDIO"

    def __init__(self, reason: str):
        super().__init__("Invalid or unreadable audio file", reason)


class AudioTooShortError(AudioError):
    code = "AUDIO_TOO_SHORT"

    def __init__(self, duration: float, minimum: float):
        super().__init__(
            "Audio is too short for processing",
            f"Duration {duration:.2f}s < minimum {minimum}s",
        )
        self.duration = duration


# =============================================================================
# Model
# =============================================================================

class ModelLoadError(RVCError):
    code = "MODEL_LOAD_FAILED"
    http_status = 500
    recoverable = False

    def __init__(self, reason: str):
        super().__init__("Failed to load RVC model", reason)


# =============================================================================
# Inference
# =============================================================================

class InferenceError(RVCError):
    code = "INFERENCE_FAILED"
    http_status = 500
    recoverable = True

    def __init__(self, reason: str):
        super().__init__("Voice conversion inference failed", reason)


class CUDAOutOfMemoryError(RVCError):
    code = "CUDA_OOM"
    http_status = 500
    recoverable = False

    def __init__(self):
        super().__init__(
            "CUDA out of memory",
            "GPU memory exhausted during inference",
        )


# =============================================================================
# Encoding
# =============================================================================

class EncodingError(RVCError):
    code = "ENCODING_FAILED"
    http_status = 500

    def __init__(self, reason: str, stderr: Optional[str] = None):
        details = reason
        if stderr:
            details += f" | stderr={stderr[:300]}"
        super().__init__("Audio encoding failed", details)
        self.stderr = stderr


class FFmpegNotAvailableError(RVCError):
    code = "FFMPEG_NOT_AVAILABLE"
    http_status = 500
    recoverable = False

    def __init__(self):
        super().__init__(
            "FFmpeg not available",
            "FFmpeg is required but was not found on the system",
        )


# =============================================================================
# Upload / S3
# =============================================================================

class S3NotConfiguredError(RVCError):
    code = "S3_NOT_CONFIGURED"
    http_status = 500
    recoverable = False

    def __init__(self, missing_vars: List[str]):
        super().__init__(
            "S3 storage not configured",
            f"Missing env vars: {', '.join(missing_vars)}",
        )


class S3UploadError(RVCError):
    code = "S3_UPLOAD_FAILED"
    http_status = 502

    def __init__(self, key: str, reason: str):
        super().__init__(
            "S3 upload failed",
            f"key={key} | {reason}",
        )
        self.key = key


# =============================================================================
# Aggregate
# =============================================================================

class AllInputsFailedError(RVCError):
    code = "ALL_INPUTS_FAILED"
    http_status = 422
    recoverable = True

    def __init__(self, total: int, errors: List[str]):
        super().__init__(
            f"All {total} input(s) failed to process",
            "; ".join(errors[:5]),
        )
        self.errors = errors
