"""
RVC Worker Schemas

Defines input validation, internal metadata tracking, and output schemas
for the RVC voice conversion serverless worker.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from datetime import datetime
import json
import uuid
import time


# =============================================================================
# UUID7 Generator (Time-ordered UUID)
# =============================================================================

def generate_uuid7() -> str:
    """
    Generate a UUID7 (time-ordered UUID).
    Format: 8-4-4-4-12 hexadecimal characters
    Uses Unix timestamp in milliseconds for time component.
    """
    # Get current timestamp in milliseconds
    timestamp_ms = int(time.time() * 1000)
    
    # UUID7 structure:
    # - 48 bits: Unix timestamp in milliseconds
    # - 4 bits: version (7)
    # - 12 bits: random
    # - 2 bits: variant (10)
    # - 62 bits: random
    
    # Generate random bits
    random_bits = uuid.uuid4().int
    
    # Construct UUID7
    uuid_int = (
        (timestamp_ms & 0xFFFFFFFFFFFF) << 80 |  # 48-bit timestamp
        (0x7 << 76) |                              # 4-bit version (7)
        ((random_bits >> 64) & 0x0FFF) << 64 |    # 12-bit random
        (0x2 << 62) |                              # 2-bit variant (10)
        (random_bits & 0x3FFFFFFFFFFFFFFF)         # 62-bit random
    )
    
    # Format as UUID string
    hex_str = f"{uuid_int:032x}"
    return f"{hex_str[:8]}-{hex_str[8:12]}-{hex_str[12:16]}-{hex_str[16:20]}-{hex_str[20:]}"


# =============================================================================
# Constants (Fixed Values)
# =============================================================================

# Fixed processing parameters
F0_METHOD = "rmvpe"
OUTPUT_FORMAT = "ogg"
OUTPUT_CODEC = "opus"
OUTPUT_BITRATE = "128k"
OUTPUT_SAMPLE_RATE = 48000

# Default values
DEFAULT_F0_UP_KEY = 0
DEFAULT_INDEX_RATE = 0.75


# =============================================================================
# Input Schema (for RunPod validation)
# =============================================================================

INPUT_SCHEMA = {
    # -------------------------------------------------------------------------
    # Required Fields
    # -------------------------------------------------------------------------
    'vocal_url': {
        'type': str,
        'required': True,
        'description': 'URL of the input vocal audio file to convert'
    },
    'model_url': {
        'type': str,
        'required': True,
        'description': 'URL of the RVC model (.pth) file'
    },
    
    # -------------------------------------------------------------------------
    # Optional Fields
    # -------------------------------------------------------------------------
    'index_url': {
        'type': str,
        'required': False,
        'default': None,
        'description': 'URL of the index (.index) file for feature retrieval'
    },
    'f0_up_key': {
        'type': int,
        'required': False,
        'default': DEFAULT_F0_UP_KEY,
        'constraints': lambda x: -24 <= x <= 24,
        'description': 'Pitch shift in semitones (-24 to +24). 0 = original pitch'
    },
    'index_rate': {
        'type': float,
        'required': False,
        'default': DEFAULT_INDEX_RATE,
        'constraints': lambda x: 0.0 <= x <= 1.0,
        'description': 'Feature index mix ratio (0.0 - 1.0). Higher = more similar to training voice'
    },
    'tag': {
        'type': str,
        'required': False,
        'default': None,
        'description': 'Client-provided tag for tracking/identification'
    },
}


# =============================================================================
# Dataclasses for Internal Metadata
# =============================================================================

@dataclass
class AudioInfo:
    """Information about an audio file."""
    url: Optional[str] = None
    file_path: Optional[str] = None
    duration_seconds: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    format: Optional[str] = None
    file_size_bytes: Optional[int] = None
    peak_amplitude: Optional[float] = None
    rms_level: Optional[float] = None


@dataclass
class ModelInfo:
    """Information about the RVC model used."""
    url: str
    file_hash: Optional[str] = None
    version: Optional[str] = None  # "v1" or "v2"
    has_f0: Optional[bool] = None
    target_sample_rate: Optional[int] = None
    speaker_count: Optional[int] = None
    cached: bool = False


@dataclass
class TimingMetrics:
    """Timing information for each processing stage."""
    total_seconds: float = 0.0
    download_vocal_seconds: float = 0.0
    download_model_seconds: float = 0.0
    download_index_seconds: float = 0.0
    load_model_seconds: float = 0.0
    pitch_extraction_seconds: float = 0.0
    inference_seconds: float = 0.0
    encoding_seconds: float = 0.0
    upload_seconds: float = 0.0


@dataclass
class ResourceMetrics:
    """Resource usage metrics."""
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_peak_mb: Optional[float] = None
    
    # Cache statistics
    model_cache_hit: bool = False
    vocal_cache_hit: bool = False
    index_cache_hit: bool = False


@dataclass
class ErrorInfo:
    """Error information for failed requests."""
    error_type: str = ""
    error_message: str = ""
    error_stage: Optional[str] = None  # Which stage failed
    stack_trace: Optional[str] = None


@dataclass
class InferenceMetadata:
    """
    Complete internal metadata for tracking and analytics.
    This is stored/logged internally for analysis.
    """
    # Identifiers
    request_id: str = ""  # UUID7
    job_id: str = ""      # RunPod job ID
    tag: Optional[str] = None  # Client-provided tag
    
    # Timestamps
    received_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    # Input information
    input_audio: Optional[AudioInfo] = None
    input_params: Optional[Dict[str, Any]] = None
    
    # Model information
    model_info: Optional[ModelInfo] = None
    index_url: Optional[str] = None
    
    # Output information
    output_audio: Optional[AudioInfo] = None
    
    # Processing settings (fixed values)
    f0_method: str = F0_METHOD
    output_format: str = OUTPUT_FORMAT
    output_codec: str = OUTPUT_CODEC
    output_bitrate: str = OUTPUT_BITRATE
    
    # Metrics
    timing: Optional[TimingMetrics] = None
    resources: Optional[ResourceMetrics] = None
    
    # Result
    success: bool = True
    error: Optional[ErrorInfo] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str, indent=2)
    
    @classmethod
    def create(cls, job_id: str, tag: Optional[str] = None) -> 'InferenceMetadata':
        """Factory method to create new metadata instance with UUID7."""
        return cls(
            request_id=generate_uuid7(),
            job_id=job_id,
            tag=tag,
            received_at=datetime.utcnow().isoformat() + "Z",
            timing=TimingMetrics(),
            resources=ResourceMetrics(),
        )


# =============================================================================
# Output Schema
# =============================================================================

@dataclass
class ConversionOutput:
    """
    Output schema returned to the client.
    """
    # Result status
    result: str  # "success" or "failure"
    message: Optional[str] = None  # Error message if failure
    
    # Request tracking
    request_id: str = ""
    tag: Optional[str] = None
    
    # Audio result (only on success)
    audio_url: Optional[str] = None
    sample_rate: Optional[int] = None
    duration_seconds: Optional[float] = None
    file_size_bytes: Optional[int] = None
    format: str = OUTPUT_FORMAT
    
    # Processing information
    processing_time_seconds: Optional[float] = None
    model_version: Optional[str] = None
    
    # Detailed timing (optional, for debugging)
    timing_details: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values for clean output."""
        result = {}
        for key, value in asdict(self).items():
            if value is not None:
                result[key] = value
        return result


# =============================================================================
# Output Factory Functions
# =============================================================================

def create_success_output(
    audio_url: str,
    metadata: InferenceMetadata,
    include_timing: bool = True,
) -> Dict[str, Any]:
    """
    Create a success output dictionary from inference metadata.
    """
    output = ConversionOutput(
        result="success",
        message=None,
        request_id=metadata.request_id,
        tag=metadata.tag,
        audio_url=audio_url,
        sample_rate=metadata.output_audio.sample_rate if metadata.output_audio else OUTPUT_SAMPLE_RATE,
        duration_seconds=metadata.output_audio.duration_seconds if metadata.output_audio else None,
        file_size_bytes=metadata.output_audio.file_size_bytes if metadata.output_audio else None,
        format=OUTPUT_FORMAT,
        processing_time_seconds=metadata.timing.total_seconds if metadata.timing else None,
        model_version=metadata.model_info.version if metadata.model_info else None,
    )
    
    if include_timing and metadata.timing:
        output.timing_details = {
            "download_vocal": metadata.timing.download_vocal_seconds,
            "download_model": metadata.timing.download_model_seconds,
            "download_index": metadata.timing.download_index_seconds,
            "load_model": metadata.timing.load_model_seconds,
            "pitch_extraction": metadata.timing.pitch_extraction_seconds,
            "inference": metadata.timing.inference_seconds,
            "encoding": metadata.timing.encoding_seconds,
            "upload": metadata.timing.upload_seconds,
        }
    
    return output.to_dict()


def create_error_output(
    error_message: str,
    metadata: InferenceMetadata,
    refresh_worker: bool = False,
) -> Dict[str, Any]:
    """
    Create an error output dictionary.
    """
    output = ConversionOutput(
        result="failure",
        message=error_message,
        request_id=metadata.request_id,
        tag=metadata.tag,
    )
    
    result = output.to_dict()
    
    # Add refresh_worker flag for RunPod
    if refresh_worker:
        result["refresh_worker"] = True
    
    return result
