"""
RVC Worker Schemas (v2)

Input validation constants, output response structures,
and internal metadata dataclasses.
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List


# =============================================================================
# UUID7 Generator
# =============================================================================

def generate_uuid7() -> str:
    """Generate a UUID7 (time-ordered UUID)."""
    timestamp_ms = int(time.time() * 1000)
    random_bits = uuid.uuid4().int
    uuid_int = (
        (timestamp_ms & 0xFFFFFFFFFFFF) << 80
        | (0x7 << 76)
        | ((random_bits >> 64) & 0x0FFF) << 64
        | (0x2 << 62)
        | (random_bits & 0x3FFFFFFFFFFFFFFF)
    )
    hex_str = f"{uuid_int:032x}"
    return f"{hex_str[:8]}-{hex_str[8:12]}-{hex_str[12:16]}-{hex_str[16:20]}-{hex_str[20:]}"


# =============================================================================
# Supported Output Configurations
# =============================================================================

SUPPORTED_FORMATS: Dict[str, Dict[str, str]] = {
    "ogg":  {"codec": "opus",      "ffmpeg_codec": "libopus",    "content_type": "audio/ogg",  "ext": "ogg"},
    "m4a":  {"codec": "aac",       "ffmpeg_codec": "aac",        "content_type": "audio/mp4",  "ext": "m4a"},
    "mp3":  {"codec": "mp3",       "ffmpeg_codec": "libmp3lame", "content_type": "audio/mpeg", "ext": "mp3"},
    "wav":  {"codec": "pcm_s16le", "ffmpeg_codec": "pcm_s16le",  "content_type": "audio/wav",  "ext": "wav"},
    "flac": {"codec": "flac",      "ffmpeg_codec": "flac",       "content_type": "audio/flac", "ext": "flac"},
    "webm": {"codec": "opus",      "ffmpeg_codec": "libopus",    "content_type": "audio/webm", "ext": "webm"},
}

SUPPORTED_BITRATES = ["32k", "64k", "96k", "128k", "192k", "256k", "320k"]

SUPPORTED_SAMPLE_RATES = [8000, 16000, 22050, 44100, 48000, 96000]

# Formats where bitrate is ignored (lossless / PCM)
LOSSLESS_FORMATS = {"wav", "flac"}

# MIME type → file extension (for data-URL decoding)
MIME_TO_EXT: Dict[str, str] = {
    "audio/wav":      ".wav",
    "audio/x-wav":    ".wav",
    "audio/wave":     ".wav",
    "audio/mpeg":     ".mp3",
    "audio/mp3":      ".mp3",
    "audio/ogg":      ".ogg",
    "audio/flac":     ".flac",
    "audio/x-flac":   ".flac",
    "audio/mp4":      ".m4a",
    "audio/aac":      ".aac",
    "audio/webm":     ".webm",
    "audio/x-m4a":    ".m4a",
    "audio/opus":     ".ogg",
}

# =============================================================================
# Defaults
# =============================================================================

DEFAULT_FORMAT = "ogg"
DEFAULT_BITRATE = "128k"
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_F0_UP_KEY = 0
DEFAULT_INDEX_RATE = 0.75

# =============================================================================
# Fixed RVC Parameters
# =============================================================================

F0_METHOD = "rmvpe"
FILTER_RADIUS = 3
RMS_MIX_RATE = 0.25
PROTECT = 0.33

# =============================================================================
# Limits
# =============================================================================

MIN_AUDIO_DURATION = 0.5   # seconds
DOWNLOAD_TIMEOUT = 120     # seconds


# =============================================================================
# Response Dataclasses
# =============================================================================

@dataclass
class FileResult:
    """Single file entry in the response `files` array."""
    key: str = ""
    size: int = 0
    format: str = ""
    codec: str = ""
    channel: int = 1
    bitrate: str = ""
    sample_rate: int = 0
    duration_sec: float = 0.0
    url: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Metadata Dataclasses
# =============================================================================

@dataclass
class TraceInfo:
    request_id: str = ""
    started_at: str = ""
    ended_at: str = ""


@dataclass
class ResourceInfo:
    gpu_model: str = ""
    peak_vram: float = 0.0   # MB
    peak_power: float = 0.0  # Watts


@dataclass
class PerformanceInfo:
    download_file: float = 0.0
    validate_input_file: float = 0.0
    load_model: float = 0.0
    convert_vocal: float = 0.0
    encode_converted: float = 0.0
    upload_results: float = 0.0


@dataclass
class InputFileDebug:
    input_original_key: str = ""
    output_original_key: str = ""
    output_target_key: str = ""


@dataclass
class FilesDebugInfo:
    model_key: str = ""
    index_key: str = ""
    inputs: List[InputFileDebug] = field(default_factory=list)


@dataclass
class DebugInfo:
    trace: TraceInfo = field(default_factory=TraceInfo)
    resource: ResourceInfo = field(default_factory=ResourceInfo)
    performance: PerformanceInfo = field(default_factory=PerformanceInfo)
    files: FilesDebugInfo = field(default_factory=FilesDebugInfo)
    logs: List[str] = field(default_factory=list)


@dataclass
class Metadata:
    """Complete metadata uploaded alongside results."""
    request: Dict[str, Any] = field(default_factory=dict)
    response: Dict[str, Any] = field(default_factory=dict)
    debug: DebugInfo = field(default_factory=DebugInfo)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str, ensure_ascii=False, indent=2)


# =============================================================================
# Response Builder Helpers
# =============================================================================

def build_success_response(
    files: List[FileResult],
    message: str = "All inputs processed successfully",
) -> Dict[str, Any]:
    return {
        "status": "success",
        "code": "OK",
        "message": message,
        "files": [f.to_dict() for f in files],
    }


def build_fail_response(
    code: str,
    message: str,
    files: Optional[List[FileResult]] = None,
) -> Dict[str, Any]:
    resp: Dict[str, Any] = {
        "status": "fail",
        "code": code,
        "message": message,
    }
    if files:
        resp["files"] = [f.to_dict() for f in files]
    else:
        resp["files"] = []
    return resp
