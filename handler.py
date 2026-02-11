"""
RunPod Serverless Handler for RVC Voice Conversion — v2

Changes from v1:
  - Multiple input URLs (list), including data:audio/... Data URLs
  - User-configurable output format / bitrate / sample_rate
  - Structured response with per-file results
  - Comprehensive metadata with debug trace / resource / performance / files / logs
  - All artifacts (inputs, outputs, model, index) uploaded to R2
"""

import os
import sys
import re
import base64
import hashlib
import time
import json
import logging
import traceback
import subprocess
import tempfile
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any, List

import torch
import numpy as np
import soundfile as sf
import boto3
from botocore.exceptions import (
    ClientError,
    NoCredentialsError,
    PartialCredentialsError,
    EndpointConnectionError,
)

# Add src to path for local imports
sys.path.insert(0, "/src")

import runpod

from schemas_new import (
    generate_uuid7,
    SUPPORTED_FORMATS,
    SUPPORTED_BITRATES,
    SUPPORTED_SAMPLE_RATES,
    LOSSLESS_FORMATS,
    MIME_TO_EXT,
    DEFAULT_FORMAT,
    DEFAULT_BITRATE,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_F0_UP_KEY,
    DEFAULT_INDEX_RATE,
    F0_METHOD,
    FILTER_RADIUS,
    RMS_MIX_RATE,
    PROTECT,
    MIN_AUDIO_DURATION,
    DOWNLOAD_TIMEOUT,
    FileResult,
    TraceInfo,
    ResourceInfo,
    PerformanceInfo,
    InputFileDebug,
    FilesDebugInfo,
    DebugInfo,
    Metadata,
    build_success_response,
    build_fail_response,
)
from exceptions_new import (
    RVCError,
    EmptyInputURLError,
    FormatNotSupportedError,
    InvalidBitrateError,
    InvalidSampleRateError,
    InvalidF0UpKeyError,
    InvalidIndexRateError,
    InvalidDataURLError,
    MissingRequiredFieldError,
    ModelDownloadError,
    IndexDownloadError,
    InputDownloadError,
    ModelLoadError,
    InferenceError,
    CUDAOutOfMemoryError,
    EncodingError,
    FFmpegNotAvailableError,
    S3NotConfiguredError,
    S3UploadError,
    InvalidAudioError,
    AudioTooShortError,
    AllInputsFailedError,
)
from src.rvc import RVCInference

torch.cuda.empty_cache()


# =============================================================================
# Constants
# =============================================================================
ASSETS_DIR = "/assets"
HUBERT_PATH = f"{ASSETS_DIR}/hubert/hubert_base.pt"
RMVPE_PATH = f"{ASSETS_DIR}/rmvpe/rmvpe.pt"
MODEL_CACHE_DIR = "/tmp/models"
AUDIO_CACHE_DIR = "/tmp/audio"
WORK_DIR = "/tmp/work"

# S3 Configuration
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
S3_KEY_PREFIX = os.environ.get("S3_KEY_PREFIX", "worker-rvc-nano")
S3_REGION = os.environ.get("S3_REGION", "auto")
S3_PUBLIC_URL = os.environ.get("S3_PUBLIC_URL")
S3_PRESIGNED_EXPIRY = int(os.environ.get("S3_PRESIGNED_EXPIRY", 60 * 60 * 24))  # 1 day


# =============================================================================
# Logging — dual output: console + in-memory list (captured into metadata)
# =============================================================================
class ListLogHandler(logging.Handler):
    """Captures formatted log lines into an internal list."""

    def __init__(self):
        super().__init__()
        self.logs: List[str] = []

    def emit(self, record):
        try:
            self.logs.append(self.format(record))
        except Exception:
            pass

    def reset(self):
        self.logs.clear()

    def snapshot(self) -> List[str]:
        return list(self.logs)


LOG_FORMAT = "[%(asctime)s] %(levelname)s | %(message)s"
LOG_DATEFMT = "%H:%M:%S"

logger = logging.getLogger("rvc_worker")
logger.setLevel(logging.DEBUG)

_console = logging.StreamHandler(sys.stdout)
_console.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
logger.addHandler(_console)

_list_handler = ListLogHandler()
_list_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
logger.addHandler(_list_handler)


# =============================================================================
# Timer
# =============================================================================
class Timer:
    """Context-manager stopwatch."""

    def __init__(self):
        self.elapsed: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start


# =============================================================================
# S3 Storage
# =============================================================================
class S3Storage:
    """S3-compatible storage client (AWS S3, Cloudflare R2, MinIO, etc.)."""

    def __init__(self):
        self._client = None
        self._configured = False
        self._check()

    # ------------------------------------------------------------------
    def _check(self):
        required = {
            "S3_ENDPOINT_URL": S3_ENDPOINT_URL,
            "S3_ACCESS_KEY": S3_ACCESS_KEY,
            "S3_SECRET_KEY": S3_SECRET_KEY,
            "S3_BUCKET_NAME": S3_BUCKET_NAME,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            logger.warning(f"S3 not configured — missing: {missing}")
            self._configured = False
        else:
            self._configured = True
            logger.info(f"S3 configured — bucket={S3_BUCKET_NAME}")

    @property
    def is_configured(self) -> bool:
        return self._configured

    @property
    def client(self):
        if self._client is None:
            if not self._configured:
                missing = [k for k, v in {
                    "S3_ENDPOINT_URL": S3_ENDPOINT_URL,
                    "S3_ACCESS_KEY": S3_ACCESS_KEY,
                    "S3_SECRET_KEY": S3_SECRET_KEY,
                    "S3_BUCKET_NAME": S3_BUCKET_NAME,
                }.items() if not v]
                raise S3NotConfiguredError(missing)
            try:
                self._client = boto3.client(
                    "s3",
                    endpoint_url=S3_ENDPOINT_URL,
                    aws_access_key_id=S3_ACCESS_KEY,
                    aws_secret_access_key=S3_SECRET_KEY,
                    region_name=S3_REGION,
                )
            except Exception as exc:
                raise S3UploadError("(client-init)", str(exc))
        return self._client

    # ------------------------------------------------------------------
    def key(self, request_id: str, filename: str) -> str:
        return f"{S3_KEY_PREFIX}/{request_id}/{filename}"

    def upload(
        self,
        local_path: str,
        request_id: str,
        filename: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload a local file and return the R2 object key."""
        obj_key = self.key(request_id, filename)
        try:
            self.client.upload_file(
                Filename=local_path,
                Bucket=S3_BUCKET_NAME,
                Key=obj_key,
                ExtraArgs={"ContentType": content_type},
            )
            logger.debug(f"S3 uploaded: {obj_key} ({os.path.getsize(local_path)} bytes)")
            return obj_key
        except NoCredentialsError:
            raise S3UploadError(obj_key, "No credentials")
        except PartialCredentialsError as exc:
            raise S3UploadError(obj_key, f"Partial credentials: {exc}")
        except EndpointConnectionError as exc:
            raise S3UploadError(obj_key, f"Endpoint unreachable: {exc}")
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "Unknown")
            msg = exc.response.get("Error", {}).get("Message", str(exc))
            raise S3UploadError(obj_key, f"{code}: {msg}")
        except Exception as exc:
            raise S3UploadError(obj_key, str(exc))

    def presigned_url(self, obj_key: str) -> str:
        """Generate a presigned GET URL (or use public URL prefix)."""
        if S3_PUBLIC_URL:
            return f"{S3_PUBLIC_URL.rstrip('/')}/{obj_key}"
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET_NAME, "Key": obj_key},
            ExpiresIn=S3_PRESIGNED_EXPIRY,
        )

    def upload_json(self, data: str, request_id: str, filename: str) -> str:
        """Write JSON string to a temp file, upload, return key."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            return self.upload(tmp_path, request_id, filename, "application/json")
        finally:
            _safe_remove(tmp_path)


# Global S3 instance
s3 = S3Storage()


# =============================================================================
# RVC Engine — global singleton
# =============================================================================
logger.info("Initializing RVC inference engine …")
try:
    rvc_engine = RVCInference(
        device="cuda:0",
        is_half=True,
        hubert_path=HUBERT_PATH,
        rmvpe_path=RMVPE_PATH,
    )
    logger.info("RVC engine ready")
except Exception as exc:
    logger.critical(f"RVC engine init failed: {exc}")
    raise

_current_model_hash: Optional[str] = None


# =============================================================================
# Utility helpers
# =============================================================================
def _safe_remove(path: str):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:16]


def _ext_from_url(url: str) -> str:
    """Extract file extension from a normal URL (ignoring query string)."""
    path = url.split("?")[0]
    ext = os.path.splitext(path)[-1]
    return ext if ext else ".bin"


def _ext_from_data_url(data_url: str) -> str:
    """Extract file extension from a data:audio/...;base64,... URL."""
    m = re.match(r"data:([^;,]+)", data_url)
    if not m:
        return ".bin"
    mime = m.group(1).lower()
    return MIME_TO_EXT.get(mime, ".bin")


def _content_type(ext: str) -> str:
    mapping = {v: k for k, v in MIME_TO_EXT.items()}
    return mapping.get(ext, "application/octet-stream")


# =============================================================================
# FFmpeg / FFprobe helpers
# =============================================================================
def _check_ffmpeg():
    """Raise FFmpegNotAvailableError if ffmpeg is missing."""
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        if r.returncode != 0:
            raise FFmpegNotAvailableError()
    except FileNotFoundError:
        raise FFmpegNotAvailableError()


def probe_audio(path: str) -> Optional[Dict[str, Any]]:
    """Return audio stream info via ffprobe, or None on failure."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        "-select_streams", "a:0",
        path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return None
        data = json.loads(r.stdout)
        stream = (data.get("streams") or [{}])[0]
        fmt = data.get("format", {})
        return {
            "duration": float(fmt.get("duration", stream.get("duration", 0))),
            "sample_rate": int(stream.get("sample_rate", 0)),
            "channels": int(stream.get("channels", 0)),
            "codec_name": stream.get("codec_name", ""),
            "bit_rate": fmt.get("bit_rate", stream.get("bit_rate", "")),
        }
    except Exception:
        return None


def convert_to_wav(input_path: str, output_path: str):
    """Convert any audio to mono PCM WAV (preserve original sample rate)."""
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-c:a", "pcm_s16le", "-ac", "1", "-vn",
        output_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise InvalidAudioError(f"WAV conversion failed: {r.stderr[:300]}")
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise InvalidAudioError("WAV conversion produced empty output")


def encode_audio(
    wav_path: str,
    output_path: str,
    fmt: str,
    bitrate: str,
    sample_rate: int,
):
    """Encode WAV to the requested output format via ffmpeg."""
    fmt_info = SUPPORTED_FORMATS[fmt]
    codec = fmt_info["ffmpeg_codec"]

    cmd = ["ffmpeg", "-y", "-i", wav_path]

    if fmt in LOSSLESS_FORMATS:
        # WAV / FLAC — bitrate is meaningless
        cmd += ["-c:a", codec, "-ar", str(sample_rate), "-ac", "1", "-vn", output_path]
    else:
        cmd += [
            "-c:a", codec,
            "-b:a", bitrate,
            "-ar", str(sample_rate),
            "-ac", "1",
            "-vn",
            output_path,
        ]

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        raise EncodingError(f"FFmpeg exited with code {r.returncode}", r.stderr)
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise EncodingError("Encoded output is empty or missing")


# =============================================================================
# GPU helpers
# =============================================================================
def _gpu_model() -> str:
    if not torch.cuda.is_available():
        return "N/A"
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return "unknown"


def _peak_vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    try:
        return torch.cuda.max_memory_allocated(0) / (1024 * 1024)
    except Exception:
        return 0.0


def _gpu_power_watts() -> float:
    """Snapshot current GPU power draw via nvidia-smi."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            return float(r.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return 0.0


# =============================================================================
# Download / decode helpers
# =============================================================================
def download_file(url: str, cache_dir: str, prefix: str = "") -> Tuple[str, float]:
    """
    Download a file from *url* into *cache_dir*.
    Returns (local_path, elapsed_seconds).
    Uses URL-hash caching so repeated downloads are free.
    """
    import requests
    from requests.exceptions import (
        ConnectionError as ReqConnError,
        Timeout as ReqTimeout,
        HTTPError as ReqHTTPError,
    )

    os.makedirs(cache_dir, exist_ok=True)
    ext = _ext_from_url(url)
    h = _url_hash(url)
    cached = os.path.join(cache_dir, f"{prefix}{h}{ext}")

    if os.path.exists(cached) and os.path.getsize(cached) > 0:
        logger.debug(f"Cache hit: {cached}")
        return cached, 0.0

    headers = {
        "User-Agent": "Mozilla/5.0 (RVC-Worker/2.0)",
        "Accept": "*/*",
    }

    with Timer() as t:
        try:
            resp = requests.get(url, headers=headers, stream=True,
                                timeout=DOWNLOAD_TIMEOUT, allow_redirects=True)
            resp.raise_for_status()
            size = 0
            with open(cached, "wb") as fp:
                for chunk in resp.iter_content(8192):
                    if chunk:
                        fp.write(chunk)
                        size += len(chunk)
            logger.info(f"Downloaded {size / 1024:.1f} KB → {cached} ({t.elapsed:.2f}s)")
        except ReqHTTPError as exc:
            _safe_remove(cached)
            status = exc.response.status_code if exc.response is not None else 0
            raise InputDownloadError(url, f"HTTP {status}")
        except ReqTimeout:
            _safe_remove(cached)
            raise InputDownloadError(url, f"Timeout after {DOWNLOAD_TIMEOUT}s")
        except ReqConnError as exc:
            _safe_remove(cached)
            raise InputDownloadError(url, f"Connection error: {exc}")
        except Exception as exc:
            _safe_remove(cached)
            if isinstance(exc, RVCError):
                raise
            raise InputDownloadError(url, str(exc))

    if not os.path.exists(cached) or os.path.getsize(cached) == 0:
        _safe_remove(cached)
        raise InputDownloadError(url, "Downloaded file is empty")

    return cached, t.elapsed


def decode_data_url(data_url: str, dest_dir: str, idx: int) -> str:
    """Decode a data:audio/…;base64,… string to a local file. Returns path."""
    m = re.match(r"data:([^;,]+);base64,(.+)", data_url, re.DOTALL)
    if not m:
        raise InvalidDataURLError("Cannot parse data URL (expected data:<mime>;base64,<data>)")
    mime = m.group(1)
    b64 = m.group(2)
    ext = MIME_TO_EXT.get(mime.lower(), ".bin")

    try:
        raw = base64.b64decode(b64)
    except Exception as exc:
        raise InvalidDataURLError(f"Base64 decode failed: {exc}")

    if len(raw) == 0:
        raise InvalidDataURLError("Decoded audio data is empty")

    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, f"data_{idx}{ext}")
    with open(path, "wb") as fp:
        fp.write(raw)

    logger.info(f"Decoded data URL ({mime}) → {path} ({len(raw)} bytes)")
    return path


def download_or_decode(url: str, cache_dir: str, idx: int) -> str:
    """Handle both normal URLs and data: URLs. Returns local file path."""
    if url.startswith("data:"):
        return decode_data_url(url, cache_dir, idx)
    path, _ = download_file(url, cache_dir, prefix=f"input_{idx}_")
    return path


# =============================================================================
# Input validation
# =============================================================================
def validate_input(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalise the raw job input.
    Raises appropriate RVCError subclass on invalid input.
    Returns a clean dict with all fields populated (defaults applied).
    """
    # --- required ---
    if "input_urls" not in job_input:
        raise MissingRequiredFieldError("input_urls")
    if "model_url" not in job_input:
        raise MissingRequiredFieldError("model_url")

    input_urls = job_input["input_urls"]
    if not isinstance(input_urls, list) or len(input_urls) == 0:
        raise EmptyInputURLError()

    model_url = job_input["model_url"]
    if not isinstance(model_url, str) or not model_url.strip():
        raise MissingRequiredFieldError("model_url")

    # --- optional with defaults ---
    index_url = job_input.get("index_url") or None
    if index_url and not isinstance(index_url, str):
        index_url = None

    fmt = str(job_input.get("format", DEFAULT_FORMAT)).lower()
    if fmt not in SUPPORTED_FORMATS:
        raise FormatNotSupportedError(fmt, list(SUPPORTED_FORMATS.keys()))

    bitrate = str(job_input.get("bitrate", DEFAULT_BITRATE)).lower()
    if fmt not in LOSSLESS_FORMATS and bitrate not in SUPPORTED_BITRATES:
        raise InvalidBitrateError(bitrate, SUPPORTED_BITRATES)

    sample_rate = job_input.get("sample_rate", DEFAULT_SAMPLE_RATE)
    try:
        sample_rate = int(sample_rate)
    except (TypeError, ValueError):
        raise InvalidSampleRateError(sample_rate, SUPPORTED_SAMPLE_RATES)
    if sample_rate not in SUPPORTED_SAMPLE_RATES:
        raise InvalidSampleRateError(sample_rate, SUPPORTED_SAMPLE_RATES)

    f0_up_key = job_input.get("f0_up_key", DEFAULT_F0_UP_KEY)
    try:
        f0_up_key = int(f0_up_key)
    except (TypeError, ValueError):
        raise InvalidF0UpKeyError(f0_up_key)
    if not (-24 <= f0_up_key <= 24):
        raise InvalidF0UpKeyError(f0_up_key)

    index_rate = job_input.get("index_rate", DEFAULT_INDEX_RATE)
    try:
        index_rate = float(index_rate)
    except (TypeError, ValueError):
        raise InvalidIndexRateError(index_rate)
    if not (0.0 <= index_rate <= 1.0):
        raise InvalidIndexRateError(index_rate)

    return {
        "input_urls": input_urls,
        "model_url": model_url,
        "index_url": index_url,
        "f0_up_key": f0_up_key,
        "index_rate": index_rate,
        "format": fmt,
        "bitrate": bitrate,
        "sample_rate": sample_rate,
    }


# =============================================================================
# Main handler
# =============================================================================
@torch.inference_mode()
def convert_voice(job: Dict[str, Any]) -> Dict[str, Any]:
    global _current_model_hash

    job_id = job.get("id", "unknown")
    job_input = job.get("input", {})

    # --- init bookkeeping ---
    request_id = generate_uuid7()
    started_at = datetime.now(timezone.utc).isoformat()
    _list_handler.reset()
    torch.cuda.reset_peak_memory_stats()

    perf = PerformanceInfo()
    files_debug = FilesDebugInfo()
    metadata = Metadata(
        request=job_input,
        debug=DebugInfo(
            trace=TraceInfo(request_id=request_id, started_at=started_at),
            performance=perf,
            files=files_debug,
        ),
    )

    logger.info(f"=== Job started | job_id={job_id} | request_id={request_id} ===")
    logger.info(f"Input payload: {json.dumps(job_input, default=str)[:1000]}")

    def _finalise(response: Dict[str, Any]) -> Dict[str, Any]:
        """Write metadata and return response."""
        metadata.debug.trace.ended_at = datetime.now(timezone.utc).isoformat()
        metadata.debug.resource = ResourceInfo(
            gpu_model=_gpu_model(),
            peak_vram=round(_peak_vram_mb(), 2),
            peak_power=round(_gpu_power_watts(), 2),
        )
        metadata.response = response
        metadata.debug.logs = _list_handler.snapshot()

        # Upload metadata to S3
        if s3.is_configured:
            try:
                s3.upload_json(metadata.to_json(), request_id, "metadata.json")
                logger.info("Metadata uploaded to S3")
            except Exception as exc:
                logger.warning(f"Metadata upload failed: {exc}")

        logger.info(f"=== Job finished | request_id={request_id} ===")
        return response

    # ─────────────────────────────────────────────────────────────────
    # STAGE 0: Validate input
    # ─────────────────────────────────────────────────────────────────
    try:
        params = validate_input(job_input)
    except RVCError as exc:
        logger.error(f"Validation failed: {exc}")
        return _finalise(build_fail_response(exc.code, str(exc)))

    input_urls: List[str] = params["input_urls"]
    model_url: str = params["model_url"]
    index_url: Optional[str] = params["index_url"]
    fmt: str = params["format"]
    bitrate: str = params["bitrate"]
    sample_rate: int = params["sample_rate"]
    f0_up_key: int = params["f0_up_key"]
    index_rate: float = params["index_rate"]

    logger.info(f"Validated | inputs={len(input_urls)} | format={fmt} | bitrate={bitrate} | sr={sample_rate}")

    # Ensure ffmpeg exists early
    try:
        _check_ffmpeg()
    except FFmpegNotAvailableError as exc:
        logger.critical("FFmpeg not found")
        return _finalise(build_fail_response(exc.code, str(exc)))

    # Prepare working directory for this request
    work = os.path.join(WORK_DIR, request_id)
    os.makedirs(work, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────
    # STAGE 1: Download all files (model, index, inputs)
    # ─────────────────────────────────────────────────────────────────
    # Each input item tracks its state through the pipeline
    items: List[Dict[str, Any]] = [
        {"idx": i, "key": url, "error": None} for i, url in enumerate(input_urls)
    ]

    try:
        with Timer() as dl_timer:
            # Model
            logger.info(f"Downloading model: {model_url[:120]}")
            try:
                model_path, _ = download_file(model_url, MODEL_CACHE_DIR, prefix="model_")
            except Exception as exc:
                raise ModelDownloadError(model_url, str(exc))

            # Index
            index_path: Optional[str] = None
            if index_url:
                logger.info(f"Downloading index: {index_url[:120]}")
                try:
                    index_path, _ = download_file(index_url, MODEL_CACHE_DIR, prefix="index_")
                except Exception as exc:
                    raise IndexDownloadError(index_url, str(exc))

            # Inputs
            for item in items:
                url = item["key"]
                try:
                    logger.info(f"Downloading input [{item['idx']}]: {url[:120]}")
                    item["input_path"] = download_or_decode(url, os.path.join(work, "inputs"), item["idx"])
                except Exception as exc:
                    logger.warning(f"Input [{item['idx']}] download failed: {exc}")
                    item["error"] = f"Download failed: {exc}"

        perf.download_file = round(dl_timer.elapsed, 4)
        logger.info(f"Downloads complete ({dl_timer.elapsed:.2f}s)")

    except RVCError as exc:
        logger.error(f"Critical download failure: {exc}")
        return _finalise(build_fail_response(exc.code, str(exc)))

    # Upload model & index originals to R2
    if s3.is_configured:
        try:
            ext = _ext_from_url(model_url)
            files_debug.model_key = s3.upload(model_path, request_id, f"model{ext}")
        except Exception as exc:
            logger.warning(f"Model upload to R2 failed: {exc}")
        if index_path and index_url:
            try:
                ext = _ext_from_url(index_url)
                files_debug.index_key = s3.upload(index_path, request_id, f"index{ext}")
            except Exception as exc:
                logger.warning(f"Index upload to R2 failed: {exc}")

    # ─────────────────────────────────────────────────────────────────
    # STAGE 2: Validate inputs & convert to WAV
    # ─────────────────────────────────────────────────────────────────
    with Timer() as val_timer:
        for item in items:
            if item["error"]:
                continue
            try:
                in_path = item["input_path"]
                info = probe_audio(in_path)
                if info is None:
                    raise InvalidAudioError("ffprobe could not read file")

                duration = info["duration"]
                if duration < MIN_AUDIO_DURATION:
                    raise AudioTooShortError(duration, MIN_AUDIO_DURATION)

                wav_path = os.path.join(work, f"{item['idx']}_input.wav")
                convert_to_wav(in_path, wav_path)
                item["wav_path"] = wav_path
                logger.info(f"Input [{item['idx']}] validated: {duration:.2f}s, {info['sample_rate']}Hz")

                # Upload original input to R2
                if s3.is_configured:
                    try:
                        if item["key"].startswith("data:"):
                            ext = _ext_from_data_url(item["key"])
                        else:
                            ext = _ext_from_url(item["key"])
                        item["input_original_key"] = s3.upload(
                            in_path, request_id, f"{item['idx']}_input{ext}",
                        )
                    except Exception as exc:
                        logger.warning(f"Input [{item['idx']}] original upload failed: {exc}")

            except RVCError as exc:
                logger.warning(f"Input [{item['idx']}] validation failed: {exc}")
                item["error"] = str(exc)
            except Exception as exc:
                logger.warning(f"Input [{item['idx']}] validation error: {exc}")
                item["error"] = f"Validation error: {exc}"

    perf.validate_input_file = round(val_timer.elapsed, 4)

    # Check if any inputs survived
    valid_items = [it for it in items if it.get("error") is None]
    if not valid_items:
        errors = [f"[{it['idx']}] {it['error']}" for it in items]
        logger.error(f"All inputs failed validation: {errors}")
        return _finalise(build_fail_response(
            "ALL_INPUTS_FAILED",
            f"All {len(items)} input(s) failed: {'; '.join(errors)}",
        ))

    # ─────────────────────────────────────────────────────────────────
    # STAGE 3: Load RVC model
    # ─────────────────────────────────────────────────────────────────
    try:
        with Timer() as mdl_timer:
            model_hash = _url_hash(model_url)
            if _current_model_hash == model_hash:
                logger.info(f"Model already loaded (hash={model_hash})")
            else:
                logger.info(f"Loading model (hash={model_hash}) …")
                rvc_engine.load_model(model_path)
                _current_model_hash = model_hash
                logger.info(f"Model loaded: v{rvc_engine.version}, tgt_sr={rvc_engine.tgt_sr}")
        perf.load_model = round(mdl_timer.elapsed, 4)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "cuda" in msg and "memory" in msg:
            raise CUDAOutOfMemoryError()
        logger.error(f"Model load failed: {exc}")
        return _finalise(build_fail_response("MODEL_LOAD_FAILED", f"Failed to load model: {exc}"))
    except Exception as exc:
        logger.error(f"Model load failed: {exc}")
        return _finalise(build_fail_response("MODEL_LOAD_FAILED", f"Failed to load model: {exc}"))

    # ─────────────────────────────────────────────────────────────────
    # STAGE 4: RVC inference (per input)
    # ─────────────────────────────────────────────────────────────────
    with Timer() as infer_timer:
        for item in items:
            if item.get("error"):
                continue
            try:
                logger.info(f"Running inference [{item['idx']}] | f0={F0_METHOD} | key={f0_up_key}")
                sr, audio_out = rvc_engine.infer(
                    input_path=item["wav_path"],
                    f0_up_key=f0_up_key,
                    f0_method=F0_METHOD,
                    file_index=index_path or "",
                    index_rate=index_rate,
                    filter_radius=FILTER_RADIUS,
                    resample_sr=0,
                    rms_mix_rate=RMS_MIX_RATE,
                    protect=PROTECT,
                )
                # Save output WAV
                out_wav = os.path.join(work, f"{item['idx']}_output.wav")
                sf.write(out_wav, audio_out, sr)
                item["output_wav"] = out_wav
                item["output_sr"] = sr
                logger.info(f"Inference [{item['idx']}] done: {len(audio_out)/sr:.2f}s @ {sr}Hz")

                # Upload output WAV to R2
                if s3.is_configured:
                    try:
                        item["output_original_key"] = s3.upload(
                            out_wav, request_id, f"{item['idx']}_output.wav", "audio/wav",
                        )
                    except Exception as exc:
                        logger.warning(f"Output WAV [{item['idx']}] upload failed: {exc}")

            except RuntimeError as exc:
                msg = str(exc).lower()
                if "cuda" in msg and "memory" in msg:
                    logger.error(f"CUDA OOM during inference [{item['idx']}]")
                    item["error"] = "CUDA out of memory"
                else:
                    logger.error(f"Inference [{item['idx']}] failed: {exc}")
                    item["error"] = f"Inference failed: {exc}"
            except Exception as exc:
                logger.error(f"Inference [{item['idx']}] failed: {exc}")
                item["error"] = f"Inference failed: {exc}"

    perf.convert_vocal = round(infer_timer.elapsed, 4)

    # ─────────────────────────────────────────────────────────────────
    # STAGE 5: Encode to target format
    # ─────────────────────────────────────────────────────────────────
    with Timer() as enc_timer:
        for item in items:
            if item.get("error"):
                continue
            try:
                ext = SUPPORTED_FORMATS[fmt]["ext"]
                encoded_path = os.path.join(work, f"{item['idx']}_output.{ext}")
                logger.info(f"Encoding [{item['idx']}] → {fmt} / {bitrate} / {sample_rate}Hz")
                encode_audio(item["output_wav"], encoded_path, fmt, bitrate, sample_rate)
                item["encoded_path"] = encoded_path
                logger.info(f"Encoded [{item['idx']}]: {os.path.getsize(encoded_path)} bytes")
            except Exception as exc:
                logger.error(f"Encoding [{item['idx']}] failed: {exc}")
                item["error"] = f"Encoding failed: {exc}"

    perf.encode_converted = round(enc_timer.elapsed, 4)

    # ─────────────────────────────────────────────────────────────────
    # STAGE 6: Upload encoded results & build response file entries
    # ─────────────────────────────────────────────────────────────────
    file_results: List[FileResult] = []

    with Timer() as upl_timer:
        for item in items:
            if item.get("error"):
                continue
            try:
                ext = SUPPORTED_FORMATS[fmt]["ext"]
                ct = SUPPORTED_FORMATS[fmt]["content_type"]
                r2_key = s3.upload(
                    item["encoded_path"], request_id, f"{item['idx']}_output.{ext}", ct,
                )
                item["output_target_key"] = r2_key
                presigned = s3.presigned_url(r2_key)

                # Probe the encoded file for accurate metadata
                info = probe_audio(item["encoded_path"]) or {}
                file_results.append(FileResult(
                    key=item["key"],
                    size=os.path.getsize(item["encoded_path"]),
                    format=fmt,
                    codec=SUPPORTED_FORMATS[fmt]["codec"],
                    channel=1,
                    bitrate=bitrate if fmt not in LOSSLESS_FORMATS else "N/A",
                    sample_rate=sample_rate,
                    duration_sec=round(info.get("duration", 0), 2),
                    url=presigned,
                ))

            except Exception as exc:
                logger.error(f"Upload [{item['idx']}] failed: {exc}")
                item["error"] = f"Upload failed: {exc}"

    perf.upload_results = round(upl_timer.elapsed, 4)

    # ─────────────────────────────────────────────────────────────────
    # Build debug.files.inputs
    # ─────────────────────────────────────────────────────────────────
    for item in items:
        files_debug.inputs.append(InputFileDebug(
            input_original_key=item.get("input_original_key", ""),
            output_original_key=item.get("output_original_key", ""),
            output_target_key=item.get("output_target_key", ""),
        ))

    # ─────────────────────────────────────────────────────────────────
    # Build final response
    # ─────────────────────────────────────────────────────────────────
    total = len(items)
    ok_count = len(file_results)
    fail_count = total - ok_count

    if ok_count == 0:
        errors = [f"[{it['idx']}] {it.get('error', 'unknown')}" for it in items]
        response = build_fail_response(
            "ALL_INPUTS_FAILED",
            f"All {total} input(s) failed: {'; '.join(errors)}",
        )
    elif fail_count > 0:
        response = build_fail_response(
            "PARTIAL_INPUT_FAILED",
            f"{fail_count}/{total} input(s) failed — returning {ok_count} successful result(s)",
            file_results,
        )
    else:
        response = build_success_response(file_results)

    logger.info(
        f"Result: {ok_count}/{total} succeeded | "
        f"perf: dl={perf.download_file:.2f}s val={perf.validate_input_file:.2f}s "
        f"model={perf.load_model:.2f}s infer={perf.convert_vocal:.2f}s "
        f"enc={perf.encode_converted:.2f}s upl={perf.upload_results:.2f}s"
    )

    # Cleanup working directory
    try:
        import shutil
        shutil.rmtree(work, ignore_errors=True)
    except Exception:
        pass

    return _finalise(response)


# =============================================================================
# Entry Point
# =============================================================================
runpod.serverless.start({"handler": convert_voice})
