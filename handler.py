"""
RunPod Serverless Handler for RVC (Retrieval-based Voice Conversion) inference.
Fixed settings: f0_method=rmvpe, output=opus/ogg/128kbps
Storage: boto3 S3-compatible storage
"""

import os
import sys
import base64
import hashlib
import time
import traceback
import subprocess
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

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
from runpod.serverless.utils.rp_validator import validate

from schemas import (
    INPUT_SCHEMA,
    InferenceMetadata,
    AudioInfo,
    ModelInfo,
    TimingMetrics,
    ErrorInfo,
    create_success_output,
    create_error_output,
    # Fixed constants
    F0_METHOD,
    OUTPUT_FORMAT,
    OUTPUT_CODEC,
    OUTPUT_BITRATE,
    OUTPUT_SAMPLE_RATE,
    DEFAULT_INDEX_RATE,
)
from exceptions import (
    RVCWorkerException,
    # Validation
    ValidationException,
    MissingRequiredFieldException,
    InvalidFieldTypeException,
    FieldConstraintException,
    # Download
    DownloadException,
    URLNotReachableException,
    URLNotFoundException,
    URLForbiddenException,
    DownloadTimeoutException,
    DownloadedFileCorruptException,
    # Audio
    AudioException,
    InvalidAudioFormatException,
    AudioCorruptedException,
    AudioTooShortException,
    AudioLoadException,
    # Model
    ModelException,
    InvalidModelFileException,
    ModelCorruptedException,
    ModelLoadException,
    InvalidIndexFileException,
    # Inference
    InferenceException,
    CUDAOutOfMemoryException,
    PitchExtractionException,
    FeatureExtractionException,
    VoiceConversionException,
    # Encoding
    EncodingException,
    FFmpegNotAvailableException,
    FFmpegEncodingException,
    OutputWriteException,
    # Upload (S3)
    UploadException,
    S3NotConfiguredException,
    S3CredentialsException,
    S3BucketException,
    S3UploadException,
    # Helper
    classify_exception,
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
OUTPUT_DIR = "/tmp/output"

# Fixed RVC parameters
FILTER_RADIUS = 3
RMS_MIX_RATE = 0.25
PROTECT = 0.33

# Limits
MIN_AUDIO_DURATION = 0.5  # seconds
DOWNLOAD_TIMEOUT = 120  # seconds

# S3 Configuration
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
S3_REGION = os.environ.get("S3_REGION", "auto")
S3_PUBLIC_URL = os.environ.get("S3_PUBLIC_URL")  # Optional: custom public URL prefix


# =============================================================================
# S3 Client
# =============================================================================
class S3Storage:
    """
    S3-compatible storage client using boto3.
    Supports AWS S3, Cloudflare R2, MinIO, etc.
    """
    
    def __init__(self):
        self._client = None
        self._configured = False
        self._check_configuration()
    
    def _check_configuration(self):
        """Check if S3 is properly configured."""
        required_vars = {
            "S3_ENDPOINT_URL": S3_ENDPOINT_URL,
            "S3_ACCESS_KEY": S3_ACCESS_KEY,
            "S3_SECRET_KEY": S3_SECRET_KEY,
            "S3_BUCKET_NAME": S3_BUCKET_NAME,
        }
        
        missing = [name for name, value in required_vars.items() if not value]
        
        if missing:
            print(f"[S3Storage] Not configured. Missing: {missing}")
            self._configured = False
        else:
            self._configured = True
            print(f"[S3Storage] Configured for bucket: {S3_BUCKET_NAME}")
    
    @property
    def is_configured(self) -> bool:
        return self._configured
    
    @property
    def client(self):
        """Lazy initialization of S3 client."""
        if self._client is None:
            if not self._configured:
                missing = []
                if not S3_ENDPOINT_URL:
                    missing.append("S3_ENDPOINT_URL")
                if not S3_ACCESS_KEY:
                    missing.append("S3_ACCESS_KEY")
                if not S3_SECRET_KEY:
                    missing.append("S3_SECRET_KEY")
                if not S3_BUCKET_NAME:
                    missing.append("S3_BUCKET_NAME")
                raise S3NotConfiguredException(missing)
            
            try:
                self._client = boto3.client(
                    "s3",
                    endpoint_url=S3_ENDPOINT_URL,
                    aws_access_key_id=S3_ACCESS_KEY,
                    aws_secret_access_key=S3_SECRET_KEY,
                    region_name=S3_REGION,
                )
            except Exception as e:
                raise S3CredentialsException(str(e))
        
        return self._client
    
    def upload_file(
        self,
        file_path: str,
        key: str,
        content_type: str = "audio/ogg",
        presigned_expiry: int = 60*60*24,  # 1 day in seconds
    ) -> str:
        """
        Upload a file to S3 and return a downloadable URL.
        
        Args:
            file_path: Local file path to upload
            key: S3 object key (path in bucket)
            content_type: MIME type of the file
            presigned_expiry: Expiry time for presigned URL in seconds (default: 7 days)
        
        Returns:
            Downloadable URL (presigned URL or custom public URL)
        
        Raises:
            S3CredentialsException, S3BucketException, S3UploadException
        """
        try:
            self.client.upload_file(
                Filename=file_path,
                Bucket=S3_BUCKET_NAME,
                Key=key,
                ExtraArgs={
                    "ContentType": content_type,
                },
            )
            
            # Generate downloadable URL
            if S3_PUBLIC_URL:
                # Use custom public URL prefix (CDN, custom domain, etc.)
                url = f"{S3_PUBLIC_URL.rstrip('/')}/{key}"
            else:
                # Generate presigned URL for direct download
                url = self.client.generate_presigned_url(
                    "get_object",
                    Params={
                        "Bucket": S3_BUCKET_NAME,
                        "Key": key,
                    },
                    ExpiresIn=presigned_expiry,
                )
            
            print(f"[S3Storage] Uploaded: {key}")
            return url
            
        except NoCredentialsError:
            raise S3CredentialsException("No credentials provided")
        except PartialCredentialsError as e:
            raise S3CredentialsException(f"Incomplete credentials: {e}")
        except EndpointConnectionError as e:
            raise S3BucketException(S3_BUCKET_NAME, f"Cannot connect to endpoint: {e}")
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_msg = e.response.get("Error", {}).get("Message", str(e))
            
            if error_code in ("AccessDenied", "InvalidAccessKeyId", "SignatureDoesNotMatch"):
                raise S3CredentialsException(f"{error_code}: {error_msg}")
            elif error_code in ("NoSuchBucket", "InvalidBucketName"):
                raise S3BucketException(S3_BUCKET_NAME, f"{error_code}: {error_msg}")
            else:
                raise S3UploadException(key, f"{error_code}: {error_msg}")
        except Exception as e:
            raise S3UploadException(key, str(e))
    
    def generate_key(self, request_id: str, filename: str) -> str:
        """
        Generate S3 key for the new structure.
        Format: worker-rvc-nano/request-{request_id}/{filename}
        """
        return f"worker-rvc-nano/request-{request_id}/{filename}"
    
    def get_request_folder_url(self, request_id: str) -> str:
        """Get the public URL for a request's folder."""
        key_prefix = f"worker-rvc-nano/request-{request_id}/"
        if S3_PUBLIC_URL:
            return f"{S3_PUBLIC_URL.rstrip('/')}/{key_prefix}"
        else:
            return f"{S3_ENDPOINT_URL.rstrip('/')}/{S3_BUCKET_NAME}/{key_prefix}"
    
    def upload_input_audio(self, file_path: str, request_id: str) -> str:
        """Upload input audio as input.opus.ogg."""
        key = self.generate_key(request_id, "input.opus.ogg")
        return self.upload_file(file_path, key, "audio/ogg")
    
    def upload_output_audio(self, file_path: str, request_id: str) -> str:
        """Upload output audio as output.opus.ogg."""
        key = self.generate_key(request_id, "output.opus.ogg")
        return self.upload_file(file_path, key, "audio/ogg")
    
    def upload_metadata(self, metadata_json: str, request_id: str) -> str:
        """Upload metadata as metadata.json."""
        import tempfile
        
        key = self.generate_key(request_id, "metadata.json")
        
        # Write metadata to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(metadata_json)
            temp_path = f.name
        
        try:
            url = self.upload_file(temp_path, key, "application/json")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        return url


# Initialize global S3 storage
S3 = S3Storage()


# =============================================================================
# Timing Context Manager
# =============================================================================
class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self):
        self.elapsed: float = 0.0
    
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start


# =============================================================================
# Model Handler
# =============================================================================
class ModelHandler:
    """
    Manages RVC model loading and caching with proper exception handling.
    """
    
    def __init__(self):
        self.rvc: Optional[RVCInference] = None
        self.current_model_hash: Optional[str] = None
        self.current_model_info: Optional[ModelInfo] = None
        self._init_rvc()
    
    def _init_rvc(self):
        """Initialize RVC inference engine with base models."""
        print("[ModelHandler] Initializing RVC inference engine...")
        try:
            self.rvc = RVCInference(
                device="cuda:0",
                is_half=True,
                hubert_path=HUBERT_PATH,
                rmvpe_path=RMVPE_PATH,
            )
            print("[ModelHandler] RVC engine initialized successfully")
        except FileNotFoundError as e:
            raise ModelCorruptedException(str(e), "Base model files not found")
        except Exception as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                raise CUDAOutOfMemoryException()
            raise ModelLoadException("base_models", str(e))
    
    def _get_file_hash(self, url: str) -> str:
        """Generate a hash for the URL to use as cache key."""
        return hashlib.md5(url.encode()).hexdigest()[:16]
    
    def _download_file(
        self, 
        url: str, 
        cache_dir: str, 
        prefix: str = "",
        resource_type: str = "file",
    ) -> Tuple[str, bool, float]:
        """
        Download a file from URL with caching and proper error handling.
        Uses requests library for better compatibility (User-Agent, redirects, etc.)
        Returns: (file_path, cache_hit, download_time)
        Raises: DownloadException subclasses
        """
        import requests
        from requests.exceptions import (
            ConnectionError as RequestsConnectionError,
            Timeout as RequestsTimeout,
            HTTPError as RequestsHTTPError,
        )
        
        os.makedirs(cache_dir, exist_ok=True)
        
        # Extract extension from URL path (before query string)
        url_path = url.split("?")[0]
        extension = os.path.splitext(url_path)[-1] or ".bin"
        
        url_hash = self._get_file_hash(url)
        cached_path = os.path.join(cache_dir, f"{prefix}{url_hash}{extension}")
        
        if os.path.exists(cached_path):
            print(f"[ModelHandler] Cache hit: {cached_path}")
            return cached_path, True, 0.0
        
        print(f"[ModelHandler] Downloading: {url}")
        
        # Headers for better compatibility with various hosts
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate",
        }
        
        try:
            with Timer() as t:
                response = requests.get(
                    url,
                    headers=headers,
                    stream=True,
                    timeout=DOWNLOAD_TIMEOUT,
                    allow_redirects=True,
                )
                response.raise_for_status()
                
                # Get total size for progress logging
                total_size = int(response.headers.get("content-length", 0))
                downloaded_size = 0
                
                with open(cached_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                
                # Log download info
                if total_size > 0:
                    print(f"[ModelHandler] Downloaded {downloaded_size / 1024 / 1024:.2f}MB")
            
            if not os.path.exists(cached_path):
                raise DownloadedFileCorruptException(url, resource_type, "File not created")
            
            if os.path.getsize(cached_path) == 0:
                os.remove(cached_path)
                raise DownloadedFileCorruptException(url, resource_type, "Downloaded file is empty")
            
            print(f"[ModelHandler] Downloaded: {cached_path} ({t.elapsed:.2f}s)")
            return cached_path, False, t.elapsed
            
        except RequestsHTTPError as e:
            if os.path.exists(cached_path):
                os.remove(cached_path)
            status_code = e.response.status_code if e.response is not None else 0
            if status_code == 404:
                raise URLNotFoundException(url, resource_type)
            elif status_code == 403:
                raise URLForbiddenException(url)
            else:
                raise URLNotReachableException(url, f"HTTP {status_code}: {str(e)}")
        
        except RequestsTimeout:
            if os.path.exists(cached_path):
                os.remove(cached_path)
            raise DownloadTimeoutException(url, DOWNLOAD_TIMEOUT)
        
        except RequestsConnectionError as e:
            if os.path.exists(cached_path):
                os.remove(cached_path)
            raise URLNotReachableException(url, f"Connection error: {str(e)}")
        
        except Exception as e:
            if os.path.exists(cached_path):
                os.remove(cached_path)
            if isinstance(e, DownloadException):
                raise
            raise URLNotReachableException(url, str(e))
    
    def load_model(self, model_url: str, metadata: InferenceMetadata) -> ModelInfo:
        """Load RVC model from URL with caching and metadata tracking."""
        model_hash = self._get_file_hash(model_url)
        cache_hit = self.current_model_hash == model_hash
        
        model_info = ModelInfo(url=model_url, file_hash=model_hash, cached=cache_hit)
        
        if cache_hit:
            print(f"[ModelHandler] Model already loaded: {model_hash}")
            if self.current_model_info:
                model_info.version = self.current_model_info.version
                model_info.has_f0 = self.current_model_info.has_f0
                model_info.target_sample_rate = self.current_model_info.target_sample_rate
            metadata.resources.model_cache_hit = True
            return model_info
        
        model_path, _, download_time = self._download_file(
            model_url, MODEL_CACHE_DIR, prefix="model_", resource_type="model"
        )
        metadata.timing.download_model_seconds = download_time
        
        try:
            with Timer() as t:
                self.rvc.load_model(model_path)
            metadata.timing.load_model_seconds = t.elapsed
            
        except FileNotFoundError:
            raise ModelCorruptedException(model_path, "Model file not found after download")
        except KeyError as e:
            raise InvalidModelFileException(model_path, f"Missing required key: {e}")
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "size mismatch" in error_msg:
                raise InvalidModelFileException(model_path, "Model weight size mismatch")
            elif "cuda" in error_msg or "out of memory" in error_msg:
                raise CUDAOutOfMemoryException()
            raise ModelLoadException(model_path, str(e))
        except Exception as e:
            raise ModelLoadException(model_path, str(e))
        
        model_info.version = self.rvc.version
        model_info.has_f0 = bool(self.rvc.if_f0)
        model_info.target_sample_rate = self.rvc.tgt_sr
        
        self.current_model_hash = model_hash
        self.current_model_info = model_info
        
        print(f"[ModelHandler] Model loaded: {model_hash} (v{model_info.version})")
        return model_info
    
    def download_vocal(self, vocal_url: str, metadata: InferenceMetadata) -> Tuple[str, AudioInfo]:
        """Download input vocal audio from URL."""
        audio_path, cache_hit, download_time = self._download_file(
            vocal_url, AUDIO_CACHE_DIR, prefix="vocal_", resource_type="vocal audio"
        )
        
        metadata.timing.download_vocal_seconds = download_time
        metadata.resources.vocal_cache_hit = cache_hit
        
        audio_info = AudioInfo(url=vocal_url, file_path=audio_path)
        
        try:
            info = sf.info(audio_path)
            audio_info.duration_seconds = info.duration
            audio_info.sample_rate = info.samplerate
            audio_info.channels = info.channels
            audio_info.format = info.format
            audio_info.file_size_bytes = os.path.getsize(audio_path)
            
        except sf.LibsndfileError as e:
            raise AudioCorruptedException(audio_path, f"Cannot read audio: {e}")
        except Exception as e:
            raise InvalidAudioFormatException(audio_path, str(e))
        
        if audio_info.duration_seconds < MIN_AUDIO_DURATION:
            raise AudioTooShortException(audio_info.duration_seconds, MIN_AUDIO_DURATION)
        
        return audio_path, audio_info
    
    def download_index(self, index_url: str, metadata: InferenceMetadata) -> str:
        """Download index file from URL."""
        index_path, cache_hit, download_time = self._download_file(
            index_url, MODEL_CACHE_DIR, prefix="index_", resource_type="index"
        )
        
        metadata.timing.download_index_seconds = download_time
        metadata.resources.index_cache_hit = cache_hit
        
        if not index_path.endswith(".index") and os.path.getsize(index_path) < 1000:
            raise InvalidIndexFileException(index_path, "File appears to be invalid or too small")
        
        return index_path


# Initialize global model handler
MODELS = ModelHandler()


# =============================================================================
# Audio Processing
# =============================================================================
def encode_to_opus_ogg(
    audio_data: np.ndarray,
    input_sample_rate: int,
    output_path: str,
    bitrate: str = OUTPUT_BITRATE,
    sample_rate: int = OUTPUT_SAMPLE_RATE,
) -> None:
    """
    Encode audio data to Opus/OGG format using ffmpeg.
    Raises: FFmpegNotAvailableException, FFmpegEncodingException
    """
    import tempfile
    
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode != 0:
            raise FFmpegNotAvailableException()
    except FileNotFoundError:
        raise FFmpegNotAvailableException()
    except Exception:
        raise FFmpegNotAvailableException()
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            sf.write(tmp_path, audio_data, input_sample_rate)
    except Exception as e:
        raise OutputWriteException(tmp_path, f"Failed to write temp WAV: {e}")
    
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", tmp_path,
            "-c:a", "libopus",
            "-b:a", bitrate,
            "-ar", str(sample_rate),
            "-ac", "1",
            "-vn",
            output_path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        
        if result.returncode != 0:
            raise FFmpegEncodingException(
                f"FFmpeg exited with code {result.returncode}",
                stderr=result.stderr
            )
        
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise FFmpegEncodingException("Output file not created or empty")
            
    except subprocess.TimeoutExpired:
        raise FFmpegEncodingException("FFmpeg encoding timed out")
    except FFmpegEncodingException:
        raise
    except Exception as e:
        raise FFmpegEncodingException(str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def analyze_audio(audio_data: np.ndarray, sample_rate: int) -> AudioInfo:
    """Analyze audio and create AudioInfo."""
    duration = len(audio_data) / sample_rate
    peak = float(np.abs(audio_data).max())
    rms = float(np.sqrt(np.mean(audio_data.astype(np.float64) ** 2)))
    
    return AudioInfo(
        duration_seconds=duration,
        sample_rate=sample_rate,
        channels=1,
        peak_amplitude=peak,
        rms_level=rms,
    )


def save_and_upload_audio(
    audio_data: np.ndarray,
    sample_rate: int,
    job_id: str,
    metadata: InferenceMetadata,
) -> Tuple[str, AudioInfo]:
    """
    Encode audio to Opus/OGG and upload to S3 or return as base64.
    S3 structure: worker-rvc-nano/request-{request_id}/output.opus.ogg
    Raises: EncodingException, UploadException
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_filename = f"output_{metadata.request_id}.{OUTPUT_FORMAT}"
    audio_path = os.path.join(OUTPUT_DIR, output_filename)
    
    with Timer() as encode_timer:
        encode_to_opus_ogg(audio_data, sample_rate, audio_path)
    metadata.timing.encoding_seconds = encode_timer.elapsed
    
    file_size = os.path.getsize(audio_path)
    
    # Upload to S3 or encode to base64
    with Timer() as upload_timer:
        if S3.is_configured:
            # Upload to S3 as output.opus.ogg
            audio_url = S3.upload_output_audio(audio_path, metadata.request_id)
        else:
            # Fallback to base64
            print("[save_and_upload_audio] S3 not configured, returning base64")
            with open(audio_path, "rb") as audio_file:
                audio_b64 = base64.b64encode(audio_file.read()).decode("utf-8")
                audio_url = f"data:audio/{OUTPUT_FORMAT};base64,{audio_b64}"
    
    metadata.timing.upload_seconds = upload_timer.elapsed
    
    # Cleanup local file
    if os.path.exists(audio_path):
        os.remove(audio_path)
    
    # Create output audio info
    audio_info = analyze_audio(audio_data, sample_rate)
    audio_info.format = OUTPUT_FORMAT
    audio_info.file_size_bytes = file_size
    audio_info.sample_rate = OUTPUT_SAMPLE_RATE
    
    return audio_url, audio_info


def encode_and_upload_input_audio(
    input_audio_path: str,
    metadata: InferenceMetadata,
) -> Optional[str]:
    """
    Encode input audio to Opus/OGG and upload to S3.
    S3 structure: worker-rvc-nano/request-{request_id}/input.opus.ogg
    Returns: S3 URL or None if S3 is not configured
    """
    if not S3.is_configured:
        return None
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    encoded_input_path = os.path.join(OUTPUT_DIR, f"input_{metadata.request_id}.{OUTPUT_FORMAT}")
    
    try:
        # Load input audio
        import librosa
        audio_data, sr = librosa.load(input_audio_path, sr=None, mono=True)
        
        # Encode to opus/ogg
        encode_to_opus_ogg(audio_data, sr, encoded_input_path)
        
        # Upload to S3
        input_url = S3.upload_input_audio(encoded_input_path, metadata.request_id)
        
        return input_url
    except Exception as e:
        print(f"[encode_and_upload_input_audio] Warning: Failed to upload input: {e}")
        return None
    finally:
        if os.path.exists(encoded_input_path):
            os.remove(encoded_input_path)


def upload_metadata_to_s3(metadata: InferenceMetadata) -> Optional[str]:
    """
    Upload metadata.json to S3.
    S3 structure: worker-rvc-nano/request-{request_id}/metadata.json
    Returns: S3 URL or None if S3 is not configured
    """
    if not S3.is_configured:
        return None
    
    try:
        metadata_json = metadata.to_json()
        metadata_url = S3.upload_metadata(metadata_json, metadata.request_id)
        return metadata_url
    except Exception as e:
        print(f"[upload_metadata_to_s3] Warning: Failed to upload metadata: {e}")
        return None


def get_gpu_memory_usage() -> Tuple[Optional[float], Optional[float]]:
    """Get current and peak GPU memory usage in MB."""
    if not torch.cuda.is_available():
        return None, None
    try:
        current = torch.cuda.memory_allocated() / 1024 / 1024
        peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        return current, peak
    except Exception:
        return None, None


# =============================================================================
# Main Handler
# =============================================================================
@torch.inference_mode()
def convert_voice(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler for RVC voice conversion.
    Fixed: f0_method=rmvpe, output=opus/ogg/128kbps
    """
    import json
    import pprint
    
    job_id = job.get("id", "unknown")
    job_input = job.get("input", {})
    
    # Initialize metadata with UUID7
    metadata = InferenceMetadata.create(
        job_id=job_id,
        tag=job_input.get("tag"),
    )
    metadata.started_at = datetime.utcnow().isoformat() + "Z"
    
    total_timer = Timer()
    total_timer.__enter__()
    
    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------
    print(f"[convert_voice] Job started | request_id={metadata.request_id} | tag={metadata.tag}")
    print("[convert_voice] Input:")
    try:
        print(json.dumps(job_input, indent=2, default=str), flush=True)
    except Exception:
        pprint.pprint(job_input, depth=4)
    
    # -------------------------------------------------------------------------
    # Input Validation
    # -------------------------------------------------------------------------
    try:
        validated_input = validate(job_input, INPUT_SCHEMA)
        
        if "errors" in validated_input:
            errors = validated_input["errors"]
            raise ValidationException(str(errors))
        
        job_input = validated_input["validated_input"]
        
    except ValidationException as e:
        print(f"[convert_voice] Validation error: {e}", flush=True)
        metadata.error = ErrorInfo(
            error_type=type(e).__name__,
            error_message=e.message,
            error_stage=e.error_stage,
        )
        metadata.success = False
        return create_error_output(e.message, metadata)
    except Exception as e:
        print(f"[convert_voice] Validation exception: {e}", flush=True)
        metadata.error = ErrorInfo(
            error_type="ValidationError",
            error_message=str(e),
            error_stage="input_validation",
            stack_trace=traceback.format_exc(),
        )
        metadata.success = False
        return create_error_output(str(e), metadata)
    
    # Store input params
    metadata.input_params = {
        "f0_up_key": job_input["f0_up_key"],
        "index_rate": job_input["index_rate"],
    }
    metadata.index_url = job_input.get("index_url")
    
    # -------------------------------------------------------------------------
    # Process Request with Structured Exception Handling
    # -------------------------------------------------------------------------
    try:
        # Stage 1: Load model
        model_info = MODELS.load_model(job_input["model_url"], metadata)
        metadata.model_info = model_info
        
        # Stage 2: Download vocal
        vocal_path, input_audio_info = MODELS.download_vocal(job_input["vocal_url"], metadata)
        metadata.input_audio = input_audio_info
        
        # Stage 3: Download index if provided
        index_path = ""
        if job_input.get("index_url"):
            index_path = MODELS.download_index(job_input["index_url"], metadata)
        
        # Stage 4: Run RVC inference
        print(f"[convert_voice] Running inference | f0_method={F0_METHOD}")
        
        try:
            with Timer() as inference_timer:
                sample_rate, audio_output = MODELS.rvc.infer(
                    input_path=vocal_path,
                    f0_up_key=job_input["f0_up_key"],
                    f0_method=F0_METHOD,
                    file_index=index_path,
                    index_rate=job_input["index_rate"],
                    filter_radius=FILTER_RADIUS,
                    resample_sr=0,
                    rms_mix_rate=RMS_MIX_RATE,
                    protect=PROTECT,
                )
            metadata.timing.inference_seconds = inference_timer.elapsed
            
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "cuda" in error_msg and "memory" in error_msg:
                raise CUDAOutOfMemoryException()
            elif "f0" in error_msg or "pitch" in error_msg:
                raise PitchExtractionException(F0_METHOD, str(e))
            elif "hubert" in error_msg or "feature" in error_msg:
                raise FeatureExtractionException(str(e))
            else:
                raise VoiceConversionException(str(e))
        except Exception as e:
            if isinstance(e, InferenceException):
                raise
            raise VoiceConversionException(str(e))
        
        print(f"[convert_voice] Inference done | sr={sample_rate} | duration={len(audio_output)/sample_rate:.2f}s")
        
        # GPU metrics
        current_mem, peak_mem = get_gpu_memory_usage()
        metadata.resources.gpu_memory_used_mb = current_mem
        metadata.resources.gpu_memory_peak_mb = peak_mem
        
        # Stage 5: Encode and upload output
        audio_url, output_audio_info = save_and_upload_audio(
            audio_output, sample_rate, job_id, metadata
        )
        metadata.output_audio = output_audio_info
        
        # Stage 6: Upload input audio to S3 (optional, non-blocking)
        input_s3_url = encode_and_upload_input_audio(vocal_path, metadata)
        if input_s3_url:
            metadata.input_audio.url = input_s3_url
            print(f"[convert_voice] Input audio uploaded: {input_s3_url}")
        
        # Finalize timing
        total_timer.__exit__(None, None, None)
        metadata.timing.total_seconds = total_timer.elapsed
        metadata.completed_at = datetime.utcnow().isoformat() + "Z"
        metadata.success = True
        
        # Stage 7: Upload metadata.json to S3 (optional, non-blocking)
        metadata_url = upload_metadata_to_s3(metadata)
        if metadata_url:
            print(f"[convert_voice] Metadata uploaded: {metadata_url}")
        
        print(f"[convert_voice] Completed | request_id={metadata.request_id} | "
              f"total={metadata.timing.total_seconds:.2f}s | "
              f"inference={metadata.timing.inference_seconds:.2f}s")
        
        # Log full metadata for analytics
        print(f"[METADATA] {metadata.to_json()}")
        
        return create_success_output(audio_url, metadata, include_timing=True)
    
    # -------------------------------------------------------------------------
    # Structured Exception Handling
    # -------------------------------------------------------------------------
    except RVCWorkerException as e:
        print(f"[ERROR] {type(e).__name__}: {e}", flush=True)
        metadata.error = ErrorInfo(
            error_type=type(e).__name__,
            error_message=e.message,
            error_stage=e.error_stage,
            stack_trace=traceback.format_exc() if not e.recoverable else None,
        )
        metadata.success = False
        
        total_timer.__exit__(None, None, None)
        metadata.timing.total_seconds = total_timer.elapsed
        metadata.completed_at = datetime.utcnow().isoformat() + "Z"
        print(f"[METADATA] {metadata.to_json()}")
        
        return create_error_output(
            str(e),
            metadata,
            refresh_worker=not e.recoverable
        )
    
    except Exception as e:
        classified = classify_exception(e)
        print(f"[ERROR] Unexpected {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        
        metadata.error = ErrorInfo(
            error_type=type(e).__name__,
            error_message=str(e),
            error_stage=classified.error_stage,
            stack_trace=traceback.format_exc(),
        )
        metadata.success = False
        
        total_timer.__exit__(None, None, None)
        metadata.timing.total_seconds = total_timer.elapsed
        metadata.completed_at = datetime.utcnow().isoformat() + "Z"
        print(f"[METADATA] {metadata.to_json()}")
        
        return create_error_output(
            str(e),
            metadata,
            refresh_worker=not classified.recoverable
        )


# =============================================================================
# Entry Point
# =============================================================================
runpod.serverless.start({"handler": convert_voice})
