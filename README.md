# RVC-Nano | RunPod Serverless Worker

Run [RVC (Retrieval-based Voice Conversion)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) as a serverless endpoint for voice conversion.

---

## 🚀 Features

- **Serverless**: Runs on RunPod's serverless infrastructure
- **GPU Accelerated**: CUDA 11.8 optimized inference
- **Fixed Quality**: Opus/OGG 128kbps output for optimal size/quality
- **Smart Caching**: Model and audio file caching for faster subsequent requests
- **UUID7 Tracking**: Time-ordered unique request IDs for easy tracing

---

## 📦 Quick Start

### Deploy to RunPod

```bash
# Build Docker image
docker build -t rvc-nano-worker .

# Push to your registry
docker tag rvc-nano-worker your-registry/rvc-nano-worker:latest
docker push your-registry/rvc-nano-worker:latest
```

Then create a Serverless Endpoint on [RunPod](https://www.runpod.io/) using your image.

---

## 📝 API Reference

### Input Parameters

| Parameter    | Type    | Required | Default | Description                                      |
| :----------- | :------ | :------: | :------ | :----------------------------------------------- |
| `vocal_url`  | `str`   |    ✅    | -       | URL of the input vocal audio file                |
| `model_url`  | `str`   |    ✅    | -       | URL of the RVC model (.pth) file                 |
| `index_url`  | `str`   |    ❌    | `null`  | URL of the index (.index) file                   |
| `f0_up_key`  | `int`   |    ❌    | `0`     | Pitch shift in semitones (-24 to +24)            |
| `index_rate` | `float` |    ❌    | `0.75`  | Feature index mix ratio (0.0 - 1.0)              |
| `tag`        | `str`   |    ❌    | `null`  | Client-provided tag for tracking                 |

### Fixed Settings (Not Configurable)

| Setting         | Value          |
| :-------------- | :------------- |
| `f0_method`     | `rmvpe`        |
| Output Format   | `opus/ogg`     |
| Output Bitrate  | `128kbps`      |
| Output Sample Rate | `48000 Hz`  |

### Example Request

```json
{
  "input": {
    "vocal_url": "https://storage.example.com/vocal.wav",
    "model_url": "https://storage.example.com/model.pth",
    "index_url": "https://storage.example.com/model.index",
    "f0_up_key": 0,
    "index_rate": 0.75,
    "tag": "user_123_song_456"
  }
}
```

### Success Response

```json
{
  "result": "success",
  "request_id": "0195db8a-7c3e-7abc-8def-1234567890ab",
  "tag": "user_123_song_456",
  "audio_url": "data:audio/ogg;base64,T2dnUw...",
  "sample_rate": 48000,
  "duration_seconds": 125.4,
  "file_size_bytes": 256000,
  "format": "ogg",
  "processing_time_seconds": 8.5,
  "model_version": "v2",
  "timing_details": {
    "download_vocal": 1.2,
    "download_model": 0.0,
    "download_index": 0.3,
    "load_model": 0.0,
    "pitch_extraction": 2.1,
    "inference": 4.2,
    "encoding": 0.5,
    "upload": 0.2
  }
}
```

### Error Response

```json
{
  "result": "failure",
  "message": "Model file not found at URL",
  "request_id": "0195db8a-7c3e-7abc-8def-1234567890ab",
  "tag": "user_123_song_456",
  "refresh_worker": true
}
```

---

## 🔧 Request ID (UUID7)

Each request is assigned a **UUID7** (`request_id`) for tracking:

- **Time-ordered**: First 48 bits are millisecond timestamp
- **Sortable**: Requests sort chronologically by ID
- **Example**: `0195db8a-7c3e-7abc-8def-1234567890ab`

Use the `tag` field to add your own tracking identifier (user ID, session ID, etc.).

---

## 📂 Project Structure

```
rvc-nano/
├── Dockerfile           # Docker build (CUDA 11.8, ffmpeg, uv)
├── handler.py           # RunPod serverless handler
├── schemas.py           # Input/Output/Metadata schemas
├── download_models.py   # HuggingFace model downloader
├── requirements.txt     # Python dependencies
├── test_input.json      # Test input example
├── main.py              # Local usage example
└── src/                 # Core RVC inference module
    ├── rvc.py           # RVCInference class
    ├── config.py        # Configuration
    ├── lib/             # Core libraries
    │   ├── audio.py     # Audio processing
    │   ├── rmvpe.py     # RMVPE pitch extractor
    │   └── infer_pack/  # Neural network models
    └── modules/         # Inference pipeline
        ├── pipeline.py  # Main pipeline
        └── utils.py     # Utilities
```

---

## 🛠 Local Development

### Prerequisites

- Python 3.10+
- CUDA 11.8 compatible GPU
- FFmpeg with libopus support

### Setup

```bash
# Clone repository
git clone https://github.com/your-repo/rvc-nano.git
cd rvc-nano

# Install dependencies
pip install -r requirements.txt

# Download base models
python download_models.py
```

### Usage Example

```python
from src.rvc import RVCInference
import soundfile as sf

# Initialize
rvc = RVCInference(
    device="cuda:0",
    is_half=True,
    hubert_path="assets/hubert/hubert_base.pt",
    rmvpe_path="assets/rmvpe/rmvpe.pt"
)

# Load model
rvc.load_model("path/to/model.pth")

# Convert voice
sr, audio = rvc.infer(
    "input.wav",
    f0_up_key=0,        # Pitch shift
    f0_method="rmvpe",  # Pitch extraction
    index_rate=0.75     # Index mix ratio
)

# Save output
sf.write("output.wav", audio, sr)
```

---

## 🌐 Environment Variables

### S3 Storage (Optional)

Configure S3-compatible storage for output file uploads. Supports AWS S3, Cloudflare R2, MinIO, etc.

| Variable          | Required | Description                                           |
| :---------------- | :------: | :---------------------------------------------------- |
| `S3_ENDPOINT_URL` |    ✅    | S3 endpoint URL (e.g., `https://s3.amazonaws.com`)    |
| `S3_ACCESS_KEY`   |    ✅    | AWS access key ID                                     |
| `S3_SECRET_KEY`   |    ✅    | AWS secret access key                                 |
| `S3_BUCKET_NAME`  |    ✅    | Bucket name for uploads                               |
| `S3_REGION`       |    ❌    | AWS region (default: `auto`)                          |
| `S3_PUBLIC_URL`   |    ❌    | Custom public URL prefix for generated URLs           |

### Examples

**AWS S3:**
```bash
S3_ENDPOINT_URL=https://s3.us-east-1.amazonaws.com
S3_ACCESS_KEY=AKIAIOSFODNN7EXAMPLE
S3_SECRET_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
S3_BUCKET_NAME=my-rvc-outputs
S3_REGION=us-east-1
```

**Cloudflare R2:**
```bash
S3_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
S3_ACCESS_KEY=<r2-access-key>
S3_SECRET_KEY=<r2-secret-key>
S3_BUCKET_NAME=rvc-outputs
S3_PUBLIC_URL=https://cdn.example.com
```

When S3 is configured, files are uploaded with the following structure:
```
{bucket}/
└── worker-rvc-nano/
    └── request-{request_id}/
        ├── input.opus.ogg     # Input audio (encoded to opus)
        ├── output.opus.ogg    # Converted output audio
        └── metadata.json      # Processing metadata
```

**Example:**
```
my-bucket/
└── worker-rvc-nano/
    └── request-0195db8a-7c3e-7abc-8def-1234567890ab/
        ├── input.opus.ogg
        ├── output.opus.ogg
        └── metadata.json
```

If S3 is not configured, output files are returned as base64-encoded data URLs.


---

## � Internal Metadata (Logged)

Each request logs comprehensive metadata for analytics:

```
request_id          # UUID7
tag                 # Client tag
job_id              # RunPod job ID

# Timestamps
received_at, started_at, completed_at

# Input Analysis
input_audio         # duration, sample_rate, channels, format, file_size

# Model Info
model_info          # url, hash, version (v1/v2), cached

# Timing Metrics
timing              # download, load, inference, encoding, upload times

# Resource Metrics
resources           # GPU memory usage, cache hit rates

# Result
success             # true/false
error               # type, message, stage, stack_trace
```

---

## �📜 License

This project extracts and optimizes the inference portion of RVC.  
See the original [RVC Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) for licensing.
