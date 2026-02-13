# RVC-Nano | RunPod Serverless Worker

Run [RVC (Retrieval-based Voice Conversion)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) as a serverless endpoint for voice conversion.

---

## 🚀 Features
# RVC-Nano (RunPod serverless worker)

한국어 README: 이 저장소는 RVC 기반 음성 변환 모델을 RunPod 같은 서버리스 환경에서 실행하기 위한 워커입니다.

주요 내용 요약
- 서버리스 엔드포인트로 음성 변환을 수행
- CUDA 11.8 기반 PyTorch 환경을 사용하여 GPU 추론 지원
- 입력 음원, 모델, 인덱스 파일을 다운로드하고 변환 결과를 S3(R2)으로 업로드하거나 base64로 반환

빠른 시작
1) Docker 이미지 빌드 (CUDA 환경이 필요한 경우):

```bash
docker build -t rvc-nano-worker .
```

2) 로컬에서 테스트 (사전 조건: Python 3.10, ffmpeg, CUDA 드라이버(옵션))

```bash
python -m pip install -r requirements.txt
python download_models.py   # hubert_base.pt, rmvpe.pt 등 기본 모델을 /assets에 저장
python main.py              # main.py 예시로 RVCInference 사용 방법 확인
```

핵심 파일 설명
- `handler.py` : RunPod 서버리스 핸들러. 입력 검증, 다운로드, 모델 로드, 추론, 인코딩, 결과 업로드를 수행합니다.
- `schemas.py` : 입력/출력 스키마, 기본값, 유효값, 메타데이터 구조 정의
- `download_models.py` : HuggingFace에서 필수 베이스 모델(hubert, rmvpe)을 다운로드하는 유틸
- `main.py` : 로컬 사용 예시 (초기화, 모델 로드, 추론 예시)
- `requirements.txt` : Python 의존성 (PyTorch는 Dockerfile에서 별도 설치)
- `test_input.json` : 핸들러 테스트에 사용할 예시 입력

핸들러 요약 (handler.py)
- 입력 필드: `input_urls` (목록), `model_url`(필수), `index_url`(선택), `format`, `bitrate`, `sample_rate`, `f0_up_key`, `index_rate`
- 처리 단계:
  1. 입력 검증
  2. 모델/인덱스/입력 오디오 다운로드 (data: URL도 지원)
  3. 입력 오디오를 WAV로 변환 (FFmpeg 사용)
  4. RVC 모델 로드 및 추론
  5. 인코딩(ogg/m4a/mp3 등) 및 S3 업로드 또는 base64 반환
  6. 메타데이터(타이밍, 리소스, 로그)를 작성하여 S3에 업로드

환경 변수 (예시: `.env` 또는 런타임 환경 변수로 설정)
- `S3_ENDPOINT_URL`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_BUCKET_NAME` (업로드를 사용하려면 필수)
- 선택 항목: `S3_REGION`, `S3_PUBLIC_URL`, `S3_KEY_PREFIX`

테스트
- `test_input.json`에 있는 예시 입력을 사용하여 핸들러 동작을 확인하세요. (로컬에서 직접 호출하거나 RunPod에 배포 후 테스트)

모델 다운로드
- `download_models.py` 실행 시 `/assets/hubert`와 `/assets/rmvpe` 디렉토리에 베이스 모델이 저장됩니다.

의존성
- `requirements.txt`의 패키지들은 CPU/라이브러리 의존성이 있으므로 플랫폼에 맞게 설치하세요. (Dockerfile은 CUDA 11.8용 PyTorch를 별도 설치합니다.)

파일 구조 (요약)

```
Dockerfile
download_models.py
handler.py
main.py
schemas.py
requirements.txt
test_input.json
src/        # RVCInference 구현 등 (이 README 생성 시 src 내부는 생략)
```

주요 사용 팁
- FFmpeg가 시스템에 설치되어 있어야 하며 핸들러는 `ffmpeg`/`ffprobe`를 호출합니다.
- S3(R2) 업로드를 사용하지 않으면 결과는 base64 data URL로 응답에 포함됩니다.
- GPU 메모리 부족 발생 시 모델 로드 또는 추론 단계에서 OOM 에러가 발생할 수 있으니 `is_half` 등 설정을 조정하세요.

라이선스 및 출처
- 이 저장소는 RVC 관련 추론 파트를 활용합니다. 원본 프로젝트 및 모델 라이선스를 확인하세요: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI

필요하면 이 README에 예제 요청/응답, 세부 환경설정, 배포 가이드 등을 더 추가해 드리겠습니다。

---

# Audio Separator – RunPod 배포 가이드

이 문서는 RunPod 서버리스 환경에서 GPU 가속 기반 오디오 스템 분리(분리기)를 배포하는 예시입니다. MDX-Net, Demucs, Roformer 등 다양한 분리 모델을 활용할 수 있으며, Cloudflare R2/S3 업로드를 지원합니다.

## 주요 기능
- 오디오를 여러 스템(보컬, 드럼, 베이스 등)으로 분리
- WAV/MP3/FLAC/M4A/OGG/OPUS/AAC 등 일반 포맷 지원
- GPU 추론(CUDA)으로 빠른 처리
- 모델 자동 다운로드 및 캐시
- Cloudflare R2 / S3 업로드 및 presigned URL 제공
- RunPod 서버리스 핸들러(REST API) 제공

## 빠른 시작 (RunPod)

### 1) Docker 이미지 빌드

```sh
docker build -f Dockerfile.runpod -t audio-separator-runpod .
```

### 2) 환경 변수 설정

`.env.example`을 복사하여 `.env`로 만들고 R2/S3 정보 등을 채워 넣으세요:

```
R2_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=your-access-key-id
R2_SECRET_ACCESS_KEY=your-secret-access-key
R2_BUCKET_NAME=your-bucket-name
R2_BASE_PREFIX="worker-audio-separator"
```

필요에 따라 `R2_PUBLIC_URL`, `R2_REGION` 등을 추가로 설정할 수 있습니다.

### 3) RunPod 배포

이미지를 레지스트리에 푸시한 뒤 RunPod Serverless 엔드포인트로 배포합니다. 런타임 환경 변수에 위에서 설정한 값을 입력하세요.

### 4) API 사용 예시

POST 본문 예시(JSON):

```json
{
  "input_urls": ["https://example.com/audio.wav"],
  "model_name": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
  "stems": ["vocals", "instrumental"],
  "format": "mp3",
  "bitrate": "192k",
  "sample_rate": 44100
}
```

성공 응답 예시:

```json
{
  "status": "success",
  "code": "OK",
  "message": "Audio separation completed successfully",
  "files": [
    {
      "stem": "vocals",
      "format": "mp3",
      "codec": "libmp3lame",
      "channel": 2,
      "bitrate": "192k",
      "sample_rate": 44100,
      "duration_sec": 180.0,
      "url": "https://.../vocals.mp3"
    }
  ]
}
```

## 핸들러 파이프라인
- `download_file`: 입력 URL 또는 data:audio/ 디코딩
- `validate_input_file`: ffprobe/ffmpeg로 검사 및 WAV 변환
- `load_model`: 모델 로드(워밍/캐시 지원)
- `separate_audio`: 분리 모델로 추론 수행
- `encode_stems`: 요청 포맷으로 인코딩(FFmpeg)
- `upload_results`: R2/S3 업로드 및 presigned URL 생성

## 에러 처리
오디오 검증 오류, 다운로드/네트워크 오류, 모델 로드/추론 오류, 스토리지 업로드 오류 등을 구분하는 커스텀 예외(`runpod_exceptions.py`)를 사용해 상세 메시지와 코드로 반환합니다.

## 입력 스키마
자세한 검증 로직은 `runpod_schemas.py`를 참조하세요. 필수/권장 필드는 다음과 같습니다:
- `input_urls`: HTTP(S) 또는 `data:audio/` 형식의 URL 목록 (필수)
- `model_name`: 사용할 모델 파일명(기본값 제공 가능)
- `stems`: 추출할 스템 목록 (예: `vocals`, `drums`) (선택)
- `format`: 출력 포맷 (예: `wav`, `mp3`, `flac`)
- `bitrate`: 출력 비트레이트 (예: `128k`)
- `sample_rate`: 출력 샘플레이트 (예: `44100`)

## 개발
- 전체 CLI/파이썬 API 사용법은 `README.md.old` 참조
- 주요 파일
  - `runpod_handler.py` : 서버리스 진입점(핸들러)
  - `runpod_exceptions.py` : 커스텀 예외 정의
  - `runpod_schemas.py` : 입력 검증 및 기본값 정의
  - `Dockerfile.runpod` : RunPod용 Dockerfile

로컬 테스트 예시:

```bash
python -m pip install -r requirements.txt
python download_models.py   # 필요한 베이스 모델 미리 다운로드
# 핸들러를 로컬에서 시뮬레이션할 수 있는 스크립트가 있다면 해당 스크립트 실행
```

## 라이선스
MIT 라이선스(또는 해당 프로젝트에 맞는 라이선스). 자세한 내용은 `LICENSE` 파일을 확인하세요.