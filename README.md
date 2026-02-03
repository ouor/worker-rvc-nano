# RVC Extraction Project (독립 실행형 RVC 추론 모듈)

이 프로젝트는 기존 RVC(Retrieval-based Voice Conversion) 코드베이스에서 추론(Inference)에 필요한 핵심 로직만을 추출하여, 외부 의존성을 최소화하고 단독으로 실행 가능하도록 재구성한 모듈입니다.

## 📂 디렉터리 구조 및 파일 설명

### 루트 파일
- **`rvc.py`**: 모듈의 메인 엔트리 포인트입니다. `RVCInference` 클래스를 제공하여 모델 로드 및 음성 변환(Inference)을 간편하게 수행할 수 있습니다.
- **`config.py`**: 추론 환경(Device, Precision 등) 및 오디오 처리 파라미터(Padding, Window size 등)를 관리합니다.
- **`requirements.txt`**: 모듈 실행에 필요한 최소 파라미터와 라이브러리 목록입니다.

### `lib/` (핵심 라이브러리)
- **`audio.py`**: `ffmpeg` 및 `av`를 사용한 오디오 로드, 리샘플링, 경로 정규화 유틸리티를 포함합니다.
- **`rmvpe.py`**: RMVPE(Recurrent Neural Network-based Pitch Estimation) 고정밀 피치 추출 모델의 구현체입니다. 기존 코드의 아키텍처 불일치 문제를 해결하고 GPU 연산을 최적화했습니다.
- **`infer_pack/`**: RVC 모델의 신경망 구조 정의
    - `models.py`: Synthesizer(Generator) 및 Encoder 모델 정의 (V1, V2 지원).
    - `attentions.py`: 모델 내부의 Attention 메커니즘 구현.
    - `modules.py`: WN(Weight Norm), ResNet 블록 등 신경망 구성 요소.
    - `commons.py`: 텐서 연산, 패딩, 마스킹 등 공통 유틸리티 함수.
    - `transforms.py`: Flow 기반 변환을 위한 수학적 연산.

### `modules/` (추론 시스템)
- **`pipeline.py`**: 전체 추론 과정을 제어하는 핵심 파이프라인입니다. 오디오 전처리, 피치 추출, 특징 벡터 검색(Index Search), 모델 추론을 조율합니다.
- **`utils.py`**: 모델 파일 로드(`load_hubert`) 및 인덱스 파일 경로 검색 기능을 제공합니다.

---

## 🛠 수행된 주요 작업

1. **코드 격리 (Isolation)**:
   - 기존의 복잡한 폴더 구조에서 추론에 꼭 필요한 파일들만 선별하여 `src` 폴더로 모았습니다.
   - 외부(`infer/`, `configs/` 등)에 대한 임포트 참조를 모두 제거하고 로컬 패키지 내부 참조로 수정했습니다.

2. **RMVPE 모델 최적화 및 버그 수정**:
   - 가중치 파일(`.pt`) 로드 시 발생하던 아키텍처 크기 불일치(Size Mismatch) 문제를 수정했습니다.
   - GPU 텐서를 Numpy로 변환하는 과정에서 발생할 수 있는 디바이스 동기화 오류를 해결했습니다.

3. **인터페이스 단일화**:
   - 여러 단계로 나뉘어 있던 추론 과정을 `RVCInference` 클래스 하나로 통합하여, `load_model()`과 `infer()` 메서드만으로 음성 변환이 가능하게 만들었습니다.

4. **독립 실행 환경 구축**:
   - 전용 런타임(`runtime/python.exe`)에서 별도의 추가 설정 없이 실행될 수 있도록 `sys.path` 조작 로직을 내장했습니다.

---

## 🚀 사용 예시

```python
import sys
import soundfile as sf
# src 폴더가 있는 경로를 추가
sys.path.append("./") 

from src.rvc import RVCInference

# 1. 초기화
rvc = RVCInference(device="cuda:0", is_half=True)

# 2. 모델 로드
rvc.load_model("weights/my_model.pth")

# 3. 변환 수행
# f0_up_key: 음정 조절 (0은 원래 음정, 12는 한 옥타브 위)
sr, audio = rvc.infer("input_audio.wav", f0_up_key=0, f0_method="rmvpe")

# 4. 결과 저장
sf.write("output.wav", audio, sr)
```
