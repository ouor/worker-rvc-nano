import sys
import os

# rvc가 있는 경로를 path에 추가 (이미 해당 폴더가 PYTHONPATH에 있다면 생략 가능)
sys.path.append("./src")

from src.rvc import RVCInference

# 초기화 (디바이스, HuBERT 경로, RMVPE 모델 경로 설정)
rvc = RVCInference(
    device="cuda:0", 
    is_half=True, 
    hubert_path="assets/hubert/hubert_base.pt", 
    rmvpe_path="assets/rmvpe/rmvpe.pt"
)

# 모델 로드
rvc.load_model(r"/path/to/model.pth")

# 추론 실행
sr, audio_data = rvc.infer(r"/path/to/audio.wav")

# 결과 저장
import soundfile as sf
sf.write("./output_voice.wav", audio_data, sr)