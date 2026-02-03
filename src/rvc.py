import os
import sys
import torch
import numpy as np
import soundfile as sf
import traceback
import logging

# Add the current directory to sys.path so that absolute imports within extracted code work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from config import Config
from lib.audio import load_audio
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from modules.pipeline import Pipeline
from modules.utils import load_hubert

logger = logging.getLogger(__name__)

class RVCInference:
    def __init__(self, device="cuda:0", is_half=True, hubert_path="assets/hubert/hubert_base.pt", rmvpe_path="assets/rmvpe/rmvpe.pt"):
        self.config = Config(device=device, is_half=is_half)
        self.hubert_path = hubert_path
        self.rmvpe_path = rmvpe_path
        
        self.net_g = None
        self.pipeline = None
        self.hubert_model = None
        self.tgt_sr = None
        self.version = "v2"
        self.if_f0 = 1

    def load_model(self, model_path):
        logger.info(f"Loading RVC model: {model_path}")
        cpt = torch.load(model_path, map_location="cpu")
        self.tgt_sr = cpt["config"][-1]
        cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        self.if_f0 = cpt.get("f0", 1)
        self.version = cpt.get("version", "v1")

        synthesizer_class = {
            ("v1", 1): SynthesizerTrnMs256NSFsid,
            ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
            ("v2", 1): SynthesizerTrnMs768NSFsid,
            ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
        }

        self.net_g = synthesizer_class.get(
            (self.version, self.if_f0), SynthesizerTrnMs256NSFsid
        )(*cpt["config"], is_half=self.config.is_half)

        del self.net_g.enc_q

        self.net_g.load_state_dict(cpt["weight"], strict=False)
        self.net_g.eval().to(self.config.device)
        if self.config.is_half:
            self.net_g = self.net_g.half()
        else:
            self.net_g = self.net_g.float()

        self.pipeline = Pipeline(self.tgt_sr, self.config)

    def infer(self, input_path, f0_up_key=0, f0_method="rmvpe", file_index="", index_rate=0.75, filter_radius=3, resample_sr=0, rms_mix_rate=0.25, protect=0.33):
        if self.net_g is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if self.hubert_model is None:
            self.hubert_model = load_hubert(self.config, self.hubert_path)

        audio = load_audio(input_path, 16000)
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio /= audio_max

        times = [0, 0, 0]
        
        audio_opt = self.pipeline.pipeline(
            self.hubert_model,
            self.net_g,
            0, # sid
            audio,
            input_path,
            times,
            f0_up_key,
            f0_method,
            file_index,
            index_rate,
            self.if_f0,
            filter_radius,
            self.tgt_sr,
            resample_sr,
            rms_mix_rate,
            self.version,
            protect,
            rmvpe_path=self.rmvpe_path
        )
        
        sr = resample_sr if (resample_sr >= 16000 and self.tgt_sr != resample_sr) else self.tgt_sr
        return sr, audio_opt

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    rvc = RVCInference()
    # rvc.load_model("path/to/model.pth")
    # sr, audio = rvc.infer("path/to/input.wav")
    # sf.write("output.wav", audio, sr)
