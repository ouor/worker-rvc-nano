import os
from fairseq import checkpoint_utils

def get_index_path_from_model(sid, index_root):
    if not index_root or not os.path.exists(index_root):
        return ""
    for root, _, files in os.walk(index_root, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                if sid.split(".")[0] in name:
                    return os.path.join(root, name)
    return ""

def load_hubert(config, hubert_path):
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [hubert_path],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()
