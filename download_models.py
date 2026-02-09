"""
Download required models from HuggingFace for RVC inference.
Downloads hubert_base.pt and rmvpe.pt to the assets directory.
"""

import os
from huggingface_hub import hf_hub_download


REPO_ID = "lj1995/VoiceConversionWebUI"
ASSETS_DIR = "/assets"

MODELS = [
    {
        "filename": "hubert_base.pt",
        "subfolder": "",
        "local_dir": f"{ASSETS_DIR}/hubert",
    },
    {
        "filename": "rmvpe.pt",
        "subfolder": "",
        "local_dir": f"{ASSETS_DIR}/rmvpe",
    },
]


def download_model(filename: str, subfolder: str, local_dir: str, max_retries: int = 3):
    """
    Download a model file from HuggingFace with retry logic.
    """
    os.makedirs(local_dir, exist_ok=True)
    
    for attempt in range(max_retries):
        try:
            print(f"Downloading {filename}... (attempt {attempt + 1}/{max_retries})")
            path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                subfolder=subfolder if subfolder else None,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
            )
            print(f"Successfully downloaded: {path}")
            return path
        except Exception as err:
            if attempt < max_retries - 1:
                print(f"Error: {err}. Retrying...")
            else:
                raise RuntimeError(f"Failed to download {filename} after {max_retries} attempts: {err}")


def download_all_models():
    """
    Download all required models for RVC inference.
    """
    print("=" * 50)
    print("Downloading RVC models from HuggingFace...")
    print(f"Repository: {REPO_ID}")
    print("=" * 50)
    
    for model in MODELS:
        download_model(
            filename=model["filename"],
            subfolder=model["subfolder"],
            local_dir=model["local_dir"],
        )
    
    print("=" * 50)
    print("All models downloaded successfully!")
    print("=" * 50)


if __name__ == "__main__":
    download_all_models()
