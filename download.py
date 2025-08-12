#python download.py
from pathlib import Path
from huggingface_hub import snapshot_download
BIGEARTH_MODELS = ["resnet50-s1-v0.2.0", "resnet50-s2-v0.2.0"]
BIGEARTH_BASE_DIR = Path("pretrained_models")
def download_bigearth_models():
    """Download BigEarthNet v2.0 ResNet models from Hugging Face"""
    BIGEARTH_BASE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(BIGEARTH_MODELS)} BigEarthNet models to: {BIGEARTH_BASE_DIR}")
    for model_name in BIGEARTH_MODELS:
        print(f"\nModel: {model_name}")
        local_dir = BIGEARTH_BASE_DIR / model_name
        if local_dir.exists() and any(local_dir.iterdir()):
            print(f"Skipping existing model: {model_name}")
            continue
        try:
            snapshot_download(
                repo_id=f"BIFOLD-BigEarthNetv2-0/{model_name}",
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print(f"Downloaded: {model_name}")
        except Exception as e:
            print(f"Failed to download {model_name}: {e}")
    print("\nBigEarthNet download complete.\n")
def main():
    download_bigearth_models()
if __name__ == "__main__":
    main()