from huggingface_hub import snapshot_download

print("Downloading datasets from Hugging Face...")

snapshot_download(
    repo_id="zadnap/cinematch-dataset",
    repo_type="dataset",
    local_dir="ml/data/raw_data",
    local_dir_use_symlinks=False
)

print("Downloading successfully!")