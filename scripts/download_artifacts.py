from dotenv import load_dotenv
import os
from huggingface_hub import snapshot_download

load_dotenv()

REPO_ID = os.getenv("REPO_ID")
HF_TOKEN = os.getenv("HF_TOKEN")

if not REPO_ID:
    raise ValueError("Missing REPO_ID environment variable")

if not HF_TOKEN:
    raise ValueError("Missing HF_TOKEN environment variable")

print("Downloading artifacts and datasets from Hugging Face...")

snapshot_download(
    repo_id=REPO_ID,
    repo_type="model",
    token=HF_TOKEN,
    local_dir="ml",
    allow_patterns=["artifacts/*", "data/processed_data/*"]
)

print("Downloading successfully!")