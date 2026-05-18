import os
import subprocess

os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
os.environ["REPO_ID"] = "zadnap/cinematch-model"
os.environ["CORS_ORIGINS"] = "https://cinematch-api-8fuj.onrender.com"
os.environ["WEB_API_URL"] = "https://cinematch-api-8fuj.onrender.com"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

def run_command(command):
    print(f"=== Executing: {command} ===")
    process = subprocess.run(command, shell=True, check=True, text=True)
    return process

run_command("pip install -r requirements.kaggle.txt")
run_command("python scripts/download_dataset.py")
run_command("python -m ml.training.train")
run_command("python scripts/upload_artifacts.py")