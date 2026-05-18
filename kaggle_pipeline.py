import os
import subprocess
import shutil

WORKING_DIR = "/kaggle/working"
REPO_NAME = "cinematch-ml-service"
REPO_PATH = os.path.join(WORKING_DIR, REPO_NAME)

if os.path.exists(WORKING_DIR):
    os.chdir(WORKING_DIR)
    
    if os.path.exists(REPO_PATH):
        shutil.rmtree(REPO_PATH)

print("=== Cloning repository from GitHub ===")
subprocess.run(f"git clone https://github.com/zadnap/{REPO_NAME}.git", shell=True, check=True)

os.chdir(REPO_PATH)
print(f"Current working directory changed to: {os.getcwd()}")

try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    os.environ["HF_TOKEN"] = user_secrets.get_secret("HF_TOKEN")
    print("=== Thành công: Đã cấu hình HF_TOKEN từ Kaggle Secrets ===")
except Exception as e:
    print(f"=== Cảnh báo: Không thể lấy HF_TOKEN từ Secrets ({e}). Thử lấy từ môi trường hiện tại ===")
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