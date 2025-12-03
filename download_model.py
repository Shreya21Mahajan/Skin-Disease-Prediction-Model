import os
import gdown

# CONFIGURATION: Add your two Google Drive direct download URLs
EFFICIENTNET_URL = "https://drive.google.com/file/d/1GkjWDB41eFROKdUaHCFjtr66DXJqiBPm/view?usp=drive_link"
MOBILENET_URL    = "https://drive.google.com/file/d/1_37-KjfaPnFuDDZCQMNSU4g1U9Xiw9g9/view?usp=drive_link"

# Save locations
MODEL_DIR = "models"
EFFICIENTNET_PATH = os.path.join(MODEL_DIR, "efficientnet_model.keras")
MOBILENET_PATH    = os.path.join(MODEL_DIR, "mobilenet_model.keras")

# DOWNLOAD FUNCTION
def download_file(url, output_path, model_name):
    """Download a file from Google Drive using gdown."""

    # Create directory if missing
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"[INFO] Created directory: {MODEL_DIR}")

    # Skip if file already exists
    if os.path.exists(output_path):
        print(f"[INFO] {model_name} already exists → {output_path}")
        return

    print(f"[INFO] Downloading {model_name}...")
    gdown.download(url, output_path, quiet=False)
    print(f"[SUCCESS] {model_name} downloaded → {output_path}\n")

# MAIN EXECUTION
if __name__ == "__main__":
    print("MODEL DOWNLOAD STARTED\n")

    download_file(EFFICIENTNET_URL, EFFICIENTNET_PATH, "EfficientNet Model")
    download_file(MOBILENET_URL, MOBILENET_PATH, "MobileNet Model")

    print("ALL MODELS READY")
