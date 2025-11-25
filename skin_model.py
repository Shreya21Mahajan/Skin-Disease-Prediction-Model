# skin_model.py – FINAL VERSION (WORKS WITH REAL MODEL)

import os
import numpy as np
from keras.models import load_model

# MODEL FILE PATH
MODEL_PATH = "dermascan_model.h5"

# IMAGE SIZE (MUST MATCH TRAINING)
IMAGE_SIZE = 160    # MobileNetV2 input size

# CLASS NAMES (MATCH YOUR FOLDER ORDER)

CLASS_NAMES = [
    "Eczema",
    "Melanoma",
    "Atopic Dermatitis",
    "Basal Cell Carcinoma (BCC)",
    "Melanocytic Nevi (NV)",
    "Benign Keratosis-like Lesions (BKL)",
    "Psoriasis / Lichen Planus",
    "Seborrheic Keratoses",
    "Tinea / Ringworm / Candidiasis",
    "Warts / Molluscum / Viral Infections"
]

# LOAD TRAINED MODEL
def load_dermascan_model():
    """Load the trained skin disease model."""
    try:
        if not os.path.exists(MODEL_PATH):
            print("\n ERROR: Model file not found!\n")
            print("Expected file:", MODEL_PATH)
            print("Please train the model first using train_model.py\n")
            return None

        print("Loading trained DermaScan model...")
        model = load_model(MODEL_PATH)

        if model is None:
            print("Model loader returned None.")
            return None

        print("Model loaded successfully.")
        print("➡ Input shape:", getattr(model, "input_shape", None))
        print("➡ Output shape:", getattr(model, "output_shape", None))
        return model

    except Exception as e:
        print("Failed to load model:", str(e))
        return None

# GLOBAL MODEL (used by app.py)
DERSCAN_MODEL = load_dermascan_model()
