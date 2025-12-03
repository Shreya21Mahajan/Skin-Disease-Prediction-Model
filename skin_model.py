import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

print(">>> Using TensorFlow-Keras Model Loader <<<")

import tensorflow as tf
import numpy as np
from PIL import Image

# Load using tf.keras because model was trained using tf.keras
print("Loading models...")

mobilenet_model = tf.keras.models.load_model("mobilenet_final.keras", compile=False)
efficientnet_model = tf.keras.models.load_model("dermascan_efficientnetb0.keras", compile=False)

print("Models loaded successfully!")

CLASS_NAMES = [
    "atopic_dermatitis",
    "basal_cell_carcinoma",
    "benign_keratosis-like_lesions",
    "eczema",
    "melanocytic_nevi",
    "melanoma",
    "psoriasis_pictures_lichen_planus",
    "seborrheic_keratoses",
    "tinea_ringworm_candidiasis",
]

IMAGE_SIZE = 224

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def ensemble_predict(image_array):
    p1 = mobilenet_model.predict(image_array, verbose=0)
    p2 = efficientnet_model.predict(image_array, verbose=0)
    return (p1 + p2) / 2

DERMACAN_ENSEMBLE = ensemble_predict
