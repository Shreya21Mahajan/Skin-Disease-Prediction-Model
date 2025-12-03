import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras.models import load_model
import numpy as np
from keras.preprocessing import image

# IMAGE SIZE
IMAGE_SIZE = 224

# LOAD MODELS USING STANDALONE KERAS
print("Loading MobileNetV2 model...")
mobilenet_model = load_model("mobilenet_final.keras", compile=False)

print("Loading EfficientNetB0 model...")
efficientnet_model = load_model("dermascan_efficientnetb0.keras", compile=False)

print("Both models loaded successfully!\n")

def preprocess(img_path):
    """Load image, resize, normalize and expand dims."""
    img = image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def ensemble_predict(img_path, class_names):
    """Predict using both models and average predictions."""
    img = preprocess(img_path)

    # predictions
    pred1 = mobilenet_model.predict(img, verbose=0)
    pred2 = efficientnet_model.predict(img, verbose=0)

    final_pred = (pred1 + pred2) / 2

    # Find best class
    class_index = np.argmax(final_pred)
    confidence = float(final_pred[0][class_index])

    return class_names[class_index], confidence
