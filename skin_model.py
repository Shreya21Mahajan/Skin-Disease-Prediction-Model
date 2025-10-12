import tensorflow as tf
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import os
import keras

# --- Model Configuration ---
# Based on the image, there are 10 detectable conditions.
NUM_CLASSES = 10
IMAGE_SIZE = (64, 64) # Small size for quick example
INPUT_SHAPE = IMAGE_SIZE + (3,) # 64x64 color images

# A map of class indices to disease names (based on the image)
CLASS_NAMES = [
    'Eczema', 'Melanoma', 'Atopic Dermatitis', 'Derma Cell', 'Warts Molluscum',
    'Melanocytic Nevi', 'Benign Keratosis', 'Psoriasis', 'Seborrheic Keratosis', 'Tinea Ringworm'
]

def build_cnn_model():
    """Defines a simple CNN model using Keras."""
    model = Sequential([
        # Advanced ResNet model mentioned in the image would be much more complex,
        # but this simple CNN demonstrates the principle.
        Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax') # Softmax for multi-class classification
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# skin_model.py (REVISED)
# ... other imports ...

def create_and_save_dummy_model(model_path='derma_model.h5'):
    """Creates dummy data and trains a minimal model for demonstration."""
    print("Creating and training a dummy model...")
    # Generate dummy data (e.g., 100 random images)
    X_train = np.random.rand(100, IMAGE_SIZE[0], IMAGE_SIZE[1], 3).astype('float32')
    
    # *** CORRECTED LINE HERE ***
    # Import the utility from tf.keras or simply use the full path if tf is imported
   # CORRECTED LINE (Uses the imported 'tf' alias)
    y_train = keras.utils.to_categorical(np.random.randint(0, NUM_CLASSES, 100), num_classes=NUM_CLASSES)

    model = build_cnn_model()
# ... rest of the function ...

    # Train for a few steps on dummy data
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose='0')

    # Save the model
    model.save(model_path)
    print(f"Dummy model saved to {model_path}")
    
    return model

# Ensure the model file exists
MODEL_FILE = 'derma_model.h5'
if not os.path.exists(MODEL_FILE):
    create_and_save_dummy_model(MODEL_FILE)

# Load the model for use in the Flask app
try:
    DERSCAN_MODEL = keras.models.load_model(MODEL_FILE)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    DERSCAN_MODEL = None # Set to None if loading fails
    
if DERSCAN_MODEL:
    print(f"Model input shape: {DERSCAN_MODEL.input_shape}")
    print(f"Number of output classes: {DERSCAN_MODEL.output_shape[-1]}")