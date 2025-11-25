import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# SETTINGS 
DATASET_DIR = r"E:\Project\Minor Project\IMG_CLASSES"
IMAGE_SIZE = 160        # Faster
BATCH_SIZE = 16         # Good for CPU / low RAM
EPOCHS = 6              # Quick and effective
NUM_CLASSES = 10        # Number of folders

# DATA LOADERS
datagen = ImageDataGenerator(
    rescale=1/255.0,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# MODEL: MobileNetV2 (FAST & ACCURATE)
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
)

# Freezing base model for FAST training
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
from sklearn.utils import class_weight
import numpy as np

# compute class weights
class_labels = train_data.classes  
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(class_labels),
    y=class_labels
)
class_weights = dict(enumerate(class_weights))
print("CLASS WEIGHTS:", class_weights)

# TRAINING
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=6
)

# SAVE MODEL
model.save("dermascan_model.h5")
print("\nTraining complete! Model saved as dermascan_model.h5")
   
