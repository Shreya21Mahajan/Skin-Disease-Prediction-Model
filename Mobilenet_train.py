import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
import numpy as np
import matplotlib.pyplot as plt

# SETTINGS
DATASET_DIR = r"E:\Project\Minor Project\IMG_CLASSES"
IMAGE_SIZE = 224
BATCH_SIZE = 32
INITIAL_EPOCHS = 6
FINETUNE_EPOCHS = 6
NUM_CLASSES = 9

# DATA AUGMENTATION PIPELINE
datagen = ImageDataGenerator(
    rescale=1/255.0,
    validation_split=0.2,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.2,
    zoom_range=0.30,
    brightness_range=[0.7, 1.4],
    horizontal_flip=True,
    fill_mode='nearest'
)

train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

print("\nDetected Classes:", train_data.class_indices)

# CLASS WEIGHTS
class_labels = train_data.classes
cw = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(class_labels),
    y=class_labels
)
class_weights = dict(enumerate(cw))

print("\nClass Weights:", class_weights)

# BUILD MODEL (MobileNetV2)
base_model = MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
)

# Freeze for Stage-1
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(base_model.input, output)

model.compile(
    optimizer=Adam(3e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2)
]

# TRAINING — STAGE 1
print("\nTRAINING STAGE 1 — Frozen Backbone\n")
history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

model.save("mobilenet_stage1.keras")

# TRAINING — STAGE 2 (FINE-TUNE)
print("\n TRAINING STAGE 2 — Fine-Tuning\n")
base_model.trainable = True

model.compile(
    optimizer=Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=FINETUNE_EPOCHS,
    callbacks=callbacks
)

model.save("mobilenet_final.keras")

print("\n Training complete! Final model saved at mobilenet_final")

# PLOT ACCURACY CURVE
train_acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc   = history1.history['val_accuracy'] + history2.history['val_accuracy']

plt.figure(figsize=(8,5))
plt.plot(train_acc, label="Train Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.title("Training + Fine-Tuning Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()