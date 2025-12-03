import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import pickle

# SETTINGS
DATASET_DIR = r"E:\Project\Minor Project\IMG_CLASSES"
IMAGE_SIZE = 224
BATCH_SIZE = 32
INITIAL_EPOCHS = 6
FINETUNE_EPOCHS = 10     # increased for better performance
NUM_CLASSES = 9

# DATA PIPELINE (Optimized for medical images)
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.10,
    horizontal_flip=True
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

# CLASS WEIGHTS (Very important for imbalanced medical datasets)
classes = train_data.classes
cw = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(classes),
    y=classes
)
class_weights = dict(enumerate(cw))

# STAGE 1: BASE MODEL (freeze)
base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
)
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(base_model.input, output)

model.compile(
    optimizer=Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nStage 1 Training (Frozen base)...\n")
history3 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weights,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.3)
    ]
)

# STAGE 2: FINE-TUNING (unfreeze layers)
print("\nFine-tuning EfficientNetB0...\n")

base_model.trainable = True

# unfreeze last 120 layers
for layer in base_model.layers[:-120]:
    layer.trainable = False

model.compile(
    optimizer=Adam(3e-5),   # optimal LR for fine-tuning
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history4 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=FINETUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.3)
    ]
)

# SAVE MODEL
model.save("dermascan_efficientnetb0.keras")
print("\nModel saved as dermascan_efficientnetb0.keras\n")

# SAVE HISTORY
with open("history.pkl", "wb") as f:
    pickle.dump({"h3": history3.history, "h4": history4.history}, f)

# PLOTS
train_acc = history3.history["accuracy"] + history4.history["accuracy"]
val_acc   = history3.history["val_accuracy"] + history4.history["val_accuracy"]

train_loss = history3.history["loss"] + history4.history["loss"]
val_loss   = history3.history["val_loss"] + history4.history["val_loss"]

epochs = range(1, len(train_acc) + 1)

plt.figure(figsize=(10,5))
plt.plot(epochs, train_acc, label="Train Accuracy")
plt.plot(epochs, val_acc, label="Validation Accuracy")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(epochs, train_loss, label="Train Loss")
plt.plot(epochs, val_loss, label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()
