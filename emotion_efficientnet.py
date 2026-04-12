# emotion_efficientnet.py
# Fine-tunes EfficientNetB2 on FER2013 for 7-class emotion recognition.
#
# Input:  48x48 RGB, pixel values in [0, 255] (preprocess_input handles normalisation)
# Output: emotion_efficientnet.keras
#
# Training is split into two phases:
#   Phase 1 — freeze the EfficientNetB2 base, train only the new classification head
#   Phase 2 — unfreeze the top 30 layers of the base for fine-tuning

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------- CONFIG ---------------- #
TRAIN_DIR = "datasets/fer2013/train"
VAL_DIR   = "datasets/fer2013/val"
IMG_SIZE   = 48
BATCH_SIZE = 64
SAVE_PATH  = "emotion_efficientnet.keras"

# ---------------- DATA GENERATORS ---------------- #
# preprocess_input expects uint8 or float [0,255] — pass it as the preprocessing function
# so augmentation happens before normalisation
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
)

val_data = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)

# ---------------- MODEL DEFINITION ---------------- #
base = EfficientNetB2(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
base.trainable = False  # frozen for phase 1

x = base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
output = layers.Dense(7, activation="softmax")(x)

model = Model(inputs=base.input, outputs=output)

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6, verbose=1),
]

# ---------------- PHASE 1: TRAIN HEAD ONLY ---------------- #
print("\n=== Phase 1: training classification head (base frozen) ===\n")
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(train_data, validation_data=val_data, epochs=10, callbacks=callbacks)

# ---------------- PHASE 2: FINE-TUNE TOP LAYERS ---------------- #
print("\n=== Phase 2: fine-tuning top 30 base layers ===\n")
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

# Lower LR for fine-tuning to avoid destroying pretrained features
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(train_data, validation_data=val_data, epochs=15, callbacks=callbacks)

# ---------------- SAVE ---------------- #
model.save(SAVE_PATH)
print(f"\nModel saved as {SAVE_PATH}")
