"""
train_emotion_cnn_improved.py

Improved CNN training with:
  - Class weight balancing (handles imbalanced data like disgust)
  - Enhanced data augmentation (shear, brightness, zoom)
  - Better regularization (L2 + higher dropout)
  - Learning rate scheduling
  - Extended training with early stopping

Expected improvement: 54% → 58-60% accuracy
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

# --- Directories ---
train_dir = "datasets/fer2013/train"
val_dir = "datasets/fer2013/val"
EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# --- Enhanced Data Augmentation ---
train_datagen = ImageDataGenerator(
    rescale=1/255.0,
    rotation_range=20,              # increased from 12
    width_shift_range=0.2,          # increased from 0.12
    height_shift_range=0.2,         # increased from 0.12
    zoom_range=0.2,                 # increased from 0.12
    shear_range=0.15,               # NEW: adds geometric distortion
    brightness_range=[0.8, 1.2],    # NEW: varies brightness
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1/255.0)

# --- Load datasets ---
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical",
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical",
    shuffle=False
)

# --- Compute class weights (handle imbalance) ---
# Get class indices
class_indices = train_data.class_indices
num_classes = len(class_indices)

# Count samples per class
class_counts = {}
for emotion_idx, emotion in enumerate(EMOTION_CLASSES):
    emotion_path = os.path.join(train_dir, emotion)
    class_counts[emotion_idx] = len(os.listdir(emotion_path))

print("\n===== Class Distribution =====")
for idx, emotion in enumerate(EMOTION_CLASSES):
    count = class_counts[idx]
    print(f"{emotion}: {count} samples")

# Compute weights
weights = compute_class_weight(
    'balanced',
    classes=np.array(list(class_counts.keys())),
    y=np.array([class_counts[i] for i in range(num_classes)])
)
class_weight_dict = {i: w for i, w in enumerate(weights)}
print(f"\nClass weights: {class_weight_dict}\n")

# --- Improved CNN Model with Regularization ---
model = Sequential([
    Conv2D(32, (3,3), activation="relu", padding="same",
           kernel_regularizer=l2(0.001), input_shape=(48,48,1)),
    BatchNormalization(),
    Conv2D(32, (3,3), activation="relu", kernel_regularizer=l2(0.001)),
    MaxPooling2D(2,2),
    Dropout(0.3),  # increased from 0.25

    Conv2D(64, (3,3), activation="relu", padding="same",
           kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Conv2D(64, (3,3), activation="relu", kernel_regularizer=l2(0.001)),
    MaxPooling2D(2,2),
    Dropout(0.3),  # increased from 0.25

    Conv2D(128, (3,3), activation="relu", padding="same",
           kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Conv2D(128, (3,3), activation="relu", kernel_regularizer=l2(0.001)),
    MaxPooling2D(2,2),
    Dropout(0.3),  # increased from 0.25

    Flatten(),
    Dense(256, activation="relu", kernel_regularizer=l2(0.001)),
    Dropout(0.5),  # keep high
    Dense(7, activation="softmax")
])

# --- Learning Rate Schedule ---
def lr_schedule(epoch):
    lr = 0.0001
    if epoch > 20:
        lr *= 0.5
    if epoch > 30:
        lr *= 0.5
    if epoch > 40:
        lr *= 0.5
    return lr

# --- Compile & Train ---
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7, verbose=1),
    LearningRateScheduler(lr_schedule, verbose=1)
]

print("\n===== Training (Improved Model) =====\n")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,  # increased from 25
    callbacks=callbacks,
    class_weight=class_weight_dict  # use weighted losses
)

# --- Save ---
model.save("emotion_cnn_improved.h5")
print("\nModel saved as emotion_cnn_improved.h5")

# --- Evaluate on validation set ---
val_loss, val_acc = model.evaluate(val_data)
print(f"\nValidation accuracy: {val_acc:.4f}")
