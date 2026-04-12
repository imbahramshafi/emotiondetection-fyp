import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# --- Directories ---
train_dir = "datasets/fer2013/train"
val_dir = "datasets/fer2013/val"

# --- Data Augmentation for stronger learning ---
train_datagen = ImageDataGenerator(
    rescale=1/255.0,
    rotation_range=12,
    width_shift_range=0.12,
    height_shift_range=0.12,
    zoom_range=0.12,
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

# --- CNN Model (upgraded with BatchNorm + Dropout) ---
model = Sequential([
    Conv2D(32, (3,3), activation="relu", padding="same", input_shape=(48,48,1)),
    BatchNormalization(),
    Conv2D(32, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Conv2D(64, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Conv2D(128, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(7, activation="softmax")
])

# --- Compile ---
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# --- Train for 25 epochs ---
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=25
)

# --- Save ---
model.save("emotion_cnn.h5")
print("Model saved as emotion_cnn.h5")
