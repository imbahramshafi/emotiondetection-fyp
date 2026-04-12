# evaluate_models.py
# Evaluates the CNN (emotion_cnn.h5) and/or EfficientNetB2 (emotion_efficientnet.keras)
# on the FER2013 test set. Produces:
#   - Classification report (precision, recall, F1 per class)
#   - Confusion matrix heatmap saved to evaluation_results/

import numpy as np
import os
import argparse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

TEST_DIR = "datasets/fer2013/test"
IMG_SIZE = 48
BATCH_SIZE = 64
EMO_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
OUTPUT_DIR = "evaluation_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def evaluate_cnn():
    print("\n========== CNN (emotion_cnn.h5) ==========\n")
    model = load_model("emotion_cnn.h5")

    datagen = ImageDataGenerator(rescale=1 / 255.0)
    test_data = datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    loss, acc = model.evaluate(test_data, verbose=1)
    print(f"\nTest loss: {loss:.4f}  |  Test accuracy: {acc:.4f}\n")

    y_true = test_data.classes
    y_pred_probs = model.predict(test_data, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    report = classification_report(y_true, y_pred, target_names=EMO_CLASSES, digits=4)
    print(report)

    report_path = os.path.join(OUTPUT_DIR, "cnn_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Model: emotion_cnn.h5\n")
        f.write(f"Test loss: {loss:.4f}  |  Test accuracy: {acc:.4f}\n\n")
        f.write(report)
    print(f"Report saved to {report_path}")

    save_confusion_matrix(y_true, y_pred, "CNN (emotion_cnn.h5)", "cnn_confusion_matrix.png")


def evaluate_efficientnet():
    print("\n========== EfficientNetB2 (emotion_efficientnet.keras) ==========\n")

    from tensorflow.keras.applications.efficientnet import preprocess_input
    from tensorflow.keras.applications import EfficientNetB2
    from tensorflow.keras import layers, Model

    # Rebuild architecture to avoid Keras version mismatch, then load weights
    print("Building EfficientNetB2 architecture...")
    base = EfficientNetB2(include_top=False, weights=None, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(7, activation="softmax")(x)
    model = Model(inputs=base.input, outputs=output)
    model.load_weights("emotion_efficientnet.keras")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_data = datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    loss, acc = model.evaluate(test_data, verbose=1)
    print(f"\nTest loss: {loss:.4f}  |  Test accuracy: {acc:.4f}\n")

    y_true = test_data.classes
    y_pred_probs = model.predict(test_data, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    report = classification_report(y_true, y_pred, target_names=EMO_CLASSES, digits=4)
    print(report)

    report_path = os.path.join(OUTPUT_DIR, "efficientnet_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Model: emotion_efficientnet.keras (EfficientNetB2)\n")
        f.write(f"Test loss: {loss:.4f}  |  Test accuracy: {acc:.4f}\n\n")
        f.write(report)
    print(f"Report saved to {report_path}")

    save_confusion_matrix(y_true, y_pred, "EfficientNetB2", "efficientnet_confusion_matrix.png")


def save_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=EMO_CLASSES, yticklabels=EMO_CLASSES, ax=axes[0])
    axes[0].set_title(f"{title} — Counts")
    axes[0].set_ylabel("True")
    axes[0].set_xlabel("Predicted")

    # Percentages
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=EMO_CLASSES, yticklabels=EMO_CLASSES, ax=axes[1])
    axes[1].set_title(f"{title} — Per-class %")
    axes[1].set_ylabel("True")
    axes[1].set_xlabel("Predicted")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"Confusion matrix saved to {path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "efficientnet", "both"], default="both",
                        help="Which model to evaluate (default: both)")
    args = parser.parse_args()

    if args.model in ("cnn", "both"):
        evaluate_cnn()
    if args.model in ("efficientnet", "both"):
        evaluate_efficientnet()

    print("Done. Results saved to evaluation_results/")
