import os
import shutil
import random

train_dir = "datasets/fer2013/train"
val_dir = "datasets/fer2013/val"

os.makedirs(val_dir, exist_ok=True)

for emotion in os.listdir(train_dir):
    emotion_path = os.path.join(train_dir, emotion)

    if not os.path.isdir(emotion_path):
        continue

    val_emotion_path = os.path.join(val_dir, emotion)
    os.makedirs(val_emotion_path, exist_ok=True)

    images = os.listdir(emotion_path)
    random.shuffle(images)

    val_count = int(len(images) * 0.1)  # 10% for validation
    val_images = images[:val_count]

    for img in val_images:
        src = os.path.join(emotion_path, img)
        dst = os.path.join(val_emotion_path, img)
        shutil.move(src, dst)

print("Validation split created successfully.")
