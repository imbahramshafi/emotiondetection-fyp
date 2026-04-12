"""
live_emotion_pipeline.py — V1: Initial Pipeline

Basic real-time emotion detection pipeline combining YOLO face detection
with CNN emotion classification. No tracking, no event logging.

Features:
  - YOLOv8 face detection
  - Grayscale CNN emotion classification (7 emotions)
  - Frame skipping for performance
  - Padding applied around detected faces

This was the starting point before adding sophisticated features like
multi-person tracking (DeepSORT) and structured session logging.
"""

from ultralytics import YOLO
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# --- config ---
YOLO_WEIGHTS = "runs/detect/train3/weights/best.pt"    # your YOLO face detector
CNN_MODEL    = "emotion_cnn.h5"                        # your Keras emotion model
EMO_CLASSES  = ["angry","disgust","fear","happy","neutral","sad","surprise"]  # directory order used for training
PROCESS_EVERY_N_FRAMES = 2  # increase fps by processing every Nth frame (set 1 to process every frame)
FACE_PAD = 0.2  # pad around face box (fraction of box size)

# --- load models ---
yolo = YOLO(YOLO_WEIGHTS)
cnn = load_model(CNN_MODEL)

# --- helpers ---
def preprocess_face(face_img):
    # face_img: BGR color crop from OpenCV
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48,48))
    scaled = resized.astype(np.float32) / 255.0
    x = scaled.reshape(1,48,48,1)
    return x

def draw_label(frame, x1,y1,x2,y2, label, prob):
    text = f"{label} {prob:.2f}"
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    # put text background
    (tw,th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1-20), (x1+tw, y1), (0,255,0), -1)
    cv2.putText(frame, text, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

# --- open webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: cannot open webcam")
    exit()

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    if frame_idx % PROCESS_EVERY_N_FRAMES != 0:
        # show without processing for speed
        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # run YOLO (fast single-call inference)
    results = yolo(frame, stream=False)  # returns list-like; results[0] holds detections
    r = results[0]

    if r.boxes is None or len(r.boxes) == 0:
        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    for box in r.boxes:  # iterate detected boxes
        # box.xyxy -> tensor [x1,y1,x2,y2]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # add padding
        w = x2 - x1
        h = y2 - y1
        pad = int(max(w,h) * FACE_PAD)
        x1p = max(0, x1 - pad)
        y1p = max(0, y1 - pad)
        x2p = min(frame.shape[1]-1, x2 + pad)
        y2p = min(frame.shape[0]-1, y2 + pad)

        face_crop = frame[y1p:y2p, x1p:x2p]
        if face_crop.size == 0:
            continue

        x = preprocess_face(face_crop)
        probs = cnn.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        label = EMO_CLASSES[idx] if idx < len(EMO_CLASSES) else str(idx)
        prob = float(probs[idx])

        draw_label(frame, x1p, y1p, x2p, y2p, label, prob)

    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
