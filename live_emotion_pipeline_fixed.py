"""
live_emotion_pipeline_fixed.py — V2: Enhanced Pipeline with Logging & Confused Emotion

Improvements over V1:
  - Frame-level CSV logging of all detections (timestamp, emotion, confidence, bbox)
  - "Confused" emotion class detection (heuristic: neutral + surprise/fear/sad mix)
  - Box tightening logic to improve crop quality for small faces
  - Label persistence across skipped frames (smoother visual feedback)
  - Fallback model selection (last.pt or best.pt)

Known limitation: No multi-person tracking yet (processes each face independently
per frame). This was the motivation for upgrading to DeepSORT in V3.
"""

from ultralytics import YOLO
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import csv
from datetime import datetime


# -------------------------
# CONFIG
# -------------------------
YOLO_WEIGHTS = "runs/detect/train3/weights/last.pt"
if not os.path.exists(YOLO_WEIGHTS):
    YOLO_WEIGHTS = "runs/detect/train3/weights/best.pt"

CNN_MODEL = "emotion_cnn.h5"   # or emotion_mobilenet.h5 if upgraded
EMO_CLASSES = ["angry","disgust","fear","happy","neutral","sad","surprise"]

IMG_SIZE = 320
CONF_THRES = 0.30
FACE_PAD = 0.20
PROCESS_EVERY_N_FRAMES = 4
PERSIST_FRAMES = 10

# -------------------------
# LOAD MODELS
# -------------------------
print("Loading YOLO:", YOLO_WEIGHTS)
yolo = YOLO(YOLO_WEIGHTS)

print("Loading CNN:", CNN_MODEL)
cnn = load_model(CNN_MODEL)

# -------------------------
# CONFUSED EMOTION LOGIC
# -------------------------
def get_final_emotion(probs, classes):
    angry, disgust, fear, happy, neutral, sad, surprise = probs
    
    max_prob = max(probs)
    main_label = classes[np.argmax(probs)]

    # Strong, confident emotion
    if max_prob > 0.45:
        return main_label, max_prob

    # CONFUSED DETECTION
    if neutral > 0.25 and (surprise > 0.20 or fear > 0.20 or sad > 0.20):
        return "confused", float(neutral + surprise + fear + sad) / 4

    # Very low confidence → unknown
    if max_prob < 0.15:
        return "unknown", max_prob

    return main_label, max_prob

def log_emotion(frame_number, label, prob, box):
    x1, y1, x2, y2 = box

    # Make logs/ folder if missing
    if not os.path.exists("logs"):
        os.makedirs("logs")

    file_path = "logs/emotion_log.csv"
    file_exists = os.path.isfile(file_path)

    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "frame", "emotion", "confidence", "x1", "y1", "x2", "y2"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            frame_number,
            label,
            round(prob, 4),
            x1, y1, x2, y2
        ])


# -------------------------
# FACE PREPROCESSING FOR CNN
# -------------------------
def preprocess_face(face_img):
    # For CNN48 model
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48,48))
    scaled = resized.astype(np.float32) / 255.0
    x = scaled.reshape(1,48,48,1)
    return x


# -------------------------
# DRAW LABEL
# -------------------------
def draw_label(frame, x1,y1,x2,y2, label, prob):
    text = f"{label} {prob:.2f}"
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    (tw,th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, max(0,y1-22)), (x1+tw+4, y1), (0,255,0), -1)
    cv2.putText(frame, text, (x1+2, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)


# -------------------------
# YOLO WARMUP
# -------------------------
print("Warming up YOLO...")
for _ in range(6):
    yolo.predict(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8),
                 imgsz=IMG_SIZE, conf=CONF_THRES, classes=[0], verbose=False)


# -------------------------
# WEBCAM LOOP
# -------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error opening webcam")
    exit()

frame_idx = 0
prev_boxes = []
prev_labels = []
prev_ttl = 0

print("Running Emotion Detection — Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # Skip frames for speed
    if frame_idx % PROCESS_EVERY_N_FRAMES != 0:
        if prev_ttl > 0:
            for (b, lab) in zip(prev_boxes, prev_labels):
                x1,y1,x2,y2 = b
                draw_label(frame, x1,y1,x2,y2, lab[0], lab[1])
        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # YOLO detect faces
    results = yolo.predict(frame, imgsz=IMG_SIZE, conf=CONF_THRES, classes=[0], verbose=False)
    r = results[0]

    detected_boxes = []
    detected_labels = []

    if r.boxes is not None and len(r.boxes) > 0:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Add padding
            w = x2 - x1
            h = y2 - y1
            pad = int(max(w,h) * FACE_PAD)
            x1p = max(0, x1 - pad)
            y1p = max(0, y1 - pad)
            x2p = min(frame.shape[1]-1, x2 + pad)
            y2p = min(frame.shape[0]-1, y2 + pad)

            face = frame[y1p:y2p, x1p:x2p]
            if face.size == 0:
                continue

            # CNN prediction
            x = preprocess_face(face)
            probs = cnn.predict(x, verbose=0)[0]

            label, prob = get_final_emotion(probs, EMO_CLASSES)
            log_emotion(frame_idx, label, prob, (x1p, y1p, x2p, y2p))

            detected_boxes.append((x1p,y1p,x2p,y2p))
            detected_labels.append((label, prob))

        prev_boxes = detected_boxes.copy()
        prev_labels = detected_labels.copy()
        prev_ttl = PERSIST_FRAMES

    else:
        if prev_ttl > 0:
            prev_ttl -= 1

    # Draw detections
    if len(detected_boxes) > 0:
        for (b, lab) in zip(detected_boxes, detected_labels):
            draw_label(frame, b[0], b[1], b[2], b[3], lab[0], lab[1])
    else:
        for (b, lab) in zip(prev_boxes, prev_labels):
            draw_label(frame, b[0], b[1], b[2], b[3], lab[0], lab[1])

    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
