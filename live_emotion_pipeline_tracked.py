"""
live_emotion_pipeline_tracked.py — V3: Production Pipeline with Multi-Person Tracking

Final, feature-complete pipeline for real-time classroom emotion detection.

Key improvements over V2:
  - DeepSORT multi-person tracking (persistent face_id across frames)
  - Batched CNN inference (single predict() call for all faces per frame)
  - Temporal emotion smoothing (sliding window + majority vote, reduces flicker)
  - Live event streaming to FastAPI backend (WebSocket)
  - Session-based logging (one CSV per run with metadata)
  - Flexible model selection (CNN or EfficientNetB2)
  - Frame-skipping with smart label persistence

Architecture:
  1. YOLO detects faces in frame
  2. DeepSORT assigns/tracks face IDs across frames
  3. Batch preprocess confirmed tracks into CNN
  4. Single batched predict() → emotion probabilities
  5. Apply "confused" heuristic and temporal smoothing
  6. Log to CSV and push live events to API

This is the recommended entry point for deployment.
"""

from ultralytics import YOLO
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import csv
from datetime import datetime
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
from collections import defaultdict, deque, Counter
import argparse
import threading
import requests

# ------------- ARGUMENT PARSING ------------- #
parser = argparse.ArgumentParser(description="Tracked emotion detection pipeline")
parser.add_argument("--input", default=None,
                    help="Video file path (e.g. classroom.mp4). Omit to use webcam.")
args = parser.parse_args()
input_source = args.input if args.input is not None else 0

# SESSION logging (auto-created per run)
SESSION_NAME = datetime.now().strftime("session_%Y-%m-%d_%H-%M-%S")
if not os.path.exists("logs"):
    os.makedirs("logs")
SESSION_LOG_FILE = os.path.join("logs", f"{SESSION_NAME}.csv")
print("Session log file:", SESSION_LOG_FILE)
log_file = open(SESSION_LOG_FILE, "a", newline="")
log_writer = csv.writer(log_file)
log_writer.writerow(["timestamp", "frame", "face_id", "emotion", "confidence", "x1", "y1", "x2", "y2"])

# ---------------- CONFIG ---------------- #
YOLO_WEIGHTS = "face_yolo.pt"
CNN_MODEL = "emotion_cnn.h5"
EFFICIENTNET_MODEL = "emotion_efficientnet.keras"

# "cnn"         — original 48x48 grayscale CNN (emotion_cnn.h5)
# "efficientnet" — EfficientNetB2 fine-tuned, 48x48 RGB (emotion_efficientnet.keras)
MODEL_TYPE = "cnn"

# Set to None to disable live pushing (e.g. when api_server is not running)
API_EVENT_URL = "http://127.0.0.1:8000/internal/event"

EMO_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

IMG_SIZE = 320
CONF_THRES = 0.30
FACE_PAD = 0.20
PROCESS_EVERY_N_FRAMES = 4
PERSIST_FRAMES = 10
SMOOTHING_WINDOW = 5

# ------------- LOAD MODELS --------------- #
print("Loading YOLO:", YOLO_WEIGHTS)
yolo = YOLO(YOLO_WEIGHTS)

if MODEL_TYPE == "efficientnet":
    from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
    from tensorflow.keras.applications import EfficientNetB2
    from tensorflow.keras import layers, Model as KerasModel

    # Rebuild architecture locally to avoid Keras version config mismatch,
    # then load only the weights from the .keras file.
    print("Building EfficientNetB2 architecture...")
    _base = EfficientNetB2(include_top=False, weights=None, input_shape=(48, 48, 3))
    _x = layers.GlobalAveragePooling2D()(_base.output)
    _x = layers.Dense(128, activation="relu")(_x)
    _x = layers.Dropout(0.3)(_x)
    _out = layers.Dense(7, activation="softmax")(_x)
    cnn = KerasModel(inputs=_base.input, outputs=_out)
    print("Loading EfficientNetB2 weights:", EFFICIENTNET_MODEL)
    cnn.load_weights(EFFICIENTNET_MODEL)
else:
    print("Loading CNN:", CNN_MODEL)
    cnn = load_model(CNN_MODEL)

# ----------- INIT DEEPSORT TRACKER ----------- #
tracker = DeepSort(max_age=30, n_init=2)


# --------- CONFUSED EMOTION LOGIC -------- #
def get_final_emotion(probs, classes):
    angry, disgust, fear, happy, neutral, sad, surprise = probs
    max_prob = max(probs)
    main_label = classes[np.argmax(probs)]

    if max_prob > 0.45:
        return main_label, max_prob

    if neutral > 0.25 and (surprise > 0.20 or fear > 0.20 or sad > 0.20):
        triggered = [neutral] + [v for v in (surprise, fear, sad) if v > 0.20]
        return "confused", float(sum(triggered)) / len(triggered)

    if max_prob < 0.15:
        return "unknown", max_prob

    return main_label, max_prob

#----------TIGHTNING BOX AROUND DETECTED FACES-----#
def tighten_box(x1,y1,x2,y2):
    w=x2-x1
    h=y2-y1
    size=min(w,h)

    if size<60:
        shrink=0.35
    elif size<120:
        shrink=0.25
    else:
        shrink=0.15

    dx=int(w*shrink/2)
    dy=int(h*shrink/2)

    x1+=dx;y1+=dy;x2-=dx;y2-=dy

    h=y2-y1
    w=x2-x1
    if h>1.3*w:
        excess=int((h-1.3*w)/2)
        y1+=excess;y2-=excess

    return x1,y1,x2,y2


# ------------- PREPROCESS FACE ------------- #
def preprocess_face(face_img):
    """For MODEL_TYPE='cnn': 48x48 grayscale, normalised to [0,1]."""
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    scaled = resized.astype(np.float32) / 255.0
    x = scaled.reshape(1, 48, 48, 1)
    return x


def preprocess_face_efficientnet(face_img):
    """For MODEL_TYPE='efficientnet': 48x48 RGB in [0,255], passed through
    EfficientNet's preprocess_input (normalisation happens inside the model)."""
    rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (48, 48))
    x = resized.astype(np.float32)
    x = efficientnet_preprocess(x)
    return x.reshape(1, 48, 48, 3)


# ---------------- LOGGING ---------------- #
def log_emotion_with_id(frame_number, face_id, label, prob, box):
    x1, y1, x2, y2 = box
    log_writer.writerow([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        frame_number,
        face_id,
        label,
        round(prob, 4),
        x1, y1, x2, y2
    ])


# ---------------- LIVE EVENT PUSH ---------------- #
def _post_event(payload):
    try:
        requests.post(API_EVENT_URL, json=payload, timeout=0.1)
    except Exception:
        pass  # api_server not running — silently skip

def push_live_event(frame_number, face_id, label, prob):
    if API_EVENT_URL is None:
        return
    payload = {
        "frame":      frame_number,
        "face_id":    face_id,
        "emotion":    label,
        "confidence": round(float(prob), 4),
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
    }
    threading.Thread(target=_post_event, args=(payload,), daemon=True).start()


# ------------- DRAW LABEL ---------------- #
def draw_label(frame, x1, y1, x2, y2, label, prob, face_id):
    text = f"ID:{face_id} {label} {prob:.2f}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + tw + 4, y1), (0, 255, 0), -1)
    cv2.putText(frame, text, (x1 + 2, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)


# ----------- WARM UP YOLO -------------- #
print("Warming up YOLO...")
for _ in range(4):
    yolo.predict(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8),
                 imgsz=IMG_SIZE, conf=CONF_THRES, classes=[0], verbose=False)

# ------------- WEBCAM LOOP ------------- #
if isinstance(input_source, str):
    print(f"Input source: file — {input_source}")
else:
    print("Input source: webcam (index 0)")
cap = cv2.VideoCapture(input_source)
if not cap.isOpened():
    print(f"Error: could not open input source: {input_source}")
    exit()

frame_idx = 0
fps = 0.0
fps_timer = time.time()
fps_frame_count = 0
prev_boxes = []
prev_labels = []
prev_ttl = 0
total_detections = 0
seen_ids = set()
emotion_history = defaultdict(lambda: deque(maxlen=SMOOTHING_WINDOW))

print("Tracking & Emotion Detection Running — Press Q to quit")

cv2.namedWindow("Tracked Emotion Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tracked Emotion Detection", 960, 720)


try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # skip frames to speed up
        if frame_idx % PROCESS_EVERY_N_FRAMES != 0:
            if prev_ttl > 0:
                for (b, lab) in zip(prev_boxes, prev_labels):
                    draw_label(frame, b[0], b[1], b[2], b[3], lab[0], lab[1], lab[2])
                prev_ttl -= 1

            fps_frame_count += 1
            if time.time() - fps_timer >= 1.0:
                fps = fps_frame_count / (time.time() - fps_timer)
                fps_frame_count = 0
                fps_timer = time.time()
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Tracked Emotion Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # YOLO detect faces
        results = yolo.predict(frame, imgsz=IMG_SIZE, conf=CONF_THRES, classes=[0], verbose=False)
        r = results[0]

        detections = []
        if r.boxes is not None:
            for b in r.boxes:
                x1,y1,x2,y2=map(int,b.xyxy[0].tolist())
                x1,y1,x2,y2=tighten_box(x1,y1,x2,y2)
                x1=max(0,x1);y1=max(0,y1)
                x2=min(frame.shape[1]-1,x2)
                y2=min(frame.shape[0]-1,y2)
                conf=float(b.conf[0].item())
                w=x2-x1
                h=y2-y1
                detections.append(([x1,y1,w,h],conf,0))

        # DeepSORT tracking
        tracks = tracker.update_tracks(detections, frame=frame)

        detected_boxes = []
        detected_labels = []

        # Pass 1: collect all valid face crops without calling CNN yet
        pending = []  # list of (track_id, tight_box, preprocessed_array)
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())

            # ---------- TIGHT BOX (for drawing + tracking) ----------
            tight_x1 = max(0, x1)
            tight_y1 = max(0, y1)
            tight_x2 = min(frame.shape[1] - 1, x2)
            tight_y2 = min(frame.shape[0] - 1, y2)

            # ---------- PADDED BOX (for emotion CNN only) ----------
            w = tight_x2 - tight_x1
            h = tight_y2 - tight_y1

            pad_x = int(w * FACE_PAD)
            pad_y = int(h * FACE_PAD)

            emo_x1 = max(0, tight_x1 - pad_x)
            emo_y1 = max(0, tight_y1 - pad_y)
            emo_x2 = min(frame.shape[1] - 1, tight_x2 + pad_x)
            emo_y2 = min(frame.shape[0] - 1, tight_y2 + pad_y)

            face = frame[emo_y1:emo_y2, emo_x1:emo_x2]
            if face.size == 0:
                continue
            if (emo_x2 - emo_x1) < 20 or (emo_y2 - emo_y1) < 20:
                continue

            preprocessed = (preprocess_face_efficientnet(face)
                            if MODEL_TYPE == "efficientnet"
                            else preprocess_face(face))
            pending.append((track_id, (tight_x1, tight_y1, tight_x2, tight_y2), preprocessed))

        # Pass 2: single batched CNN call, then log and store results
        if pending:
            batch = np.concatenate([p[2] for p in pending], axis=0)
            all_probs = cnn.predict(batch, verbose=0)
            for (track_id, tight_box, _), probs in zip(pending, all_probs):
                label, prob = get_final_emotion(probs, EMO_CLASSES)
                emotion_history[track_id].append(label)
                smoothed_label = Counter(emotion_history[track_id]).most_common(1)[0][0]
                detected_boxes.append(tight_box)
                detected_labels.append((smoothed_label, prob, track_id))
                log_emotion_with_id(frame_idx, track_id, smoothed_label, prob, tight_box)
                push_live_event(frame_idx, track_id, smoothed_label, prob)
                total_detections += 1
                seen_ids.add(track_id)

        # persistence
        if len(detected_boxes) > 0:
            prev_boxes = detected_boxes
            prev_labels = detected_labels
            prev_ttl = PERSIST_FRAMES
        else:
            if prev_ttl > 0:
                prev_ttl -= 1

        # draw
        if len(detected_boxes) > 0:
            for (b, lab) in zip(detected_boxes, detected_labels):
                draw_label(frame, b[0], b[1], b[2], b[3], lab[0], lab[1], lab[2])
        else:
            for (b, lab) in zip(prev_boxes, prev_labels):
                draw_label(frame, b[0], b[1], b[2], b[3], lab[0], lab[1], lab[2])

        fps_frame_count += 1
        if time.time() - fps_timer >= 1.0:
            fps = fps_frame_count / (time.time() - fps_timer)
            fps_frame_count = 0
            fps_timer = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Tracked Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("\n===== SESSION SUMMARY =====")
    print(f"Log file:          {SESSION_LOG_FILE}")
    print(f"Frames captured:   {frame_idx}")
    print(f"Total detections:  {total_detections}")
    print(f"Unique faces seen: {len(seen_ids)}")
    print("===========================\n")
    cap.release()
    cv2.destroyAllWindows()
    log_file.close()
