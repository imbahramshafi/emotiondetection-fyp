# Classroom Emotion Detection System

**Real-time AI-powered emotion recognition for educational analytics**

![License](https://img.shields.io/badge/license-MIT-blue) ![Python](https://img.shields.io/badge/python-3.8+-blue) ![Status](https://img.shields.io/badge/status-production-brightgreen)

---

## 📌 Overview

A production-ready system that detects and tracks student emotions in real-time using computer vision and deep learning. Built for classroom engagement monitoring, this system provides teachers with actionable insights through automated reports powered by Claude AI.

**Key Results:**
- ✅ **54.3% accuracy** on FER2013 (7-class emotion recognition)
- ✅ **Multi-person tracking** across frames (DeepSORT)
- ✅ **Real-time processing** at 15+ FPS
- ✅ **Live WebSocket streaming** to frontend
- ✅ **AI-generated teacher reports** (Claude integration)

---

## 🎯 What It Does

1. **Detects faces** in real-time using YOLOv8
2. **Tracks individuals** across frames with persistent IDs
3. **Classifies emotions** into 7 categories (happy, sad, angry, neutral, fear, surprise, disgust)
4. **Detects confusion** using custom heuristics (neutral + surprise/fear/sad mix)
5. **Streams live events** to frontend via WebSocket
6. **Generates engagement reports** using Claude AI
7. **Logs all detections** to session-based CSV files

---

## 🏗️ Architecture

```
Webcam/Video
    ↓
[YOLO Face Detection] (YOLOv8)
    ↓
[DeepSORT Tracking] (multi-person ID assignment)
    ↓
[Batched CNN Inference] (emotion_cnn.h5)
    ↓
[Temporal Smoothing] (majority vote over 5 frames)
    ↓
[Analytics Backend] (FastAPI + WebSocket)
    ↓
[Live Dashboard] (frontend integration)
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Webcam or video file
- 4GB+ RAM

### Installation
```bash
git clone https://github.com/imbahramshafi/emotiondetection-fyp.git
cd emotiondetection-fyp
pip install -r requirements.txt
```

### Run Pipeline
```bash
# Start backend API
python api_server.py

# In another terminal, run emotion detection
python live_emotion_pipeline_tracked.py

# Optional: specify video file instead of webcam
python live_emotion_pipeline_tracked.py --input classroom.mp4
```

The system will:
- Display live detections with face IDs and emotion labels
- Log all detections to `logs/session_YYYY-MM-DD_HH-MM-SS.csv`
- Stream events to `http://localhost:8000/ws/live` (WebSocket)

---

## 📊 API Endpoints

The FastAPI backend provides:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/sessions` | GET | List all recorded sessions |
| `/class_summary?session=...` | GET | Emotion distribution for a session |
| `/timeline?session=...` | GET | Engagement timeline (over time) |
| `/report?session=...` | GET | Claude AI-generated teacher report |
| `/ws/live` | WebSocket | Live event stream (real-time emotions) |

**Example:**
```bash
curl http://localhost:8000/sessions
curl "http://localhost:8000/report?session=session_2026-04-12_12-00-00.csv"
```

---

## 📈 Model Performance

### CNN (emotion_cnn.h5)
- **Test Accuracy:** 54.3%
- **Best Class:** Happy (F1=0.81)
- **Challenging Class:** Disgust (F1=0.00)

### Evaluation
Full metrics available in `evaluation_results/`:
- Confusion matrices (raw counts + percentages)
- Per-class precision, recall, F1 scores
- 7,178 test samples across 7 emotions

---

## 🔧 Configuration

Edit `live_emotion_pipeline_tracked.py` to customize:

```python
MODEL_TYPE = "cnn"              # or "efficientnet"
CONF_THRES = 0.30              # YOLO confidence threshold
FACE_PAD = 0.20                # padding around faces (fraction)
PROCESS_EVERY_N_FRAMES = 4     # frame skipping for speed
SMOOTHING_WINDOW = 5           # temporal smoothing window
API_EVENT_URL = "http://..."   # backend WebSocket URL
```

---

## 📁 Project Structure

```
emotion-fyp/
├── live_emotion_pipeline_tracked.py  # V3: Production pipeline (START HERE)
├── live_emotion_pipeline_fixed.py    # V2: With logging & confusion detection
├── live_emotion_pipeline.py          # V1: Basic detection
├── api_server.py                     # FastAPI backend + WebSocket
├── evaluate_models.py                # Test set evaluation & metrics
├── analytics.py                      # Per-student & class summaries
├── engagement_timeline.py            # Visualization of engagement over time
├── emotion_cnn.h5                    # Trained CNN model
├── emotion_efficientnet.keras        # EfficientNetB2 model
├── shared.py                         # Shared constants (engagement weights)
├── requirements.txt                  # Dependencies
├── train_colab.ipynb                 # Google Colab training notebook
├── evaluation_results/               # Confusion matrices & reports
├── scripts/                          # One-time utilities & older scripts
└── datasets/                         # FER2013 (train/val/test splits)
```

---

## 🎓 Iterative Development

This project shows clear engineering progression:

1. **V1** (`live_emotion_pipeline.py`) — Basic YOLO + CNN pipeline
2. **V2** (`live_emotion_pipeline_fixed.py`) — Added logging, confused emotion class, persistence
3. **V3** (`live_emotion_pipeline_tracked.py`) — DeepSORT tracking, batched inference, live API events

Each version solves specific limitations of the previous one.

---

## 💡 Key Features

### Smart Emotion Detection
- **Confused Class:** Custom heuristic detects ambiguous expressions
- **Temporal Smoothing:** Reduces flicker using 5-frame majority vote
- **Confidence Scores:** All predictions include confidence levels

### Performance Optimizations
- **Frame Skipping:** Process every Nth frame, display every frame
- **Batched Inference:** Single CNN call for all faces per frame
- **Label Persistence:** Show previous detections during skipped frames

### Production-Ready
- **Session Logging:** Unique CSV per run with timestamps
- **WebSocket Streaming:** Real-time events to frontend
- **Class Weights:** Handles imbalanced emotion classes during training

---

## 🤝 Integration with Frontend

Your frontend team can:

1. **Connect to WebSocket** at `ws://localhost:8000/ws/live`
2. **Receive events** with format:
   ```json
   {
     "frame": 1234,
     "face_id": 5,
     "emotion": "happy",
     "confidence": 0.87,
     "timestamp": "2026-04-12 12:34:56.789123"
   }
   ```
3. **Query REST endpoints** for historical data (sessions, reports, timelines)

---

## 📊 Use Cases

- **Classroom Engagement Monitoring** — Teachers see real-time student emotions
- **Online Course Analytics** — Platforms track learner sentiment during videos
- **Interview Assessment** — HR tools measure candidate emotional responses
- **User Experience Research** — See how users react to interfaces/products
- **Customer Service Training** — Monitor trainee emotions during calls

---

## 🛠️ Tech Stack

- **Computer Vision:** YOLOv8 (face detection), OpenCV
- **Tracking:** DeepSORT (multi-object tracking)
- **Deep Learning:** TensorFlow/Keras (CNN, EfficientNetB2)
- **Backend:** FastAPI, WebSocket
- **Data Processing:** Pandas, NumPy, Scikit-learn
- **AI Integration:** Anthropic Claude API
- **Training:** Google Colab (GPU)

---

## 📝 Training Your Own Model

See `train_colab.ipynb` for step-by-step Colab training with GPU. Or locally:

```bash
python train_emotion_cnn_improved.py  # Improved version with class weights & augmentation
```

---

## 📄 License

MIT License — feel free to use for research, education, and commercial projects.

---

## 👨‍💻 About

Built as a **Final Year Project (FYP)** in computer science, combining real-world applications of computer vision, deep learning, and educational analytics.

For questions or collaborations, feel free to reach out!

---

## 🚨 Notes

- **Privacy:** Always obtain consent before recording/analyzing people
- **Fairness:** FER2013 dataset is limited; model may have gender/age biases
- **Classroom Use:** Get institutional approval before deploying in schools

---

**⭐ If you find this useful, please star the repo!**
