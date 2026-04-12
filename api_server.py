# api_server.py

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import glob
import os
import json
import pandas as pd
import numpy as np
import anthropic

app = FastAPI()

# Allow frontend on any device to connect (React Native needs this)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from shared import calculate_engagement, LOGS_DIR, LOGS_GLOB

# ------------------------------
# WebSocket connection manager
# ------------------------------
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)

    async def broadcast(self, message: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(json.dumps(message))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active.remove(ws)

manager = ConnectionManager()


# ------------------------------
# Endpoint: Live WebSocket stream
# ------------------------------
@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep connection alive; pipeline pushes events
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ------------------------------
# Internal: receive event from pipeline
# ------------------------------
@app.post("/internal/event")
async def receive_event(event: dict):
    await manager.broadcast(event)
    return {"ok": True}


# ------------------------------
# Endpoint: Get all session files
# ------------------------------
@app.get("/sessions")
def get_sessions():
    files = sorted(glob.glob(LOGS_GLOB), key=os.path.getctime)
    sessions = [os.path.basename(f) for f in files]
    return {"sessions": sessions}


# ------------------------------
# Endpoint: Class summary
# ------------------------------
@app.get("/class_summary")
def class_summary(session: str = Query(...)):
    file_path = f"{LOGS_DIR}/{session}"
    if not os.path.exists(file_path):
        return {"error": "Session not found"}

    df = pd.read_csv(file_path)

    df["engagement"] = df["emotion"].apply(calculate_engagement)
    class_engagement = float(df["engagement"].mean())

    emotion_dist = df["emotion"].value_counts().to_dict()

    return {
        "session": session,
        "class_engagement": round(class_engagement, 3),
        "emotion_distribution": emotion_dist,
        "total_records": len(df)
    }


# ------------------------------
# Endpoint: Engagement timeline
# ------------------------------
@app.get("/timeline")
def engagement_timeline(session: str = Query(...)):
    file_path = f"{LOGS_DIR}/{session}"
    if not os.path.exists(file_path):
        return {"error": "Session not found"}

    df = pd.read_csv(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["engagement"] = df["emotion"].apply(calculate_engagement)

    timeline = df.groupby("timestamp")["engagement"].mean().reset_index()
    timeline["timestamp"] = timeline["timestamp"].astype(str)

    return {
        "session": session,
        "timeline": timeline.to_dict(orient="records")
    }


# ------------------------------
# Endpoint: Claude engagement report
# ------------------------------
@app.get("/report")
def engagement_report(session: str = Query(...)):
    file_path = f"{LOGS_DIR}/{session}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Session not found")

    df = pd.read_csv(file_path)
    df["engagement"] = df["emotion"].apply(calculate_engagement)

    # Per-student summary
    student_rows = []
    for face_id, group in df.groupby("face_id"):
        counts = group["emotion"].value_counts().to_dict()
        eng = round(float(group["engagement"].mean()), 3)
        dominant = group["emotion"].mode()[0]
        student_rows.append(
            f"  Student {face_id}: engagement={eng}, dominant emotion={dominant}, "
            f"emotion counts={counts}"
        )

    # Class-level summary
    class_eng = round(float(df["engagement"].mean()), 3)
    emotion_dist = df["emotion"].value_counts().to_dict()
    duration_s = len(df["frame"].unique())

    stats_block = "\n".join([
        f"Session: {session}",
        f"Total detections: {len(df)}",
        f"Unique students tracked: {df['face_id'].nunique()}",
        f"Processed frames: {duration_s}",
        f"Class engagement score (0–1): {class_eng}",
        f"Class emotion distribution: {emotion_dist}",
        "Per-student breakdown:",
        *student_rows,
    ])

    prompt = (
        "You are an educational analytics assistant. "
        "Below is automated emotion-detection data collected during a classroom session. "
        "Write a concise, natural-language engagement summary for the teacher. "
        "Cover: overall class engagement, which students appeared most/least engaged, "
        "any notable emotional patterns, and one or two practical suggestions. "
        "Be factual, constructive, and under 200 words.\n\n"
        f"{stats_block}"
    )

    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    summary = message.content[0].text

    return {
        "session": session,
        "class_engagement": class_eng,
        "emotion_distribution": emotion_dist,
        "summary": summary,
    }
