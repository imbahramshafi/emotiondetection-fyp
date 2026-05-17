# Frontend Integration Guide — Classroom Emotion Detection System

> Everything your frontend team needs to connect to the backend, display live data, and build the dashboard.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Authentication](#2-authentication)
3. [API Endpoints Reference](#3-api-endpoints-reference)
4. [WebSocket — Live Stream](#4-websocket--live-stream)
5. [Emotion Categories & Engagement Scoring](#5-emotion-categories--engagement-scoring)
6. [Teacher Tracking Data](#6-teacher-tracking-data)
7. [Frame Streaming (Live Video)](#7-frame-streaming-live-video)
8. [Session & Analytics Data](#8-session--analytics-data)
9. [Error Handling](#9-error-handling)
10. [Configuration & Environment](#10-configuration--environment)
11. [Code Snippets (React/JS)](#11-code-snippets-reactjs)

---

## 1. Quick Start

### Backend URL

```
http://localhost:8000
```

### Start the API server

```bash
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```

### Default login credentials

| Field    | Value           |
|----------|-----------------|
| Email    | `teacher@local` |
| Password | `changeme`      |

### CORS

The backend allows **all origins** — no proxy config needed during development. Your React/Vue/mobile app can call it directly.

---

## 2. Authentication

### How it works

- The system uses **JWT (JSON Web Tokens)** with HS256 signing.
- Tokens expire after **24 hours** (1440 minutes) by default.
- Send the token in the `Authorization` header for every protected request.

### Login flow

**1. POST `/auth/login`**

Content type must be `application/x-www-form-urlencoded` (NOT JSON).

```
POST /auth/login
Content-Type: application/x-www-form-urlencoded

username=teacher@local&password=changeme
```

Response:

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

**2. Use the token in all subsequent requests:**

```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Get current user profile

```
GET /auth/me
Authorization: Bearer <token>
```

Response:

```json
{
  "id": 1,
  "email": "teacher@local",
  "name": "Teacher"
}
```

### Important notes

- The `username` field in the login form is actually the **email address** (FastAPI's OAuth2 form uses "username" as the field name).
- If the token is expired or invalid, you'll get a `401` with `{"detail": "Invalid or expired token"}`. Redirect to login.
- There is no `/auth/register` endpoint — user accounts are created on the backend only.

---

## 3. API Endpoints Reference

All endpoints except `/auth/login`, `/internal/event`, and `/ws/live` require the `Authorization: Bearer <token>` header.

### Auth

| Method | Path          | Auth | Description             |
|--------|---------------|------|-------------------------|
| POST   | `/auth/login` | No   | Login, get JWT token    |
| GET    | `/auth/me`    | Yes  | Get current user profile|

### Live Data

| Method    | Path              | Auth | Description                          |
|-----------|-------------------|------|--------------------------------------|
| WebSocket | `/ws/live`        | No   | Real-time emotion + teacher + frames |
| GET       | `/teacher/state`  | Yes  | Poll latest teacher state            |

### Sessions & Analytics

| Method | Path             | Auth | Query Params               | Description                    |
|--------|------------------|------|-----------------------------|--------------------------------|
| GET    | `/sessions`      | Yes  | —                           | List all session CSV files     |
| GET    | `/class_summary` | Yes  | `session=<filename>`        | Emotion distribution + score   |
| GET    | `/timeline`      | Yes  | `session=<filename>`        | Engagement over time           |
| GET    | `/report`        | Yes  | `session=<filename>`        | AI-generated analysis (Claude) |

### Internal (pipeline only — don't call from frontend)

| Method | Path              | Auth | Description                    |
|--------|-------------------|------|--------------------------------|
| POST   | `/internal/event` | No   | Pipeline pushes events to API  |

---

## 4. WebSocket — Live Stream

### Connecting

```javascript
const ws = new WebSocket("ws://localhost:8000/ws/live");
```

No authentication needed for the WebSocket connection.

### Message types

The WebSocket sends JSON messages. Parse each one and check the `type` field:

```javascript
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  switch (data.type) {
    case "student":  // emotion detection event
    case "teacher":  // teacher tracking event
    case "frame":    // live camera frame (base64 JPEG)
  }
};
```

### Student emotion event

```json
{
  "type": "student",
  "frame": 42,
  "face_id": 1,
  "emotion": "happy",
  "confidence": 0.8732,
  "timestamp": "2025-05-16 14:32:45.123456"
}
```

| Field        | Type   | Description                                          |
|--------------|--------|------------------------------------------------------|
| `type`       | string | Always `"student"`                                   |
| `frame`      | int    | Video frame number                                   |
| `face_id`    | int    | Persistent student ID (tracked across frames)        |
| `emotion`    | string | One of the 7 emotions, `"confused"`, or `"unknown"`  |
| `confidence` | float  | Model confidence 0.0–1.0                             |
| `timestamp`  | string | ISO datetime when detected                           |

### Teacher tracking event

```json
{
  "type": "teacher",
  "frame": 42,
  "timestamp": "2025-05-16 14:32:45.123456",
  "detected": true,
  "position": { "x": 0.52, "y": 0.61 },
  "posture": "standing",
  "gesture": "pointing",
  "movement": "gesturing",
  "facing": "students"
}
```

(Full breakdown in [Section 6](#6-teacher-tracking-data))

### Frame event (live video)

```json
{
  "type": "frame",
  "camera": "students",
  "frame": 42,
  "jpeg": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ...",
  "timestamp": "2025-05-16 14:32:45.123456"
}
```

(Full breakdown in [Section 7](#7-frame-streaming-live-video))

### Keeping the connection alive

The server keeps the WebSocket open as long as the client is connected. The client only needs to listen — no need to send data. If the connection drops, implement a reconnect:

```javascript
function connectWS() {
  const ws = new WebSocket("ws://localhost:8000/ws/live");
  ws.onclose = () => setTimeout(connectWS, 2000); // retry after 2s
  ws.onmessage = (event) => { /* handle */ };
}
```

---

## 5. Emotion Categories & Engagement Scoring

### The 7 base emotions

| Emotion     | Engagement Score | Colour suggestion |
|-------------|-----------------|-------------------|
| `happy`     | **1.0**         | Green             |
| `surprise`  | **0.7**         | Yellow            |
| `neutral`   | **0.6**         | Grey / Blue       |
| `sad`       | **0.2**         | Blue              |
| `fear`      | **0.1**         | Purple            |
| `angry`     | **0.1**         | Red               |
| `disgust`   | **0.1**         | Dark Red          |

### Special values

| Value       | Engagement Score | When it appears                                                |
|-------------|-----------------|----------------------------------------------------------------|
| `confused`  | **0.4**         | Model detects mixed signals (neutral + surprise/fear/sad)      |
| `unknown`   | **0.0**         | Model can't confidently classify any emotion (<15% all classes)|

### How engagement is calculated

Each detected emotion maps directly to a score:

```
engagement_score = WEIGHTS[emotion]
```

**Class engagement** = average of all individual scores in a session (0.0 to 1.0).

### Suggested engagement level labels

| Score Range | Label        | Suggested UI                |
|-------------|-------------|------------------------------|
| 0.8 – 1.0  | High        | Green indicator              |
| 0.5 – 0.79 | Moderate    | Yellow/amber indicator       |
| 0.2 – 0.49 | Low         | Orange indicator             |
| 0.0 – 0.19 | Very Low    | Red indicator                |

---

## 6. Teacher Tracking Data

### Full teacher state object

```json
{
  "type": "teacher",
  "frame": 42,
  "timestamp": "2025-05-16 14:32:45.123456",
  "detected": true,
  "position": { "x": 0.52, "y": 0.61 },
  "posture": "standing",
  "gesture": "pointing",
  "movement": "gesturing",
  "facing": "students"
}
```

### Position

- `x` and `y` are **normalised 0.0–1.0** (percentage of frame width/height).
- `(0, 0)` = top-left corner, `(1, 1)` = bottom-right corner.
- Computed from the midpoint of shoulders and hips.

### Posture values

| Value       | Meaning                                  |
|-------------|------------------------------------------|
| `standing`  | Upright, normal teaching position        |
| `sitting`   | Seated (hips below knees)                |
| `leaning`   | Leaning to one side                      |
| `unknown`   | Pose not detected                        |

### Gesture values

| Value         | Meaning                                  |
|---------------|------------------------------------------|
| `arms_raised` | Both hands above shoulders               |
| `pointing`    | One arm extended horizontally            |
| `writing`     | Hand(s) raised (writing on board)        |
| `arms_down`   | Arms relaxed at sides (default)          |
| `unknown`     | Pose not detected                        |

### Movement values

| Value        | Meaning                                   |
|--------------|-------------------------------------------|
| `pacing`     | Walking around the classroom              |
| `gesturing`  | Small movements (hand gestures, shifting) |
| `stationary` | Standing still                            |
| `unknown`    | Pose not detected                         |

### Facing values

| Value      | Meaning                          |
|------------|----------------------------------|
| `students` | Facing toward the camera/class   |
| `board`    | Facing away (toward board/wall)  |
| `unknown`  | Ambiguous orientation            |

### Polling vs WebSocket

You can get teacher state two ways:

1. **WebSocket** (`ws://localhost:8000/ws/live`) — real-time push, listen for `type: "teacher"` events.
2. **REST polling** (`GET /teacher/state`) — returns the most recent state. Use if WebSocket isn't practical. Requires auth token.

When no teacher is detected: `{"detected": false}`.

---

## 7. Frame Streaming (Live Video)

### How frames arrive

Frames come through the WebSocket as base64-encoded JPEG images:

```json
{
  "type": "frame",
  "camera": "students",
  "frame": 42,
  "jpeg": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "timestamp": "2025-05-16 14:32:45.123456"
}
```

### Camera values

| Value      | Description                            |
|------------|----------------------------------------|
| `students` | Student-facing camera (top-down view)  |
| `teacher`  | Teacher-facing camera (frontal view)   |

### Displaying in an `<img>` tag

The `jpeg` field is already a complete data URL — assign it directly:

```javascript
document.getElementById("live-feed").src = data.jpeg;
```

Or in React:

```jsx
<img src={frameData.jpeg} alt="Live classroom feed" />
```

### Frame rate

- Pipeline runs at ~30 fps but only **streams 1 out of every 3 frames** (~10 fps).
- Frames are downscaled to **640px width** and compressed at **60% JPEG quality**.
- Each frame is roughly **15–40 KB** of base64 data.

---

## 8. Session & Analytics Data

### Listing sessions

```
GET /sessions
Authorization: Bearer <token>
```

```json
{
  "sessions": [
    "session_2025-05-16_14-30-00.csv",
    "session_2025-05-16_13-45-22.csv"
  ]
}
```

Sessions are sorted oldest-first. The filename encodes the start date and time.

### Class summary

```
GET /class_summary?session=session_2025-05-16_14-30-00.csv
Authorization: Bearer <token>
```

```json
{
  "session": "session_2025-05-16_14-30-00.csv",
  "class_engagement": 0.682,
  "emotion_distribution": {
    "happy": 245,
    "neutral": 189,
    "surprise": 54,
    "confused": 32,
    "sad": 18,
    "fear": 8,
    "angry": 5,
    "disgust": 2,
    "unknown": 1
  },
  "total_records": 554
}
```

- `class_engagement`: float 0.0–1.0 (average engagement across all detections)
- `emotion_distribution`: count of each emotion detected in the session
- `total_records`: total number of face detections in the session

### Engagement timeline

```
GET /timeline?session=session_2025-05-16_14-30-00.csv
Authorization: Bearer <token>
```

```json
{
  "session": "session_2025-05-16_14-30-00.csv",
  "timeline": [
    { "timestamp": "2025-05-16 14:30:05.123456", "engagement": 0.65 },
    { "timestamp": "2025-05-16 14:30:06.234567", "engagement": 0.68 },
    { "timestamp": "2025-05-16 14:30:07.345678", "engagement": 0.71 }
  ]
}
```

- Each entry = average engagement of all students at that timestamp.
- Use this for a **line chart** of engagement over time.

### AI-generated report

```
GET /report?session=session_2025-05-16_14-30-00.csv
Authorization: Bearer <token>
```

```json
{
  "session": "session_2025-05-16_14-30-00.csv",
  "class_engagement": 0.682,
  "emotion_distribution": {
    "happy": 245,
    "neutral": 189
  },
  "summary": "Overall, the class maintained a moderate engagement level of 0.68. Students 2 and 5 showed consistently high engagement..."
}
```

- `summary` is a natural-language paragraph generated by Claude AI.
- This endpoint is **slower** (~2–5 seconds) because it calls the Claude API.
- Show a loading spinner while waiting.

---

## 9. Error Handling

### Error response format

All errors return JSON with a `detail` field:

```json
{ "detail": "Error message here" }
```

The `/class_summary` and `/timeline` endpoints use `error` instead of `detail` for 404:

```json
{ "error": "Session not found" }
```

### Status codes you'll encounter

| Code | When                                           | Action                    |
|------|------------------------------------------------|---------------------------|
| 200  | Success                                        | Parse response normally   |
| 401  | Bad credentials or expired/invalid token       | Redirect to login         |
| 404  | Session file doesn't exist                     | Show "session not found"  |
| 422  | Missing/invalid query parameter                | Check request format      |

### Handling 401 globally

Set up an HTTP interceptor that catches 401 responses and redirects to your login page:

```javascript
// Axios example
axios.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem("token");
      window.location.href = "/login";
    }
    return Promise.reject(err);
  }
);
```

---

## 10. Configuration & Environment

### Backend defaults

| Setting               | Default Value     | Notes                                  |
|-----------------------|-------------------|----------------------------------------|
| API port              | `8000`            | FastAPI/Uvicorn                        |
| WebSocket path        | `/ws/live`        | Same port as API                       |
| Token expiry          | 24 hours          | Configurable via `JWT_EXPIRE_MINUTES`  |
| Default admin email   | `teacher@local`   | Configurable via `ADMIN_EMAIL`         |
| Default admin pass    | `changeme`        | Configurable via `ADMIN_PASSWORD`      |
| Frame stream FPS      | ~10 fps           | 1 in 3 frames from 30fps source       |
| Frame width           | 640px             | Downscaled before streaming            |
| JPEG quality          | 60%               | Compression for frame streaming        |

### What the frontend needs

Your frontend only needs:

1. **API base URL**: `http://localhost:8000` (or wherever the backend is deployed)
2. **WebSocket URL**: `ws://localhost:8000/ws/live`

No API keys, no environment variables, no SDK setup. Just HTTP and WebSocket.

---

## 11. Code Snippets (React/JS)

### API helper with auth

```javascript
// api.js
const API_BASE = "http://localhost:8000";

export async function login(email, password) {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({ username: email, password }),
  });
  if (!res.ok) throw new Error("Login failed");
  const data = await res.json();
  localStorage.setItem("token", data.access_token);
  return data;
}

export function getToken() {
  return localStorage.getItem("token");
}

export async function apiFetch(path, options = {}) {
  const token = getToken();
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      ...options.headers,
      Authorization: `Bearer ${token}`,
    },
  });
  if (res.status === 401) {
    localStorage.removeItem("token");
    window.location.href = "/login";
    throw new Error("Unauthorized");
  }
  return res.json();
}
```

### Usage examples

```javascript
// Get current user
const user = await apiFetch("/auth/me");

// List sessions
const { sessions } = await apiFetch("/sessions");

// Get class summary
const summary = await apiFetch(`/class_summary?session=${sessions[0]}`);

// Get timeline for a chart
const { timeline } = await apiFetch(`/timeline?session=${sessions[0]}`);

// Get AI report
const report = await apiFetch(`/report?session=${sessions[0]}`);
```

### WebSocket hook (React)

```javascript
// useWebSocket.js
import { useEffect, useRef, useState, useCallback } from "react";

export function useLiveStream(url = "ws://localhost:8000/ws/live") {
  const wsRef = useRef(null);
  const [students, setStudents] = useState({});   // face_id -> latest event
  const [teacher, setTeacher] = useState(null);
  const [frame, setFrame] = useState(null);        // latest frame data URL

  const connect = useCallback(() => {
    const ws = new WebSocket(url);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === "student") {
        setStudents((prev) => ({ ...prev, [data.face_id]: data }));
      } else if (data.type === "teacher") {
        setTeacher(data);
      } else if (data.type === "frame") {
        setFrame(data);
      }
    };

    ws.onclose = () => setTimeout(connect, 2000);
    wsRef.current = ws;
  }, [url]);

  useEffect(() => {
    connect();
    return () => wsRef.current?.close();
  }, [connect]);

  return { students, teacher, frame };
}
```

### Using the hook in a component

```jsx
function LiveDashboard() {
  const { students, teacher, frame } = useLiveStream();

  return (
    <div>
      {/* Live camera feed */}
      {frame && <img src={frame.jpeg} alt="Live feed" />}

      {/* Student emotions */}
      <h2>Students</h2>
      {Object.values(students).map((s) => (
        <div key={s.face_id}>
          Student {s.face_id}: {s.emotion} ({(s.confidence * 100).toFixed(0)}%)
        </div>
      ))}

      {/* Teacher state */}
      <h2>Teacher</h2>
      {teacher?.detected ? (
        <div>
          Posture: {teacher.posture} | Gesture: {teacher.gesture} |
          Movement: {teacher.movement} | Facing: {teacher.facing}
        </div>
      ) : (
        <div>Teacher not detected</div>
      )}
    </div>
  );
}
```

### Engagement score helper (replicate backend logic)

```javascript
// engagement.js
const WEIGHTS = {
  happy: 1.0,
  surprise: 0.7,
  neutral: 0.6,
  confused: 0.4,
  sad: 0.2,
  fear: 0.1,
  angry: 0.1,
  disgust: 0.1,
  unknown: 0.0,
};

export function getEngagement(emotion) {
  return WEIGHTS[emotion] ?? 0.0;
}

export function getEngagementLabel(score) {
  if (score >= 0.8) return "High";
  if (score >= 0.5) return "Moderate";
  if (score >= 0.2) return "Low";
  return "Very Low";
}

export function getEngagementColor(score) {
  if (score >= 0.8) return "#22c55e"; // green
  if (score >= 0.5) return "#eab308"; // yellow
  if (score >= 0.2) return "#f97316"; // orange
  return "#ef4444";                    // red
}
```

---

## 12. Real-Time Alerts (Teacher Suggestions)

The backend monitors student engagement and teacher behavior in real-time and pushes alert events through the WebSocket when something needs attention.

### Alert event format

```json
{
  "type": "alert",
  "alert_type": "low_engagement",
  "severity": "warning",
  "message": "Class engagement is declining",
  "suggestion": "Try asking an open-ended question, showing a visual, or giving students a quick pair-discussion task.",
  "timestamp": "2025-05-16 14:35:22"
}
```

### Alert types

| `alert_type`           | Severity   | Trigger                                           |
|------------------------|------------|---------------------------------------------------|
| `very_low_engagement`  | `critical` | Class avg engagement < 0.2 for 15+ seconds       |
| `low_engagement`       | `warning`  | Class avg engagement < 0.35 for 30+ seconds      |
| `teacher_stationary`   | `info`     | Teacher hasn't moved for 2+ minutes              |
| `teacher_facing_board` | `info`     | Teacher facing board for 60+ seconds             |
| `teacher_no_gesture`   | `info`     | Teacher arms down (no gestures) for 3+ minutes   |

### Severity levels

| Severity   | Meaning                              | Suggested UI                    |
|------------|--------------------------------------|---------------------------------|
| `critical` | Immediate attention needed           | Red banner / toast notification |
| `warning`  | Something is trending badly          | Yellow/amber notification       |
| `info`     | Gentle suggestion for improvement    | Blue/grey subtle notification   |

### Cooldowns

Each alert type has a **2-minute cooldown** — it won't fire the same alert repeatedly. Once triggered, the teacher has time to act before getting the same suggestion again.

### Polling endpoint (optional)

```
GET /alerts/status
Authorization: Bearer <token>
```

```json
{
  "current_engagement": 0.542,
  "tracked_students": 5,
  "teacher_stationary_seconds": 45,
  "teacher_facing_board_seconds": 0
}
```

Use this to show a live dashboard widget of the alert engine state.

### Handling alerts in the frontend

```javascript
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.type === "alert") {
    showNotification({
      severity: data.severity,
      title: data.message,
      body: data.suggestion,
    });
  }
};
```

---

## Quick Architecture Diagram

```
+------------------+       +------------------+       +------------------+
|  Student Camera  |       |  Teacher Camera  |       |    Frontend      |
|  (YOLO + CNN +   |       |  (MediaPipe Pose |       |  (React/Vue/     |
|   DeepSORT)      |       |   Tracking)      |       |   Mobile App)    |
+--------+---------+       +--------+---------+       +--------+---------+
         |                          |                           |
         |  POST /internal/event    |  POST /internal/event     |
         v                          v                           |
+--------+-------------------------------------------+          |
|              FastAPI Server (:8000)                 |<---------+
|                                                     | REST API + WS
|  - /auth/login, /auth/me  (JWT auth)               |
|  - /ws/live               (WebSocket broadcast)     |
|  - /sessions              (list CSV files)          |
|  - /class_summary         (aggregated stats)        |
|  - /timeline              (engagement over time)    |
|  - /report                (Claude AI summary)       |
|  - /teacher/state         (latest teacher state)    |
+-----------------------------------------------------+
         |
         v
+-----------------------------------------------------+
|  Storage                                             |
|  - auth.db (SQLite) — user accounts                  |
|  - logs/session_*.csv — emotion detection logs       |
+-----------------------------------------------------+
```

---

## Checklist for Frontend Team

- [ ] Set up API base URL as an environment variable (`REACT_APP_API_URL` or similar)
- [ ] Implement login page (POST form-urlencoded to `/auth/login`)
- [ ] Store JWT token in localStorage or secure cookie
- [ ] Add auth interceptor to redirect on 401
- [ ] Connect WebSocket to `/ws/live` with auto-reconnect
- [ ] Build live dashboard: camera feed, student emotion cards, teacher state panel
- [ ] Build session list page (GET `/sessions`)
- [ ] Build session detail page with:
  - [ ] Emotion distribution pie/bar chart (from `/class_summary`)
  - [ ] Engagement timeline line chart (from `/timeline`)
  - [ ] AI report section (from `/report` — show loading state)
- [ ] Handle edge cases: no sessions, teacher not detected, WebSocket disconnect
- [ ] Use engagement scoring weights to show colour-coded emotion indicators
