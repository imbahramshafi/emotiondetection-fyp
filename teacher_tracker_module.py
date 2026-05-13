"""
teacher_tracker_module.py — MediaPipe Pose-based teacher tracking

Tracks the teacher from a frontal camera (camera 0) and produces per-frame
state snapshots:
  - position   : normalised (x, y) centre-of-mass in frame
  - posture    : "standing" | "sitting" | "leaning"
  - movement   : "stationary" | "pacing" | "gesturing"
  - gesture    : "arms_down" | "pointing" | "arms_raised" | "writing"
  - facing     : "students" | "board" | "unknown"

Designed to run in its own thread, pushing events to the same FastAPI
/internal/event endpoint that the student emotion pipeline uses.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import requests
import base64
from datetime import datetime
from collections import deque

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

POSITION_HISTORY_LEN = 15
MOVEMENT_THRESHOLD = 0.015
GESTURE_SHOULDER_RATIO = 0.35

FRAME_STREAM_EVERY_N = 3
FRAME_STREAM_QUALITY = 60
FRAME_STREAM_WIDTH = 640


class TeacherTracker:
    def __init__(self, camera_index=0, api_url="http://127.0.0.1:8000/internal/event",
                 show_window=True, stream_frames=True):
        self.camera_index = camera_index
        self.api_url = api_url
        self.show_window = show_window
        self.stream_frames = stream_frames

        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.position_history = deque(maxlen=POSITION_HISTORY_LEN)
        self._running = False
        self._thread = None
        self.frame_idx = 0
        self.latest_state = {}

    def _classify_posture(self, landmarks):
        """Classify standing/sitting/leaning from hip-shoulder-ankle geometry."""
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_y = (left_hip.y + right_hip.y) / 2
        torso_height = hip_y - shoulder_y

        shoulder_cx = (left_shoulder.x + right_shoulder.x) / 2
        hip_cx = (left_hip.x + right_hip.x) / 2
        lateral_offset = abs(shoulder_cx - hip_cx)

        if lateral_offset > 0.06:
            return "leaning"

        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        knee_y = (left_knee.y + right_knee.y) / 2

        leg_ratio = (knee_y - hip_y) / max(torso_height, 0.01)
        if leg_ratio < 0.6:
            return "sitting"

        return "standing"

    def _classify_gesture(self, landmarks):
        """Classify arm gesture from wrist/elbow/shoulder positions."""
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]

        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2

        left_raised = left_wrist.y < left_shoulder.y - GESTURE_SHOULDER_RATIO
        right_raised = right_wrist.y < right_shoulder.y - GESTURE_SHOULDER_RATIO

        if left_raised and right_raised:
            return "arms_raised"

        left_extended = abs(left_wrist.x - left_shoulder.x) > 0.25
        right_extended = abs(right_wrist.x - right_shoulder.x) > 0.25
        left_level = abs(left_wrist.y - left_shoulder.y) < 0.15
        right_level = abs(right_wrist.y - right_shoulder.y) < 0.15

        if (left_extended and left_level) or (right_extended and right_level):
            return "pointing"

        left_high = left_wrist.y < shoulder_y
        right_high = right_wrist.y < shoulder_y
        if left_high or right_high:
            high_wrist = left_wrist if left_high else right_wrist
            if high_wrist.y < shoulder_y - 0.1:
                return "writing"

        return "arms_down"

    def _classify_movement(self):
        """Classify movement from position history."""
        if len(self.position_history) < 5:
            return "stationary"

        positions = list(self.position_history)
        recent = positions[-5:]
        dx = max(p[0] for p in recent) - min(p[0] for p in recent)
        dy = max(p[1] for p in recent) - min(p[1] for p in recent)
        total_displacement = np.sqrt(dx**2 + dy**2)

        if total_displacement > MOVEMENT_THRESHOLD * 3:
            return "pacing"
        if total_displacement > MOVEMENT_THRESHOLD:
            return "gesturing"
        return "stationary"

    def _classify_facing(self, landmarks):
        """Estimate facing direction from nose-shoulder geometry."""
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        shoulder_cx = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_width = abs(right_shoulder.x - left_shoulder.x)

        if shoulder_width < 0.05:
            return "board"

        nose_offset = nose.x - shoulder_cx
        if abs(nose_offset) < 0.03:
            return "students"

        return "board" if abs(nose_offset) > 0.06 else "unknown"

    def _compute_centre(self, landmarks):
        """Normalised centre-of-mass from hips and shoulders."""
        pts = [
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
        ]
        cx = np.mean([p.x for p in pts])
        cy = np.mean([p.y for p in pts])
        return round(float(cx), 4), round(float(cy), 4)

    def _push_event(self, state):
        if self.api_url is None:
            return
        payload = {
            "type": "teacher",
            "frame": self.frame_idx,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
            **state,
        }
        try:
            requests.post(self.api_url, json=payload, timeout=0.5)
        except Exception:
            pass

    def _push_frame(self, frame_bgr):
        if self.api_url is None or not self.stream_frames:
            return
        h, w = frame_bgr.shape[:2]
        if w > FRAME_STREAM_WIDTH:
            new_h = int(h * FRAME_STREAM_WIDTH / w)
            frame_bgr = cv2.resize(frame_bgr, (FRAME_STREAM_WIDTH, new_h))
        ok, buf = cv2.imencode(".jpg", frame_bgr,
                               [cv2.IMWRITE_JPEG_QUALITY, FRAME_STREAM_QUALITY])
        if not ok:
            return
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        payload = {
            "type": "frame",
            "camera": "teacher",
            "frame": self.frame_idx,
            "jpeg": f"data:image/jpeg;base64,{b64}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        }
        try:
            requests.post(self.api_url, json=payload, timeout=0.5)
        except Exception:
            pass

    def _run_loop(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"[TeacherTracker] Cannot open camera {self.camera_index}")
            return

        print(f"[TeacherTracker] Running on camera {self.camera_index}")

        if self.show_window:
            cv2.namedWindow("Teacher Tracker", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Teacher Tracker", 640, 480)

        fps = 0.0
        fps_timer = time.time()
        fps_count = 0

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    break

                self.frame_idx += 1
                if self.frame_idx % 2 != 0:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self._running = False
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb)

                state = {
                    "position": None,
                    "posture": "unknown",
                    "movement": "unknown",
                    "gesture": "unknown",
                    "facing": "unknown",
                    "detected": False,
                }

                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark
                    cx, cy = self._compute_centre(lm)
                    self.position_history.append((cx, cy))

                    state["detected"] = True
                    state["position"] = {"x": cx, "y": cy}
                    state["posture"] = self._classify_posture(lm)
                    state["gesture"] = self._classify_gesture(lm)
                    state["movement"] = self._classify_movement()
                    state["facing"] = self._classify_facing(lm)

                    if self.show_window:
                        mp_drawing.draw_landmarks(
                            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                        info = f"{state['posture']} | {state['gesture']} | {state['movement']} | {state['facing']}"
                        cv2.putText(frame, info, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

                self.latest_state = state
                threading.Thread(target=self._push_event, args=(state,), daemon=True).start()

                fps_count += 1
                if time.time() - fps_timer >= 1.0:
                    fps = fps_count / (time.time() - fps_timer)
                    fps_count = 0
                    fps_timer = time.time()

                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if self.frame_idx % FRAME_STREAM_EVERY_N == 0:
                    threading.Thread(target=self._push_frame, args=(frame.copy(),),
                                     daemon=True).start()

                if self.show_window:
                    cv2.imshow("Teacher Tracker", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self._running = False
        finally:
            cap.release()
            if self.show_window:
                cv2.destroyWindow("Teacher Tracker")

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def is_running(self):
        return self._running and self._thread and self._thread.is_alive()
