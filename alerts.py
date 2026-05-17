# alerts.py — Real-time alert engine for teacher suggestions
#
# Monitors student engagement and teacher behavior, fires alerts
# when thresholds are breached for sustained periods.

import time
from collections import deque
from shared import WEIGHTS

# --------------- CONFIGURATION --------------- #

# Engagement alerts
LOW_ENGAGEMENT_THRESHOLD = 0.35
LOW_ENGAGEMENT_DURATION = 30        # seconds of sustained low engagement
VERY_LOW_ENGAGEMENT_THRESHOLD = 0.2
VERY_LOW_ENGAGEMENT_DURATION = 15   # seconds — fires faster for critical drops

# Teacher behavior alerts
TEACHER_STATIONARY_DURATION = 120   # seconds idle before suggesting movement
TEACHER_BOARD_DURATION = 60         # seconds facing board before suggesting turn around
TEACHER_NO_GESTURE_DURATION = 180   # seconds with arms_down before suggesting gestures

# Cooldowns (don't spam the same alert)
ALERT_COOLDOWN = 120                # seconds between same alert type


# --------------- ALERT ENGINE --------------- #

class AlertEngine:
    def __init__(self):
        # Engagement tracking
        self.engagement_history = deque(maxlen=300)  # (timestamp, score) pairs
        self.student_emotions = {}  # face_id -> latest emotion

        # Teacher tracking
        self.teacher_stationary_since = None
        self.teacher_board_since = None
        self.teacher_no_gesture_since = None
        self.last_teacher_state = None

        # Cooldown tracking: alert_type -> last_fired_timestamp
        self.last_alert_time = {}

    def _cooldown_ok(self, alert_type: str) -> bool:
        last = self.last_alert_time.get(alert_type, 0)
        return (time.time() - last) >= ALERT_COOLDOWN

    def _fire(self, alert_type: str, severity: str, message: str, suggestion: str) -> dict:
        self.last_alert_time[alert_type] = time.time()
        return {
            "type": "alert",
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
            "suggestion": suggestion,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def process_event(self, event: dict) -> list[dict]:
        """Process an incoming event and return any triggered alerts."""
        alerts = []

        if event.get("type") == "student":
            alerts.extend(self._process_student(event))
        elif event.get("type") == "teacher":
            alerts.extend(self._process_teacher(event))

        return alerts

    # ---- Student engagement monitoring ---- #

    def _process_student(self, event: dict) -> list[dict]:
        alerts = []
        now = time.time()

        face_id = event.get("face_id")
        emotion = event.get("emotion", "unknown")
        self.student_emotions[face_id] = emotion

        # Calculate current class average engagement
        if self.student_emotions:
            scores = [WEIGHTS.get(e, 0.0) for e in self.student_emotions.values()]
            avg_engagement = sum(scores) / len(scores)
            self.engagement_history.append((now, avg_engagement))

            # Check for very low engagement (critical)
            alert = self._check_engagement(
                "very_low_engagement",
                VERY_LOW_ENGAGEMENT_THRESHOLD,
                VERY_LOW_ENGAGEMENT_DURATION,
                "critical",
                "Class engagement has dropped critically low",
                "Consider a quick energiser activity, ask a direct question, "
                "or switch to a group discussion to re-engage students.",
            )
            if alert:
                alerts.append(alert)

            # Check for low engagement (warning)
            elif not alerts:
                alert = self._check_engagement(
                    "low_engagement",
                    LOW_ENGAGEMENT_THRESHOLD,
                    LOW_ENGAGEMENT_DURATION,
                    "warning",
                    "Class engagement is declining",
                    "Try asking an open-ended question, showing a visual, "
                    "or giving students a quick pair-discussion task.",
                )
                if alert:
                    alerts.append(alert)

        return alerts

    def _check_engagement(self, alert_type, threshold, duration, severity, message, suggestion):
        if not self._cooldown_ok(alert_type):
            return None

        now = time.time()
        cutoff = now - duration

        recent = [(t, s) for t, s in self.engagement_history if t >= cutoff]
        if len(recent) < 3:
            return None

        avg = sum(s for _, s in recent) / len(recent)
        if avg < threshold:
            return self._fire(alert_type, severity, message, suggestion)
        return None

    # ---- Teacher behavior monitoring ---- #

    def _process_teacher(self, event: dict) -> list[dict]:
        alerts = []
        now = time.time()

        if not event.get("detected"):
            self._reset_teacher_timers()
            return alerts

        movement = event.get("movement", "unknown")
        facing = event.get("facing", "unknown")
        gesture = event.get("gesture", "unknown")

        # --- Stationary too long ---
        if movement == "stationary":
            if self.teacher_stationary_since is None:
                self.teacher_stationary_since = now
            elif (now - self.teacher_stationary_since) >= TEACHER_STATIONARY_DURATION:
                if self._cooldown_ok("teacher_stationary"):
                    alerts.append(self._fire(
                        "teacher_stationary",
                        "info",
                        "You've been stationary for a while",
                        "Moving around the classroom can help maintain "
                        "student attention and create a more dynamic environment.",
                    ))
        else:
            self.teacher_stationary_since = None

        # --- Facing board too long ---
        if facing == "board":
            if self.teacher_board_since is None:
                self.teacher_board_since = now
            elif (now - self.teacher_board_since) >= TEACHER_BOARD_DURATION:
                if self._cooldown_ok("teacher_facing_board"):
                    alerts.append(self._fire(
                        "teacher_facing_board",
                        "info",
                        "You've been facing the board for a while",
                        "Turning to face students helps maintain eye contact "
                        "and keeps them engaged. Consider pausing to check understanding.",
                    ))
        else:
            self.teacher_board_since = None

        # --- No gestures for too long ---
        if gesture == "arms_down":
            if self.teacher_no_gesture_since is None:
                self.teacher_no_gesture_since = now
            elif (now - self.teacher_no_gesture_since) >= TEACHER_NO_GESTURE_DURATION:
                if self._cooldown_ok("teacher_no_gesture"):
                    alerts.append(self._fire(
                        "teacher_no_gesture",
                        "info",
                        "Consider using more gestures",
                        "Hand gestures and pointing can emphasise key points "
                        "and help visual learners follow along better.",
                    ))
        else:
            self.teacher_no_gesture_since = None

        self.last_teacher_state = event
        return alerts

    def _reset_teacher_timers(self):
        self.teacher_stationary_since = None
        self.teacher_board_since = None
        self.teacher_no_gesture_since = None

    def get_status(self) -> dict:
        """Return current alert engine state for debugging/polling."""
        now = time.time()
        current_engagement = None
        if self.engagement_history:
            recent = [(t, s) for t, s in self.engagement_history if t >= now - 10]
            if recent:
                current_engagement = round(sum(s for _, s in recent) / len(recent), 3)

        return {
            "current_engagement": current_engagement,
            "tracked_students": len(self.student_emotions),
            "teacher_stationary_seconds": (
                round(now - self.teacher_stationary_since)
                if self.teacher_stationary_since else 0
            ),
            "teacher_facing_board_seconds": (
                round(now - self.teacher_board_since)
                if self.teacher_board_since else 0
            ),
        }
