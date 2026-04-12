# analytics.py
import pandas as pd
import numpy as np
import os

from shared import calculate_engagement, select_session_file

# ---------------- ANALYTICS FUNCTIONS ---------------- #
def compute_student_summary(df):
    summaries = {}

    for face_id, group in df.groupby("face_id"):
        emotions = group["emotion"].value_counts().to_dict()

        engagement_values = [calculate_engagement(e) for e in group["emotion"]]
        engagement_score = np.mean(engagement_values)

        summaries[face_id] = {
            "emotion_counts": emotions,
            "total_detections": len(group),
            "engagement_score": round(float(engagement_score), 3)
        }

    return summaries


def compute_class_summary(df):
    engagement_values = [calculate_engagement(e) for e in df["emotion"]]
    class_engagement = np.mean(engagement_values)

    emotion_distribution = df["emotion"].value_counts().to_dict()

    return {
        "class_engagement": round(float(class_engagement), 3),
        "emotion_distribution": emotion_distribution,
        "total_entries": len(df)
    }


# ---------------- MAIN PROGRAM ---------------- #
if __name__ == "__main__":
    log_file = select_session_file()
    if not log_file:
        exit()

    df = pd.read_csv(log_file)
    print(f"\nLoaded Session: {log_file}\n")

    print("===== PER-STUDENT SUMMARY =====")
    student_summary = compute_student_summary(df)
    for sid, info in student_summary.items():
        print(f"\nStudent ID: {sid}")
        print("Emotion Counts:", info["emotion_counts"])
        print("Engagement Score:", info["engagement_score"])

    print("\n===== CLASS SUMMARY =====")
    class_summary = compute_class_summary(df)
    print("Class Engagement:", class_summary["class_engagement"])
    print("Emotion Distribution:", class_summary["emotion_distribution"])
    print("Total Records:", class_summary["total_entries"])

