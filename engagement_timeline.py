# engagement_timeline.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from shared import calculate_engagement, select_session_file

# ---------------- TIMELINE GENERATION ---------------- #
if __name__ == "__main__":
    file_path = select_session_file()
    if not file_path:
        exit()

    print(f"\nLoading {file_path}...\n")
    df = pd.read_csv(file_path)

    # Extract session name from filename
    session_name = os.path.splitext(os.path.basename(file_path))[0]  # e.g. session_2025-12-10_15-22-30

    # Ensure directory exists
    if not os.path.exists("timeline_graphs"):
        os.makedirs("timeline_graphs")

    # Convert timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Compute engagement for each row
    df["engagement"] = df["emotion"].apply(calculate_engagement)

    # Average engagement per timestamp (class-level)
    timeline = df.groupby("timestamp")["engagement"].mean().reset_index()

    # Smooth curve
    timeline["smooth_engagement"] = timeline["engagement"].rolling(window=10, min_periods=1).mean()

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(timeline["timestamp"], timeline["smooth_engagement"], label="Engagement (smoothed)", color="blue")
    plt.plot(timeline["timestamp"], timeline["engagement"], alpha=0.3, label="Raw Engagement", color="gray")

    plt.title(f"Engagement Timeline – {session_name}")
    plt.xlabel("Time")
    plt.ylabel("Engagement Score (0-1)")
    plt.legend()
    plt.grid(True)

    # Save graph with same session name
    output_path = f"timeline_graphs/{session_name}.png"
    plt.savefig(output_path, dpi=300)

    print(f"\nTimeline graph saved to:\n{output_path}\n")

    plt.show()
