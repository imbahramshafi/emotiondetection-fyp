# shared.py — shared constants used across analytics, api_server, and engagement_timeline

import glob
import os

LOGS_DIR = "logs"
LOGS_GLOB = "logs/session_*.csv"

WEIGHTS = {
    "happy": 1.0,
    "surprise": 0.7,
    "neutral": 0.6,
    "confused": 0.4,
    "sad": 0.2,
    "fear": 0.1,
    "angry": 0.1,
    "disgust": 0.1,
    "unknown": 0.0
}

def calculate_engagement(emotion):
    return WEIGHTS.get(emotion, 0.0)


def select_session_file():
    files = sorted(glob.glob(LOGS_GLOB), key=os.path.getctime)

    if not files:
        print("\n❌ No session files found.\nRun the tracked pipeline first to generate a session CSV.\n")
        return None

    print("\nAvailable Sessions:\n")
    for i, f in enumerate(files):
        print(f"{i}: {os.path.basename(f)}")

    while True:
        try:
            choice = int(input("\nSelect session number: "))
            if 0 <= choice < len(files):
                return files[choice]
            print("Invalid choice, try again.")
        except EOFError:
            return None
        except ValueError:
            print("Invalid choice, try again.")
