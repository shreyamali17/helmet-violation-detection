"""
Central configuration for the helmet violation detection pipeline.

Paths resolve relative to the project root, so this works on the Jetson,
in Colab, or anywhere else the repo is cloned.
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best.pt")
VIDEO_PATH = os.path.join(PROJECT_ROOT, "sample_videos", "sample.mp4")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

CLASS_NAMES = {
    0: "motorcycle",
    1: "rider",
    2: "helmet",
    3: "no_helmet",
    4: "license_plate",
}

FRAME_LIMIT = None  # None = process the entire video, no cutoff
TRACKER = "bytetrack.yaml"

RIDER_MOTO_MAX_COST = 0.9  # requires >= 0.1 IoU to accept a pairing
HEAD_RIDER_MAX_COST = 60
PLATE_MOTO_MAX_COST = 100

MIN_OBSERVATIONS = 3
MIN_CONFIDENCE = 0.7
