"""
Central configuration for the helmet violation detection pipeline.

"""

MODEL_PATH = "/home/emertxe/Desktop/Emertxe/Shreya/helmet-detection-violation/models/best.pt"
VIDEO_PATH = "/home/emertxe/Desktop/Emertxe/Shreya/helmet-detection-violation/sample_videos/20211125082353_0060.mp4"

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

OUTPUT_DIR = "/home/emertxe/Desktop/Emertxe/Shreya/helmet-detection-violation/output"