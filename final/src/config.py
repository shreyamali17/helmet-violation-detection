
"""
Central configuration for the helmet violation detection pipeline.

Paths are resolved relative to the project root, so the same file works
on a Jetson device, in Google Colab, or in any other location.
"""

# The os module supplies path utilities that behave correctly on every
# operating system, which keeps the project portable.
import os

# src root is the directory containing this file.
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
# project root is the directory containing the src folder, 
# which is the parent of this file.
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# Location of the trained detection weights the pipeline loads.
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best.pt")
# Location of the input video.
VIDEO_PATH = os.path.join(PROJECT_ROOT, "sample_videos", "sample.mp4")
# Directory into which reports and violation snapshots are written.
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Table for mapping class IDs to class names
# Model generates one of these IDs for each detected object
CLASS_NAMES = {
    # The person travelling on the motorcycle.
    0: "rider",
    # The motorcycle itself.
    1: "motorcycle",
    # A head that is wearing a helmet.
    2: "helmet",
    # A head that is not wearing a helmet.
    3: "no_helmet",
    # The number plate mounted on the motorcycle.
    4: "license_plate",
}

# Maximum number of frames to process; None means process the whole video.
FRAME_LIMIT = None
# Configuration file naming the tracking algorithm used to assign track IDs.
TRACKER = "bytetrack.yaml"

# If the paired rider and motorcycle boxes has IoU or xdist cost > this value,
# the pairing is rejected.
RIDER_MOTO_MAX_COST_IOU = 0.9
RIDER_MOTO_MAX_COST_XDIST = 300
# Rejection ceiling, in pixels, between a rider's head position and a helmet.
HEAD_RIDER_MAX_COST = 60
# Rejection ceiling, in pixels, between a motorcycle and its number plate.
# Plates are small and often partly obscured, so the allowance is larger.
PLATE_MOTO_MAX_COST = 100

# Frames a rider must be observed before the system will produce any verdict.
MIN_OBSERVATIONS = 3
# Averaged detector confidence a verdict must reach before it is accepted.
MIN_CONFIDENCE = 0.7
