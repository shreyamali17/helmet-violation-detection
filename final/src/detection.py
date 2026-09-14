
"""
Stage 1: convert a raw tracking result into clean per-class dictionaries.

The model reports every object in a frame as one flat list. This module
reorganises that list so any later stage can address an object directly,
for example as detections["rider"][7].
"""

# The class-number to class-name mapping defined in the configuration module.
from src.config import CLASS_NAMES

# Reorganise one frame's tracking result into per-class lookup tables.
def get_frame_detections(r, class_names=None):
    """Return a pair of dictionaries sharing the same structure.

    detections[class_name][track_id]  is the bounding box [x1, y1, x2, y2].
    confidences[class_name][track_id] is the model's confidence, 0 to 1.
    """
    # Fall back to the configured mapping when the caller supplies none.
    class_names = class_names or CLASS_NAMES
    # Pre-create one empty dictionary per class so every class key always
    # exists, allowing callers to write det["helmet"] without checking first.
    detections = {name: {} for name in class_names.values()}
    # Build a parallel structure that will hold the confidence of each detection.
    confidences = {name: {} for name in class_names.values()}

    # A frame containing no tracked objects yields None rather than an empty
    # list, and calling .tolist() on None would raise an exception.
    if r.boxes.id is None:
        # Return the empty tables, because an empty frame is a normal event.
        return detections, confidences

    # Class number of every detected object, converted to list
    classes = r.boxes.cls.tolist()
    # Tracking identifier of every object, which remains stable across frames.
    track_ids = r.boxes.id.tolist()
    # Corner coordinates of every object in [x1, y1, x2, y2] pixel form.
    boxes = r.boxes.xyxy.tolist()
    # Detection confidence of every object, expressed between 0 and 1.
    confs = r.boxes.conf.tolist()

    # The four lists are aligned by position, so zip walks one object at a time.
    for cls, tid, box, conf in zip(classes, track_ids, boxes, confs):
        # Translate the numeric class into its readable name.
        # since cls is in float, convert it to int for lookup
        name = class_names.get(int(cls))
        # Silently skip any class the configuration does not define.
        if name:
            # File the box under its class name and tracking identifier.
            detections[name][int(tid)] = box
            # File the matching confidence in the parallel structure.
            confidences[name][int(tid)] = conf

    # Return both tables to the caller.
    return detections, confidences
