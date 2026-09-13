"""raw model.track() result -> clean per-class dicts."""

from config import CLASS_NAMES

def get_frame_detections(r, class_names=None):
    """Returns (detections, confidences) - two dicts with the same
    structure. detections[name][track_id] = box (unchanged, existing
    behavior). confidences[name][track_id] = the model's own
    confidence score for that specific detection (0 to 1)."""
    class_names = class_names or CLASS_NAMES
    detections = {name: {} for name in class_names.values()}
    confidences = {name: {} for name in class_names.values()}

    if r.boxes.id is None:
        return detections, confidences

    classes = r.boxes.cls.tolist()
    track_ids = r.boxes.id.tolist()
    boxes = r.boxes.xyxy.tolist()
    confs = r.boxes.conf.tolist()

    for cls, tid, box, conf in zip(classes, track_ids, boxes, confs):
        name = class_names.get(int(cls))
        if name:
            detections[name][int(tid)] = box
            confidences[name][int(tid)] = conf

    return detections, confidences