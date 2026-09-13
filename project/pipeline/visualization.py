"""Stage 10: plate cropping + drawing pipeline state onto frames."""

import cv2

def crop_plate(frame_image, plate_box, padding=5):
    h, w = frame_image.shape[:2]
    x1, y1, x2, y2 = plate_box
    x1, y1 = max(0, int(x1)-padding), max(0, int(y1)-padding)
    x2, y2 = min(w, int(x2)+padding), min(h, int(y2)+padding)
    return frame_image[y1:y2, x1:x2]


def draw_rm_instances(frame_image, det, instances_by_rider, highlight_rider_id=None):
    """If highlight_rider_id is given, only that rider gets a real
    status-colored box (for a clean, single-violation snapshot);
    everyone else is drawn dim/gray for context only. If
    highlight_rider_id is None, every rider is drawn with their own
    real status color (original behavior)."""
    img = frame_image.copy()

    for rider_id, box in det["rider"].items():
        x1, y1, x2, y2 = map(int, box)
        inst = instances_by_rider.get(rider_id)

        if highlight_rider_id is not None and rider_id != highlight_rider_id:
            color, label, thickness = (140, 140, 140), "", 1
        elif inst is None:
            color, label, thickness = (150, 150, 150), f"rider {rider_id}", 2
        elif inst.is_violation is True:
            color, label, thickness = (0, 0, 255), f"rider {rider_id}: NO HELMET", 3
        elif inst.is_violation is False:
            color, label, thickness = (0, 200, 0), f"rider {rider_id}: helmet OK", 2
        else:
            color, label, thickness = (0, 165, 255), f"rider {rider_id}: pending", 2

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if label:
            cv2.putText(img, label, (x1, max(y1-10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    for moto_id, box in det["motorcycle"].items():
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 150, 0), 1)

    return img
