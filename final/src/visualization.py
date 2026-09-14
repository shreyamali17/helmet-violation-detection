
"""
Plate cropping and the rendering of pipeline state onto video frames.

A violation stored as True in a dictionary is not evidence. This module
produces the images a human reviewer actually examines.
"""

# OpenCV supplies the drawing primitives and the image array conventions.
import cv2


# Cut a tight image of a number plate out of a full frame.
def crop_plate(frame_image, plate_box, padding=5):
    """The padding widens the box slightly, because detection boxes sit tight
    against the plate and later text recognition performs better with a margin."""
    # Frame height and width, required to keep the crop inside the image.
    h, w = frame_image.shape[:2]
    # The plate's detected corner coordinates.
    x1, y1, x2, y2 = plate_box
    # Expand the top-left corner outwards, clamped so it cannot become negative.
    x1, y1 = max(0, int(x1)-padding), max(0, int(y1)-padding)
    # Expand the bottom-right corner outwards, clamped to the frame's edges.
    x2, y2 = min(w, int(x2)+padding), min(h, int(y2)+padding)
    # Array slicing is rows first and columns second, so y precedes x here.
    return frame_image[y1:y2, x1:x2]


# Draw every rider and motorcycle in a frame, coloured by current status.
def draw_rm_instances(frame_image, det, instances_by_rider, highlight_rider_id=None):
    """When highlight_rider_id is supplied, only that rider receives a status
    colour and every other rider is dimmed to grey, producing an unambiguous
    evidence image. When it is None, every rider is drawn in their own status
    colour, which is the view used for debugging."""
    # Work on a copy, because OpenCV draws in place and the original frame is
    # reused elsewhere in the pipeline.
    img = frame_image.copy()

    # Examine every rider detected in this frame.
    for rider_id, box in det["rider"].items():
        # Convert the box corners to integers, as drawing requires pixel indices.
        x1, y1, x2, y2 = map(int, box)
        # The accumulated record for this rider, if one exists yet.
        inst = instances_by_rider.get(rider_id)

        # Evidence mode: every rider other than the subject is drawn faintly.
        if highlight_rider_id is not None and rider_id != highlight_rider_id:
            # Neutral grey, no label, and the thinnest outline.
            color, label, thickness = (140, 140, 140), "", 1
        # A rider detected but not yet paired with a motorcycle has no record.
        elif inst is None:
            # Grey with a plain identifier label and no status claim.
            color, label, thickness = (150, 150, 150), f"rider {rider_id}", 2
        # Confirmed violation. The explicit comparison against True matters.
        elif inst.is_violation is True:
            # Red in blue-green-red channel order, drawn thickest for emphasis.
            color, label, thickness = (0, 0, 255), f"rider {rider_id}: NO HELMET", 3
        # Confirmed compliant. Testing against False keeps None out of this branch.
        elif inst.is_violation is False:
            # Green, indicating the helmet was verified across enough frames.
            color, label, thickness = (0, 200, 0), f"rider {rider_id}: helmet OK", 2
        # The remaining case is is_violation being None, meaning undetermined.
        else:
            # Orange, marking a rider the system is still gathering evidence on.
            color, label, thickness = (0, 165, 255), f"rider {rider_id}: pending", 2

        # Outline the rider using the colour and thickness selected above.
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        # Dimmed riders carry no label, so draw text only when one was set.
        if label:
            # Place the label above the box, pushed down when the box is near
            # the top edge and the text would otherwise fall outside the frame.
            cv2.putText(img, label, (x1, max(y1-10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Draw every motorcycle thinly, as supporting context rather than a subject.
    for moto_id, box in det["motorcycle"].items():
        # Integer corner coordinates for drawing.
        x1, y1, x2, y2 = map(int, box)
        # A thin outline in a colour distinct from every rider status colour.
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 150, 0), 1)

    # Return the annotated copy, leaving the original frame untouched.
    return img
