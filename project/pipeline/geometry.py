"""Geometry helpers shared across association stages."""


def x_center(box):
    """Used inassociate_riders_to_motorcycles stage - applied to
    BOTH the rider's box and the motorcycle's box, since horizontal
    alignment is what matters for that match, on both sides equally."""
    x1, _, x2, _ = box
    return (x1 + x2) / 2


def top_center(box):
    """Used in associate_heads_to_riders stage - applied only to
    the RIDER's box, since a head sits at the top of a rider's body,
    not at its vertical center."""
    x1, y1, x2, _ = box
    return ((x1 + x2) / 2, y1)


def bottom_center(box):
    """Used in associate_plates_to_motorcycles stage - applied only
    to the MOTORCYCLE's box, since a license plate is mounted low on
    the bike, near the bottom of its box."""
    x1, _, x2, y2 = box
    return ((x1 + x2) / 2, y2)


def center(box):
    """Used in associate_helmets_to_riders stage - applied to
    the helmet/no_helmet box, and in associate_plates_to_motorcycles stage - applied to
    the license_plate box, the 'other side' of each match, where the
    object has no directional bias of its own, so its plain center is
    the right reference point."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def iou(box_a, box_b):
    """Intersection-over-Union between two [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
