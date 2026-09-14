
"""
Geometric helpers shared across every association stage.

Each function reduces a bounding box to the single point, orthe single score,
that makes a particular comparison meaningful.
"""

# Horizontal midpoint of a box.
def x_center(box):
    """Implemented to check the baseline method."""
    # Only the two horizontal edges are required; the vertical edges are discarded.
    x1, _, x2, _ = box
    # The point midway between the left and right edges.
    return (x1 + x2) / 2


# Midpoint of the top edge of a box.
def top_center(box):
    """Applied to the RIDER box when matching heads, because a head sits at
    the top of a rider's body rather than at its vertical centre."""
    # The top edge y1 is retained; the bottom edge is not required.
    x1, y1, x2, _ = box
    # Horizontal midpoint paired with the top edge.
    return ((x1 + x2) / 2, y1)


# Midpoint of the bottom edge of a box.
def bottom_center(box):
    """Applied to the MOTORCYCLE box when matching number plates, because a
    plate is mounted low on the vehicle, near the bottom of its box."""
    # The bottom edge y2 is retained; the top edge is not required.
    x1, _, x2, y2 = box
    # Horizontal midpoint paired with the bottom edge.
    return ((x1 + x2) / 2, y2)


# Geometric centre of a box.
def center(box):
    """Applied to the helmet box and to the number plate box, which form the
    opposite side of each match. These objects carry no directional bias of
    their own, so their plain centre is the correct reference point."""
    # All four edges are required to compute a true centre.
    x1, y1, x2, y2 = box
    # Midpoint taken horizontally and vertically.
    return ((x1 + x2) / 2, (y1 + y2) / 2)


# Intersection over Union between two boxes given as [x1, y1, x2, y2].
def iou(box_a, box_b):
    """Returns the shared area divided by the combined area, which is 0.0 for
    boxes that do not touch and 1.0 for boxes that coincide exactly."""
    # Corner coordinates of the first box.
    ax1, ay1, ax2, ay2 = box_a
    # Corner coordinates of the second box.
    bx1, by1, bx2, by2 = box_b
    # Top-left corner of the shared region
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    # Bottom-right corner of the shared region
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    # Clamp both dimensions at zero. Boxes that do not touch produce an
    # inverted rectangle whose two negative dimensions would otherwise
    # multiply into a positive area.
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    # Area shared by the two boxes.
    inter = iw * ih
    # Total area of the first box.
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    # Total area of the second box.
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    # Combined area, subtracting the shared region that both areas included.
    union = area_a + area_b - inter
    # Guard against division by zero for degenerate boxes enclosing no area.
    return inter / union if union > 0 else 0.0
