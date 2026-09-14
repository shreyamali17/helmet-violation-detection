
"""
Stage 2: build relationships between the objects detection reported.

Three matches are produced here, all sharing the same three-step machinery:
construct a cost matrix, solve it optimally with the Hungarian algorithm,
then discard any pairing that remains too expensive to accept.
"""

# NumPy supplies the two-dimensional array used for every cost matrix.
import numpy as np
# SciPy supplies the Hungarian algorithm, which solves assignment optimally.
from scipy.optimize import linear_sum_assignment

# Geometric metrics from the geometry module.
from src.geometry import x_center, top_center, bottom_center, center, iou
# Rejection ceilings, defined centrally rather than repeated in this file.
from src.config import RIDER_MOTO_MAX_COST_IOU, RIDER_MOTO_MAX_COST_XDIST, HEAD_RIDER_MAX_COST, PLATE_MOTO_MAX_COST


# Original cost definition: horizontal pixel distance between box centres.
def build_cost_matrix_xdist(riders, motorcycles):
    """Retained for comparison only. Because it ignores vertical position
    entirely, it rates a distant object as close whenever the horizontal
    positions happen to align."""
    # Fixed ordering of rider identifiers, mapping each row back to a rider.
    rider_ids = list(riders.keys())
    # Fixed ordering of motorcycle identifiers, mapping each column back.
    moto_ids = list(motorcycles.keys())
    # Allocate one row per rider and one column per motorcycle.
    cost_matrix = np.zeros((len(rider_ids), len(moto_ids)))
    # Walk every rider, retaining its row index.
    for i, rid in enumerate(rider_ids):
        # Horizontal midpoint of this rider, computed once per row.
        rx = x_center(riders[rid])
        # Walk every motorcycle, retaining its column index.
        for j, mid in enumerate(moto_ids):
            # Cost is the absolute horizontal gap between the two midpoints.
            cost_matrix[i, j] = abs(rx - x_center(motorcycles[mid]))
    # Return the matrix together with the orderings needed to decode it.
    return cost_matrix, rider_ids, moto_ids


# Current cost definition: one minus the overlap between the two boxes.
def build_cost_matrix_iou(riders, motorcycles):
    """Overlap preserves the whole rectangle, so a rider and a motorcycle that
    never touch receive the maximum cost of 1.0 regardless of alignment."""
    # Fixed ordering of rider identifiers for the matrix rows.
    rider_ids = list(riders.keys())
    # Fixed ordering of motorcycle identifiers for the matrix columns.
    moto_ids = list(motorcycles.keys())
    # Allocate the cost matrix of the required shape.
    cost_matrix = np.zeros((len(rider_ids), len(moto_ids)))
    # Walk every rider row.
    for i, rid in enumerate(rider_ids):
        # Walk every motorcycle column.
        for j, mid in enumerate(moto_ids):
            # Subtracting from one converts high overlap into low cost, which
            # is what the minimising solver requires.
            cost_matrix[i, j] = 1.0 - iou(riders[rid], motorcycles[mid])
    # Return the matrix with its row and column orderings.
    return cost_matrix, rider_ids, moto_ids

# Pair each rider with the motorcycle they are travelling on.
def associate_riders_to_motorcycles(riders, motorcycles, method="iou", max_cost=None):
    """method selects "xdist" (baseline) or "iou" (current default).
    max_cost overrides the default ceiling for the chosen method."""
    # The solver raises on an empty matrix, and frames without riders are routine.
    if not riders or not motorcycles:
        # No pairing is possible, so report an empty list and continue.
        return []

    # Construct the matrix from horizontal distance when that method is chosen.
    if method == "xdist":
        max_cost = RIDER_MOTO_MAX_COST_XDIST
        cost_matrix, rider_ids, moto_ids = build_cost_matrix_xdist(riders, motorcycles)
    # Construct the matrix from box overlap for the default method.
    elif method == "iou":
        max_cost = RIDER_MOTO_MAX_COST_IOU
        cost_matrix, rider_ids, moto_ids = build_cost_matrix_iou(riders, motorcycles)
    # Any other value indicates a programming error and is reported at once.
    else:
        raise ValueError(f"Unknown method: {method}")

    # Solve for the pairing set with the lowest total cost across all riders.
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # Retain only those pairings whose individual cost is acceptable, because
    # the best available match is not necessarily a good match.
    return [(rider_ids[r], moto_ids[c]) for r, c in zip(row_ind, col_ind) if cost_matrix[r, c] <= max_cost]

# Determine, for each rider, whether their nearest head detection is helmeted.
def associate_heads_to_riders(riders, helmets, no_helmets, helmet_confs=None, no_helmet_confs=None, max_cost=None):
    """Returns {rider_id: (status, model_confidence)}. model_confidence is the
    detector's own score for the matched head box, or None when the confidence
    dictionaries are not supplied, which keeps older callers working."""
    # Apply the configured pixel ceiling unless the caller overrides it.
    max_cost = max_cost if max_cost is not None else HEAD_RIDER_MAX_COST
    # Treat a missing helmet confidence table as an empty one.
    helmet_confs = helmet_confs or {}
    # Treat a missing no-helmet confidence table the same way.
    no_helmet_confs = no_helmet_confs or {}

    # Three parallel tables holding the head boxes, their class, and their score.
    heads, head_type, head_conf = {}, {}, {}
    # Add every helmet detection into the shared pool of candidate heads.
    for hid, box in helmets.items():
        # Record the box, mark it as a helmet, and carry its confidence across.
        heads[hid] = box
        head_type[hid] = "helmet"
        head_conf[hid] = helmet_confs.get(hid)
    # Add every no-helmet detection into the same pool, because a rider has
    # exactly one head and the two classes must therefore compete directly.
    for hid, box in no_helmets.items():
        # Record the box, mark it as a no-helmet, and carry its confidence across.
        heads[hid] = box
        head_type[hid] = "no_helmet"
        head_conf[hid] = no_helmet_confs.get(hid)

    # No assignment is possible when either side of the match is empty.
    if not riders or not heads:
        # An empty result is a valid outcome for a frame with no visible heads.
        return {}
    # Fixed orderings so matrix indices can be decoded back into identifiers.
    rider_ids, head_ids = list(riders.keys()), list(heads.keys())
    # Allocate one row per rider and one column per candidate head.
    cost_matrix = np.zeros((len(rider_ids), len(head_ids)))
    # Walk every rider row.
    for i, rid in enumerate(rider_ids):
        # The rider's head position is the midpoint of the top edge of the box.
        rx, ry = top_center(riders[rid])
        # Walk every candidate head column.
        for j, hid in enumerate(head_ids):
            # A head box carries no directional bias, so its centre is used.
            hx, hy = center(heads[hid])
            # Cost is the straight-line pixel distance between the two points.
            cost_matrix[i, j] = ((rx-hx)**2 + (ry-hy)**2) ** 0.5 
    # Solve for the globally cheapest set of rider-to-head pairings.
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # Return the accepted pairings, reporting both the class and its confidence.
    return {
        rider_ids[r]: (head_type[head_ids[c]], head_conf[head_ids[c]])
        for r, c in zip(row_ind, col_ind) if cost_matrix[r, c] <= max_cost
    }


# Pair each motorcycle with its number plate.
def associate_plates_to_motorcycles(motorcycles, plates, max_cost=None):
    """Structurally identical to head matching, except the motorcycle is
    measured from its bottom edge because plates are mounted low."""
    # Apply the configured pixel ceiling unless the caller overrides it.
    max_cost = max_cost if max_cost is not None else PLATE_MOTO_MAX_COST
    # No assignment is possible when either side of the match is empty.
    if not motorcycles or not plates:
        # Report no matches and allow the pipeline to continue.
        return {}
    # Fixed orderings for the rows, which are motorcycles, and the columns,
    # which are number plates.
    moto_ids, plate_ids = list(motorcycles.keys()), list(plates.keys())
    # Allocate the cost matrix of the required shape.
    cost_matrix = np.zeros((len(moto_ids), len(plate_ids)))
    # Walk every motorcycle row.
    for i, mid in enumerate(moto_ids):
        # The expected plate position is the midpoint of the bottom edge.
        mx, my = bottom_center(motorcycles[mid])
        # Walk every plate column.
        for j, pid in enumerate(plate_ids):
            # The plate itself is represented by its geometric centre.
            px, py = center(plates[pid])
            # Cost is the straight-line pixel distance between the two points.
            cost_matrix[i, j] = ((mx-px)**2 + (my-py)**2) ** 0.5
    # Solve for the cheapest overall set of motorcycle-to-plate pairings.
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # Return only the pairings that fall within the distance ceiling.
    return {moto_ids[r]: plate_ids[c] for r, c in zip(row_ind, col_ind) if cost_matrix[r, c] <= max_cost}
