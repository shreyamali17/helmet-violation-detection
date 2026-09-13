"""Hungarian-algorithm association functions."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from geometry import x_center, top_center, bottom_center, center, iou
from config import RIDER_MOTO_MAX_COST, HEAD_RIDER_MAX_COST, PLATE_MOTO_MAX_COST


def build_cost_matrix_xdist(riders, motorcycles):
    """ORIGINAL method: cost = horizontal pixel distance between box centers."""
    rider_ids = list(riders.keys())
    moto_ids = list(motorcycles.keys())
    cost_matrix = np.zeros((len(rider_ids), len(moto_ids)))
    for i, rid in enumerate(rider_ids):
        rx = x_center(riders[rid])
        for j, mid in enumerate(moto_ids):
            cost_matrix[i, j] = abs(rx - x_center(motorcycles[mid]))
    return cost_matrix, rider_ids, moto_ids


def build_cost_matrix_iou(riders, motorcycles):
    """NEW method: cost = 1 - IoU between rider and motorcycle boxes."""
    rider_ids = list(riders.keys())
    moto_ids = list(motorcycles.keys())
    cost_matrix = np.zeros((len(rider_ids), len(moto_ids)))
    for i, rid in enumerate(rider_ids):
        for j, mid in enumerate(moto_ids):
            cost_matrix[i, j] = 1.0 - iou(riders[rid], motorcycles[mid])
    return cost_matrix, rider_ids, moto_ids


# default thresholds per method - xdist uses pixels, iou uses 1-min_IoU
DEFAULT_MAX_COST = {
    "xdist": 300,
    "iou": 0.9,
}


def associate_riders_to_motorcycles(riders, motorcycles, method="iou", max_cost=None):
    """
    method: "xdist" (original) or "iou" (new).
    max_cost: override the default threshold for the chosen method.
    """
    max_cost = max_cost if max_cost is not None else DEFAULT_MAX_COST[method]
    if not riders or not motorcycles:
        return []

    if method == "xdist":
        cost_matrix, rider_ids, moto_ids = build_cost_matrix_xdist(riders, motorcycles)
    elif method == "iou":
        cost_matrix, rider_ids, moto_ids = build_cost_matrix_iou(riders, motorcycles)
    else:
        raise ValueError(f"Unknown method: {method}")

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return [(rider_ids[r], moto_ids[c]) for r, c in zip(row_ind, col_ind) if cost_matrix[r, c] <= max_cost]


def associate_heads_to_riders(riders, helmets, no_helmets, helmet_confs=None, no_helmet_confs=None, max_cost=None):
    """Returns {rider_id: (status, model_confidence)}. model_confidence
    is the YOLO detection's own confidence score for the matched
    helmet/no_helmet box - None if confidence dicts aren't provided
    (keeps this function backward-compatible)."""
    max_cost = max_cost if max_cost is not None else HEAD_RIDER_MAX_COST
    helmet_confs = helmet_confs or {}
    no_helmet_confs = no_helmet_confs or {}

    heads, head_type, head_conf = {}, {}, {}
    for hid, box in helmets.items():
        heads[hid] = box; head_type[hid] = "helmet"; head_conf[hid] = helmet_confs.get(hid)
    for hid, box in no_helmets.items():
        heads[hid] = box; head_type[hid] = "no_helmet"; head_conf[hid] = no_helmet_confs.get(hid)

    if not riders or not heads:
        return {}
    rider_ids, head_ids = list(riders.keys()), list(heads.keys())
    cost_matrix = np.zeros((len(rider_ids), len(head_ids)))
    for i, rid in enumerate(rider_ids):
        rx, ry = top_center(riders[rid])
        for j, hid in enumerate(head_ids):
            hx, hy = center(heads[hid])
            cost_matrix[i, j] = ((rx-hx)**2 + (ry-hy)**2) ** 0.5
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return {
        rider_ids[r]: (head_type[head_ids[c]], head_conf[head_ids[c]])
        for r, c in zip(row_ind, col_ind) if cost_matrix[r, c] <= max_cost
    }


def associate_plates_to_motorcycles(motorcycles, plates, max_cost=None):
    max_cost = max_cost if max_cost is not None else PLATE_MOTO_MAX_COST
    if not motorcycles or not plates:
        return {}
    moto_ids, plate_ids = list(motorcycles.keys()), list(plates.keys())
    cost_matrix = np.zeros((len(moto_ids), len(plate_ids)))
    for i, mid in enumerate(moto_ids):
        mx, my = bottom_center(motorcycles[mid])
        for j, pid in enumerate(plate_ids):
            px, py = center(plates[pid])
            cost_matrix[i, j] = ((mx-px)**2 + (my-py)**2) ** 0.5
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return {moto_ids[r]: plate_ids[c] for r, c in zip(row_ind, col_ind) if cost_matrix[r, c] <= max_cost}