"""temporal smoothing and violation decision."""

from collections import Counter
from config import MIN_OBSERVATIONS, MIN_CONFIDENCE


def smooth_helmet_status(observations, rider_id, min_observations=None):
    """observations[rider_id] is a list of (status, model_confidence)
    tuples - one per frame this rider was matched to a helmet/no_helmet
    detection. Returns (majority_status, avg_model_confidence), where
    avg_model_confidence is the average of the YOLO model's own
    confidence scores, taken only over the frames that agree with
    the majority status (not a vote-agreement fraction)."""
    min_observations = min_observations if min_observations is not None else MIN_OBSERVATIONS
    votes = observations.get(rider_id, [])
    if len(votes) < min_observations:
        return None

    statuses = [v[0] for v in votes]
    status, _ = Counter(statuses).most_common(1)[0]

    agreeing_confs = [v[1] for v in votes if v[0] == status and v[1] is not None]
    avg_confidence = sum(agreeing_confs) / len(agreeing_confs) if agreeing_confs else None

    return status, avg_confidence


def decide_violation(smoothed_result, min_confidence=None):
    min_confidence = min_confidence if min_confidence is not None else MIN_CONFIDENCE
    if smoothed_result is None:
        return None
    status, confidence = smoothed_result
    if confidence is None or confidence < min_confidence:
        return None
    return status == "no_helmet"
