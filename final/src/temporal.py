
"""
Temporal smoothing and the violation decision.

A single frame cannot be trusted, because a rider may be blurred, partly
occluded, or captured at an unhelpful angle. These two functions combine many
frames into one verdict, and decline to answer when evidence is insufficient.
"""

# Counter is used to identify the status that occurs most frequently.
from collections import Counter
# Evidence thresholds are defined centrally rather than inside this module.
from src.config import MIN_OBSERVATIONS, MIN_CONFIDENCE


# Summarise every observation recorded for one rider.
def smooth_helmet_status(observations, rider_id, min_observations=None):
    """observations[rider_id] is a list of (status, model_confidence) tuples,
    with one entry per frame in which that rider was matched to a head.

    Returns (majority_status, average_model_confidence). The average is taken
    over the agreeing frames only, and is the detector's own confidence rather
    than a vote-agreement fraction.
    """
    # Apply the configured minimum unless the caller supplies another.
    min_observations = min_observations if min_observations is not None else MIN_OBSERVATIONS
    # Every observation recorded for this rider, or an empty list if none exist.
    votes = observations.get(rider_id, [])
    # Too little evidence exists to state anything at all.
    if len(votes) < min_observations:
        # None is a genuine third outcome: the system declines to decide.
        return None

    # Discard the confidence values and retain only the status labels.
    statuses = [v[0] for v in votes]
    # The label occurring most often becomes this rider's smoothed status.
    status, _ = Counter(statuses).most_common(1)[0]

    # Collect the detector's scores from the frames agreeing with the majority.
    agreeing_confs = [v[1] for v in votes if v[0] == status and v[1] is not None]
    # Average those scores, reporting None when no usable score was recorded.
    avg_confidence = sum(agreeing_confs) / len(agreeing_confs) if agreeing_confs else None

    # Return the status together with the confidence supporting it.
    return status, avg_confidence


# Convert a smoothed summary into a final verdict.
def decide_violation(smoothed_result, min_confidence=None):
    """Returns True for a confirmed violation, False for a confirmed helmet,
    and None when the system declines to judge. All three are distinct."""
    # Apply the configured confidence floor unless the caller overrides it.
    min_confidence = min_confidence if min_confidence is not None else MIN_CONFIDENCE
    # Smoothing already refused to answer, usually for want of observations.
    if smoothed_result is None:
        # Pass that refusal through unchanged.
        return None
    # Separate the majority status from the confidence supporting it.
    status, confidence = smoothed_result
    # Reject a verdict that no score supports, or one the detector was unsure of.
    if confidence is None or confidence < min_confidence:
        # None again, here meaning insufficient confidence rather than compliance.
        return None
    # Only a confirmed absence of a helmet constitutes a violation.
    return status == "no_helmet"
