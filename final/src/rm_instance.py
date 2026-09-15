
"""
The pipeline's memory: the RMInstance data model and its manager.

Detection and association operate one frame at a time and retain nothing.
A violation, however, is a property of a rider accumulated across the whole
period they were visible. These two classes hold that accumulating record.
"""

# Counter tallies repeated values; deque provides a fixed-length rolling window.
from collections import Counter, deque

# One record per rider, holding everything the pipeline learns about them.
class RMInstance:
    # Create the record the first time a rider is paired with a motorcycle.
    def __init__(self, motorcycle_id, rider_id, buffer_size=5):
        # Tracking identifier of the rider, which also keys this record.
        self.rider_id = rider_id
        # Current best estimate of which motorcycle this rider is travelling on.
        self.motorcycle_id = motorcycle_id
        # Rolling window of recent motorcycle assignments. Appending beyond the
        # maximum length automatically discards the oldest entry.
        self._moto_votes = deque(maxlen=buffer_size)
        # Seed the window with the assignment that created this record.
        self._moto_votes.append(motorcycle_id)
        # Helmet verdict, written later by the temporal stage.
        self.helmet_status = None
        # Averaged detector confidence supporting that verdict.
        self.confidence = None
        # Recognised plate text, reserved for a future text recognition step.
        self.plate_number = None
        # Bounding box of the matched number plate, supplied by association.
        self.plate_box = None
        # Final verdict: True, False, or None when evidence is insufficient.
        self.is_violation = None

    # Record one further frame's opinion of which motorcycle this rider is on.
    def vote_motorcycle(self, motorcycle_id):
        # Append the new observation, automatically discarding the oldest.
        self._moto_votes.append(motorcycle_id)
        # The majority of the recent window becomes the accepted answer, so a
        # single misassigned frame cannot change the result on its own.
        self.motorcycle_id = Counter(self._moto_votes).most_common(1)[0][0]

    # Readable one-line summary, used when writing the final report to disk.
    def __repr__(self):
        # Report the identifiers alongside every decision made so far.
        return (f"RMInstance(moto={self.motorcycle_id}, rider={self.rider_id}, "
                f"helmet={self.helmet_status}, plate={self.plate_number}, "
                f"violation={self.is_violation})")


# Owns every RMInstance and returns the correct one on request.
class RMInstanceManager:
    # Start with no records; they are created as riders appear in the video.
    def __init__(self):
        # Keyed by rider identifier, because the violation belongs to the rider.
        self.instances_by_rider = {}

    # Return this rider's record, creating it the first time they are seen.
    def get_or_create(self, rider_id, motorcycle_id):
        # A rider not encountered before requires a new record.
        if rider_id not in self.instances_by_rider:
            # Create the record and store it against the rider's identifier.
            self.instances_by_rider[rider_id] = RMInstance(motorcycle_id=motorcycle_id, rider_id=rider_id)
        # A rider already known simply contributes one further vote.
        else:
            # Add this frame's motorcycle assignment to the rolling window.
            self.instances_by_rider[rider_id].vote_motorcycle(motorcycle_id)
        # In either case, return the record so the caller can read or update it.
        return self.instances_by_rider[rider_id]

    # Every record currently held, returned as a list for reporting.
    def all_instances(self):
        # Dictionary values converted to a list for stable iteration.
        return list(self.instances_by_rider.values())

    # Aggregate counts describing the run as a whole.
    def summary(self):
        # Total number of distinct rider tracking identifiers encountered.
        total = len(self.instances_by_rider)
        # Riders for which the temporal stage reached a helmet verdict.
        with_helmet_status = sum(1 for i in self.instances_by_rider.values() if i.helmet_status)
        # Riders confirmed as violations.
        violations = sum(1 for i in self.instances_by_rider.values() if i.is_violation)
        # Returned as a dictionary so the caller can print or store it directly.
        return {"total_instances": total, "confident_helmet_status": with_helmet_status, "confirmed_violations": violations}
