"""RMInstance data model + manager."""

from collections import Counter, deque


class RMInstance:
    def __init__(self, motorcycle_id, rider_id, buffer_size=5):
        self.rider_id = rider_id
        self.motorcycle_id = motorcycle_id
        self._moto_votes = deque(maxlen=buffer_size)
        self._moto_votes.append(motorcycle_id)
        self.helmet_status = None
        self.confidence = None
        self.plate_number = None
        self.plate_box = None
        self.is_violation = None

    def vote_motorcycle(self, motorcycle_id):
        self._moto_votes.append(motorcycle_id)
        self.motorcycle_id = Counter(self._moto_votes).most_common(1)[0][0]

    def __repr__(self):
        return (f"RMInstance(moto={self.motorcycle_id}, rider={self.rider_id}, "
                f"helmet={self.helmet_status}, plate={self.plate_number}, "
                f"violation={self.is_violation})")


class RMInstanceManager:
    def __init__(self):
        self.instances_by_rider = {}

    def get_or_create(self, rider_id, motorcycle_id):
        if rider_id not in self.instances_by_rider:
            self.instances_by_rider[rider_id] = RMInstance(motorcycle_id=motorcycle_id, rider_id=rider_id)
        else:
            self.instances_by_rider[rider_id].vote_motorcycle(motorcycle_id)
        return self.instances_by_rider[rider_id]

    def all_instances(self):
        return list(self.instances_by_rider.values())

    def summary(self):
        total = len(self.instances_by_rider)
        with_helmet_status = sum(1 for i in self.instances_by_rider.values() if i.helmet_status)
        violations = sum(1 for i in self.instances_by_rider.values() if i.is_violation)
        return {"total_instances": total, "confident_helmet_status": with_helmet_status, "confirmed_violations": violations}