"""Process-local review requests shared by observe and code submitters."""

import threading
import time


class ReviewRegistry:
    """A lock-protected pending queue; decisions are one-way and first-wins."""

    def __init__(self):
        self._lock = threading.Lock()
        self._next = 0
        self._records = {}

    def create(self, code, intent, origin):
        with self._lock:
            self._next += 1
            review_id = f"review-{self._next}"
            record = {
                "review_id": review_id,
                "code": code,
                "intent": intent,
                "origin": origin,
                "created": time.time(),
                "state": "pending",
            }
            self._records[review_id] = record
            return dict(record)

    def get(self, review_id):
        with self._lock:
            record = self._records.get(review_id)
            return dict(record) if record else None

    def pending(self):
        with self._lock:
            return [dict(r) for r in self._records.values() if r["state"] == "pending"]

    def decide(self, review_id, decision):
        with self._lock:
            record = self._records.get(review_id)
            if record is None:
                return None, False
            if record["state"] != "pending":
                return dict(record), False
            record["state"] = decision
            return dict(record), True

    def begin_submission(self, review_id):
        """Claim an approved review for one submitter, or return its current state."""
        with self._lock:
            record = self._records.get(review_id)
            if record is None:
                return None, False
            if record["state"] == "approved":
                record["state"] = "submitting"
                return dict(record), True
            return dict(record), False

    def finish_submission(self, review_id, result):
        with self._lock:
            record = self._records.get(review_id)
            if record is not None:
                record["state"] = "submitted"
                record["result"] = result

    def clear(self):
        with self._lock:
            self._next = 0
            self._records.clear()


registry = ReviewRegistry()
