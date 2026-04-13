import numpy as np

RISE_RATE         = 0.25
DECAY_RATE        = 0.25
STABLE_THRESHOLD  = 0.45   # FIX: lowered from 0.55
                            # 0.55 needed ~4 consecutive frames to confirm a member.
                            # 0.45 needs ~3 frames — still stable but counts
                            # edge members who are slightly off the MST spine.
FORGET_THRESHOLD  = 0.05


class TemporalSmoother:
    """
    Per-person EMA queue confidence score.

    Score rises toward 1.0 when person is in the queue graph,
    decays toward 0.0 when they leave. Asymmetric rates reflect
    that people join queues faster than they leave them.

    Update rule:
        in queue  : score = score + rise_rate  * (1 - score)
        not in queue: score = score * (1 - decay_rate)

    Equivalent to a single-gate GRU where the hidden state is
    the confidence score.
    """

    def __init__(self,
                 rise_rate=RISE_RATE,
                 decay_rate=DECAY_RATE,
                 stable_threshold=STABLE_THRESHOLD,
                 forget_threshold=FORGET_THRESHOLD):

        self.rise_rate        = rise_rate
        self.decay_rate       = decay_rate
        self.stable_threshold = stable_threshold
        self.forget_threshold = forget_threshold
        self._scores          = {}

    def update(self, current_queue_ids):
        """
        Call once per frame with the set of track IDs currently in the queue.
        Returns dict of tid -> score for all tracked persons.
        """
        current_queue_ids = set(current_queue_ids)

        for tid in current_queue_ids:
            prev = self._scores.get(tid, 0.0)
            self._scores[tid] = prev + self.rise_rate * (1.0 - prev)

        for tid in list(self._scores.keys()):
            if tid not in current_queue_ids:
                self._scores[tid] *= (1.0 - self.decay_rate)
                if self._scores[tid] < self.forget_threshold:
                    del self._scores[tid]

        return dict(self._scores)

    def get_stable_ids(self):
        """Track IDs whose confidence is above stable_threshold."""
        return [tid for tid, score in self._scores.items()
                if score >= self.stable_threshold]

    def get_score(self, tid):
        return self._scores.get(tid, 0.0)

    def get_all_scores(self):
        return dict(self._scores)

    def remove(self, tid):
        self._scores.pop(tid, None)

    def reset(self):
        self._scores.clear()