"""
Implementation of exponential histogram from Vertical Pod Autoscaler (VPA).

VPA source: https://github.com/kubernetes/autoscaler/tree/master/vertical-pod-autoscaler
"""
import collections
import math
from typing import Any
from typing import Dict


EPSILON = 0.001
MAX_CHECKPOINT_WEIGHT = 10000

# When the decay factor exceeds 2^maxDecayExponent the histogram is
# renormalized by shifting the decay start time forward.
MAX_DECAY_EXPONENT = 100


class DecayingExponentialHistogram:

    """
    Histogram represents an approximate distribution of some variable.

    https:#github.com/kubernetes/autoscaler/blob/master/vertical-pod-autoscaler/pkg/recommender/util/histogram.go
    """

    bucket_weights: Dict[int, float]
    total_weight: float

    def __init__(self, first_bucket_size: float, ratio: float, half_life: float):
        self.first_bucket_size = first_bucket_size
        self.ratio = ratio
        self.bucket_weights = collections.defaultdict(float)
        self.total_weight = 0.0
        self.half_life = half_life
        self.reference_time = 0

    def add_sample(self, value: float, weight: float, time: float):
        # Max timestamp before the exponent grows too large.
        max_allowed_time = self.reference_time + (self.half_life * MAX_DECAY_EXPONENT)
        if time > max_allowed_time:
            # The exponent has grown too large. Renormalize the histogram by
            # shifting the referenceTimestamp to the current timestamp and rescaling
            # the weights accordingly.
            self.shift_reference_time(time)
        decay_factor = 2 ** ((time - self.reference_time) / self.half_life)
        new_weight = weight * decay_factor
        bucket = self.find_bucket(value)
        self.bucket_weights[bucket] += new_weight
        self.total_weight += new_weight

    def scale(self, factor: float):
        for i, v in self.bucket_weights.items():
            self.bucket_weights[i] = v * factor
        self.total_weight *= factor

    def shift_reference_time(self, new_reference_time: float):
        # Make sure the decay start is an integer multiple of halfLife.
        new_reference_time = int(
            (new_reference_time // self.half_life) * self.half_life
        )
        exponent = round((self.reference_time - new_reference_time) / self.half_life)
        self.scale(2 ** exponent)  # Scale all weights by 2^exponent.
        self.reference_time = new_reference_time

    def find_bucket(self, value: float) -> int:
        if value < self.first_bucket_size:
            return 0
        bucket = int(
            math.log(value * (self.ratio - 1) / self.first_bucket_size + 1, self.ratio)
        )
        return bucket

    def get_bucket_start(self, bucket: int) -> float:
        """Return the start of the bucket with given index."""
        if bucket == 0:
            return 0.0
        return self.first_bucket_size * ((self.ratio ** bucket) - 1) / (self.ratio - 1)

    def get_percentile(self, percentile: float) -> float:
        if self.is_empty():
            return 0.0
        partial_sum = 0.0
        threshold = percentile * self.total_weight
        bucket = None
        for i, w in sorted(self.bucket_weights.items()):
            partial_sum += w
            if partial_sum >= threshold:
                bucket = i
                break
        if bucket is None:
            # last bucket
            bucket = max(self.bucket_weights.keys())
        # Return the end of the bucket.
        return self.get_bucket_start(bucket + 1)

    def is_empty(self) -> bool:
        return (
            len(self.bucket_weights) == 0 or max(self.bucket_weights.values()) < EPSILON
        )

    def get_checkpoint(self) -> Dict[str, Any]:
        return {
            "total_weight": self.total_weight,
            "bucket_weights": {
                b: w for b, w in self.bucket_weights.items() if w > EPSILON
            },
            "reference_time": self.reference_time,
        }

    def from_checkpoint(self, checkpoint: Dict[str, Any]):
        total_weight = checkpoint["total_weight"]
        if total_weight < 0.0:
            raise ValueError(
                f"Invalid checkpoint data with negative weight {total_weight}"
            )
        for bucket_str, weight in checkpoint["bucket_weights"].items():
            # JSON keys are always strings, convert to int
            bucket = int(bucket_str)
            self.bucket_weights[bucket] += weight
        self.total_weight += total_weight
        self.reference_time = checkpoint["reference_time"]
