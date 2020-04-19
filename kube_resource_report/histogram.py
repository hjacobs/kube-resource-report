"""
Implementation of exponential histogram from Vertical Pod Autoscaler (VPA).

VPA source: https://github.com/kubernetes/autoscaler/tree/master/vertical-pod-autoscaler
"""
import collections
import math
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

    num_buckets: int
    min_bucket: int
    max_bucket: int
    bucket_weights: Dict[int, float]
    total_weight: float

    def __init__(
        self, max_value: float, first_bucket_size: float, ratio: float, half_life: float
    ):
        self.max_value = max_value
        self.first_bucket_size = first_bucket_size
        self.ratio = ratio
        num_buckets = (
            int(
                math.ceil(
                    math.log(max_value * (ratio - 1) / first_bucket_size + 1, ratio)
                )
            )
            + 1
        )
        self.num_buckets = num_buckets
        self.min_bucket = num_buckets - 1
        self.max_bucket = 0
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

        if bucket < self.min_bucket and self.bucket_weights[bucket] >= EPSILON:
            self.min_bucket = bucket
        if bucket > self.max_bucket and self.bucket_weights[bucket] >= EPSILON:
            self.max_bucket = bucket

    def update_min_and_max_bucket(self):
        lastBucket = self.num_buckets - 1
        while (
            self.bucket_weights[self.min_bucket] < EPSILON
            and self.min_bucket < lastBucket
        ):
            self.min_bucket += 1
        while self.bucket_weights[self.max_bucket] < EPSILON and self.max_bucket > 0:
            self.max_bucket -= 1

    def scale(self, factor: float):
        for i, v in self.bucket_weights.items():
            self.bucket_weights[i] = v * factor
        self.total_weight *= factor
        # Some buckets might become empty (weight < epsilon), so adjust min and max buckets.
        self.update_min_and_max_bucket()

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
        if bucket >= self.num_buckets:
            return self.num_buckets - 1
        return bucket

    def get_bucket_start(self, bucket):
        """Return the start of the bucket with given index."""
        if bucket == 0:
            return 0.0
        return self.first_bucket_size * ((self.ratio ** bucket) - 1) / (self.ratio - 1)

    def get_percentile(self, percentile: float):
        if self.is_empty():
            return 0.0
        partial_sum = 0.0
        threshold = percentile * self.total_weight
        bucket = self.min_bucket
        for i, w in sorted(self.bucket_weights.items()):
            if i >= self.max_bucket:
                bucket = i
                break
            elif i >= bucket:
                partial_sum += w
                if partial_sum >= threshold:
                    bucket = i
                    break
        if bucket < self.num_buckets - 1:
            # Return the end of the bucket.
            return self.get_bucket_start(bucket + 1)
        # Return the start of the last bucket (note that the last bucket
        # doesn't have an upper bound).
        return self.get_bucket_start(bucket)

    def is_empty(self) -> bool:
        return self.bucket_weights[self.min_bucket] < EPSILON

    def get_checkpoint(self):
        return {
            "total_weight": self.total_weight,
            "bucket_weights": {
                b: w for b, w in self.bucket_weights.items() if w > EPSILON
            },
            "reference_time": self.reference_time,
        }

    def from_checkpoint(self, checkpoint):
        total_weight = checkpoint["total_weight"]
        if total_weight < 0.0:
            raise ValueError(
                f"Invalid checkpoint data with negative weight {total_weight}"
            )
        for bucket_str, weight in checkpoint["bucket_weights"].items():
            # JSON keys are always strings, convert to int
            bucket = int(bucket_str)
            if bucket < self.min_bucket:
                self.min_bucket = bucket
            if bucket > self.max_bucket:
                self.max_bucket = bucket
            self.bucket_weights[bucket] += weight
        self.total_weight += total_weight
        self.reference_time = checkpoint["reference_time"]
