"""
Recommend CPU/memory resource values for Pods based on some (hard-coded) aggregation logic with decaying exponential histograms.

The histogram decay algorithm is similar to VerticalPodAutoscaler (VPA), i.e. resource metrics are weighted half after one day (24 hours).
"""
import collections
import json
import logging
import time
from pathlib import Path

from .histogram import DecayingExponentialHistogram

CHECKPOINT_FILE_NAME = "checkpoint.json"

CPU_PERCENTILE = 0.9
CPU_SAFETY_MARGIN_FACTOR = 1.15
MEMORY_PERCENTILE = 1.0
MEMORY_SAFETY_MARGIN_FACTOR = 1.15
MIN_SAMPLE_WEIGHT = 0.1

ONE_DAY = 3600 * 24
CPU_HISTOGRAM_MIN_VALUE = 0.001  # 1m
CPU_HISTOGRAM_DECAY_HALF_LIFE = ONE_DAY
MEMORY_HISTOGRAM_MIN_VALUE = 10.0 * 1024 * 1024  # 10 MiB
MEMORY_HISTOGRAM_DECAY_HALF_LIFE = ONE_DAY

AGGREGATION_KEY_LENGTH = 4

# delete checkpoint files without updates after 14 days
MAX_STALE_FILE_AGE_SECONDS = 14 * 24 * 3600

logger = logging.getLogger(__name__)


def new_cpu_histogram():
    # CPU histograms use exponential bucketing scheme with the smallest bucket
    # size of 0.001 core, max of 100.0 cores
    return DecayingExponentialHistogram(
        CPU_HISTOGRAM_MIN_VALUE, 1.05, CPU_HISTOGRAM_DECAY_HALF_LIFE
    )


def new_memory_histogram():
    # Memory histograms use exponential bucketing scheme with the smallest
    # bucket size of 10MB, max of 1TB
    return DecayingExponentialHistogram(
        MEMORY_HISTOGRAM_MIN_VALUE, 1.05, MEMORY_HISTOGRAM_DECAY_HALF_LIFE
    )


class Recommender:
    def __init__(self):
        self.cpu_histograms = collections.defaultdict(new_cpu_histogram)
        self.memory_histograms = collections.defaultdict(new_memory_histogram)
        self.first_sample_times = collections.defaultdict(int)
        self.total_sample_counts = collections.defaultdict(int)
        self._stale_aggregation_keys = set()
        self._updated_aggregation_keys = set()

    def get_aggregation_key(self, namespace: str, name: str, pod: dict):
        aggregation_key = pod.get("aggregation_key")
        if aggregation_key and len(aggregation_key) == AGGREGATION_KEY_LENGTH:
            # custom aggregation key was set for this Pod
            return aggregation_key
        aggregation_key = (
            namespace,
            pod["application"],
            pod["component"],
            ",".join(sorted(pod["container_names"])),
        )
        return aggregation_key

    def update_pods(self, pods: dict):
        pods_by_aggregation_key = collections.defaultdict(list)
        now = time.time()

        for namespace_name, pod in pods.items():
            namespace, name = namespace_name
            aggregation_key = self.get_aggregation_key(namespace, name, pod)
            pods_by_aggregation_key[aggregation_key].append(pod)

            cpu_usage = pod["usage"]["cpu"]
            memory_usage = pod["usage"]["memory"]

            if cpu_usage <= 0 and memory_usage <= 0:
                # ignore pods without usage metrics
                continue

            cpu_histogram = self.cpu_histograms[aggregation_key]
            cpu_histogram.add_sample(
                cpu_usage, max(pod["requests"]["cpu"], MIN_SAMPLE_WEIGHT), now
            )

            memory_histogram = self.memory_histograms[aggregation_key]
            memory_histogram.add_sample(memory_usage, 1.0, now)
            self.total_sample_counts[aggregation_key] += 1
            if aggregation_key not in self.first_sample_times:
                self.first_sample_times[aggregation_key] = int(now)
            self._updated_aggregation_keys.add(aggregation_key)

        for aggregation_key, pods_ in pods_by_aggregation_key.items():
            cpu_histogram = self.cpu_histograms[aggregation_key]
            cpu_recommendation = (
                cpu_histogram.get_percentile(CPU_PERCENTILE) * CPU_SAFETY_MARGIN_FACTOR
            )

            memory_histogram = self.memory_histograms[aggregation_key]
            memory_recommendation = (
                memory_histogram.get_percentile(MEMORY_PERCENTILE)
                * MEMORY_SAFETY_MARGIN_FACTOR
            )

            for pod in pods_:
                # don't overwrite any existing recommendations (e.g. from VPA)
                if "recommendation" not in pod:
                    pod["recommendation"] = {
                        "cpu": cpu_recommendation,
                        "memory": memory_recommendation,
                    }

    def load_from_file(self, data_path: Path):
        oldest_mtime = time.time() - MAX_STALE_FILE_AGE_SECONDS
        for path in data_path.rglob(CHECKPOINT_FILE_NAME):
            aggregation_key = tuple(
                ("" if p == "-" else p)
                for p in path.parent.parts[-AGGREGATION_KEY_LENGTH:]
            )
            try:
                if path.stat().st_mtime < oldest_mtime:
                    self._stale_aggregation_keys.add(aggregation_key)
                with path.open() as fd:
                    data = json.load(fd)
                self.first_sample_times[aggregation_key] = data["first_sample_time"]
                self.total_sample_counts[aggregation_key] = data["total_samples_count"]
                self.cpu_histograms[aggregation_key].from_checkpoint(
                    data["cpu_histogram"]
                )
                self.memory_histograms[aggregation_key].from_checkpoint(
                    data["memory_histogram"]
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load recommender checkpoint from {path}: {e}"
                )

    def _get_folder(self, data_path: Path, aggregation_key):
        folder = data_path
        for part in aggregation_key:
            folder /= part if part else "-"
        return folder

    def save_to_file(self, data_path: Path):
        for aggregation_key in self._updated_aggregation_keys:
            if aggregation_key in self._stale_aggregation_keys:
                self._stale_aggregation_keys.remove(aggregation_key)

            folder = self._get_folder(data_path, aggregation_key)
            folder.mkdir(parents=True, exist_ok=True)
            data = {
                "first_sample_time": self.first_sample_times[aggregation_key],
                "total_samples_count": self.total_sample_counts[aggregation_key],
                "cpu_histogram": self.cpu_histograms[aggregation_key].get_checkpoint(),
                "memory_histogram": self.memory_histograms[
                    aggregation_key
                ].get_checkpoint(),
            }
            path = folder / CHECKPOINT_FILE_NAME
            with path.open("w") as fd:
                json.dump(data, fd)

        for aggregation_key in self._stale_aggregation_keys:
            folder = self._get_folder(data_path, aggregation_key)
            path = folder / CHECKPOINT_FILE_NAME
            path.unlink(missing_ok=True)
