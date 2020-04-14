from pytest import approx

from kube_resource_report.histogram import ExponentialHistogram


def test_histogram_empty():
    hist = ExponentialHistogram(1000.0, 0.01, 1.05)
    assert hist.is_empty()
    assert hist.get_percentile(0.9) == 0


def test_histogram_checkpoint_empty():
    hist = ExponentialHistogram(1000.0, 0.01, 1.05)
    assert hist.get_checkpoint() == {"total_weight": 0, "bucket_weights": {}}


def test_histogram_checkpoint_single_bucket():
    hist = ExponentialHistogram(1000.0, 0.01, 1.05)
    hist.add_sample(0.001, 1, None)
    assert hist.get_checkpoint() == {"total_weight": 1, "bucket_weights": {0: 10000}}


def test_histogram_percentile():
    hist = ExponentialHistogram(1000.0, 0.01, 1.05)
    hist.add_sample(1, 1, None)
    hist.add_sample(2, 1, None)
    hist.add_sample(3, 1, None)
    assert hist.get_percentile(0.5) == approx(2, rel=0.1)

    hist = ExponentialHistogram(1000.0, 0.01, 1.05)
    for i in range(1, 11):
        hist.add_sample(i, 1, None)
    assert hist.get_percentile(0.5) == approx(5, rel=0.1)
