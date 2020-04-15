import time

from pytest import approx

from kube_resource_report.histogram import DecayingExponentialHistogram

ONE_DAY = 3600 * 24


def test_histogram_empty():
    hist = DecayingExponentialHistogram(1000.0, 0.01, 1.05, ONE_DAY)
    assert hist.is_empty()
    assert hist.get_percentile(0.9) == 0


def test_histogram_checkpoint_empty():
    hist = DecayingExponentialHistogram(1000.0, 0.01, 1.05, ONE_DAY)
    assert hist.get_checkpoint() == {
        "total_weight": 0,
        "bucket_weights": {},
        "reference_time": 0,
    }


def test_histogram_checkpoint_single_bucket():
    hist = DecayingExponentialHistogram(1000.0, 0.01, 1.05, ONE_DAY)
    now = time.time()
    hist.add_sample(0.001, 1, now)
    assert hist.get_checkpoint() == {
        "total_weight": hist.total_weight,
        "bucket_weights": {0: 10000},
        "reference_time": (now // ONE_DAY) * ONE_DAY,
    }


def test_histogram_from_checkpoint():
    hist = DecayingExponentialHistogram(1000.0, 0.01, 1.05, ONE_DAY)
    hist.from_checkpoint(
        {"total_weight": 1, "bucket_weights": {0: 10000}, "reference_time": 123}
    )
    assert hist.reference_time == 123


def test_histogram_decay():
    hist = DecayingExponentialHistogram(1000.0, 0.01, 1.05, ONE_DAY)
    now = time.time()
    hist.add_sample(10, 1, now)
    expected_ref_time = (now // ONE_DAY) * ONE_DAY
    assert hist.reference_time == expected_ref_time
    assert hist.get_percentile(0.5) == approx(10, rel=0.1)
    hist.add_sample(10, 1, now + 60)
    assert hist.reference_time == expected_ref_time
    assert hist.get_percentile(0.5) == approx(10, rel=0.2)
    hist.add_sample(1, 1, now + ONE_DAY)
    assert hist.reference_time == expected_ref_time
    hist.add_sample(1, 1, now + ONE_DAY + 1)
    assert hist.reference_time == expected_ref_time
    assert hist.get_percentile(0.5) == approx(1, rel=0.2)


def test_histogram_percentile():
    hist = DecayingExponentialHistogram(1000.0, 0.01, 1.05, ONE_DAY)
    now = time.mktime((2020, 4, 15, 21, 34, 0, 2, 0, 0))
    hist.add_sample(1, 1, now)
    hist.add_sample(2, 1, now)
    hist.add_sample(3, 1, now)
    assert hist.get_percentile(0.5) == approx(2, rel=0.1)

    hist = DecayingExponentialHistogram(1000.0, 0.01, 1.05, ONE_DAY)
    for i in range(1, 11):
        hist.add_sample(i, 1, now)
    assert hist.get_percentile(0.5) == approx(5, rel=0.1)
    assert hist.get_percentile(1.0) == approx(10, rel=0.2)
