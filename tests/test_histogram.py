import math
import time

from pytest import approx

from kube_resource_report.histogram import DecayingExponentialHistogram

ONE_DAY = 3600 * 24


def percentile(N, percent):
    """
    Find the percentile of a list of values.

    @parameter N - is a list of values. Note N MUST BE already sorted.
    @parameter percent - a float value from 0.0 to 1.0.

    @return - the percentile of the values
    """
    if not N:
        return None
    k = (len(N) - 1) * percent
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return N[int(k)]
    d0 = N[int(f)] * (c - k)
    d1 = N[int(c)] * (k - f)
    return d0 + d1


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


def test_histogram_percentile_large1():
    hist = DecayingExponentialHistogram(100.0, 0.001, 1.05, ONE_DAY)
    now = time.mktime((2020, 4, 15, 21, 34, 0, 2, 0, 0))

    values = [
        0.00356826,
        0.003571567,
        0.00357581,
        0.00357730,
        0.003580761,
        0.00359196,
        0.003596697,
        0.0035993,
        0.00361249,
        0.003615085,
        0.003617967,
        0.003621645,
        0.003624581,
        0.003628672,
        0.00363173,
        0.003633299,
        0.00363402,
        0.00363437,
        0.003634643,
        0.003634752,
        0.003641667,
        0.003641921,
        0.003644168,
        0.00365697,
        0.00365703,
        0.003660491,
        0.00366627,
        0.00366672,
        0.00366859,
        0.003671890,
        0.003674470,
        0.003674653,
        0.003678485,
        0.0036906,
        0.003692064,
        0.003692147,
        0.003692247,
        0.00369417,
        0.003694798,
        0.003697279,
        0.00370017,
        0.00370552,
        0.003706270,
        0.003708455,
        0.00371141,
        0.00371142,
        0.00371282,
        0.00371546,
        0.003717293,
        0.00372156,
        0.00372266,
        0.00372328,
        0.003731720,
        0.00373524,
        0.003738866,
        0.003739554,
        0.003740255,
        0.003740514,
        0.003742887,
        0.003743114,
        0.00374609,
        0.003748427,
        0.00375339,
        0.003754504,
        0.003759066,
        0.003762933,
        0.003763028,
        0.003770094,
        0.003772507,
        0.003773164,
        0.003784076,
        0.00378883,
        0.003791513,
        0.00379167,
        0.003797098,
        0.0037978,
        0.003799059,
        0.003800420,
        0.003803184,
        0.00380564,
        0.00380581,
        0.00381127,
        0.00381342,
        0.00381523,
        0.003815614,
        0.00382284,
        0.003827261,
        0.00382847,
        0.003831219,
        0.003836552,
        0.003837597,
        0.00383817,
        0.003838523,
        0.003843782,
        0.00384448,
        0.003847642,
        0.00384810,
        0.003849126,
        0.003852803,
        0.003858588,
        0.003859859,
        0.00386054,
        0.003861486,
        0.00386349,
        0.00448897,
        0.004490609,
        0.00449483,
        0.00449538,
        0.00449559,
        0.00449842,
        0.00449850,
        0.00449870,
        0.00449938,
        0.00450397,
        0.0045061,
        0.00451622,
        0.00451764,
        0.00451884,
        0.00452040,
        0.00452098,
        0.004521002,
        0.00452261,
        0.0045254,
    ]
    values.sort()
    for i in values:
        hist.add_sample(i, 1, now)

    assert hist.get_percentile(0.5) == approx(percentile(values, 0.5), abs=0.02)
    assert hist.get_percentile(0.75) == approx(percentile(values, 0.75), abs=0.02)
    assert hist.get_percentile(0.9) == approx(percentile(values, 0.9), abs=0.02)


def test_histogram_percentile_large2():
    hist = DecayingExponentialHistogram(100.0, 0.001, 1.05, ONE_DAY)
    now = time.mktime((2020, 4, 15, 21, 34, 0, 2, 0, 0))

    values = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.42, 0.51, 0.7, 1.2, 1.4]
    values.sort()
    for i in values:
        hist.add_sample(i, 1, now)

    assert hist.get_percentile(0.5) == approx(percentile(values, 0.5), abs=0.02)
    assert hist.get_percentile(0.75) == approx(percentile(values, 0.75), abs=0.15)
    assert hist.get_percentile(0.9) == approx(percentile(values, 0.9), abs=0.05)


def test_histogram_max_decay():
    min_value = 10.0 * 1024 * 1024  # 10 MiB
    max_value = 1024.0 ** 4  # 1 TiB
    hist = DecayingExponentialHistogram(max_value, min_value, 1.05, ONE_DAY)
    now = time.mktime((2020, 4, 15, 21, 34, 0, 2, 0, 0))

    old_max = 900.0 * 1024 * 1024
    new_max = 100.0 * 1014 * 1024
    hist.add_sample(old_max, 1, now)
    assert hist.get_percentile(1.0) == approx(old_max, rel=0.01)

    ts = now
    for i in range(60):
        ts += ONE_DAY
        hist.add_sample(new_max, 1, ts)
        if i < 53:
            assert hist.get_percentile(1.0) == approx(old_max, rel=0.01)
        else:
            # old max decayed too much, the new value prevails
            assert hist.get_percentile(1.0) == approx(new_max, rel=0.15)


def test_histogram_save_load_checkpoint():
    now = time.mktime((2020, 4, 19, 21, 34, 0, 2, 0, 0))

    start = time.time()
    hist = DecayingExponentialHistogram(100.0, 0.001, 1.05, ONE_DAY)
    hist.add_sample(0.5, 1, now)
    checkpoint = hist.get_checkpoint()
    for i in range(100):
        hist = DecayingExponentialHistogram(100.0, 0.001, 1.05, ONE_DAY)
        hist.from_checkpoint(checkpoint)
        hist.add_sample(0.1 * i, 1, now + (i * 60))
        checkpoint = hist.get_checkpoint()
    delta = time.time() - start
    assert delta < 0
