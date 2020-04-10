import pytest

from kube_resource_report.metrics import get_ema


@pytest.mark.parametrize(
    "curr_value,prev_value,alpha,expected",
    [
        # no previous values
        (1, None, 0, 1),
        (2, None, 0.5, 2),
        (3, None, 1, 3),
        # alpha zero, we should get prev_value only
        (1, 2, 0, 2),
        # alpha one, we should get curr_value only
        (1, 2, 1, 1),
        # large alpha, discount older observations faster
        (10, 2, 0.9, 9.2),
        # small alpha, discount older observations slower
        (10, 2, 0.1, 2.8),
    ],
)
def test_ema_func(curr_value, prev_value, alpha, expected):
    assert get_ema(curr_value, prev_value, alpha) == expected


@pytest.mark.parametrize(
    "values,alpha",
    [
        ([100, 1, 1, 1], 2 / (4 + 1)),
        ([1, 1, 1, 100], 2 / (4 + 1)),
        ([1, 1, 2, 2], 2 / (4 + 1)),
        ([1000, 1, 1, 1, 1, 1, 1], 2 / (7 + 1)),
        ([1, 1, 2, 2, 3, 3, 2, 2, 1, 1], 2 / (10 + 1)),
        (
            [5, 5, 5, 6, 10, 15, 15, 20, 15, 15, 21, 22, 23, 30, 10, 8, 5, 5, 6, 6],
            2 / (20 + 1),
        ),
    ],
)
def test_ema_like_sma(values, alpha):
    """You can treat EMA like SMA, let's check if the function is correctly written."""
    sma = sum(values) / len(values)

    ema = None

    for v in values:
        ema = get_ema(v, ema, alpha)

    # EMA reacts faster than SMA, and this ratio should not be below 0.86
    assert (ema / sma) > 0.86
