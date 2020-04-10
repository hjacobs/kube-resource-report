import re

ONE_MEBI = 1024 ** 2
ONE_GIBI = 1024 ** 3

# assume minimal requests even if no user requests are set
MIN_CPU_USER_REQUESTS = 1 / 1000
MIN_MEMORY_USER_REQUESTS = 1 / 1000

# we show costs per month by default as it leads to easily digestable numbers (for humans)
AVG_DAYS_PER_MONTH = 30.4375
HOURS_PER_DAY = 24
HOURS_PER_MONTH = HOURS_PER_DAY * AVG_DAYS_PER_MONTH

RESOURCE_PATTERN = re.compile(r"^(\d*)(\D*)$")

FACTORS = {
    "n": 1 / 1000000000,
    "u": 1 / 1000000,
    "m": 1 / 1000,
    "": 1,
    "k": 1000,
    "M": 1000 ** 2,
    "G": 1000 ** 3,
    "T": 1000 ** 4,
    "P": 1000 ** 5,
    "E": 1000 ** 6,
    "Ki": 1024,
    "Mi": 1024 ** 2,
    "Gi": 1024 ** 3,
    "Ti": 1024 ** 4,
    "Pi": 1024 ** 5,
    "Ei": 1024 ** 6,
}


def parse_resource(v):
    """
    Parse a Kubernetes resource value.

    >>> parse_resource('100m')
    0.1
    >>> parse_resource('100M')
    1000000000
    >>> parse_resource('2Gi')
    2147483648
    >>> parse_resource('2k')
    2048
    """
    match = RESOURCE_PATTERN.match(v)
    factor = FACTORS[match.group(2)]
    return int(match.group(1)) * factor
