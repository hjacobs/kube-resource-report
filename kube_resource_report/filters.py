def money(value):
    return "{:,.02f}".format(value)


def cpu(value):
    return "{:,.01f}".format(value)


def memory(value, fmt):
    if fmt == "GiB":
        return "{:,.01f}".format(value / (1024 ** 3))
    elif fmt == "MiB":
        return "{:,.0f}".format(value / (1024 ** 2))
    else:
        return value
