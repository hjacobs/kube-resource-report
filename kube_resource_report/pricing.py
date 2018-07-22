import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

NODE_COSTS_MONTHLY = {}

# CSVs downloaded from https://ec2instances.info/
for path in Path(__file__).parent.glob("aws-ec2-costs-hourly-*.csv"):
    region = path.stem.split("-", 4)[4]
    with path.open() as fd:
        reader = csv.DictReader(fd)
        for row in reader:
            cost = row["Linux On Demand cost"]
            if cost == "unavailable":
                continue
            elif cost.startswith("$") and cost.endswith(" hourly"):
                monthly_cost = float(cost.split()[0].strip("$")) * 24 * 30
                NODE_COSTS_MONTHLY[(region, row["API Name"])] = monthly_cost
            else:
                raise Exception("Invalid price data: {}".format(cost))


def get_node_cost(region, instance_type, is_spot):
    if is_spot:
        # https://aws.amazon.com/ec2/spot/instance-advisor/
        discount = 0.60
    else:
        discount = 0

    cost = NODE_COSTS_MONTHLY.get((region, instance_type))
    if cost is None:
        logger.warning("No cost information for {} in {}".format(instance_type, region))
        cost = 0
    cost *= 1 - discount
    return cost
