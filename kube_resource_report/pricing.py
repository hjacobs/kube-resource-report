#!/usr/bin/env python3
import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

AVG_DAYS_PER_MONTH = 30.4375

# just assume 60% spot discount
# see also https://aws.amazon.com/ec2/spot/instance-advisor/
DEFAULT_SPOT_DISCOUNT = 0.6

NODE_COSTS_MONTHLY = {}
NODE_SPOT_COSTS_MONTHLY = {}

_path = Path(__file__).parent / "aws-ec2-costs-monthly.csv"
_spot_path = Path(__file__).parent / "aws-ec2-spot-costs-monthly.csv"


def load_data():
    # CSV generated by calling this script (see __main__ section below)
    with _path.open() as fd:
        reader = csv.reader(fd)
        for row in reader:
            region, instance_type, monthly_cost = row
            NODE_COSTS_MONTHLY[(region, instance_type)] = float(monthly_cost)

    with _spot_path.open() as fd:
        reader = csv.reader(fd)
        for row in reader:
            region, instance_type, monthly_cost = row
            NODE_SPOT_COSTS_MONTHLY[(region, instance_type)] = float(monthly_cost)


def regenerate_cost_dict(pricing_file):
    # Reset the costs dict and fill it with data from the external pricing file
    NODE_COSTS_MONTHLY.clear()
    with pricing_file.open() as fd:
        reader = csv.reader(fd)
        for row in reader:
            region, instance_type, monthly_cost = row
            NODE_COSTS_MONTHLY[(region, instance_type)] = float(monthly_cost)


def get_node_cost(region, instance_type, is_spot):
    if is_spot:
        cost = NODE_SPOT_COSTS_MONTHLY.get((region, instance_type))
        if cost is None:
            cost = NODE_COSTS_MONTHLY.get((region, instance_type))
            if cost is not None:
                # https://aws.amazon.com/ec2/spot/instance-advisor/
                discount = DEFAULT_SPOT_DISCOUNT
                cost *= 1 - discount
    else:
        cost = NODE_COSTS_MONTHLY.get((region, instance_type))

    if cost is None:
        logger.warning(f"No cost information for {instance_type} in {region}")
        cost = 0
    return cost


def generate_price_list():
    # hack to update AWS price list
    import boto3
    import datetime
    import json

    # from https://docs.aws.amazon.com/general/latest/gr/rande.html
    # did not find a mapping of region names elsewhere :-(
    LOCATIONS = {
        "US East (Ohio)": "us-east-2",
        "US East (N. Virginia)": "us-east-1",
        "US West (N. California)": "us-west-1",
        "US West (Oregon)": "us-west-2",
        "Asia Pacific (Tokyo)": "ap-northeast-1",
        "Asia Pacific (Seoul)": "ap-northeast-2",
        "Asia Pacific (Osaka-Local)": "ap-northeast-3",
        "Asia Pacific (Mumbai)": "ap-south-1",
        "Asia Pacific (Singapore)": "ap-southeast-1",
        "Asia Pacific (Sydney)": "ap-southeast-2",
        "Canada (Central)": "ca-central-1",
        "China (Beijing)": "cn-north-1",
        "China (Ningxia)": "cn-northwest-1",
        "EU (Frankfurt)": "eu-central-1",
        "EU (Ireland)": "eu-west-1",
        "EU (London)": "eu-west-2",
        "EU (Paris)": "eu-west-3",
        # note: Sao Paulo is returned as ASCII (not "São Paulo")
        "South America (Sao Paulo)": "sa-east-1",
        "AWS GovCloud (US)": "us-gov-west-1",
    }

    max_price = {}
    for location in sorted(LOCATIONS.values()):
        # some regions are not available
        if location in ("ap-northeast-3", "cn-north-1", "cn-northwest-1", "us-gov-west-1"):
            continue
        print(location)
        ec2 = boto3.client("ec2", location)

        today = datetime.date.today()
        start = today - datetime.timedelta(days=3)

        instance_types_required = set([x[1] for x in NODE_COSTS_MONTHLY.keys() if x[0] == location])
        # instances not available as Spot..
        instance_types_required -= set(['hs1.8xlarge', 't2.nano'])
        instance_types_seen = set()

        next_token = ""
        i = 0
        while next_token is not None:
            data = ec2.describe_spot_price_history(Filters=[{"Name": "product-description", "Values": ["Linux/UNIX"]}],
                                                   StartTime=start.isoformat(), EndTime=today.isoformat(), NextToken=next_token)
            print('. ', end='')
            for entry in data["SpotPriceHistory"]:
                print(entry)
                instance_type = entry["InstanceType"]
                instance_types_seen.add(instance_type)
                price = float(entry["SpotPrice"])
                monthly_price = price * 24 * AVG_DAYS_PER_MONTH
                max_price[(location, instance_type)] = max(max_price.get((location, instance_type), 0), monthly_price)
            i += 1
            if instance_types_seen >= instance_types_required or i > 4:
                next_token = None
            else:
                print(f"Waiting to see instance types {instance_types_required - instance_types_seen}")
                next_token = data.get("NextToken")

    with _spot_path.open("w") as fd:
        writer = csv.writer(fd, lineterminator="\n")
        for loc_instance_type, price in sorted(max_price.items()):
            location, instance_type = loc_instance_type
            writer.writerow([location, instance_type, "{:.4f}".format(price)])

    filters = [
        {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
        {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
    ]

    pricing = boto3.client("pricing", "us-east-1")

    rows = []
    next_token = ""
    while next_token is not None:
        data = pricing.get_products(
            ServiceCode="AmazonEC2", Filters=filters, NextToken=next_token
        )
        print('. ', end='')
        for entry in data["PriceList"]:
            entry = json.loads(entry)
            tenancy = entry["product"]["attributes"].get("tenancy")
            os = entry["product"]["attributes"].get("operatingSystem")
            location = entry["product"]["attributes"].get("location", "")
            location = LOCATIONS.get(location, location)
            sw = entry["product"]["attributes"].get("preInstalledSw", "")
            if tenancy == "Shared" and os == "Linux" and sw == "NA":
                for k, v in entry["terms"]["OnDemand"].items():
                    for k_, v_ in v["priceDimensions"].items():
                        if v_["unit"] == "Hrs":
                            price = float(v_["pricePerUnit"]["USD"])
                            monthly_price = price * 24 * AVG_DAYS_PER_MONTH
                            rows.append([location, entry["product"]["attributes"]["instanceType"], "{:.4f}".format(monthly_price)])
        next_token = data.get("NextToken")

    with _path.open("w") as fd:
        writer = csv.writer(fd, lineterminator="\n")
        for row in sorted(rows):
            writer.writerow(row)


if __name__ == "__main__":
    generate_price_list()
else:
    load_data()
