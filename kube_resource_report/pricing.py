#!/usr/bin/env python3
import csv
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

AVG_DAYS_PER_MONTH = 30.4375
ONE_GIBI = 1024 ** 3

# just assume 60% spot discount
# see also https://aws.amazon.com/ec2/spot/instance-advisor/
DEFAULT_SPOT_DISCOUNT = 0.6

NODE_COSTS_MONTHLY = {}
NODE_SPOT_COSTS_MONTHLY = {}

_path = Path(__file__).parent / "aws-ec2-costs-monthly.csv"
_path_gcp = Path(__file__).parent / "gcp-costs-monthly.csv"
_spot_path = Path(__file__).parent / "aws-ec2-spot-costs-monthly.csv"


# from https://docs.aws.amazon.com/general/latest/gr/ec2-service.html
# did not find a mapping of region names elsewhere :-(
# entries are sorted!
AWS_LOCATIONS = {
    "Africa (Cape Town)": "af-south-1",
    "Asia Pacific (Hong Kong)": "ap-east-1",
    "Asia Pacific (Mumbai)": "ap-south-1",
    "Asia Pacific (Osaka-Local)": "ap-northeast-3",
    "Asia Pacific (Seoul)": "ap-northeast-2",
    "Asia Pacific (Singapore)": "ap-southeast-1",
    "Asia Pacific (Sydney)": "ap-southeast-2",
    "Asia Pacific (Tokyo)": "ap-northeast-1",
    "AWS GovCloud (US-East)": "us-gov-east-1",
    "AWS GovCloud (US-West)": "us-gov-west-1",
    "Canada (Central)": "ca-central-1",
    "China (Beijing)": "cn-north-1",
    "China (Ningxia)": "cn-northwest-1",
    "EU (Frankfurt)": "eu-central-1",
    "EU (Ireland)": "eu-west-1",
    "EU (London)": "eu-west-2",
    "EU (Paris)": "eu-west-3",
    "EU (Stockholm)": "eu-north-1",
    "Middle East (Bahrain)": "me-south-1",
    # note: Sao Paulo is returned as ASCII (not "São Paulo")
    "South America (Sao Paulo)": "sa-east-1",
    "US East (N. Virginia)": "us-east-1",
    "US East (Ohio)": "us-east-2",
    # https://aws.amazon.com/blogs/aws/aws-now-available-from-a-local-zone-in-los-angeles/
    # https://news.ycombinator.com/item?id=21695232
    "US West (Los Angeles)": "us-west-2-lax-1a",
    "US West (N. California)": "us-west-1",
    "US West (Oregon)": "us-west-2",
}


def load_data():
    # CSV generated by calling this script (see __main__ section below)
    for p in (_path, _path_gcp):
        with p.open() as fd:
            reader = csv.reader(fd)
            for row in reader:
                region, instance_type, monthly_cost = row[:3]
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


def get_node_cost(region, instance_type, is_spot, cpu, memory):
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

    if cost is None and instance_type.startswith("custom-"):
        per_cpu = NODE_COSTS_MONTHLY.get((region, "custom-per-cpu-core"))
        per_memory = NODE_COSTS_MONTHLY.get((region, "custom-per-memory-gib"))
        if per_cpu and per_memory:
            cost = (cpu * per_cpu) + (memory / ONE_GIBI * per_memory)

    elif cost is None and re.match("[a-z][0-9][a-z]?-", instance_type):
        if re.match("[a-z][0-9][a-z]?-custom-", instance_type):
            instance_prefix = re.sub(
                "([a-z][0-9][a-z]?-custom-).*", r"\1", instance_type
            )
        else:
            instance_prefix = re.sub(
                "([a-z][0-9][a-z]?)-.*", r"\1-predefined-", instance_type
            )

        per_cpu = NODE_COSTS_MONTHLY.get((region, instance_prefix + "vm-core"))
        per_standard_memory = NODE_COSTS_MONTHLY.get(
            (region, instance_prefix + "vm-ram")
        )
        per_extended_memory = NODE_COSTS_MONTHLY.get(
            (region, instance_prefix + "extended-ram")
        )
        if instance_type.endswith("-preemptible"):
            per_cpu = NODE_COSTS_MONTHLY.get(
                (region, instance_prefix + "vm-core-preemptible")
            )
            per_standard_memory = NODE_COSTS_MONTHLY.get(
                (region, instance_prefix + "vm-ram-preemptible")
            )
            per_extended_memory = NODE_COSTS_MONTHLY.get(
                (region, instance_prefix + "vm-extended-ram-preemptible")
            )
        if per_cpu and per_standard_memory:
            logger.debug(
                f"Monthly per-cpu cost for {instance_type} in {region} is {per_cpu}"
            )
            logger.debug(
                f"Monthly per-standard-memory cost for {instance_type} in {region} is {per_standard_memory}"
            )

            # standard memory is up to 8GB per vCPU
            standard_memory = cpu * 8 * ONE_GIBI

            if memory <= standard_memory:
                cost = (cpu * per_cpu) + (memory / ONE_GIBI * per_standard_memory)
            else:
                if per_extended_memory:
                    logger.debug(
                        f"Monthly per-extended-memory cost for {instance_type} in {region} is {per_extended_memory}"
                    )

                    cost = (cpu * per_cpu) + (
                        standard_memory / ONE_GIBI * per_standard_memory
                    )

                    # extended memory is over 8GB per vCPU
                    extended_memory = memory - standard_memory
                    cost += extended_memory / ONE_GIBI * per_extended_memory

    if cost is None:
        logger.warning(f"No cost information for {instance_type} in {region}")
        cost = 0
    else:
        logger.debug(f"Monthly cost for {instance_type} in {region} is {cost}")
    return cost


def generate_gcp_price_list():
    import requests

    response = requests.get(
        "https://cloudpricingcalculator.appspot.com/static/data/pricelist.json?v=1563953523378"
    )
    prefix = "CP-COMPUTEENGINE-VMIMAGE-"
    custom_prefix = "CP-COMPUTEENGINE-CUSTOM-VM-"

    with _path_gcp.open("w") as fd:
        writer = csv.writer(fd, lineterminator="\n")
        for product, data in sorted(response.json()["gcp_price_list"].items()):
            if product.startswith(prefix):
                instance_type = product[len(prefix) :].lower()
                for region, hourly_price in sorted(data.items()):
                    if "-" in region and isinstance(hourly_price, float):
                        monthly_price = hourly_price * 24 * AVG_DAYS_PER_MONTH
                        writer.writerow(
                            [
                                region,
                                instance_type,
                                "{:.4f}".format(monthly_price),
                                data.get("cores"),
                                data.get("memory"),
                            ]
                        )

            elif product.startswith(custom_prefix):
                _type = product[len(custom_prefix) :].lower()
                if _type == "core":
                    instance_type = "custom-per-cpu-core"
                elif _type == "ram":
                    # note: GCP prices are per GiB (2^30 bytes)
                    # https://cloud.google.com/compute/all-pricing
                    instance_type = "custom-per-memory-gib"
                elif _type == "core-preemptible":
                    instance_type = "custom-preemptible-per-cpu-core"
                elif _type == "ram-preemptible":
                    instance_type = "custom-preemptible-per-memory-gib"
                else:
                    instance_type = None
                if instance_type:
                    for region, hourly_price in sorted(data.items()):
                        if "-" in region and isinstance(hourly_price, float):
                            monthly_price = hourly_price * 24 * AVG_DAYS_PER_MONTH
                            writer.writerow(
                                [region, instance_type, "{:.4f}".format(monthly_price)]
                            )

            elif re.match("^CP-COMPUTEENGINE-[A-Z0-9]+-(CUSTOM|PREDEFINED)-", product):
                instance_type = product[17:].lower()
                if instance_type:
                    for region, hourly_price in sorted(data.items()):
                        if "-" in region and isinstance(hourly_price, float):
                            monthly_price = hourly_price * 24 * AVG_DAYS_PER_MONTH
                            writer.writerow(
                                [region, instance_type, "{:.4f}".format(monthly_price)]
                            )


def generate_ec2_spot_price_list():
    import boto3
    import datetime

    max_price = {}
    for location in sorted(AWS_LOCATIONS.values()):
        # some regions are not available
        if location in (
            "af-south-1",
            "ap-east-1",
            "ap-northeast-3",
            "cn-north-1",
            "cn-northwest-1",
            "me-south-1",
            "us-gov-east-1",
            "us-gov-west-1",
            "us-west-2-lax-1a",
        ):
            continue
        print(location)
        try:
            ec2 = boto3.client("ec2", location)

            today = datetime.date.today()
            start = today - datetime.timedelta(days=3)

            instance_types_required = set(
                [x[1] for x in NODE_COSTS_MONTHLY.keys() if x[0] == location]
            )
            # instances not available as Spot..
            instance_types_required -= set(["hs1.8xlarge", "t2.nano"])
            instance_types_seen = set()

            next_token = ""
            i = 0
            while next_token is not None:
                data = ec2.describe_spot_price_history(
                    Filters=[{"Name": "product-description", "Values": ["Linux/UNIX"]}],
                    StartTime=start.isoformat(),
                    EndTime=today.isoformat(),
                    NextToken=next_token,
                )
                print(". ", end="")
                for entry in data["SpotPriceHistory"]:
                    print(entry)
                    instance_type = entry["InstanceType"]
                    instance_types_seen.add(instance_type)
                    price = float(entry["SpotPrice"])
                    monthly_price = price * 24 * AVG_DAYS_PER_MONTH
                    max_price[(location, instance_type)] = max(
                        max_price.get((location, instance_type), 0), monthly_price
                    )
                i += 1
                if instance_types_seen >= instance_types_required or i > 4:
                    next_token = None
                else:
                    print(
                        f"Waiting to see instance types {instance_types_required - instance_types_seen}"
                    )
                    next_token = data.get("NextToken")
        except Exception as e:
            print(f"Could not get EC2 Spot prices for {location}: {e}")

    with _spot_path.open("w") as fd:
        writer = csv.writer(fd, lineterminator="\n")
        for loc_instance_type, price in sorted(max_price.items()):
            location, instance_type = loc_instance_type
            writer.writerow([location, instance_type, "{:.4f}".format(price)])


def generate_ec2_price_list():
    # hack to update AWS price list
    import boto3
    import json

    try:
        generate_ec2_spot_price_list()
    except Exception as e:
        print(f"Could not load AWS EC2 Spot prices: {e}")

    filters = [
        {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
        {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
    ]

    pricing = boto3.client("pricing", "us-east-1")

    pricing_data = {}

    next_token = ""
    while next_token is not None:
        data = pricing.get_products(
            ServiceCode="AmazonEC2", Filters=filters, NextToken=next_token
        )
        print(". ", end="")
        for entry in data["PriceList"]:
            entry = json.loads(entry)
            tenancy = entry["product"]["attributes"].get("tenancy")
            os = entry["product"]["attributes"].get("operatingSystem")
            location = entry["product"]["attributes"].get("location", "")
            location = AWS_LOCATIONS.get(location, location)
            sw = entry["product"]["attributes"].get("preInstalledSw", "")
            usagetype = entry["product"]["attributes"]["usagetype"]

            if (
                tenancy == "Shared"
                and os == "Linux"
                and sw == "NA"
                and "BoxUsage:" in usagetype
            ):
                for _k, v in entry["terms"]["OnDemand"].items():
                    for _, v_ in v["priceDimensions"].items():
                        if v_["unit"] == "Hrs":
                            price = float(v_["pricePerUnit"]["USD"])
                            if price == 0:
                                # AWS GovCloud (US-West) has some zero prices
                                # we don't care about them..
                                # print(entry["product"]["attributes"], k, v, k_, v_)
                                continue
                            monthly_price = price * 24 * AVG_DAYS_PER_MONTH
                            instance_type = entry["product"]["attributes"][
                                "instanceType"
                            ]
                            key = (location, instance_type)

                            previous_price = pricing_data.get(key)
                            if (
                                previous_price is not None
                                and previous_price != monthly_price
                            ):
                                raise Exception(
                                    "Duplicate data for {}/{}: {:.4f} and {:.4f}".format(
                                        location,
                                        instance_type,
                                        previous_price,
                                        monthly_price,
                                    )
                                )

                            pricing_data[key] = monthly_price
        next_token = data.get("NextToken")

    with _path.open("w") as fd:
        writer = csv.writer(fd, lineterminator="\n")
        for (location, instance_type), price in sorted(pricing_data.items()):
            writer.writerow([location, instance_type, "{:.4f}".format(price)])


if __name__ == "__main__":
    generate_gcp_price_list()
    generate_ec2_price_list()
else:
    load_data()
