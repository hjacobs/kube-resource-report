import logging
import os
import time
from pathlib import Path

import click

from .cluster_discovery import DEFAULT_CLUSTERS
from .report import generate_report


class CommaSeparatedValues(click.ParamType):
    name = "comma_separated_values"

    def convert(self, value, param, ctx):
        if isinstance(value, str):
            values = list(filter(None, value.split(",")))
        else:
            values = value
        return values


@click.command()
@click.option(
    "--clusters",
    type=CommaSeparatedValues(),
    help=f"Comma separated list of Kubernetes API server URLs (default: {DEFAULT_CLUSTERS})",
    envvar="CLUSTERS",
)
@click.option(
    "--cluster-registry",
    metavar="URL",
    help="URL of Cluster Registry to discover clusters to report on",
)
@click.option(
    "--kubeconfig-path", type=click.Path(exists=True), help="Path to kubeconfig file"
)
@click.option(
    "--kubeconfig-contexts",
    type=CommaSeparatedValues(),
    help="List of kubeconfig contexts to use (default: use all defined contexts)",
    envvar="KUBECONFIG_CONTEXTS",
)
@click.option(
    "--application-registry",
    metavar="URL",
    help="URL of Application Registry to look up team by application ID",
)
@click.option(
    "--use-cache", is_flag=True, help="Use cached data (mostly for development)"
)
@click.option(
    "--no-ingress-status", is_flag=True, help="Do not check Ingress HTTP status"
)
@click.option(
    "--system-namespaces",
    type=CommaSeparatedValues(),
    metavar="NS1,NS2",
    default="kube-system",
    help="Comma separated list of system/infrastructure namespaces (default: kube-system)",
)
@click.option(
    "--include-clusters", metavar="PATTERN", help="Include clusters matching the regex"
)
@click.option(
    "--exclude-clusters", metavar="PATTERN", help="Exclude clusters matching the regex"
)
@click.option(
    "--additional-cost-per-cluster",
    type=float,
    help="Additional fixed costs per cluster (e.g. etcd nodes, ELBs, ..)",
    default=0,
)
@click.option(
    "--alpha-ema",
    type=float,
    help="""
    Alpha for Exponential Moving Average (EMA).

    The coefficient alpha represents the degree of weighting decrease, a constant smoothing
    factor between 0 and 1. A higher alpha discounts older observations faster.

    More info about EMA: https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average

    Note that there is no "accepted" value that should be chosen for alpha, although
    there are some recommended values based on the application.

    You can use EMA as SMA (Simple Moving Average) by choosing `alpha = 2 / (N+1)`,
    where N is just number of periods (remember that it should behave like SMA; it is not SMA).
    For example, if your update interval is a minute, by choosing N to 60 you will have "the average" from an hour.
    By choosing N to 10 you will have "an average" from ten minutes, and so on...
    """,
    default=1.0,
)
@click.option(
    "--update-interval-minutes",
    type=float,
    help="Update the report every X minutes (default: run once and exit)",
    default=0,
)
@click.option(
    "--pricing-file",
    type=click.Path(exists=True),
    help="Path to alternate pricing file",
)
@click.option(
    "--links-file",
    type=click.Path(exists=True),
    help="Path to YAML file defining custom links for resources",
)
@click.option(
    "--node-labels",
    type=CommaSeparatedValues(),
    help="Values for the kubernetes.io/role label (e.g. 'worker' if nodes are labeled kubernetes.io/role=worker)",
    default="worker",
)
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.option("--data-path", type=click.Path(exists=True))
@click.option(
    "--templates-path",
    type=click.Path(exists=True),
    help="Path to directory with custom HTML/Jinja2 templates",
)
@click.argument("output_dir", type=click.Path(exists=True))
def main(
    clusters,
    cluster_registry,
    kubeconfig_path,
    kubeconfig_contexts,
    application_registry,
    use_cache,
    no_ingress_status,
    output_dir,
    data_path,
    system_namespaces,
    include_clusters,
    exclude_clusters,
    additional_cost_per_cluster,
    alpha_ema,
    update_interval_minutes,
    pricing_file,
    links_file,
    node_labels,
    debug,
    templates_path,
):
    """Kubernetes Resource Report generates a static HTML report to OUTPUT_DIR for all clusters in ~/.kube/config or Cluster Registry."""

    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if kubeconfig_path:
        kubeconfig_path = Path(kubeconfig_path)
    else:
        kubeconfig_path = Path(os.path.expanduser("~/.kube/config"))

    if pricing_file:
        pricing_file = Path(pricing_file)

    if data_path:
        data_path = Path(data_path)
    else:
        data_path = Path(str(output_dir)) / "data"

    if templates_path:
        templates_path = Path(templates_path)

    cluster_summaries = {}

    while True:
        cluster_summaries = generate_report(
            clusters,
            cluster_registry,
            kubeconfig_path,
            set(kubeconfig_contexts or []),
            application_registry,
            use_cache,
            no_ingress_status,
            output_dir,
            data_path,
            set(system_namespaces),
            include_clusters,
            exclude_clusters,
            additional_cost_per_cluster,
            alpha_ema,
            cluster_summaries,
            pricing_file,
            links_file,
            node_labels,
            templates_path,
        )
        if update_interval_minutes > 0:
            time.sleep(update_interval_minutes * 60)
        else:
            break
