import click
from .cluster_discovery import DEFAULT_CLUSTERS
from pathlib import Path
import logging
import os
import time

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
    "--update-interval-minutes",
    type=float,
    help="Update the report every X minutes (default: run once and exit)",
    default=0,
)
@click.option(
    "--pricing-file",
    type=click.Path(exists=True),
    help="Path to alternate pricing file"
)
@click.option(
    "--links-file",
    type=click.Path(exists=True),
    help="Path to YAML file defining custom links for resources"
)
@click.option(
    "--node-label",
    help="Value for the kubernetes.io/role label (e.g. 'worker' if nodes are labeled kubernetes.io/role=worker)",
    default="worker",
)
@click.option(
    "--debug", is_flag=True, help="Enable debug logging"
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
    system_namespaces,
    include_clusters,
    exclude_clusters,
    additional_cost_per_cluster,
    update_interval_minutes,
    pricing_file,
    links_file,
    node_label,
    debug,
):
    """Kubernetes Resource Report

    Generate a static HTML report to OUTPUT_DIR for all clusters in ~/.kube/config or Cluster Registry.
    """

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

    while True:
        generate_report(
            clusters,
            cluster_registry,
            kubeconfig_path,
            set(kubeconfig_contexts or []),
            application_registry,
            use_cache,
            no_ingress_status,
            output_dir,
            set(system_namespaces),
            include_clusters,
            exclude_clusters,
            additional_cost_per_cluster,
            pricing_file,
            links_file,
            node_label,
        )
        if update_interval_minutes > 0:
            time.sleep(update_interval_minutes * 60)
        else:
            break
