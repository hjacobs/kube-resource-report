import argparse
import logging
import os
import time
from pathlib import Path

from .cluster_discovery import DEFAULT_CLUSTERS
from .report import generate_report


def comma_separated_values(value):
    if isinstance(value, str):
        values = list(filter(None, value.split(",")))
    else:
        values = value
    return values


def existing_path(value):
    path = Path(value)
    if not path.exists():
        raise ValueError(f"Path does not exist: {path}")
    return path


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clusters",
        type=comma_separated_values,
        help=f"Comma separated list of Kubernetes API server URLs (default: {DEFAULT_CLUSTERS})",
        default=os.getenv("CLUSTERS"),
    )
    parser.add_argument(
        "--cluster-registry",
        metavar="URL",
        help="URL of Cluster Registry to discover clusters to report on",
    )
    parser.add_argument(
        "--kubeconfig-path", type=existing_path, help="Path to kubeconfig file"
    )
    parser.add_argument(
        "--kubeconfig-contexts",
        type=comma_separated_values,
        help="List of kubeconfig contexts to use (default: use all defined contexts)",
        default=os.getenv("KUBECONFIG_CONTEXTS"),
    )
    parser.add_argument(
        "--application-registry",
        metavar="URL",
        help="URL of Application Registry to look up team by application ID",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use cached data (mostly for development)",
    )
    parser.add_argument(
        "--no-ingress-status",
        action="store_true",
        help="Do not check Ingress HTTP status",
    )
    parser.add_argument(
        "--system-namespaces",
        type=comma_separated_values,
        metavar="NS1,NS2",
        default="kube-system",
        help="Comma separated list of system/infrastructure namespaces (default: kube-system)",
    )
    parser.add_argument(
        "--include-clusters",
        metavar="PATTERN",
        help="Include clusters matching the regex",
    )
    parser.add_argument(
        "--exclude-clusters",
        metavar="PATTERN",
        help="Exclude clusters matching the regex",
    )
    parser.add_argument(
        "--additional-cost-per-cluster",
        type=float,
        help="Additional fixed costs per cluster (e.g. etcd nodes, ELBs, ..)",
        default=0,
    )
    parser.add_argument(
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
    parser.add_argument(
        "--update-interval-minutes",
        type=float,
        help="Update the report every X minutes (default: run once and exit)",
        default=0,
    )
    parser.add_argument(
        "--pricing-file", type=existing_path, help="Path to alternate pricing file",
    )
    parser.add_argument(
        "--links-file",
        type=existing_path,
        help="Path to YAML file defining custom links for resources",
    )
    parser.add_argument(
        "--node-labels",
        type=comma_separated_values,
        help="Values for the kubernetes.io/role label (e.g. 'worker' if nodes are labeled kubernetes.io/role=worker)",
        default="worker",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--data-path",
        type=existing_path,
        help="Path where to store data such as recommendation histograms",
    )
    parser.add_argument(
        "--templates-path",
        type=existing_path,
        help="Path to directory with custom HTML/Jinja2 templates",
    )
    parser.add_argument("output_dir", type=existing_path)
    return parser


def main():
    """Kubernetes Resource Report generates a static HTML report to OUTPUT_DIR for all clusters in ~/.kube/config or Cluster Registry."""

    parser = get_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    if args.kubeconfig_path:
        kubeconfig_path = args.kubeconfig_path
    else:
        kubeconfig_path = Path(os.path.expanduser("~/.kube/config"))

    if args.data_path:
        data_path = args.data_path
    else:
        data_path = args.output_dir / "data"

    cluster_summaries = {}

    while True:
        cluster_summaries = generate_report(
            args.clusters,
            args.cluster_registry,
            kubeconfig_path,
            set(args.kubeconfig_contexts or []),
            args.application_registry,
            args.use_cache,
            args.no_ingress_status,
            args.output_dir,
            data_path,
            set(args.system_namespaces),
            args.include_clusters,
            args.exclude_clusters,
            args.additional_cost_per_cluster,
            args.alpha_ema,
            cluster_summaries,
            args.pricing_file,
            args.links_file,
            args.node_labels,
            args.templates_path,
        )
        if args.update_interval_minutes > 0:
            time.sleep(args.update_interval_minutes * 60)
        else:
            break
