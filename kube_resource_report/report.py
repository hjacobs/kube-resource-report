#!/usr/bin/env python3
import collections
import concurrent.futures
import csv
import datetime
import json
import logging
import pickle
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import requests
import yaml
from pykube import Node
from pykube import Pod
from requests_futures.sessions import FuturesSession

from .output import OutputManager
from .query import query_cluster
from .utils import HOURS_PER_MONTH
from .utils import MIN_CPU_USER_REQUESTS
from .utils import MIN_MEMORY_USER_REQUESTS
from .utils import ONE_GIBI
from .utils import ONE_MEBI
from kube_resource_report import __version__
from kube_resource_report import cluster_discovery
from kube_resource_report import pricing

MAX_WORKERS = 8


session = requests.Session()
# set a friendly user agent for outgoing HTTP requests
session.headers["User-Agent"] = f"kube-resource-report/{__version__}"


def json_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(obj)


logger = logging.getLogger(__name__)


def get_cluster_summaries(
    clusters: list,
    cluster_registry: str,
    kubeconfig_path: Path,
    kubeconfig_contexts: set,
    include_clusters: str,
    exclude_clusters: str,
    system_namespaces: set,
    notifications: list,
    additional_cost_per_cluster: float,
    alpha_ema: float,
    prev_cluster_summaries: dict,
    no_ingress_status: bool,
    node_labels: list,
    data_path: Path,
    map_node_hook=None,
    map_pod_hook=None,
):
    cluster_summaries = {}

    discoverer: Any

    if cluster_registry:
        discoverer = cluster_discovery.ClusterRegistryDiscoverer(cluster_registry)
    elif clusters or not kubeconfig_path.exists():
        api_server_urls = clusters or []
        discoverer = cluster_discovery.StaticClusterDiscoverer(api_server_urls)
    else:
        discoverer = cluster_discovery.KubeconfigDiscoverer(
            kubeconfig_path, kubeconfig_contexts
        )

    include_pattern = include_clusters and re.compile(include_clusters)
    exclude_pattern = exclude_clusters and re.compile(exclude_clusters)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_cluster = {}
        for cluster in discoverer.get_clusters():
            if (not include_pattern or include_pattern.match(cluster.id)) and (
                not exclude_pattern or not exclude_pattern.match(cluster.id)
            ):
                cluster_data_path = data_path / cluster.id
                cluster_data_path.mkdir(parents=True, exist_ok=True)
                future_to_cluster[
                    executor.submit(
                        query_cluster,
                        cluster,
                        executor,
                        system_namespaces,
                        additional_cost_per_cluster,
                        alpha_ema,
                        prev_cluster_summaries.get(cluster.id, {}),
                        no_ingress_status,
                        node_labels,
                        cluster_data_path,
                        map_node_hook,
                        map_pod_hook,
                    )
                ] = cluster

        for future in concurrent.futures.as_completed(future_to_cluster):
            cluster = future_to_cluster[future]
            try:
                summary = future.result()
                cluster_summaries[cluster.id] = summary
            except Exception as e:
                notifications.append(
                    ["error", f"Failed to query cluster {cluster.id}: {e}"]
                )
                logger.exception(e)

    sorted_by_name = sorted(
        cluster_summaries.values(), key=lambda summary: summary["cluster"].name
    )
    return {summary["cluster"].id: summary for summary in sorted_by_name}


def resolve_application_ids(applications: dict, application_registry: str):
    with FuturesSession(max_workers=10, session=session) as futures_session:
        auth = cluster_discovery.OAuthTokenAuth("read-only")

        future_to_app = {}
        for app_id, app in applications.items():
            if app_id:
                future_to_app[
                    futures_session.get(
                        application_registry + "/apps/" + app_id, auth=auth, timeout=5
                    )
                ] = app

        for future in concurrent.futures.as_completed(future_to_app):
            app = future_to_app[future]
            try:
                response = future.result()
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    data = {}
            except Exception as e:
                logger.warning(f"Failed to look up application {app['id']}: {e}")
                data = {}
            team_id = data.get("team_id", "")
            app["team"] = team_id
            app["active"] = data.get("active")


def aggregate_by_team(applications: dict, teams: dict):
    for app in applications.values():
        team_id = app["team"]
        team = teams.get(
            team_id,
            {
                "clusters": set(),
                "applications": set(),
                "cost": 0,
                "pods": 0,
                "requests": {},
                "usage": {},
                "slack_cost": 0,
            },
        )
        team["applications"].add(app["id"])
        team["clusters"] |= app["clusters"]
        team["pods"] += app["pods"]
        for r in "cpu", "memory":
            team["requests"][r] = team["requests"].get(r, 0) + app["requests"][r]
            team["usage"][r] = team["usage"].get(r, 0) + app.get("usage", {}).get(r, 0)
        team["cost"] += app["cost"]
        team["slack_cost"] += app["slack_cost"]
        teams[team_id] = team


def calculate_metrics(context: dict) -> dict:
    metrics = {
        "clusters": len(context["cluster_summaries"]),
        "teams": len(context["teams"]),
        "applications": len(context["applications"]),
        "now": context["now"].isoformat(),
    }
    for k in (
        "total_worker_nodes",
        "total_allocatable",
        "total_requests",
        "total_usage",
        "total_user_requests",
        "total_pods",
        "total_cost",
        "total_cost_per_user_request_hour",
        "total_slack_cost",
        "duration",
        "version",
    ):
        if k.startswith("total_"):
            metrics_key = k[6:]
        else:
            metrics_key = k
        metrics[metrics_key] = context[k]
    return metrics


def aggregate_recommendation(pod, component):
    if "recommendation" in pod and "recommendation" in component:
        for r in "cpu", "memory":
            component["recommendation"][r] = component["recommendation"].get(
                r, 0
            ) + pod["recommendation"].get(r, 0)
    elif "recommendation" in component:
        # only recommend resources for the component if all Pods have recommendations
        del component["recommendation"]


def generate_report(
    clusters,
    cluster_registry,
    kubeconfig_path,
    kubeconfig_contexts: set,
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
    cluster_summaries,
    pricing_file,
    links_file,
    node_labels,
    templates_path: Optional[Path] = None,
    prerender_hook: Optional[Callable[[str, dict], None]] = None,
    map_node_hook: Optional[Callable[[Node, dict], None]] = None,
    map_pod_hook: Optional[Callable[[Pod, dict], None]] = None,
):
    notifications: List[tuple] = []

    if pricing_file:
        pricing.regenerate_cost_dict(pricing_file)

    if links_file:
        with open(links_file, "rb") as fd:
            links = yaml.safe_load(fd)
    else:
        links = {}

    start = datetime.datetime.utcnow()

    out = OutputManager(Path(output_dir), templates_path, prerender_hook)
    # the data collection might take a long time, so first write index.html
    # to give users feedback that Kubernetes Resource Report has started
    # first copy CSS/JS/..
    out.copy_static_assets()
    write_loading_page(out)

    pickle_file_name = "dump.pickle"

    if use_cache and out.exists(pickle_file_name):
        with out.open(pickle_file_name, "rb") as fd:
            data = pickle.load(fd)
        cluster_summaries = data["cluster_summaries"]
        teams = data["teams"]

    else:
        cluster_summaries = get_cluster_summaries(
            clusters,
            cluster_registry,
            kubeconfig_path,
            kubeconfig_contexts,
            include_clusters,
            exclude_clusters,
            system_namespaces,
            notifications,
            additional_cost_per_cluster,
            alpha_ema,
            cluster_summaries,
            no_ingress_status,
            node_labels,
            data_path,
            map_node_hook,
            map_pod_hook,
        )
        teams = {}

    applications: Dict[str, dict] = {}
    namespace_usage: Dict[tuple, dict] = {}
    nodes: Dict[str, dict] = {}

    for cluster_id, summary in sorted(cluster_summaries.items()):
        for _k, pod in summary["pods"].items():
            app = applications.get(
                pod["application"],
                {
                    "id": pod["application"],
                    "cost": 0,
                    "slack_cost": 0,
                    "pods": 0,
                    "components": {},
                    "requests": {},
                    "usage": {},
                    "recommendation": {},
                    "clusters": set(),
                    "team": "",
                    "active": None,
                },
            )
            component = app["components"].get(
                pod["component"],
                {
                    "cost": 0,
                    "slack_cost": 0,
                    "pods": 0,
                    "requests": {},
                    "usage": {},
                    "recommendation": {},
                    "clusters": set(),
                },
            )
            for r in "cpu", "memory":
                for key in "requests", "usage":
                    app[key][r] = app[key].get(r, 0) + pod.get(key, {}).get(r, 0)
                    component[key][r] = component[key].get(r, 0) + pod.get(key, {}).get(
                        r, 0
                    )
            aggregate_recommendation(pod, app)
            aggregate_recommendation(pod, component)
            app["cost"] += pod["cost"]
            app["slack_cost"] += pod.get("slack_cost", 0)
            app["pods"] += 1
            app["clusters"].add(cluster_id)
            app["team"] = pod["team"]

            component["cost"] += pod["cost"]
            component["slack_cost"] += pod.get("slack_cost", 0)
            component["pods"] += 1
            component["clusters"].add(cluster_id)

            app["components"][pod["component"]] = component
            applications[pod["application"]] = app

        for ns_pod, pod in summary["pods"].items():
            namespace = namespace_usage.get(
                (ns_pod[0], cluster_id),
                {
                    "id": ns_pod[0],
                    "cost": 0,
                    "slack_cost": 0,
                    "pods": 0,
                    "requests": {},
                    "usage": {},
                    "recommendation": {},
                    "cluster": "",
                    "email": "",
                    "status": "",
                },
            )
            for r in "cpu", "memory":
                namespace["requests"][r] = (
                    namespace["requests"].get(r, 0) + pod["requests"][r]
                )
                namespace["usage"][r] = namespace["usage"].get(r, 0) + pod.get(
                    "usage", {}
                ).get(r, 0)
            aggregate_recommendation(pod, namespace)
            namespace["cost"] += pod["cost"]
            namespace["slack_cost"] += pod.get("slack_cost", 0)
            namespace["pods"] += 1
            namespace["cluster"] = summary["cluster"]
            namespace_usage[(ns_pod[0], cluster_id)] = namespace

        for node_name, node in summary["nodes"].items():
            node["node_name"] = node_name
            node["cluster"] = cluster_id
            node["cluster_name"] = cluster_summaries[cluster_id]["cluster"].name
            nodes[f"{cluster_id}.{node_name}"] = node

    if application_registry:
        resolve_application_ids(applications, application_registry)

    aggregate_by_team(applications, teams)

    for team in teams.values():

        def cluster_name(cluster_id):
            try:
                return cluster_summaries[cluster_id]["cluster"].name
            except KeyError:
                return None

        team["clusters"] = sorted(team["clusters"], key=cluster_name)

    for _cluster_id, summary in sorted(cluster_summaries.items()):
        for _k, pod in summary["pods"].items():
            app = applications[pod["application"]]
            pod["team"] = app["team"]

    for cluster_id, summary in sorted(cluster_summaries.items()):
        for ns, ns_values in summary["namespaces"].items():
            namespace_ = namespace_usage.get((ns, cluster_id))
            if namespace_:
                namespace_["email"] = ns_values["email"]
                namespace_["status"] = ns_values["status"]

    if not use_cache:
        try:
            with out.open(pickle_file_name, "wb") as fd:
                pickle.dump(
                    {
                        "cluster_summaries": cluster_summaries,
                        "teams": teams,
                        "applications": applications,
                        "namespace_usage": namespace_usage,
                    },
                    fd,
                )
        except Exception as e:
            logger.error(f"Could not dump pickled cache data: {e}")

    write_report(
        out,
        start,
        notifications,
        cluster_summaries,
        nodes,
        namespace_usage,
        applications,
        teams,
        node_labels,
        links,
        alpha_ema,
    )

    return cluster_summaries


def write_loading_page(out):
    file_name = "index.html"

    if not out.exists(file_name):
        now = datetime.datetime.utcnow()
        context = {
            "now": now,
            "version": __version__,
        }
        out.render_template("loading.html", context, file_name)


def write_tsv_files(
    out: OutputManager,
    cluster_summaries,
    nodes,
    namespace_usage,
    applications,
    teams,
    node_labels,
):
    """Write Tab-Separated-Values (TSV) files."""

    resource_categories = ["capacity", "allocatable", "requests", "usage"]
    with out.open("clusters.tsv") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        headers = [
            "Cluster ID",
            "API Server URL",
            "Master Nodes",
            "Worker Nodes",
            "Worker Instance Type",
            "Kubelet Version",
        ]
        for x in resource_categories:
            headers.extend([f"CPU {x.capitalize()}", f"Memory {x.capitalize()} [MiB]"])
        headers.append("Cost [USD]")
        headers.append("Slack Cost [USD]")
        writer.writerow(headers)
        for cluster_id, summary in sorted(cluster_summaries.items()):
            worker_instance_type = set()
            kubelet_version = set()
            for node in summary["nodes"].values():
                if node["role"] in node_labels:
                    worker_instance_type.add(node["instance_type"])
                kubelet_version.add(node["kubelet_version"])
            fields = [
                cluster_id,
                summary["cluster"].api_server_url,
                summary["master_nodes"],
                summary["worker_nodes"],
                ",".join(sorted(worker_instance_type)),
                ",".join(sorted(kubelet_version)),
            ]
            for x in resource_categories:
                fields += [
                    round(summary[x]["cpu"], 2),
                    int(summary[x]["memory"] / ONE_MEBI),
                ]
            fields += [round(summary["cost"], 2)]
            fields += [round(summary["slack_cost"], 2)]
            writer.writerow(fields)

    with out.open("nodes.tsv") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        headers = [
            "Cluster ID",
            "Node",
            "Role",
            "Instance Type",
            "Spot Instance",
            "Kubelet Version",
        ]
        for x in resource_categories:
            headers.extend([f"CPU {x.capitalize()}", f"Memory {x.capitalize()} [MiB]"])
        headers.append("Cost [USD]")
        writer.writerow(headers)
        for _, node in sorted(nodes.items()):
            instance_type = set()
            kubelet_version = set()
            if node["role"] in node_labels:
                instance_type.add(node["instance_type"])
            kubelet_version.add(node["kubelet_version"])

            fields = [
                node["cluster"],
                node["node_name"],
                node["role"],
                node["instance_type"],
                "Yes" if node["spot"] else "No",
                node["kubelet_version"],
            ]
            for x in resource_categories:
                fields += [
                    round(node[x]["cpu"], 2),
                    int(node[x]["memory"] / ONE_MEBI),
                ]
            fields += [round(node["cost"], 2)]
            writer.writerow(fields)

    with out.open("ingresses.tsv") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        writer.writerow(
            [
                "Cluster ID",
                "API Server URL",
                "Namespace",
                "Name",
                "Application",
                "Host",
                "Status",
            ]
        )
        for cluster_id, summary in sorted(cluster_summaries.items()):
            for ingress in summary["ingresses"]:
                writer.writerow(
                    [cluster_id, summary["cluster"].api_server_url] + ingress
                )

    with out.open("teams.tsv") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        writer.writerow(
            [
                "ID",
                "Clusters",
                "Applications",
                "Pods",
                "CPU Requests",
                "Memory Requests",
                "CPU Usage",
                "Memory Usage",
                "Cost [USD]",
                "Slack Cost [USD]",
            ]
        )
        for team_id, team in sorted(teams.items()):
            writer.writerow(
                [
                    team_id,
                    len(team["clusters"]),
                    len(team["applications"]),
                    team["pods"],
                    round(team["requests"]["cpu"], 2),
                    round(team["requests"]["memory"], 2),
                    round(team["usage"]["cpu"], 2),
                    round(team["usage"]["memory"], 2),
                    round(team["cost"], 2),
                    round(team["slack_cost"], 2),
                ]
            )

    with out.open("applications.tsv") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        writer.writerow(
            [
                "ID",
                "Team",
                "Clusters",
                "Pods",
                "CPU Requests",
                "Memory Requests",
                "CPU Usage",
                "Memory Usage",
                "Cost [USD]",
                "Slack Cost [USD]",
            ]
        )
        for app_id, app in sorted(applications.items()):
            writer.writerow(
                [
                    app_id,
                    app["team"],
                    len(app["clusters"]),
                    app["pods"],
                    round(app["requests"]["cpu"], 2),
                    round(app["requests"]["memory"], 2),
                    round(app["usage"]["cpu"], 2),
                    round(app["usage"]["memory"], 2),
                    round(app["cost"], 2),
                    round(app["slack_cost"], 2),
                ]
            )

    with out.open("namespaces.tsv") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        writer.writerow(
            [
                "Name",
                "Status",
                "Cluster",
                "Pods",
                "CPU Requests",
                "Memory Requests",
                "CPU Usage",
                "Memory Usage",
                "Cost [USD]",
                "Slack Cost [USD]",
            ]
        )
        for _cluster_id, namespace_item in sorted(namespace_usage.items()):
            fields = [
                namespace_item["id"],
                namespace_item["status"],
                namespace_item["cluster"].id,
                namespace_item["pods"],
                round(namespace_item["requests"]["cpu"], 2),
                round(namespace_item["requests"]["memory"], 2),
                round(namespace_item["usage"]["cpu"], 2),
                round(namespace_item["usage"]["memory"], 2),
                round(namespace_item["cost"], 2),
                round(namespace_item["slack_cost"], 2),
            ]
            writer.writerow(fields)

    with out.open("pods.tsv") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        writer.writerow(
            [
                "Cluster ID",
                "API Server URL",
                "Namespace",
                "Name",
                "Application",
                "Component",
                "Container Images",
                "CPU Requests",
                "Memory Requests",
                "CPU Usage",
                "Memory Usage",
                "CPU Recommendation",
                "Memory Recommendation",
                "Cost [USD]",
                "Slack Cost [USD]",
            ]
        )
        with out.open("slack.tsv") as csvfile2:
            slackwriter = csv.writer(csvfile2, delimiter="\t")
            for cluster_id, summary in sorted(cluster_summaries.items()):
                cpu_slack: collections.Counter = collections.Counter()
                memory_slack: collections.Counter = collections.Counter()
                for k, pod in summary["pods"].items():
                    namespace, name = k
                    requests = pod["requests"]
                    application = pod["application"] or name.rsplit("-", 1)[0]
                    usage = pod.get("usage", collections.defaultdict(float))
                    recommendation = pod.get("recommendation")
                    cpu_slack[(namespace, application)] += (
                        requests["cpu"] - usage["cpu"]
                    )
                    memory_slack[(namespace, application)] += (
                        requests["memory"] - usage["memory"]
                    )
                    writer.writerow(
                        [
                            cluster_id,
                            summary["cluster"].api_server_url,
                            namespace,
                            name,
                            pod["application"],
                            pod["component"],
                            ", ".join(pod["container_images"]),
                            requests["cpu"],
                            requests["memory"],
                            usage["cpu"],
                            usage["memory"],
                            recommendation["cpu"] if recommendation else "",
                            recommendation["memory"] if recommendation else "",
                            pod["cost"],
                            pod["slack_cost"],
                        ]
                    )
                cost_per_cpu = summary["cost"] / summary["allocatable"]["cpu"]
                cost_per_memory = summary["cost"] / summary["allocatable"]["memory"]
                for namespace_name, slack in cpu_slack.most_common(20):
                    namespace, name = namespace_name
                    slackwriter.writerow(
                        [
                            cluster_id,
                            summary["cluster"].api_server_url,
                            namespace,
                            name,
                            "cpu",
                            "{:3.2f}".format(slack),
                            "${:.2f} potential monthly savings".format(
                                slack * cost_per_cpu
                            ),
                        ]
                    )
                for namespace_name, slack in memory_slack.most_common(20):
                    namespace, name = namespace_name
                    slackwriter.writerow(
                        [
                            cluster_id,
                            summary["cluster"].api_server_url,
                            namespace,
                            name,
                            "memory",
                            "{:6.0f}Mi".format(slack / ONE_MEBI),
                            "${:.2f} potential monthly savings".format(
                                slack * cost_per_memory
                            ),
                        ]
                    )


def write_json_files(
    out: OutputManager,
    metrics,
    cluster_summaries,
    applications,
    teams,
    ingresses_by_application,
    pods_by_application,
):
    with out.open("metrics.json") as fd:
        json.dump(metrics, fd)

    with out.open("cluster-metrics.json") as fd:
        json.dump(
            {
                cluster_id: {
                    key: {
                        k if isinstance(k, str) else "/".join(k): v
                        for k, v in value.items()
                    }
                    if hasattr(value, "items")
                    else value
                    for key, value in summary.items()
                    if key != "cluster"
                }
                for cluster_id, summary in cluster_summaries.items()
            },
            fd,
            default=json_default,
        )

    with out.open("team-metrics.json") as fd:
        json.dump(
            {
                team_id: {
                    **team,
                    "application": {
                        app_id: app
                        for app_id, app in applications.items()
                        if app["team"] == team_id
                    },
                }
                for team_id, team in teams.items()
            },
            fd,
            default=json_default,
        )

    with out.open("application-metrics.json") as fd:
        json.dump(applications, fd, default=json_default)

    for app_id, application in applications.items():
        file_name = f"application-{app_id}.json"
        with out.open(file_name) as fd:
            json.dump(
                {
                    **application,
                    "ingresses": [
                        {
                            "cluster": row["cluster_id"],
                            "namespace": row["namespace"],
                            "name": row["name"],
                            "host": row["host"],
                            "status": row["status"],
                        }
                        for row in ingresses_by_application[app_id]
                    ],
                    "pods": [
                        {
                            **row["pod"],
                            "cluster": row["cluster_id"],
                            "namespace": row["namespace"],
                            "name": row["name"],
                        }
                        for row in pods_by_application[app_id]
                    ],
                },
                fd,
                default=json_default,
            )


def write_html_files(
    out,
    context,
    alpha_ema,
    cluster_summaries,
    nodes,
    pods_by_node,
    teams,
    applications,
    ingresses_by_application,
    pods_by_application,
):
    for page in [
        "index",
        "clusters",
        "nodes",
        "ingresses",
        "teams",
        "applications",
        "namespaces",
        "pods",
    ]:
        file_name = f"{page}.html"
        context["page"] = page
        context["alpha_ema"] = alpha_ema
        out.render_template(file_name, context, file_name)

    for cluster_id, summary in cluster_summaries.items():
        page = "clusters"
        file_name = f"cluster-{cluster_id}.html"
        context["page"] = page
        context["cluster_id"] = cluster_id
        context["summary"] = summary
        out.render_template("cluster.html", context, file_name)

    for node_id, node in nodes.items():
        page = "nodes"
        file_name = f"node-{node_id}.html"
        context["page"] = page
        context["cluster_id"] = cluster_id
        context["node"] = node
        context["pods"] = pods_by_node[node_id]
        out.render_template("node.html", context, file_name)

    for team_id, team in teams.items():
        page = "teams"
        file_name = f"team-{team_id}.html"
        context["page"] = page
        context["team_id"] = team_id
        context["team"] = team
        out.render_template("team.html", context, file_name)

    for app_id, application in applications.items():
        page = "applications"
        file_name = f"application-{app_id}.html"
        context["page"] = page
        context["application"] = application
        context["ingresses_by_application"] = ingresses_by_application
        context["pods_by_application"] = pods_by_application
        out.render_template("application.html", context, file_name)


def write_report(
    out: OutputManager,
    start,
    notifications,
    cluster_summaries,
    nodes,
    namespace_usage,
    applications,
    teams,
    node_labels,
    links,
    alpha_ema: float,
):
    write_tsv_files(
        out, cluster_summaries, nodes, namespace_usage, applications, teams, node_labels
    )

    total_allocatable: dict = collections.defaultdict(int)
    total_requests: dict = collections.defaultdict(int)
    total_user_requests: dict = collections.defaultdict(int)

    for summary in cluster_summaries.values():
        for r in "cpu", "memory":
            total_allocatable[r] += summary["allocatable"][r]
            total_requests[r] += summary["requests"][r]
            total_user_requests[r] += summary["user_requests"][r]

    ingresses_by_application: Dict[str, list] = collections.defaultdict(list)
    for cluster_id, summary in cluster_summaries.items():
        for ingress in summary["ingresses"]:
            ingresses_by_application[ingress[2]].append(
                {
                    "cluster_id": cluster_id,
                    "cluster_summary": summary,
                    "namespace": ingress[0],
                    "name": ingress[1],
                    "host": ingress[3],
                    "status": ingress[4],
                }
            )

    pods_by_application: Dict[str, list] = collections.defaultdict(list)
    for cluster_id, summary in cluster_summaries.items():
        for namespace_name, pod in summary["pods"].items():
            namespace, name = namespace_name
            pods_by_application[pod["application"]].append(
                {
                    "cluster_id": cluster_id,
                    "cluster_summary": summary,
                    "namespace": namespace,
                    "name": name,
                    "pod": pod,
                }
            )

    pods_by_node: Dict[str, list] = collections.defaultdict(list)
    for cluster_id, summary in cluster_summaries.items():
        for namespace_name, pod in summary["pods"].items():
            namespace, name = namespace_name
            if "node" not in pod.keys():
                continue
            node_name = pod["node"]
            node_id = f"{cluster_id}.{node_name}"
            pods_by_node[node_id].append(
                {
                    "cluster_id": cluster_id,
                    "node": nodes[node_id],
                    "namespace": namespace,
                    "name": name,
                    "pod": pod,
                }
            )

    total_cost = sum([s["cost"] for s in cluster_summaries.values()])
    total_hourly_cost = total_cost / HOURS_PER_MONTH
    now = datetime.datetime.utcnow()
    context = {
        "links": links,
        "notifications": notifications,
        "cluster_summaries": cluster_summaries,
        "nodes": nodes,
        "teams": teams,
        "applications": applications,
        "namespace_usage": namespace_usage,
        "total_worker_nodes": sum(
            [s["worker_nodes"] for s in cluster_summaries.values()]
        ),
        "total_allocatable": total_allocatable,
        "total_requests": total_requests,
        "total_usage": {
            "cpu": sum(s["usage"]["cpu"] for s in cluster_summaries.values()),
            "memory": sum(s["usage"]["memory"] for s in cluster_summaries.values()),
        },
        "total_user_requests": total_user_requests,
        "total_pods": sum([len(s["pods"]) for s in cluster_summaries.values()]),
        "total_cost": total_cost,
        "total_cost_per_user_request_hour": {
            "cpu": 0.5
            * total_hourly_cost
            / max(total_user_requests["cpu"], MIN_CPU_USER_REQUESTS),
            "memory": 0.5
            * total_hourly_cost
            / max(total_user_requests["memory"] / ONE_GIBI, MIN_MEMORY_USER_REQUESTS),
        },
        "total_slack_cost": sum([a["slack_cost"] for a in applications.values()]),
        "now": now,
        "duration": (now - start).total_seconds(),
        "version": __version__,
    }

    metrics = calculate_metrics(context)
    write_json_files(
        out,
        metrics,
        cluster_summaries,
        applications,
        teams,
        ingresses_by_application,
        pods_by_application,
    )

    write_html_files(
        out,
        context,
        alpha_ema,
        cluster_summaries,
        nodes,
        pods_by_node,
        teams,
        applications,
        ingresses_by_application,
        pods_by_application,
    )

    out.clean_up_stale_files()
