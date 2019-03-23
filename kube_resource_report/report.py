#!/usr/bin/env python3

import collections
import csv
import pickle
import datetime
import json
import logging
import re
import requests
import shutil
import yaml
from pathlib import Path

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from requests_futures.sessions import FuturesSession

from jinja2 import Environment, FileSystemLoader, select_autoescape

import pykube
from pykube import Namespace, Pod, Node, Ingress
from pykube.objects import APIObject, NamespacedAPIObject

from kube_resource_report import cluster_discovery, pricing, filters, __version__

# TODO: this should be configurable
NODE_LABEL_SPOT = "aws.amazon.com/spot"
NODE_LABEL_ROLE = "kubernetes.io/role"
NODE_LABEL_REGION = "failure-domain.beta.kubernetes.io/region"
NODE_LABEL_INSTANCE_TYPE = "beta.kubernetes.io/instance-type"

OBJECT_LABEL_APPLICATION = ["application", "app"]

ONE_MEBI = 1024 ** 2
ONE_GIBI = 1024 ** 3

# we show costs per month by default as it leads to easily digestable numbers (for humans)
AVG_DAYS_PER_MONTH = 30.4375
HOURS_PER_DAY = 24
HOURS_PER_MONTH = HOURS_PER_DAY * AVG_DAYS_PER_MONTH

# assume minimal requests even if no user requests are set
MIN_CPU_USER_REQUESTS = 1 / 1000
MIN_MEMORY_USER_REQUESTS = 1 / 1000

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

RESOURCE_PATTERN = re.compile(r"^(\d*)(\D*)$")


def new_resources():
    return {"cpu": 0, "memory": 0}


def parse_resource(v):
    """
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


def get_application_from_labels(labels):
    for label_name in OBJECT_LABEL_APPLICATION:
        if label_name in labels:
            return labels[label_name]
    return ""


session = requests.Session()
# set a friendly user agent for outgoing HTTP requests
session.headers["User-Agent"] = f"kube-resource-report/{__version__}"


def json_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(obj)


logger = logging.getLogger(__name__)


# https://github.com/kubernetes/community/blob/master/contributors/design-proposals/instrumentation/resource-metrics-api.md
class NodeMetrics(APIObject):

    version = 'metrics.k8s.io/v1beta1'
    endpoint = 'nodes'
    kind = 'NodeMetrics'


# https://github.com/kubernetes/community/blob/master/contributors/design-proposals/instrumentation/resource-metrics-api.md
class PodMetrics(NamespacedAPIObject):

    version = 'metrics.k8s.io/v1beta1'
    endpoint = 'pods'
    kind = 'PodMetrics'


def get_node_usage(cluster, nodes: dict):
    try:
        for node_metrics in NodeMetrics.objects(cluster.client):
            key = node_metrics.name
            node = nodes.get(key)
            if node:
                usage = collections.defaultdict(float)
                for k, v in node_metrics.obj.get("usage", {}).items():
                    usage[k] += parse_resource(v)
                node["usage"] = usage
    except Exception:
        logger.exception("Failed to get node usage metrics")


def get_pod_usage(cluster, pods: dict):
    try:
        for pod_metrics in PodMetrics.objects(cluster.client):
            key = (pod_metrics.namespace, pod_metrics.name)
            pod = pods.get(key)
            if pod:
                usage = collections.defaultdict(float)
                for container in pod_metrics.obj["containers"]:
                    for k, v in container.get("usage", {}).items():
                        usage[k] += parse_resource(v)
                pod["usage"] = usage
    except Exception:
        logger.exception("Failed to get pod usage metrics")


def query_cluster(
    cluster, executor, system_namespaces, additional_cost_per_cluster, no_ingress_status, node_label
):
    logger.info(f"Querying cluster {cluster.id} ({cluster.api_server_url})..")
    pods = {}
    nodes = {}
    namespaces = {}

    for namespace in Namespace.objects(cluster.client):
        email = namespace.annotations.get('email')
        namespaces[namespace.name] = {
            "status": namespace.obj['status']['phase'],
            "email": email,
        }

    cluster_capacity = collections.defaultdict(float)
    cluster_allocatable = collections.defaultdict(float)
    cluster_requests = collections.defaultdict(float)
    user_requests = collections.defaultdict(float)
    node_count = collections.defaultdict(int)
    cluster_cost = additional_cost_per_cluster

    for _node in Node.objects(cluster.client):
        node = _node.obj
        nodes[_node.name] = node
        node["capacity"] = {}
        node["allocatable"] = {}
        node["requests"] = new_resources()
        node["usage"] = new_resources()
        for k, v in node["status"].get("capacity", {}).items():
            parsed = parse_resource(v)
            node["capacity"][k] = parsed
            cluster_capacity[k] += parsed
        for k, v in node["status"].get("allocatable", {}).items():
            parsed = parse_resource(v)
            node["allocatable"][k] = parsed
            cluster_allocatable[k] += parsed
        role = _node.labels.get(NODE_LABEL_ROLE) or "worker"
        node_count[role] += 1
        region = _node.labels.get(NODE_LABEL_REGION, "unknown")
        instance_type = _node.labels.get(NODE_LABEL_INSTANCE_TYPE, "unknown")
        is_spot = _node.labels.get(NODE_LABEL_SPOT) == "true"
        node["spot"] = is_spot
        node["kubelet_version"] = (
            node["status"].get("nodeInfo", {}).get("kubeletVersion", "")
        )
        node["role"] = role
        node["instance_type"] = instance_type
        node["cost"] = pricing.get_node_cost(region, instance_type, is_spot)
        cluster_cost += node["cost"]

    get_node_usage(cluster, nodes)

    cluster_usage = collections.defaultdict(float)
    for node in nodes.values():
        for k, v in node['usage'].items():
            cluster_usage[k] += v

    cost_per_cpu = cluster_cost / cluster_allocatable["cpu"]
    cost_per_memory = cluster_cost / cluster_allocatable["memory"]

    for pod in Pod.objects(cluster.client, namespace=pykube.all):
        if pod.obj["status"].get("phase") != "Running":
            # ignore unschedulable/completed pods
            continue
        application = get_application_from_labels(pod.labels)
        requests = collections.defaultdict(float)
        ns = pod.namespace
        for container in pod.obj["spec"]["containers"]:
            for k, v in container["resources"].get("requests", {}).items():
                pv = parse_resource(v)
                requests[k] += pv
                cluster_requests[k] += pv
                if ns not in system_namespaces:
                    user_requests[k] += pv
        if "nodeName" in pod.obj["spec"] and pod.obj["spec"]["nodeName"] in nodes:
            for k in ("cpu", "memory"):
                nodes[pod.obj["spec"]["nodeName"]]["requests"][k] += requests.get(k, 0)
        cost = max(requests["cpu"] * cost_per_cpu, requests["memory"] * cost_per_memory)
        pods[(ns, pod.name)] = {
            "requests": requests,
            "application": application,
            "cost": cost,
            "usage": new_resources(),
        }

    hourly_cost = cluster_cost / HOURS_PER_MONTH

    cluster_summary = {
        "cluster": cluster,
        "nodes": nodes,
        "pods": pods,
        "namespaces": namespaces,
        "user_pods": len([p for ns, p in pods if ns not in system_namespaces]),
        "master_nodes": node_count["master"],
        "worker_nodes": node_count[node_label],
        "kubelet_versions": set(
            [n["kubelet_version"] for n in nodes.values() if n["role"] == node_label]
        ),
        "worker_instance_types": set(
            [n["instance_type"] for n in nodes.values() if n["role"] == node_label]
        ),
        "worker_instance_is_spot": any(
            [n["spot"] for n in nodes.values() if n["role"] == node_label]
        ),
        "capacity": cluster_capacity,
        "allocatable": cluster_allocatable,
        "requests": cluster_requests,
        "user_requests": user_requests,
        "usage": cluster_usage,
        "cost": cluster_cost,
        "cost_per_user_request_hour": {
            "cpu": 0.5 * hourly_cost / max(user_requests["cpu"], MIN_CPU_USER_REQUESTS),
            "memory": 0.5 * hourly_cost / max(user_requests["memory"] / ONE_GIBI, MIN_MEMORY_USER_REQUESTS),
        },
        "ingresses": [],
    }

    get_pod_usage(cluster, pods)

    cluster_slack_cost = 0
    for pod in pods.values():
        usage_cost = max(
            pod["usage"]["cpu"] * cost_per_cpu,
            pod["usage"]["memory"] * cost_per_memory,
        )
        pod["slack_cost"] = pod["cost"] - usage_cost
        cluster_slack_cost += pod["slack_cost"]

    cluster_summary["slack_cost"] = min(cluster_cost, cluster_slack_cost)

    with FuturesSession(max_workers=10, session=session) as futures_session:
        futures_by_host = {}  # hostname -> future
        futures = collections.defaultdict(list)  # future -> [ingress]

        for _ingress in Ingress.objects(cluster.client, namespace=pykube.all):
            application = get_application_from_labels(_ingress.labels)
            for rule in _ingress.obj["spec"].get("rules", []):
                host = rule.get('host', '')
                ingress = [_ingress.namespace, _ingress.name, application, host, 0]
                if host and not no_ingress_status:
                    try:
                        future = futures_by_host[host]
                    except KeyError:
                        future = futures_session.get(f"https://{host}/", timeout=5)
                        futures_by_host[host] = future
                    futures[future].append(ingress)
                cluster_summary["ingresses"].append(ingress)

        if not no_ingress_status:
            logger.info(f'Waiting for ingress status for {cluster.id} ({cluster.api_server_url})..')
            for future in concurrent.futures.as_completed(futures):
                ingresses = futures[future]
                try:
                    response = future.result()
                    status = response.status_code
                except:
                    status = 999
                for ingress in ingresses:
                    ingress[4] = status

    return cluster_summary


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
    no_ingress_status: bool,
    node_label: str,
):
    cluster_summaries = {}

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

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_cluster = {}
        for cluster in discoverer.get_clusters():
            if (not include_pattern or include_pattern.match(cluster.id)) and (
                not exclude_pattern or not exclude_pattern.match(cluster.id)
            ):
                future_to_cluster[
                    executor.submit(
                        query_cluster,
                        cluster,
                        executor,
                        system_namespaces,
                        additional_cost_per_cluster,
                        no_ingress_status,
                        node_label,
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

    sorted_by_name = sorted(cluster_summaries.values(), key=lambda summary: summary["cluster"].name)
    return {summary["cluster"].id: summary for summary in sorted_by_name}


def resolve_application_ids(applications: dict, teams: dict, application_registry: str):
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
                team["usage"][r] = team["usage"].get(r, 0) + app.get("usage", {}).get(
                    r, 0
                )
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


def generate_report(
    clusters,
    cluster_registry,
    kubeconfig_path,
    kubeconfig_contexts: set,
    application_registry,
    use_cache,
    no_ingress_status,
    output_dir,
    system_namespaces,
    include_clusters,
    exclude_clusters,
    additional_cost_per_cluster,
    pricing_file,
    links_file,
    node_label,
):
    notifications = []

    output_path = Path(output_dir)

    if pricing_file:
        pricing.regenerate_cost_dict(pricing_file)

    if links_file:
        with open(links_file, 'rb') as fd:
            links = yaml.safe_load(fd)
    else:
        links = {}

    start = datetime.datetime.utcnow()

    pickle_path = output_path / "dump.pickle"

    if use_cache and pickle_path.exists():
        with pickle_path.open("rb") as fd:
            data = pickle.load(fd)
        cluster_summaries = data["cluster_summaries"]
        teams = data["teams"]
        applications = data["applications"]

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
            no_ingress_status,
            node_label,
        )
        teams = {}
        applications = {}
        namespace_usage = {}

    for cluster_id, summary in sorted(cluster_summaries.items()):
        for k, pod in summary["pods"].items():
            app = applications.get(
                pod["application"],
                {
                    "id": pod["application"],
                    "cost": 0,
                    "slack_cost": 0,
                    "pods": 0,
                    "requests": {},
                    "usage": {},
                    "clusters": set(),
                    "team": "",
                    "active": None,
                },
            )
            for r in "cpu", "memory":
                app["requests"][r] = app["requests"].get(r, 0) + pod["requests"][r]
                app["usage"][r] = app["usage"].get(r, 0) + pod.get("usage", {}).get(
                    r, 0
                )
            app["cost"] += pod["cost"]
            app["slack_cost"] += pod.get("slack_cost", 0)
            app["pods"] += 1
            app["clusters"].add(cluster_id)
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
                    "cluster": "",
                    "email": "",
                    "status": "",
                },
            )
            for r in "cpu", "memory":
                namespace["requests"][r] = namespace["requests"].get(r, 0) + pod["requests"][r]
                namespace["usage"][r] = namespace["usage"].get(r, 0) + pod.get("usage", {}).get(
                    r, 0
                )
            namespace["cost"] += pod["cost"]
            namespace["slack_cost"] += pod.get("slack_cost", 0)
            namespace["pods"] += 1
            namespace["cluster"] = summary["cluster"]
            namespace_usage[(ns_pod[0], cluster_id)] = namespace

    if application_registry:
        resolve_application_ids(applications, teams, application_registry)

    for team in teams.values():
        def cluster_name(cluster_id):
            try:
                return cluster_summaries[cluster_id]["cluster"].name
            except KeyError:
                return None
        team["clusters"] = sorted(team["clusters"], key=cluster_name)

    for cluster_id, summary in sorted(cluster_summaries.items()):
        for k, pod in summary["pods"].items():
            app = applications.get(pod["application"])
            pod["team"] = app["team"]

    for cluster_id, summary in sorted(cluster_summaries.items()):
        for ns, ns_values in summary["namespaces"].items():
            namespace = namespace_usage.get((ns, cluster_id))
            if namespace:
                namespace["email"] = ns_values["email"]
                namespace["status"] = ns_values["status"]

    if not use_cache:
        try:
            with pickle_path.open("wb") as fd:
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
            logger.error(f'Could not dump pickled cache data: {e}')

    write_report(output_path, start, notifications, cluster_summaries, namespace_usage, applications, teams, node_label, links)

    return cluster_summaries


def write_report(output_path: Path, start, notifications, cluster_summaries, namespace_usage, applications, teams, node_label, links):
    total_allocatable = collections.defaultdict(int)
    total_requests = collections.defaultdict(int)
    total_user_requests = collections.defaultdict(int)

    for cluster_id, summary in sorted(cluster_summaries.items()):
        for r in "cpu", "memory":
            total_allocatable[r] += summary["allocatable"][r]
            total_requests[r] += summary["requests"][r]
            total_user_requests[r] += summary["user_requests"][r]

    logger.info("Writing namespaces.tsv..")
    with (output_path / "namespaces.tsv").open("w") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        for cluster_id, namespace_item in sorted(namespace_usage.items()):
            fields = [
                namespace_item["id"],
                namespace_item["status"],
                namespace_item["cluster"],
                namespace_item["pods"],
                namespace_item["requests"]["cpu"],
                namespace_item["requests"]["memory"],
                round(namespace_item["usage"]["cpu"], 2),
                round(namespace_item["usage"]["memory"], 2),
                round(namespace_item["cost"], 2),
                round(namespace_item["slack_cost"], 2)
            ]
            writer.writerow(fields)

    logger.info("Writing clusters.tsv..")
    with (output_path / "clusters.tsv").open("w") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        for cluster_id, summary in sorted(cluster_summaries.items()):
            worker_instance_type = set()
            kubelet_version = set()
            for node in summary["nodes"].values():
                if node["role"] == node_label:
                    worker_instance_type.add(node["instance_type"])
                kubelet_version.add(node["kubelet_version"])
            fields = [
                cluster_id,
                summary["cluster"].api_server_url,
                summary["master_nodes"],
                summary["worker_nodes"],
                ",".join(worker_instance_type),
                ",".join(kubelet_version),
            ]
            for x in ["capacity", "allocatable", "requests", "usage"]:
                fields += [
                    round(summary[x]["cpu"], 2),
                    int(summary[x]["memory"] / ONE_MEBI),
                ]
            fields += [round(summary["cost"], 2)]
            writer.writerow(fields)

    logger.info("Writing ingresses.tsv..")
    with (output_path / "ingresses.tsv").open("w") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        for cluster_id, summary in sorted(cluster_summaries.items()):
            for ingress in summary["ingresses"]:
                writer.writerow(
                    [cluster_id, summary["cluster"].api_server_url] + ingress
                )

    logger.info("Writing pods.tsv..")
    with (output_path / "pods.tsv").open("w") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        with (output_path / "slack.tsv").open("w") as csvfile2:
            slackwriter = csv.writer(csvfile2, delimiter="\t")
            for cluster_id, summary in sorted(cluster_summaries.items()):
                cpu_slack = collections.Counter()
                memory_slack = collections.Counter()
                for k, pod in summary["pods"].items():
                    namespace, name = k
                    requests = pod["requests"]
                    application = pod["application"] or name.rsplit("-", 1)[0]
                    usage = pod.get("usage", collections.defaultdict(float))
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
                            requests["cpu"],
                            requests["memory"],
                            usage["cpu"],
                            usage["memory"],
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

    templates_path = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(templates_path)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    env.filters["money"] = filters.money
    env.filters["cpu"] = filters.cpu
    env.filters["memory"] = filters.memory
    total_cost = sum([s["cost"] for s in cluster_summaries.values()])
    total_hourly_cost = total_cost / HOURS_PER_MONTH
    now = datetime.datetime.utcnow()
    context = {
        "links": links,
        "notifications": notifications,
        "cluster_summaries": cluster_summaries,
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
            "cpu": 0.5 * total_hourly_cost / max(total_user_requests["cpu"], MIN_CPU_USER_REQUESTS),
            "memory": 0.5 * total_hourly_cost / max(
                total_user_requests["memory"] / ONE_GIBI, MIN_MEMORY_USER_REQUESTS),
        },
        "total_slack_cost": sum([a["slack_cost"] for a in applications.values()]),
        "now": now,
        "duration": (now - start).total_seconds(),
        "version": __version__,
    }

    metrics = calculate_metrics(context)

    with (output_path / "metrics.json").open("w") as fd:
        json.dump(metrics, fd)

    for page in ["index", "clusters", "ingresses", "teams", "applications", "namespaces", "pods"]:
        file_name = f"{page}.html"
        logger.info(f"Generating {file_name}..")
        template = env.get_template(file_name)
        context["page"] = page
        template.stream(**context).dump(str(output_path / file_name))

    for cluster_id, summary in cluster_summaries.items():
        page = "clusters"
        file_name = f"cluster-{cluster_id}.html"
        logger.info(f"Generating {file_name}..")
        template = env.get_template("cluster.html")
        context["page"] = page
        context["cluster_id"] = cluster_id
        context["summary"] = summary
        template.stream(**context).dump(str(output_path / file_name))

    with (output_path / "cluster-metrics.json").open("w") as fd:
        json.dump(
            {
                cluster_id: {
                    key: {
                        k if isinstance(k, str) else '/'.join(k): v
                        for k, v in value.items()
                    } if hasattr(value, 'items') else value
                    for key, value in summary.items()
                    if key != 'cluster'
                }
                for cluster_id, summary in cluster_summaries.items()
            },
            fd,
            default=json_default
        )

    for team_id, team in teams.items():
        page = "teams"
        file_name = f"team-{team_id}.html"
        logger.info(f"Generating {file_name}..")
        template = env.get_template("team.html")
        context["page"] = page
        context["team_id"] = team_id
        context["team"] = team
        template.stream(**context).dump(str(output_path / file_name))

    with (output_path / "team-metrics.json").open("w") as fd:
        json.dump(
            {
                team_id: {
                    **team,
                    "application": {
                        app_id: app
                        for app_id, app in applications.items()
                        if app["team"] == team_id
                    }
                }
                for team_id, team in teams.items()
            },
            fd,
            default=json_default
        )

    with (output_path / "application-metrics.json").open("w") as fd:
        json.dump(applications, fd, default=json_default)

    assets_path = output_path / "assets"
    assets_path.mkdir(exist_ok=True)

    assets_source_path = templates_path / "assets"

    for path in assets_source_path.iterdir():
        if path.match("*.js") or path.match("*.css") or path.match("*.png"):
            shutil.copy(str(path), str(assets_path / path.name))
