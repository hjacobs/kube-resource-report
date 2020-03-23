#!/usr/bin/env python3

import collections
import csv
import os
import pickle
import datetime
import json
import logging
import re
import requests
import yaml
from pathlib import Path

from typing import Dict, Any, List

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from requests_futures.sessions import FuturesSession

import pykube
from pykube import Namespace, Pod, Node, Ingress, Service, ObjectDoesNotExist
from pykube.objects import APIObject, NamespacedAPIObject

from kube_resource_report import cluster_discovery, pricing, __version__

from .output import OutputManager

NODE_LABEL_SPOT = os.environ.get("NODE_LABEL_SPOT", "aws.amazon.com/spot")
NODE_LABEL_ROLE = os.environ.get("NODE_LABEL_ROLE", "kubernetes.io/role")
# the following labels are used by both AWS and GKE
NODE_LABEL_REGION = os.environ.get(
    "NODE_LABEL_REGION", "failure-domain.beta.kubernetes.io/region"
)
NODE_LABEL_INSTANCE_TYPE = os.environ.get(
    "NODE_LABEL_INSTANCE_TYPE", "beta.kubernetes.io/instance-type"
)

# https://kubernetes.io/docs/concepts/overview/working-with-objects/common-labels/#labels
OBJECT_LABEL_APPLICATION = os.environ.get(
    "OBJECT_LABEL_APPLICATION", "application,app,app.kubernetes.io/name"
).split(",")
OBJECT_LABEL_COMPONENT = os.environ.get(
    "OBJECT_LABEL_COMPONENT", "component,app.kubernetes.io/component"
).split(",")
OBJECT_LABEL_TEAM = os.environ.get("OBJECT_LABEL_TEAM", "team,owner").split(",")

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


def get_component_from_labels(labels):
    for label_name in OBJECT_LABEL_COMPONENT:
        if label_name in labels:
            return labels[label_name]
    return ""


def get_team_from_labels(labels):
    for label_name in OBJECT_LABEL_TEAM:
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

    version = "metrics.k8s.io/v1beta1"
    endpoint = "nodes"
    kind = "NodeMetrics"


# https://github.com/kubernetes/community/blob/master/contributors/design-proposals/instrumentation/resource-metrics-api.md
class PodMetrics(NamespacedAPIObject):

    version = "metrics.k8s.io/v1beta1"
    endpoint = "pods"
    kind = "PodMetrics"


def get_ema(curr_value, prev_value, alpha=1.0):
    """
    More info about EMA: https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average

    The coefficient alpha represents the degree of weighting decrease, a constant smoothing
    factor between 0 and 1. A higher alpha discounts older observations faster.

    Alpha 1 - only the current observation.
    Alpha 0 - only the previous observation.

    Choosing the initial smoothed value - https://en.wikipedia.org/wiki/Exponential_smoothing#Choosing_the_initial_smoothed_value
    """
    if prev_value is None:
        # it is the first run, we do not have any information about the past
        return curr_value

    return prev_value + alpha * (curr_value - prev_value)


def get_node_usage(cluster, nodes: dict, prev_nodes: dict, alpha_ema: float):
    try:
        for node_metrics in NodeMetrics.objects(cluster.client):
            key = node_metrics.name
            node = nodes.get(key)
            prev_node = prev_nodes.get(key, {})

            if node:
                usage: dict = collections.defaultdict(float)
                prev_usage = prev_node.get("usage", {})

                for k, v in node_metrics.obj.get("usage", {}).items():
                    curr_value = parse_resource(v)
                    prev_value = prev_usage.get(k)
                    usage[k] = get_ema(curr_value, prev_value, alpha_ema)
                node["usage"] = usage
    except Exception:
        logger.exception("Failed to get node usage metrics")


def get_pod_usage(cluster, pods: dict, prev_pods: dict, alpha_ema: float):
    try:
        for pod_metrics in PodMetrics.objects(cluster.client, namespace=pykube.all):
            key = (pod_metrics.namespace, pod_metrics.name)
            pod = pods.get(key)
            prev_pod = prev_pods.get(key, {})

            if pod:
                usage: dict = collections.defaultdict(float)
                prev_usage = prev_pod.get("usage", {})

                for container in pod_metrics.obj["containers"]:
                    for k, v in container.get("usage", {}).items():
                        usage[k] += parse_resource(v)

                for k, v in usage.items():
                    curr_value = v
                    prev_value = prev_usage.get(k)
                    usage[k] = get_ema(curr_value, prev_value, alpha_ema)

                pod["usage"] = usage
    except Exception:
        logger.exception("Failed to get pod usage metrics")


def find_backend_application(client, ingress, rule):
    """
    The Ingress object might not have a "application" label, so let's try to find the application by looking at the backend service and its pods
    """
    paths = rule.get("http", {}).get("paths", [])
    selectors = []
    for path in paths:
        service_name = path.get("backend", {}).get("serviceName")
        if service_name:
            try:
                service = Service.objects(client, namespace=ingress.namespace).get(
                    name=service_name
                )
            except ObjectDoesNotExist:
                logger.debug(
                    f"Referenced service does not exist: {ingress.namespace}/{service_name}"
                )
            else:
                selector = service.obj["spec"].get("selector", {})
                selectors.append(selector)
                application = get_application_from_labels(selector)
                if application:
                    return application
    # we still haven't found the application, let's look up pods by label selectors
    for selector in selectors:
        application_candidates = set()
        for pod in Pod.objects(client).filter(
            namespace=ingress.namespace, selector=selector
        ):
            application = get_application_from_labels(pod.labels)
            if application:
                application_candidates.add(application)

        if len(application_candidates) == 1:
            return application_candidates.pop()
    return ""


def pod_active(pod):
    pod_status = pod.obj["status"]
    phase = pod_status.get("phase")

    if phase == "Running":
        return True
    elif phase == "Pending":
        for condition in pod_status.get("conditions", []):
            if condition.get("type") == "PodScheduled":
                return condition.get("status") == "True"

    return False


def query_cluster(
    cluster,
    executor,
    system_namespaces,
    additional_cost_per_cluster,
    alpha_ema,
    prev_cluster_summaries,
    no_ingress_status,
    node_labels,
):
    logger.info(f"Querying cluster {cluster.id} ({cluster.api_server_url})..")
    pods = {}
    nodes = {}
    namespaces = {}

    for namespace in Namespace.objects(cluster.client):
        email = namespace.annotations.get("email")
        namespaces[namespace.name] = {
            "status": namespace.obj["status"]["phase"],
            "email": email,
        }

    cluster_capacity = collections.defaultdict(float)
    cluster_allocatable = collections.defaultdict(float)
    cluster_requests = collections.defaultdict(float)
    user_requests = collections.defaultdict(float)
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
        region = _node.labels.get(NODE_LABEL_REGION, "unknown")
        instance_type = _node.labels.get(NODE_LABEL_INSTANCE_TYPE, "unknown")
        is_spot = _node.labels.get(NODE_LABEL_SPOT) == "true"
        node["spot"] = is_spot
        node["kubelet_version"] = (
            node["status"].get("nodeInfo", {}).get("kubeletVersion", "")
        )
        node["role"] = role
        node["instance_type"] = instance_type
        node["cost"] = pricing.get_node_cost(
            region,
            instance_type,
            is_spot,
            cpu=node["capacity"].get("cpu"),
            memory=node["capacity"].get("memory"),
        )
        cluster_cost += node["cost"]

    get_node_usage(cluster, nodes, prev_cluster_summaries.get("nodes", {}), alpha_ema)

    cluster_usage = collections.defaultdict(float)
    for node in nodes.values():
        for k, v in node["usage"].items():
            cluster_usage[k] += v

    cost_per_cpu = cluster_cost / cluster_allocatable["cpu"]
    cost_per_memory = cluster_cost / cluster_allocatable["memory"]

    for pod in Pod.objects(cluster.client, namespace=pykube.all):
        # ignore unschedulable/completed pods
        if not pod_active(pod):
            continue
        application = get_application_from_labels(pod.labels)
        component = get_component_from_labels(pod.labels)
        team = get_team_from_labels(pod.labels)
        requests = collections.defaultdict(float)
        ns = pod.namespace
        container_images = []
        for container in pod.obj["spec"]["containers"]:
            # note that the "image" field is optional according to Kubernetes docs
            image = container.get("image")
            if image:
                container_images.append(image)
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
            "component": component,
            "container_images": container_images,
            "cost": cost,
            "usage": new_resources(),
            "team": team,
        }

    hourly_cost = cluster_cost / HOURS_PER_MONTH

    cluster_summary = {
        "cluster": cluster,
        "nodes": nodes,
        "pods": pods,
        "namespaces": namespaces,
        "user_pods": len([p for ns, p in pods if ns not in system_namespaces]),
        "master_nodes": len([n for n in nodes.values() if n["role"] == "master"]),
        "worker_nodes": len([n for n in nodes.values() if n["role"] in node_labels]),
        "kubelet_versions": set(
            [n["kubelet_version"] for n in nodes.values() if n["role"] in node_labels]
        ),
        "worker_instance_types": set(
            [n["instance_type"] for n in nodes.values() if n["role"] in node_labels]
        ),
        "worker_instance_is_spot": any(
            [n["spot"] for n in nodes.values() if n["role"] in node_labels]
        ),
        "capacity": cluster_capacity,
        "allocatable": cluster_allocatable,
        "requests": cluster_requests,
        "user_requests": user_requests,
        "usage": cluster_usage,
        "cost": cluster_cost,
        "cost_per_user_request_hour": {
            "cpu": 0.5 * hourly_cost / max(user_requests["cpu"], MIN_CPU_USER_REQUESTS),
            "memory": 0.5
            * hourly_cost
            / max(user_requests["memory"] / ONE_GIBI, MIN_MEMORY_USER_REQUESTS),
        },
        "ingresses": [],
    }

    get_pod_usage(cluster, pods, prev_cluster_summaries.get("pods", {}), alpha_ema)

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
                host = rule.get("host", "")
                if not application:
                    # find the application by getting labels from pods
                    backend_application = find_backend_application(
                        cluster.client, _ingress, rule
                    )
                else:
                    backend_application = None
                ingress = [
                    _ingress.namespace,
                    _ingress.name,
                    application or backend_application,
                    host,
                    0,
                ]
                if host and not no_ingress_status:
                    try:
                        future = futures_by_host[host]
                    except KeyError:
                        future = futures_session.get(f"https://{host}/", timeout=5)
                        futures_by_host[host] = future
                    futures[future].append(ingress)
                cluster_summary["ingresses"].append(ingress)

        if not no_ingress_status:
            logger.info(
                f"Waiting for ingress status for {cluster.id} ({cluster.api_server_url}).."
            )
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
    alpha_ema: float,
    prev_cluster_summaries: dict,
    no_ingress_status: bool,
    node_labels: list,
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
                        alpha_ema,
                        prev_cluster_summaries.get(cluster.id, {}),
                        no_ingress_status,
                        node_labels,
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
    alpha_ema,
    cluster_summaries,
    pricing_file,
    links_file,
    node_labels,
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

    out = OutputManager(Path(output_dir))
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
        )
        teams = {}

    applications: Dict[str, dict] = {}
    namespace_usage: Dict[tuple, dict] = {}

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
            app["team"] = pod["team"]
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
                namespace["requests"][r] = (
                    namespace["requests"].get(r, 0) + pod["requests"][r]
                )
                namespace["usage"][r] = namespace["usage"].get(r, 0) + pod.get(
                    "usage", {}
                ).get(r, 0)
            namespace["cost"] += pod["cost"]
            namespace["slack_cost"] += pod.get("slack_cost", 0)
            namespace["pods"] += 1
            namespace["cluster"] = summary["cluster"]
            namespace_usage[(ns_pod[0], cluster_id)] = namespace

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

    for cluster_id, summary in sorted(cluster_summaries.items()):
        for k, pod in summary["pods"].items():
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


def write_report(
    out: OutputManager,
    start,
    notifications,
    cluster_summaries,
    namespace_usage,
    applications,
    teams,
    node_labels,
    links,
    alpha_ema: float,
):
    total_allocatable: dict = collections.defaultdict(int)
    total_requests: dict = collections.defaultdict(int)
    total_user_requests: dict = collections.defaultdict(int)

    for cluster_id, summary in sorted(cluster_summaries.items()):
        for r in "cpu", "memory":
            total_allocatable[r] += summary["allocatable"][r]
            total_requests[r] += summary["requests"][r]
            total_user_requests[r] += summary["user_requests"][r]

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
        for cluster_id, namespace_item in sorted(namespace_usage.items()):
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

    with out.open("metrics.json") as fd:
        json.dump(metrics, fd)

    for page in [
        "index",
        "clusters",
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

    for team_id, team in teams.items():
        page = "teams"
        file_name = f"team-{team_id}.html"
        context["page"] = page
        context["team_id"] = team_id
        context["team"] = team
        out.render_template("team.html", context, file_name)

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

    for app_id, application in applications.items():
        page = "applications"
        file_name = f"application-{app_id}.html"
        context["page"] = page
        context["application"] = application
        context["ingresses_by_application"] = ingresses_by_application
        context["pods_by_application"] = pods_by_application
        out.render_template("application.html", context, file_name)

    out.clean_up_stale_files()
