"""Query information from a Kubernetes cluster."""
import collections
import concurrent.futures
import logging
import os
import time
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

import pykube
import requests
from pykube import Ingress
from pykube import Namespace
from pykube import Node
from pykube import ObjectDoesNotExist
from pykube import Pod
from pykube import Service
from requests_futures.sessions import FuturesSession

from .recommender import Recommender
from .utils import HOURS_PER_MONTH
from .utils import MIN_CPU_USER_REQUESTS
from .utils import MIN_MEMORY_USER_REQUESTS
from .utils import ONE_GIBI
from .utils import parse_resource
from .vpa import get_vpas_by_match_labels
from kube_resource_report import __version__
from kube_resource_report import metrics
from kube_resource_report import pricing

NODE_LABEL_SPOT = os.environ.get("NODE_LABEL_SPOT", "aws.amazon.com/spot")
NODE_LABEL_SPOT_VALUE = os.environ.get("NODE_LABEL_SPOT_VALUE", "true")
NODE_LABEL_PREEMPTIBLE = os.environ.get(
    "NODE_LABEL_PREEMPTIBLE", "cloud.google.com/gke-preemptible"
)
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


logger = logging.getLogger(__name__)

session = requests.Session()
# set a friendly user agent for outgoing HTTP requests
session.headers["User-Agent"] = f"kube-resource-report/{__version__}"


def new_resources():
    return {"cpu": 0, "memory": 0}


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


def find_backend_application(client: pykube.HTTPClient, ingress: Ingress, rule):
    """
    Find the application ID for a given Ingress object.

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


def convert(lst):
    res_dct = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)}
    return res_dct


def exclude_node(node, node_exclude_labels):
    logger.debug(f"node_exclusions are {node_exclude_labels}")
    for label in node_exclude_labels:
        label_pair = label.split('=')
        label_dict = convert(label_pair)
        logger.debug(f"node_exclusion label_dict is {label_dict}")
        for k in label_dict:
            logger.debug(f"label_pair k is {k} value is {label_dict[k]}")
            node_labels = node.labels.get(k)
            if node_labels is not None and node_labels in label_dict[k]:
                logger.debug(f"node_labels {node_labels} found on node {node}. Excluding this node.")
                return True
    return False


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


def map_node(_node: Node):
    """Map a Kubernetes Node object to our internal structure."""

    node: Dict[str, Any] = {}
    node["capacity"] = {}
    node["allocatable"] = {}
    node["requests"] = new_resources()
    node["usage"] = new_resources()

    status = _node.obj["status"]
    for k, v in status.get("capacity", {}).items():
        parsed = parse_resource(v)
        node["capacity"][k] = parsed

    for k, v in status.get("allocatable", {}).items():
        parsed = parse_resource(v)
        node["allocatable"][k] = parsed

    role = _node.labels.get(NODE_LABEL_ROLE) or "worker"
    region = _node.labels.get(NODE_LABEL_REGION, "unknown")
    instance_type = _node.labels.get(NODE_LABEL_INSTANCE_TYPE, "unknown")
    is_spot = _node.labels.get(NODE_LABEL_SPOT) == NODE_LABEL_SPOT_VALUE
    is_preemptible = _node.labels.get(NODE_LABEL_PREEMPTIBLE, "false") == "true"
    if is_preemptible:
        instance_type = instance_type + "-preemptible"
    node["spot"] = is_spot or is_preemptible
    node["kubelet_version"] = status.get("nodeInfo", {}).get("kubeletVersion", "")
    node["role"] = role
    node["instance_type"] = instance_type
    node["cost"] = pricing.get_node_cost(
        region,
        instance_type,
        is_spot,
        cpu=node["capacity"].get("cpu"),
        memory=node["capacity"].get("memory"),
    )
    return node


def map_pod(pod: Pod, cost_per_cpu: float, cost_per_memory: float):
    """Map a Kubernetes Pod object to our internal structure."""

    application = get_application_from_labels(pod.labels)
    component = get_component_from_labels(pod.labels)
    team = get_team_from_labels(pod.labels)
    requests: Dict[str, float] = collections.defaultdict(float)
    container_images = []
    container_names = []
    for container in pod.obj["spec"]["containers"]:
        container_names.append(container.get("name", ""))
        # note that the "image" field is optional according to Kubernetes docs
        image = container.get("image")
        if image:
            container_images.append(image)
        for k, v in container["resources"].get("requests", {}).items():
            pv = parse_resource(v)
            requests[k] += pv
    cost = max(requests["cpu"] * cost_per_cpu, requests["memory"] * cost_per_memory)
    return {
        "requests": requests,
        "application": application,
        "component": component,
        "container_names": container_names,
        "container_images": container_images,
        "cost": cost,
        "usage": new_resources(),
        "team": team,
    }


def query_cluster(
    cluster,
    executor,
    system_namespaces,
    additional_cost_per_cluster,
    alpha_ema,
    prev_cluster_summaries,
    no_ingress_status,
    node_labels,
    node_exclude_labels,
    data_path: Path,
    map_node_hook=Optional[Callable[[Node, dict], None]],
    map_pod_hook=Optional[Callable[[Pod, dict], None]],
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

    cluster_capacity: Dict[str, float] = collections.defaultdict(float)
    cluster_allocatable: Dict[str, float] = collections.defaultdict(float)
    cluster_requests: Dict[str, float] = collections.defaultdict(float)
    user_requests: Dict[str, float] = collections.defaultdict(float)
    cluster_cost = additional_cost_per_cluster

    for _node in Node.objects(cluster.client):
        # skip/hide nodes which contain the node_exclude_labels labels
        if exclude_node(_node, node_exclude_labels):
            continue
        node = map_node(_node)
        if map_node_hook:
            map_node_hook(_node, node)
        nodes[_node.name] = node

        for k, v in node["capacity"].items():
            cluster_capacity[k] += v
        for k, v in node["allocatable"].items():
            cluster_allocatable[k] += v
        cluster_cost += node["cost"]

    metrics.get_node_usage(
        cluster, nodes, prev_cluster_summaries.get("nodes", {}), alpha_ema
    )

    cluster_usage: Dict[str, float] = collections.defaultdict(float)
    for node in nodes.values():
        for k, v in node["usage"].items():
            cluster_usage[k] += v

    try:
        vpas_by_namespace_label = get_vpas_by_match_labels(cluster.client)
    except Exception as e:
        logger.warning(f"Failed to query VPAs in cluster {cluster.id}: {e}")
        vpas_by_namespace_label = collections.defaultdict(list)

    cost_per_cpu = cluster_cost / cluster_allocatable["cpu"]
    cost_per_memory = cluster_cost / cluster_allocatable["memory"]

    for pod in Pod.objects(cluster.client, namespace=pykube.all):
        # ignore unschedulable/completed pods
        if not pod_active(pod):
            continue
        pod_ = map_pod(pod, cost_per_cpu, cost_per_memory)
        if map_pod_hook:
            map_pod_hook(pod, pod_)
        for k, v in pod_["requests"].items():
            cluster_requests[k] += v
            if pod.namespace not in system_namespaces:
                user_requests[k] += v
        node_name = pod.obj["spec"].get("nodeName")
        if node_name and node_name in nodes:
            for k in ("cpu", "memory"):
                nodes[node_name]["requests"][k] += pod_["requests"].get(k, 0)
        found_vpa = False
        for k, v in pod.labels.items():
            vpas = vpas_by_namespace_label[(pod.namespace, k, v)]
            for vpa in vpas:
                if vpa.matches_pod(pod):
                    recommendation = new_resources()
                    container_names = set()
                    for container in pod.obj["spec"]["containers"]:
                        container_names.add(container["name"])
                    for container in vpa.container_recommendations:
                        # VPA might contain recommendations for containers which are no longer there!
                        if container["containerName"] in container_names:
                            for k in ("cpu", "memory"):
                                recommendation[k] += parse_resource(
                                    container["target"][k]
                                )
                    pod_["recommendation"] = recommendation
                    found_vpa = True
                    break
            if found_vpa:
                break
        pods[(pod.namespace, pod.name)] = pod_

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

    metrics.get_pod_usage(
        cluster, pods, prev_cluster_summaries.get("pods", {}), alpha_ema
    )
    start = time.time()
    recommender = Recommender()
    recommender.load_from_file(data_path)
    recommender.update_pods(pods)
    recommender.save_to_file(data_path)
    delta = time.time() - start
    logger.debug(
        f"Calculated {len(recommender.cpu_histograms)} resource recommendations for cluster {cluster.id} in {delta:0.3f}s"
    )

    cluster_slack_cost = 0
    for pod in pods.values():
        usage_cost = max(
            pod["recommendation"]["cpu"] * cost_per_cpu,
            pod["recommendation"]["memory"] * cost_per_memory,
        )
        pod["slack_cost"] = max(min(pod["cost"] - usage_cost, pod["cost"]), 0)
        cluster_slack_cost += pod["slack_cost"]

    cluster_summary["slack_cost"] = min(cluster_cost, cluster_slack_cost)

    with FuturesSession(max_workers=10, session=session) as futures_session:
        futures_by_host: Dict[str, Any] = {}  # hostname -> future
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
                except Exception:
                    status = 999
                for ingress in ingresses:
                    ingress[4] = status

    return cluster_summary
