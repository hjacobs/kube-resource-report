import collections
import logging

import pykube
from pykube.objects import APIObject
from pykube.objects import NamespacedAPIObject

from .utils import parse_resource

logger = logging.getLogger(__name__)


class NodeMetrics(APIObject):

    """
    Kubernetes API object for Node metrics.

    See https://github.com/kubernetes/community/blob/master/contributors/design-proposals/instrumentation/resource-metrics-api.md
    """

    version = "metrics.k8s.io/v1beta1"
    endpoint = "nodes"
    kind = "NodeMetrics"


class PodMetrics(NamespacedAPIObject):

    """
    Kubernetes API object for Pod metrics.

    See https://github.com/kubernetes/community/blob/master/contributors/design-proposals/instrumentation/resource-metrics-api.md
    """

    version = "metrics.k8s.io/v1beta1"
    endpoint = "pods"
    kind = "PodMetrics"


def get_ema(curr_value: float, prev_value: float, alpha: float = 1.0):
    """
    Calculate the Exponential Moving Average.

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
