from kube_resource_report.cluster_discovery import Cluster
from kube_resource_report.metrics import get_node_usage
from kube_resource_report.metrics import get_pod_usage
from kube_resource_report.query import new_resources


def test_get_pod_usage(monkeypatch, fake_pod_metric_responses, helpers):
    mock_client = helpers.get_mock_client(fake_pod_metric_responses)
    cluster = Cluster(
        "test-cluster-1",
        "test-cluster-1",
        "https://test-cluster-1.example.org",
        mock_client,
    )
    pods = {("default", "pod-1"): {"usage": new_resources()}}
    get_pod_usage(cluster, pods, {}, 1.0)
    assert pods[("default", "pod-1")]["usage"]["cpu"] == 0.05


def test_get_node_usage(monkeypatch, fake_node_metric_responses, helpers):
    mock_client = helpers.get_mock_client(fake_node_metric_responses)
    cluster = Cluster(
        "test-cluster-1",
        "test-cluster-1",
        "https://test-cluster-1.example.org",
        mock_client,
    )
    nodes = {"node-1": {"usage": new_resources()}}
    get_node_usage(cluster, nodes, {}, 1.0)
    assert nodes["node-1"]["usage"]["cpu"] == 16
