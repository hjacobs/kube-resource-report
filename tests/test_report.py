import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from kube_resource_report.cluster_discovery import Cluster
from kube_resource_report.report import aggregate_by_team
from kube_resource_report.report import generate_report
from kube_resource_report.report import get_node_usage
from kube_resource_report.report import get_pod_usage
from kube_resource_report.report import HOURS_PER_MONTH
from kube_resource_report.report import new_resources
from kube_resource_report.report import NODE_LABEL_ROLE
from kube_resource_report.report import parse_resource


@pytest.fixture
def fake_responses():
    fake_pod_spec = {
        "containers": [
            {
                "resources": {
                    "requests": {
                        # 1/20 of 1 core (node capacity)
                        "cpu": "50m",
                        # 1/4 of 1Gi (node capacity)
                        "memory": "256Mi",
                    }
                }
            }
        ]
    }

    return {
        ("v1", "nodes"): {
            "items": [
                {
                    "metadata": {"name": "node-1", "labels": {}},
                    "status": {
                        "capacity": {"cpu": "1", "memory": "1Gi"},
                        "allocatable": {"cpu": "1", "memory": "1Gi"},
                    },
                }
            ]
        },
        ("v1", "pods"): {
            "items": [
                {
                    "metadata": {
                        "name": "pod-1",
                        "namespace": "default",
                        "labels": {"app": "myapp"},
                    },
                    "spec": fake_pod_spec,
                    "status": {"phase": "Running"},
                },
                {
                    "metadata": {
                        "name": "pod-failed",
                        "namespace": "default",
                        "labels": {"app": "myapp"},
                    },
                    "spec": fake_pod_spec,
                    "status": {"phase": "Failed"},
                },
                {
                    "metadata": {
                        "name": "pod-pending-scheduled",
                        "namespace": "default",
                        "labels": {"app": "myapp"},
                    },
                    "spec": fake_pod_spec,
                    "status": {
                        "phase": "Pending",
                        "conditions": [{"type": "PodScheduled", "status": "True"}],
                    },
                },
                {
                    "metadata": {
                        "name": "pod-pending-no-conditions",
                        "namespace": "default",
                        "labels": {"app": "myapp"},
                    },
                    "spec": fake_pod_spec,
                    "status": {"phase": "Pending"},
                },
                {
                    "metadata": {
                        "name": "pod-pending-not-scheduled",
                        "namespace": "default",
                        "labels": {"app": "myapp"},
                    },
                    "spec": fake_pod_spec,
                    "status": {
                        "phase": "Pending",
                        "conditions": [{"type": "PodScheduled", "status": "False"}],
                    },
                },
            ]
        },
        ("extensions/v1beta1", "ingresses"): {
            "items": [
                {
                    "metadata": {"name": "ing-1", "namespace": "default"},
                    "spec": {
                        "rules": [
                            # no "host" field!
                            {"http": {}}
                        ]
                    },
                }
            ]
        },
        ("v1", "namespaces"): {"items": []},
    }


@pytest.fixture
def fake_responses_with_two_different_nodes(fake_responses):
    fake_responses[("v1", "nodes")] = {
        "items": [
            {
                "metadata": {"name": "node-1", "labels": {NODE_LABEL_ROLE: "worker"}},
                "status": {
                    "capacity": {"cpu": "1", "memory": "1Gi"},
                    "allocatable": {"cpu": "1", "memory": "1Gi"},
                },
            },
            {
                "metadata": {"name": "node-2", "labels": {NODE_LABEL_ROLE: "node"}},
                "status": {
                    "capacity": {"cpu": "1", "memory": "1Gi"},
                    "allocatable": {"cpu": "1", "memory": "1Gi"},
                },
            },
        ]
    }
    return fake_responses


@pytest.fixture
def fake_pod_metric_responses():
    return {
        ("metrics.k8s.io/v1beta1", "pods"): {
            "items": [
                {
                    "metadata": {"namespace": "default", "name": "pod-1"},
                    "containers": [
                        {
                            # 50% of requested resources are used
                            "usage": {"cpu": "50m", "memory": "256Mi"}
                        }
                    ],
                }
            ]
        }
    }


@pytest.fixture
def fake_node_metric_responses():
    return {
        ("metrics.k8s.io/v1beta1", "nodes"): {
            "items": [
                {
                    "metadata": {"name": "node-1"},
                    "usage": {"cpu": "16", "memory": "128Gi"},
                },
            ]
        }
    }


@pytest.fixture
def fake_applications():
    return {
        "some-app": {
            "id": "some-app",
            "team": "some-team",
            "clusters": {"some-cluster"},
            "pods": 1,
            "cost": 40,
            "slack_cost": 10,
            "requests": {"cpu": 1, "memory": 1024},
        },
        "some-other-app": {
            "id": "some-other-app",
            "team": "some-team",
            "clusters": {"some-cluster"},
            "pods": 1,
            "cost": 10,
            "slack_cost": 5,
            "requests": {"cpu": 0.2, "memory": 512},
        },
    }


@pytest.fixture
def output_dir(tmpdir):
    return tmpdir.mkdir("output")


def get_mock_client(responses: dict):
    mock_client = MagicMock()

    def mock_get(version, url, **kwargs):
        response = responses.get((version, url))
        return MagicMock(json=lambda: response)

    mock_client.get = mock_get
    return mock_client


@pytest.fixture
def fake_generate_report(output_dir, monkeypatch):

    monkeypatch.setattr(
        "kube_resource_report.cluster_discovery.tokens.get", lambda x: "mytok"
    )

    def wrapper(responses):
        mock_client = get_mock_client(responses)

        monkeypatch.setattr(
            "kube_resource_report.cluster_discovery.ClusterRegistryDiscoverer.get_clusters",
            lambda x: [
                Cluster(
                    "test-cluster-1",
                    "test-cluster-1",
                    "https://test-cluster-1.example.org",
                    mock_client,
                )
            ],
        )

        cluster_summaries = generate_report(
            [],
            "https://cluster-registry",
            None,
            set(),
            None,
            False,
            False,
            str(output_dir),
            set(["kube-system"]),
            None,
            None,
            100.0,  # additional_cost_per_cluster
            1.0,  # alpha ema
            {},
            None,
            None,
            ["worker", "node"],
        )
        assert len(cluster_summaries) == 1
        return cluster_summaries

    return wrapper


def test_parse_resource():
    assert parse_resource("500m") == 0.5


def test_ingress_without_host(fake_generate_report, fake_responses):
    cluster_summaries = fake_generate_report(fake_responses)
    assert len(cluster_summaries["test-cluster-1"]["ingresses"]) == 1


def test_cluster_cost(fake_generate_report, fake_responses):
    cluster_summaries = fake_generate_report(fake_responses)

    cluster_cost = 100.0
    cost_per_hour = cluster_cost / HOURS_PER_MONTH
    cost_per_user_request_hour_cpu = 10 * cost_per_hour / 2
    cost_per_user_request_hour_memory = 2 * cost_per_hour / 2

    assert cluster_summaries["test-cluster-1"]["cost"] == cluster_cost

    assert (
        cluster_summaries["test-cluster-1"]["cost_per_user_request_hour"]["cpu"]
        == cost_per_user_request_hour_cpu
    )
    assert (
        cluster_summaries["test-cluster-1"]["cost_per_user_request_hour"]["memory"]
        == cost_per_user_request_hour_memory
    )

    # assert cost_per_hour == cost_per_user_request_hour_cpu + cost_per_user_request_hour_memory


def test_application_report(
    output_dir, fake_generate_report, fake_responses, fake_pod_metric_responses
):

    # merge responses to get usage metrics and slack costs
    all_responses = {**fake_responses, **fake_pod_metric_responses}
    fake_generate_report(all_responses)

    expected = set(["index.html", "applications.html", "application-metrics.json"])
    paths = set()
    for f in Path(str(output_dir)).iterdir():
        paths.add(f.name)

    assert expected <= paths

    with (Path(str(output_dir)) / "application-metrics.json").open() as fd:
        data = json.load(fd)

    assert data["myapp"]["id"] == "myapp"
    assert data["myapp"]["pods"] == 2
    assert data["myapp"]["requests"] == {"cpu": 0.1, "memory": 512 * 1024 ** 2}
    # the "myapp" pod consumes 1/2 of cluster capacity (512Mi of 1Gi memory)
    assert data["myapp"]["cost"] == 50.0
    # only 1/2 of requested resources are used => 50% of costs are slack
    assert data["myapp"]["slack_cost"] == 25.0


def test_get_pod_usage(monkeypatch, fake_pod_metric_responses):
    mock_client = get_mock_client(fake_pod_metric_responses)
    cluster = Cluster(
        "test-cluster-1",
        "test-cluster-1",
        "https://test-cluster-1.example.org",
        mock_client,
    )
    pods = {("default", "pod-1"): {"usage": new_resources()}}
    get_pod_usage(cluster, pods, {}, 1.0)
    assert pods[("default", "pod-1")]["usage"]["cpu"] == 0.05


def test_get_node_usage(monkeypatch, fake_node_metric_responses):
    mock_client = get_mock_client(fake_node_metric_responses)
    cluster = Cluster(
        "test-cluster-1",
        "test-cluster-1",
        "https://test-cluster-1.example.org",
        mock_client,
    )
    nodes = {"node-1": {"usage": new_resources()}}
    get_node_usage(cluster, nodes, {}, 1.0)
    assert nodes["node-1"]["usage"]["cpu"] == 16


def test_more_than_one_label(
    fake_generate_report, fake_responses_with_two_different_nodes
):
    cluster_summaries = fake_generate_report(fake_responses_with_two_different_nodes)
    assert cluster_summaries["test-cluster-1"]["worker_nodes"] == 2


def test_aggregate_by_team(fake_applications):
    teams = {}
    aggregate_by_team(fake_applications, teams)
    assert len(teams) == 1
    team = teams["some-team"]
    assert team["cost"] == 50
    assert team["slack_cost"] == 15
    assert team["pods"] == 2
    assert team["requests"] == {"cpu": 1.2, "memory": 512 + 1024}
    assert team["clusters"] == {"some-cluster"}
