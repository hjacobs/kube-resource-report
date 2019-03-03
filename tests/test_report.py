import json
import pytest

from pathlib import Path

from kube_resource_report.cluster_discovery import Cluster
from kube_resource_report.report import generate_report, HOURS_PER_MONTH, parse_resource, get_pod_usage, new_resources

from unittest.mock import MagicMock


@pytest.fixture
def fake_responses():
    return {
        ('v1', "nodes"): {
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
        ('v1', "pods"): {
            "items": [
                {
                    "metadata": {"name": "pod-1", "namespace": "default", "labels": {"app": "myapp"}},
                    "spec": {
                        "containers": [
                            {
                                "resources": {
                                    "requests": {
                                        # 1/10 of 1 core (node capacity)
                                        "cpu": "100m",
                                        # 1/2 of 1Gi (node capacity)
                                        "memory": "512Mi"
                                    }
                                }
                            }
                        ]
                    },
                    "status": {
                        "phase": "Running"
                    }
                }
            ]
        },
        ('extensions/v1beta1', "ingresses"): {
            "items": [
                {
                    "metadata": {"name": "ing-1", "namespace": "default"},
                    "spec": {
                        "rules": [
                            # no "host" field!
                            {"http": {}}
                        ]
                    }

                }
            ]
        },
        ('v1', "namespaces"): {"items": []},
    }


@pytest.fixture
def fake_metric_responses():
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
                    ]
                }
            ]
        }
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

    monkeypatch.setattr("kube_resource_report.cluster_discovery.tokens.get", lambda x: "mytok")

    def wrapper(responses):
        mock_client = get_mock_client(responses)

        monkeypatch.setattr(
            "kube_resource_report.cluster_discovery.ClusterRegistryDiscoverer.get_clusters",
            lambda x: [Cluster("test-cluster-1", "test-cluster-1", "https://test-cluster-1.example.org", mock_client)],
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
            100.0,
            None,
            None,
            "worker"
        )
        assert len(cluster_summaries) == 1
        return cluster_summaries

    return wrapper


def test_parse_resource():
    assert parse_resource('500m') == 0.5


def test_ingress_without_host(fake_generate_report, fake_responses):
    cluster_summaries = fake_generate_report(fake_responses)
    assert len(cluster_summaries['test-cluster-1']['ingresses']) == 1


def test_cluster_cost(fake_generate_report, fake_responses):
    cluster_summaries = fake_generate_report(fake_responses)

    cluster_cost = 100.0
    cost_per_hour = cluster_cost / HOURS_PER_MONTH
    cost_per_user_request_hour_cpu = 10 * cost_per_hour / 2
    cost_per_user_request_hour_memory = 2 * cost_per_hour / 2

    assert cluster_summaries['test-cluster-1']['cost'] == cluster_cost

    assert cluster_summaries['test-cluster-1']['cost_per_user_request_hour']['cpu'] == cost_per_user_request_hour_cpu
    assert cluster_summaries['test-cluster-1']['cost_per_user_request_hour']['memory'] == cost_per_user_request_hour_memory

    # assert cost_per_hour == cost_per_user_request_hour_cpu + cost_per_user_request_hour_memory


def test_application_report(output_dir, fake_generate_report, fake_responses, fake_metric_responses):

    # merge responses to get usage metrics and slack costs
    all_responses = {**fake_responses, **fake_metric_responses}
    fake_generate_report(all_responses)

    expected = set(['index.html', 'applications.html', 'application-metrics.json'])
    paths = set()
    for f in Path(str(output_dir)).iterdir():
        paths.add(f.name)

    assert expected <= paths

    with (Path(str(output_dir)) / 'application-metrics.json').open() as fd:
        data = json.load(fd)

    assert data['myapp']['id'] == 'myapp'
    assert data['myapp']['pods'] == 1
    assert data['myapp']['requests'] == {'cpu': 0.1, 'memory': 512 * 1024**2}
    # the "myapp" pod consumes 1/2 of cluster capacity (512Mi of 1Gi memory)
    assert data['myapp']['cost'] == 50.0
    # only 1/2 of requested resources are used => 50% of costs are slack
    assert data['myapp']['slack_cost'] == 25.0


def test_get_pod_usage(monkeypatch, fake_metric_responses):
    mock_client = get_mock_client(fake_metric_responses)
    cluster = Cluster("test-cluster-1", "test-cluster-1", "https://test-cluster-1.example.org", mock_client)
    pods = {('default', 'pod-1'): {'usage': new_resources()}}
    get_pod_usage(cluster, pods)
    assert pods[('default', 'pod-1')]['usage']['cpu'] == 0.05
