from unittest.mock import MagicMock

import pytest

from kube_resource_report.query import NODE_LABEL_ROLE


class Helpers:
    @staticmethod
    def get_mock_client(responses: dict):
        mock_client = MagicMock()

        def mock_get(version, url, **kwargs):
            response = responses.get((version, url))
            return MagicMock(json=lambda: response)

        mock_client.get = mock_get
        return mock_client


@pytest.fixture
def helpers():
    return Helpers


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
        ("zalando.org/v1", "routegroups"): {
            "items": [
                {
                    "metadata": {"name": "rg-1", "namespace": "default"},
                    "spec": {
                        "backends": [],
                        "default_backends": [],
                        "hosts": [],
                        "routes": [],
                    },
                }
            ]
        },
        ("v1", "namespaces"): {"items": []},
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
