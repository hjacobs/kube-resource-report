from kube_resource_report.cluster_discovery import Cluster
from kube_resource_report.report import generate_report

from unittest.mock import MagicMock


def test_generate_report(tmpdir, monkeypatch):
    output_dir = tmpdir.mkdir("output")
    monkeypatch.setattr("kube_resource_report.cluster_discovery.tokens.get", lambda x: "mytok")
    monkeypatch.setattr(
        "kube_resource_report.cluster_discovery.ClusterRegistryDiscoverer.get_clusters",
        lambda x: [Cluster("test-cluster-1", "test-cluster-1", "https://test-cluster-1.example.org")],
    )

    responses = {
        "/api/v1/nodes": {
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
        "/api/v1/pods": {"items": []},
        "/apis/extensions/v1beta1/ingresses": {
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
        "/api/v1/namespaces": {"items": []},
    }

    monkeypatch.setattr(
        "kube_resource_report.report.request",
        lambda cluster, path: MagicMock(json=lambda: responses.get(path)),
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
        0,
        None,
        "worker"
    )
    assert len(cluster_summaries) == 1
    assert len(cluster_summaries['test-cluster-1']['ingresses']) == 1
