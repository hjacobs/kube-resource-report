from cluster_discovery import Cluster
from report import generate_report

from unittest.mock import MagicMock


def test_generate_report(tmpdir, monkeypatch):
    output_dir = tmpdir.mkdir("output")
    monkeypatch.setattr("cluster_discovery.tokens.get", lambda x: "mytok")
    monkeypatch.setattr(
        "cluster_discovery.ClusterRegistryDiscoverer.get_clusters",
        lambda x: [Cluster("test-cluster-1", "https://test-cluster-1.example.org")],
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
        "/apis/extensions/v1beta1/ingresses": {"items": []},
    }

    monkeypatch.setattr(
        "report.request",
        lambda cluster, path: MagicMock(json=lambda: responses.get(path)),
    )
    cluster_summaries = generate_report(
        "https://cluster-registry", None, False, str(output_dir), set(['kube-system']),
        None, None
    )
    assert len(cluster_summaries) == 1
