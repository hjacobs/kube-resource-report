import json
from pathlib import Path

import pytest

from kube_resource_report.cluster_discovery import Cluster
from kube_resource_report.report import aggregate_by_team
from kube_resource_report.report import generate_report
from kube_resource_report.utils import HOURS_PER_MONTH
from kube_resource_report.utils import parse_resource


@pytest.fixture
def fake_generate_report(output_dir, monkeypatch, helpers):

    monkeypatch.setattr(
        "kube_resource_report.cluster_discovery.tokens.get", lambda x: "mytok"
    )

    def wrapper(responses):
        mock_client = helpers.get_mock_client(responses)

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
            Path(str(output_dir)) / "data",
            set(["kube-system"]),
            None,
            None,
            100.0,  # additional_cost_per_cluster
            1.0,  # alpha ema
            {},
            None,
            None,
            ["worker", "node"],
            None,
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
    # assert data["myapp"]["recommendation"] == {"cpu": 0.05*2, "memory": 512 * 1024 ** 2}
    # the "myapp" pod consumes 1/2 of cluster capacity (512Mi of 1Gi memory)
    assert data["myapp"]["cost"] == 50.0
    # only 1/2 of requested resources are used => 50% of costs are slack,
    # but the recommendation applies to both pods (one pod is failing and has zero usage)
    assert data["myapp"]["slack_cost"] == 0


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
