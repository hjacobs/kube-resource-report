from pykube.objects import Node
from pykube.objects import Pod

from kube_resource_report.query import map_node
from kube_resource_report.query import map_pod


def test_map_empty_node():
    node = Node(None, {"metadata": {}, "status": {}})
    assert map_node(node) == {
        "capacity": {},
        "allocatable": {},
        "usage": {"cpu": 0, "memory": 0},
        "requests": {"cpu": 0, "memory": 0},
        "cost": 0,
        "slack_cost": 0,
        "instance_type": "unknown",
        "kubelet_version": "",
        "role": "worker",
        "spot": False,
        "pods": {},
    }


def test_map_gke_preemptible_node():
    node = Node(
        None,
        {
            "metadata": {
                "labels": {
                    "beta.kubernetes.io/instance-type": "1-standard-2",
                    "cloud.google.com/gke-preemptible": "true",
                }
            },
            "status": {},
        },
    )
    assert map_node(node) == {
        "capacity": {},
        "allocatable": {},
        "usage": {"cpu": 0, "memory": 0},
        "requests": {"cpu": 0, "memory": 0},
        "cost": 0,
        "slack_cost": 0,
        "instance_type": "1-standard-2-preemptible",
        "kubelet_version": "",
        "role": "worker",
        "spot": True,
        "pods": {},
    }


def test_map_empty_pod():
    pod = Pod(None, {"metadata": {}, "spec": {"containers": []}, "status": {}})
    assert map_pod(pod, 0, 0) == {
        "application": "",
        "component": "",
        "container_names": [],
        "container_images": [],
        "team": "",
        "requests": {"cpu": 0, "memory": 0},
        "usage": {"cpu": 0, "memory": 0},
        "cost": 0,
    }


def test_map_pod_with_resources():
    pod = Pod(
        None,
        {
            "metadata": {
                "labels": {"app": "myapp", "component": "mycomp", "team": "myteam"}
            },
            "spec": {
                "containers": [
                    {
                        "name": "main",
                        "image": "hjacobs/kube-downscaler:latest",
                        "resources": {"requests": {"cpu": "5m", "memory": "200Mi"}},
                    }
                ]
            },
            "status": {},
        },
    )
    assert map_pod(pod, 0, 0) == {
        "application": "myapp",
        "component": "mycomp",
        "team": "myteam",
        "container_names": ["main"],
        "container_images": ["hjacobs/kube-downscaler:latest"],
        "requests": {"cpu": 0.005, "memory": 200 * 1024 * 1024},
        "usage": {"cpu": 0, "memory": 0},
        "cost": 0,
    }
