from pykube.objects import Node

from kube_resource_report.query import map_node


def test_map_node():
    node = Node(None, {"metadata": {}, "status": {}})
    assert map_node(node) == {
        "capacity": {},
        "allocatable": {},
        "usage": {"cpu": 0, "memory": 0},
        "requests": {"cpu": 0, "memory": 0},
        "cost": 0,
        "instance_type": "unknown",
        "kubelet_version": "",
        "role": "worker",
        "spot": False,
    }
