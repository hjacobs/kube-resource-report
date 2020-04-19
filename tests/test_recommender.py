from pathlib import Path

from pytest import approx

from kube_resource_report.recommender import Recommender


def test_save_load(output_dir):
    data_path = Path(str(output_dir))
    recommender = Recommender()
    pod = {
        "application": "my-app",
        "component": "",
        "requests": {"cpu": 0.5, "memory": 500 * 1024 * 10124},
        "usage": {"cpu": 0.1, "memory": 200 * 1024 * 1024},
    }
    pods = {("my-ns", "my-pod"): pod}
    recommender.update_pods(pods)
    assert pod["recommendation"]["cpu"] == approx(0.115, rel=0.02)
    recommender.save_to_file(data_path)

    new_recommender = Recommender()
    new_recommender.load_from_file(data_path)
    # set new pod usage
    pod = {
        "application": "my-app",
        "component": "",
        "requests": {"cpu": 0.5, "memory": 500 * 1024 * 10124},
        "usage": {"cpu": 0.01, "memory": 200 * 1024 * 1024},
    }
    pods = {("my-ns", "my-pod"): pod}
    new_recommender.update_pods(pods)
    assert pod["recommendation"]["cpu"] == approx(0.115, rel=0.02)
    new_recommender.save_to_file(data_path)
