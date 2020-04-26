import collections
import logging

import pykube
from pykube import CronJob
from pykube import DaemonSet
from pykube import Deployment
from pykube import Job
from pykube import Pod
from pykube import ReplicaSet
from pykube import StatefulSet
from pykube.objects import NamespacedAPIObject

# see supported VPA controllers
# https://github.com/kubernetes/autoscaler/blob/932d62fba78e4f04f73d3bfb86ccd53cd46bc20f/vertical-pod-autoscaler/pkg/target/fetcher.go#L82
CONTROLLER_CLASSES = {
    clazz.kind: clazz
    for clazz in [Deployment, StatefulSet, DaemonSet, ReplicaSet, Job, CronJob]
}

logger = logging.getLogger(__name__)


class VerticalPodAutoscaler(NamespacedAPIObject):

    """
    Kubernetes API object for VerticalPodAutoscaler (VPA).

    See https://github.com/kubernetes/autoscaler/blob/master/vertical-pod-autoscaler/pkg/apis/autoscaling.k8s.io/v1beta2/types.go
    """

    version = "autoscaling.k8s.io/v1beta2"
    endpoint = "verticalpodautoscalers"
    kind = "VerticalPodAutoscaler"

    target = None

    def get_target_ref(self):
        target_ref = self.obj["spec"].get("targetRef", {})
        clazz = CONTROLLER_CLASSES.get(target_ref["kind"])
        if not clazz:
            raise ValueError(f"Unsupported controller kind: {target_ref['kind']}")

        return (clazz, target_ref["name"])

    @property
    def match_labels(self):
        if self.target:
            return self.target.obj["spec"]["selector"]["matchLabels"]
        else:
            return None

    def matches_pod(self, pod: Pod):
        if not self.match_labels:
            return False
        for k, v in self.match_labels.items():
            if pod.labels.get(k) != v:
                return False
        return True

    @property
    def container_recommendations(self):
        return (
            self.obj.get("status", {})
            .get("recommendation", {})
            .get("containerRecommendations", [])
        )


def get_vpas_by_match_labels(api: pykube.HTTPClient):
    vpas_by_namespace_target_ref = collections.defaultdict(list)
    clazzes = set()
    for vpa in VerticalPodAutoscaler.objects(api, namespace=pykube.all):
        try:
            clazz, target_name = vpa.get_target_ref()
        except Exception as e:
            logger.warning(
                f"Failed to get target ref of VPA {vpa.namespace}/{vpa.name}: {e}"
            )
            continue
        clazzes.add(clazz)
        vpas_by_namespace_target_ref[(vpa.namespace, clazz, target_name)].append(vpa)

    vpas_by_namespace_label = collections.defaultdict(list)
    for clazz in clazzes:
        for target in clazz.objects(api, namespace=pykube.all):
            for vpa in vpas_by_namespace_target_ref[
                (target.namespace, clazz, target.name)
            ]:
                vpa.target = target

                if vpa.match_labels:
                    for k, v in vpa.match_labels.items():
                        vpas_by_namespace_label[(vpa.namespace, k, v)].append(vpa)

    return vpas_by_namespace_label
