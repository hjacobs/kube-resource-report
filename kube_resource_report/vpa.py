import pykube
from pykube import Deployment
from pykube import Pod
from pykube import StatefulSet
from pykube.objects import NamespacedAPIObject

CONTROLLER_CLASSES = {Deployment.kind: Deployment, StatefulSet.kind: StatefulSet}


class VerticalPodAutoscaler(NamespacedAPIObject):

    """
    Kubernetes API object for VerticalPodAutoscaler (VPA).

    See https://github.com/kubernetes/autoscaler/blob/master/vertical-pod-autoscaler/pkg/apis/autoscaling.k8s.io/v1beta2/types.go
    """

    version = "autoscaling.k8s.io/v1beta2"
    endpoint = "verticalpodautoscalers"
    kind = "VerticalPodAutoscaler"

    target = None

    @property
    def match_labels(self):
        if self.target is None:
            target_ref = self.obj["spec"].get("targetRef", {})
            clazz = CONTROLLER_CLASSES.get(target_ref["kind"])
            if not clazz:
                raise ValueError(f"Unsupported controller kind: {target_ref['kind']}")

            try:
                self.target = (
                    clazz.objects(self.api)
                    .filter(namespace=self.namespace)
                    .get_by_name(target_ref["name"])
                )
            except pykube.exceptions.ObjectDoesNotExist:
                self.target = False

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
