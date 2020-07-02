from pykube.objects import NamespacedAPIObject


class RouteGroup(NamespacedAPIObject):

    """
    Kubernetes API objct for RouteGroup.

    See https://opensource.zalando.com/skipper/kubernetes/routegroups
    """

    version = "zalando.org/v1"
    kind = "RouteGroup"
    endpoint = "routegroups"
