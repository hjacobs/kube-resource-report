import pykube
from pykube.objects import NamespacedAPIObject

class RouteGroup(NamespacedAPIObject):
    """
    Kubernetes API objct for RouteGroup.

    See https://opensource.zalando.com/skipper/kubernetes/routegroups
    """
    version = "zalando.org/v1"
    kind = "RouteGroup"
    endpoint = "routegroups"

### TODO(sszuecs) cleanup
# def list_routegroups(api: pykube.HTTPClient):
#     return RouteGroup.objects(api).get()

# def get_routegroup(name, namespace, api: pykube.HTTPClient):
#     return RouteGroup.objects(api).get(name=name, namespace=namespace)

# def get_routegroup_hosts(name, namespace, api: pykube.HTTPClient):
#         obj = get_routegroup(name, namespace, api=api)
#         return obj.obj['spec']['hosts']

# test
# import pykube
# config = pykube.KubeConfig.from_file()
# api = pykube.HTTPClient(config)

# In : from pykube.objects import NamespacedAPIObject
# In : class RouteGroup(NamespacedAPIObject):
# ...:     version = "zalando.org/v1"
# ...:     kind = "RouteGroup"
# ...:     endpoint = "routegroups"

# In : obj = RouteGroup.objects(api).get(name="rg-loadtest-shadow")
# In : obj.metadata
# {'annotations': {'kubectl.kubernetes.io/last-applied-configuration': '{"apiVersion":"zalando.org/v1","kind":"RouteGroup","metadata":{"annotations":{},"name":"rg-loadtest-shadow","namespace":"default"},"spec":{"backends":[{"name":"loadtest","serviceName":"loadtest-backend","servicePort":80,"type":"service"},{"name":"shadow-fast","serviceName":"loadtest-low-latency-backend","servicePort":80,"type":"service"},{"name":"shadow-slow","serviceName":"loadtest-high-latency-backend","servicePort":80,"type":"service"}],"hosts":["loadtest.teapot.zalan.do"],"routes":[{"backends":[{"backendName":"loadtest"}],"pathSubtree":"/"},{"backends":[{"backendName":"loadtest"}],"filters":["teeLoopback(\\"slow\\")"],"pathSubtree":"/","predicates":["Traffic(0.3)"]},{"backends":[{"backendName":"shadow-slow"}],"pathSubtree":"/","predicates":["Tee(\\"slow\\")","True()"]},{"backends":[{"backendName":"loadtest"}],"filters":["teeLoopback(\\"fast\\")"],"pathSubtree":"/","predicates":["Traffic(0.3)"]},{"backends":[{"backendName":"shadow-fast"}],"pathSubtree":"/","predicates":["Tee(\\"fast\\")","True()"]}]}}\n'}, 'creationTimestamp': '2020-06-23T17:45:04Z', 'generation': 17, 'name': 'rg-loadtest-shadow', 'namespace': 'default', 'resourceVersion': '589834085', 'selfLink': '/apis/zalando.org/v1/namespaces/default/routegroups/rg-loadtest-shadow', 'uid': '754ad7b3-3580-4d9c-9316-1cd9c4d50600'}
# obj.obj['spec']['hosts']
# ['loadtest.teapot.zalan.do']
