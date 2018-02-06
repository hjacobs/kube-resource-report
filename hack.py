#!/usr/bin/env python3

import collections
import json
import logging
import kubernetes.client
import kubernetes.config
import re
import requests
from urllib.parse import urljoin

from requests.auth import AuthBase

FACTORS = {
    'm': 1 / 1000,
    'K': 1000,
    'M': 1000**2,
    'G': 1000**3,
    'T': 1000**4,
    'P': 1000**5,
    'E': 1000**6,
    'Ki': 1024,
    'Mi': 1024**2,
    'Gi': 1024**3,
    'Ti': 1024**4,
    'Pi': 1024**5,
    'Ei': 1024**6
}

RESOURCE_PATTERN = re.compile('^(\d*)(\D*)$')


def parse_resource(v):
    match = RESOURCE_PATTERN.match(v)
    factor = FACTORS.get(match.group(2), 1)
    return int(match.group(1)) * factor


class StaticAuthorizationHeaderAuth(AuthBase):
    '''Static authentication with given "Authorization" header'''

    def __init__(self, authorization):
        self.authorization = authorization

    def __call__(self, request):
        request.headers['Authorization'] = self.authorization
        return request


class Cluster:
    def __init__(self, id, api_server_url, ssl_ca_cert=None, auth=None, cert_file=None, key_file=None):
        self.id = id
        self.api_server_url = api_server_url
        self.ssl_ca_cert = ssl_ca_cert
        self.auth = auth
        self.cert_file = cert_file
        self.key_file = key_file


session = requests.Session()


def request(cluster, path, **kwargs):
    if 'timeout' not in kwargs:
        # sane default timeout
        kwargs['timeout'] = (5, 15)
    if cluster.cert_file and cluster.key_file:
        kwargs['cert'] = (cluster.cert_file, cluster.key_file)
    return session.get(urljoin(cluster.api_server_url, path), auth=cluster.auth, verify=cluster.ssl_ca_cert, **kwargs)


logging.basicConfig()

kubernetes.config.load_kube_config()
config = kubernetes.client.configuration.Configuration()
authorization = config.api_key.get('authorization')
if authorization:
    auth = StaticAuthorizationHeaderAuth(authorization)
else:
    auth = None
cluster = Cluster(
    'default',
    config.host,
    ssl_ca_cert=config.ssl_ca_cert,
    cert_file=config.cert_file,
    key_file=config.key_file,
    auth=auth)

pods = {}

response = request(cluster, '/api/v1/pods')
response.raise_for_status()
for pod in response.json()['items']:
    requests = collections.defaultdict(float)
    for container in pod['spec']['containers']:
        for k, v in container['resources'].get('requests', {}).items():
            requests[k] += parse_resource(v)
    pods[(pod['metadata']['namespace'], pod['metadata']['name'])] = {'requests': requests}

response = request(cluster, '/api/v1/namespaces/kube-system/services/heapster/proxy/apis/metrics/v1alpha1/pods')
response.raise_for_status()
for item in response.json()['items']:
    key = (item['metadata']['namespace'], item['metadata']['name'])
    pod = pods.get(key)
    if pod:
        usage = collections.defaultdict(float)
        for container in item['containers']:
            for k, v in container.get('usage', {}).items():
                usage[k] += parse_resource(v)
        pod['usage'] = usage

for k, pod in sorted(pods.items()):
    namespace, name = k
    requests = pod['requests']
    usage = pod.get('usage', collections.defaultdict(float))
    print(namespace, '\t', name, '\t', requests['cpu'], '\t', requests['memory'], '\t', usage['cpu'], '\t', usage['memory'])
