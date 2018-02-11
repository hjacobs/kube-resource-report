#!/usr/bin/env python3

import collections
import csv
import logging
import os
import re
import requests
import cluster_discovery
from urllib.parse import urljoin
from pathlib import Path

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

NODE_COSTS_MONTHLY = {}

# CSVs downloaded from https://ec2instances.info/
for path in Path('.').glob('aws-ec2-costs-hourly-*.csv'):
    region = path.stem.split('-', 4)[4]
    with path.open() as fd:
        reader = csv.DictReader(fd)
        for row in reader:
            cost = row['Linux On Demand cost']
            if cost == 'unavailable':
                continue
            elif cost.startswith('$') and cost.endswith(' hourly'):
                monthly_cost = float(cost.split()[0].strip('$')) * 24 * 30
                NODE_COSTS_MONTHLY[(region, row['API Name'])] = monthly_cost
            else:
                raise Exception('Invalid')


session = requests.Session()


def request(cluster, path, **kwargs):
    if 'timeout' not in kwargs:
        # sane default timeout
        kwargs['timeout'] = (5, 15)
    if cluster.cert_file and cluster.key_file:
        kwargs['cert'] = (cluster.cert_file, cluster.key_file)
    return session.get(urljoin(cluster.api_server_url, path), auth=cluster.auth, verify=cluster.ssl_ca_cert, **kwargs)


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def query_cluster(cluster):
    pods = {}
    nodes = {}

    response = request(cluster, '/api/v1/nodes')
    response.raise_for_status()
    cluster_capacity = collections.defaultdict(float)
    cluster_allocatable = collections.defaultdict(float)
    cluster_requests = collections.defaultdict(float)
    cluster_usage = collections.defaultdict(float)
    node_count = collections.defaultdict(int)
    cluster_cost = 0
    for node in response.json()['items']:
        nodes[node['metadata']['name']] = node
        for k, v in node['status'].get('capacity', {}).items():
            cluster_capacity[k] += parse_resource(v)
        for k, v in node['status'].get('allocatable', {}).items():
            cluster_allocatable[k] += parse_resource(v)
        role = node['metadata']['labels'].get('kubernetes.io/role')
        node_count[role] += 1
        region = node['metadata']['labels']['failure-domain.beta.kubernetes.io/region']
        instance_type = node['metadata']['labels']['beta.kubernetes.io/instance-type']
        node['role'] = role
        node['instance_type'] = instance_type
        node['cost'] = NODE_COSTS_MONTHLY.get((region, instance_type))
        cluster_cost += node['cost']

    response = request(cluster, '/api/v1/namespaces/kube-system/services/heapster/proxy/apis/metrics/v1alpha1/nodes')
    response.raise_for_status()
    for item in response.json()['items']:
        key = item['metadata']['name']
        node = nodes.get(key)
        if node:
            usage = collections.defaultdict(float)
            for k, v in item.get('usage', {}).items():
                usage[k] += parse_resource(v)
                cluster_usage[k] += parse_resource(v)
            node['usage'] = usage


    response = request(cluster, '/api/v1/pods')
    response.raise_for_status()
    for pod in response.json()['items']:
        if pod['status'].get('phase') in ('Succeeded', 'Failed'):
            # ignore completed pods
            continue
        requests = collections.defaultdict(float)
        for container in pod['spec']['containers']:
            for k, v in container['resources'].get('requests', {}).items():
                requests[k] += parse_resource(v)
                cluster_requests[k] += parse_resource(v)
        pods[(pod['metadata']['namespace'], pod['metadata']['name'])] = {'requests': requests}

    cluster_summary = {
        'cluster': cluster,
        'nodes': nodes,
        'pods': pods,
        'master_nodes': node_count['master'],
        'worker_nodes': node_count['worker'],
        'capacity': cluster_capacity,
        'allocatable': cluster_allocatable,
        'requests': cluster_requests,
        'usage': cluster_usage,
        'cost': cluster_cost
    }

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

    return cluster_summary

cluster_summaries = {}
discoverer = cluster_discovery.KubeconfigDiscoverer(Path(os.path.expanduser('~/.kube/config')), set())
for cluster in discoverer.get_clusters():
    try:
        logger.debug('Querying cluster {} ({})..'.format(cluster.id, cluster.api_server_url))
        summary = query_cluster(cluster)
        cluster_summaries[cluster.id] = summary
    except Exception as e:
        print(e)

for cluster_id, summary in sorted(cluster_summaries.items()):
    worker_instance_type = set()
    for node in summary['nodes'].values():
        if node['role'] == 'worker':
            worker_instance_type.add(node['instance_type'])
    fields = [cluster_id, summary['master_nodes'], summary['worker_nodes'], ','.join(worker_instance_type)]
    for x in ['capacity', 'allocatable', 'requests', 'usage']:
        fields += [round(summary[x]['cpu'], 2), int(summary[x]['memory'] / (1024*1024))]
    fields += [round(summary['cost'], 2)]
    for f in fields:
        print(f, '\t', end='')
    print()

