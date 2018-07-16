#!/usr/bin/env python3

import click
import collections
import csv
import pickle
import datetime
import logging
import os
import re
import requests
import cluster_discovery
import shutil
from urllib.parse import urljoin
from pathlib import Path

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from requests_futures.sessions import FuturesSession

from jinja2 import Environment, FileSystemLoader, select_autoescape

VERSION = 'v0.1'

ONE_MEBI = 1024**2

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
    '''
    >>> parse_resource('100m')
    0.1
    >>> parse_resource('2Gi')
    2147483648
    '''
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
                raise Exception('Invalid price data: {}'.format(cost))


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


def query_cluster(cluster, executor, system_namespaces):
    logger.info('Querying cluster {} ({})..'.format(cluster.id, cluster.api_server_url))
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
        node['capacity'] = {}
        node['allocatable'] = {}
        for k, v in node['status'].get('capacity', {}).items():
            parsed = parse_resource(v)
            node['capacity'][k] = parsed
            cluster_capacity[k] += parsed
        for k, v in node['status'].get('allocatable', {}).items():
            parsed = parse_resource(v)
            node['allocatable'][k] = parsed
            cluster_allocatable[k] += parsed
        role = node['metadata']['labels'].get('kubernetes.io/role') or 'worker'
        node_count[role] += 1
        region = node['metadata']['labels'].get('failure-domain.beta.kubernetes.io/region', 'unknown')
        instance_type = node['metadata']['labels'].get('beta.kubernetes.io/instance-type', 'unknown')
        node['kubelet_version'] = node['status'].get('nodeInfo', {}).get('kubeletVersion', '')
        node['role'] = role
        node['instance_type'] = instance_type
        node['cost'] = NODE_COSTS_MONTHLY.get((region, instance_type))
        if node['cost'] is None:
            logger.warning('No cost information for {} in {}'.format(instance_type, region))
            node['cost'] = 0
        cluster_cost += node['cost']

    try:
        # https://github.com/kubernetes/community/blob/master/contributors/design-proposals/instrumentation/resource-metrics-api.md
        response = request(cluster, '/apis/metrics.k8s.io/v1beta1/nodes')
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
    except Exception as e:
        logger.exception('Failed to query Heapster metrics')

    cost_per_cpu = cluster_cost / cluster_allocatable['cpu']
    cost_per_memory = cluster_cost / cluster_allocatable['memory']

    response = request(cluster, '/api/v1/pods')
    response.raise_for_status()
    for pod in response.json()['items']:
        if pod['status'].get('phase') in ('Succeeded', 'Failed'):
            # ignore completed pods
            continue
        labels = pod['metadata'].get('labels', {})
        application = labels.get('application', labels.get('app', ''))
        requests = collections.defaultdict(float)
        for container in pod['spec']['containers']:
            for k, v in container['resources'].get('requests', {}).items():
                requests[k] += parse_resource(v)
                cluster_requests[k] += parse_resource(v)
        cost = max(requests['cpu'] * cost_per_cpu, requests['memory'] * cost_per_memory)
        pods[(pod['metadata']['namespace'], pod['metadata']['name'])] = {'requests': requests, 'application': application, 'cost': cost}

    cluster_summary = {
        'cluster': cluster,
        'nodes': nodes,
        'pods': pods,
        'user_pods': len([p for ns, p in pods if ns not in system_namespaces]),
        'master_nodes': node_count['master'],
        'worker_nodes': node_count['worker'],
        'kubelet_versions': set([n['kubelet_version'] for n in nodes.values() if n['role'] == 'worker']),
        'worker_instance_types': set([n['instance_type'] for n in nodes.values() if n['role'] == 'worker']),
        'capacity': cluster_capacity,
        'allocatable': cluster_allocatable,
        'requests': cluster_requests,
        'usage': cluster_usage,
        'cost': cluster_cost,
        'ingresses': []
    }

    try:
        # https://github.com/kubernetes/community/blob/master/contributors/design-proposals/instrumentation/resource-metrics-api.md
        response = request(cluster, '/apis/metrics.k8s.io/v1beta1/pods')
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
    except Exception as e:
        logger.exception('Failed to query Heapster metrics')

    response = request(cluster, '/apis/extensions/v1beta1/ingresses')
    response.raise_for_status()

    with FuturesSession(max_workers=10) as futures_session:
        futures = {}
        for item in response.json()['items']:
            namespace, name = item['metadata']['namespace'], item['metadata']['name']
            labels = item['metadata'].get('labels', {})
            application = labels.get('application', labels.get('app', ''))
            for rule in item['spec']['rules']:
                ingress = [namespace, name, application, rule['host'], 0]
                futures[futures_session.get('https://{}/'.format(rule['host']), timeout=5)] = ingress
                cluster_summary['ingresses'].append(ingress)

        logger.info('Waiting for ingress status..')
        for future in concurrent.futures.as_completed(futures):
            ingress = futures[future]
            try:
                status = future.result().status_code
            except:
                status = 999
            ingress[4] = status

    return cluster_summary


@click.command()
@click.option('--cluster-registry', metavar='URL', help='URL of Cluster Registry to discover clusters to report on')
@click.option('--application-registry', metavar='URL', help='URL of Application Registry to look up team by application ID')
@click.option('--use-cache', is_flag=True, help='Use cached data (mostly for development)')
@click.option('--system-namespaces', metavar='NS1,NS2', default='kube-system',
              help='Comma separated list of system/infrastructure namespaces (default: kube-system)')
@click.option('--include-clusters', metavar='PATTERN', help='Include clusters matching the regex')
@click.option('--exclude-clusters', metavar='PATTERN', help='Exclude clusters matching the regex')
@click.argument('output_dir', type=click.Path(exists=True))
def main(cluster_registry, application_registry, use_cache, output_dir, system_namespaces, include_clusters, exclude_clusters):
    '''Kubernetes Resource Report

    Generate a static HTML report to OUTPUT_DIR for all clusters in ~/.kube/config or Cluster Registry.
    '''

    generate_report(cluster_registry, application_registry, use_cache, output_dir, set(system_namespaces.split(',')), include_clusters, exclude_clusters)


def get_cluster_summaries(cluster_registry: str, include_clusters: str, exclude_clusters: str, system_namespaces: set, notifications: list):
    cluster_summaries = {}

    if cluster_registry:
        discoverer = cluster_discovery.ClusterRegistryDiscoverer(cluster_registry)
    else:
        discoverer = cluster_discovery.KubeconfigDiscoverer(Path(os.path.expanduser('~/.kube/config')), set())

    include_pattern = include_clusters and re.compile(include_clusters)
    exclude_pattern = exclude_clusters and re.compile(exclude_clusters)

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_cluster = {}
        for cluster in discoverer.get_clusters():
            if (not include_pattern or include_pattern.match(cluster.id)) and (not exclude_pattern or not exclude_pattern.match(cluster.id)):
                future_to_cluster[executor.submit(query_cluster, cluster, executor, system_namespaces)] = cluster

        for future in concurrent.futures.as_completed(future_to_cluster):
            cluster = future_to_cluster[future]
            try:
                summary = future.result()
                cluster_summaries[cluster.id] = summary
            except Exception as e:
                notifications.append(['error', 'Failed to query cluster {}: {}'.format(cluster.id, e)])
                logger.exception(e)
    return cluster_summaries


def generate_report(cluster_registry, application_registry, use_cache, output_dir, system_namespaces, include_clusters, exclude_clusters):
    notifications = []

    output_path = Path(output_dir)

    pickle_path = output_path / 'dump.pickle'

    if use_cache and pickle_path.exists():
        with pickle_path.open('rb') as fd:
            cluster_summaries = pickle.load(fd)

    else:
        cluster_summaries = get_cluster_summaries(cluster_registry, include_clusters, exclude_clusters, system_namespaces, notifications)

    with pickle_path.open('wb') as fd:
        pickle.dump(cluster_summaries, fd)

    teams = {}
    applications = {}
    total_allocatable = collections.defaultdict(int)
    total_requests = collections.defaultdict(int)

    for cluster_id, summary in sorted(cluster_summaries.items()):
        for r in 'cpu', 'memory':
            total_allocatable[r] += summary['allocatable'][r]
            total_requests[r] += summary['requests'][r]

        cost_per_cpu = summary['cost'] / summary['allocatable']['cpu']
        cost_per_memory = summary['cost'] / summary['allocatable']['memory']
        for k, pod in summary['pods'].items():
            app = applications.get(pod['application'], {'id': pod['application'], 'cost': 0, 'pods': 0, 'requests': {}, 'usage': {}, 'clusters': set()})
            for r in 'cpu', 'memory':
                app['requests'][r] = app['requests'].get(r, 0) + pod['requests'][r]
                app['usage'][r] = app['usage'].get(r, 0) + pod.get('usage', {}).get(r, 0)
            app['team'] = ''
            app['active'] = None
            app['cost'] += pod['cost']
            app['pods'] += 1
            app['clusters'].add(cluster_id)
            app['slack_cost'] = max(
                (app['requests']['cpu'] - app['usage']['cpu']) * cost_per_cpu,
                (app['requests']['memory'] - app['usage']['memory']) * cost_per_memory)
            applications[pod['application']] = app

    if application_registry:
        with FuturesSession(max_workers=10) as futures_session:
            futures_session.auth = cluster_discovery.OAuthTokenAuth('read-only')

            future_to_app = {}
            for app_id, app in applications.items():
                if app_id:
                    future_to_app[futures_session.get(application_registry + '/apps/' + app_id, timeout=5)] = app

            for future in concurrent.futures.as_completed(future_to_app):
                app = future_to_app[future]
                try:
                    response = future.result()
                    response.raise_for_status()
                    data = response.json()
                    if not isinstance(data, dict):
                        data = {}
                except Exception as e:
                    logger.warning('Failed to look up application {}: {}'.format(app['id'], e))
                    data = {}
                team_id = data.get('team_id', '')
                app['team'] = team_id
                app['active'] = data.get('active')
                team = teams.get(team_id, {'clusters': set(), 'applications': set(), 'cost': 0, 'pods': 0, 'requests': {}, 'usage': {}, 'slack_cost': 0})
                team['applications'].add(app['id'])
                team['clusters'] |= app['clusters']
                team['pods'] += app['pods']
                for r in 'cpu', 'memory':
                    team['requests'][r] = team['requests'].get(r, 0) + app['requests'][r]
                    team['usage'][r] = team['usage'].get(r, 0) + app.get('usage', {}).get(r, 0)
                team['cost'] += app['cost']
                team['slack_cost'] += app['slack_cost']
                teams[team_id] = team

    for cluster_id, summary in sorted(cluster_summaries.items()):
        for k, pod in summary['pods'].items():
            app = applications.get(pod['application'])
            pod['team'] = app['team']

    logger.info('Writing clusters.tsv..')
    with (output_path / 'clusters.tsv').open('w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        for cluster_id, summary in sorted(cluster_summaries.items()):
            worker_instance_type = set()
            kubelet_version = set()
            for node in summary['nodes'].values():
                if node['role'] == 'worker':
                    worker_instance_type.add(node['instance_type'])
                kubelet_version.add(node['kubelet_version'])
            fields = [cluster_id, summary['cluster'].api_server_url, summary['master_nodes'], summary['worker_nodes'],
                      ','.join(worker_instance_type), ','.join(kubelet_version)]
            for x in ['capacity', 'allocatable', 'requests', 'usage']:
                fields += [round(summary[x]['cpu'], 2), int(summary[x]['memory'] / ONE_MEBI)]
            fields += [round(summary['cost'], 2)]
            writer.writerow(fields)

    logger.info('Writing ingresses.tsv..')
    with (output_path / 'ingresses.tsv').open('w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        for cluster_id, summary in sorted(cluster_summaries.items()):
            for ingress in summary['ingresses']:
                writer.writerow([cluster_id, summary['cluster'].api_server_url] + ingress)

    logger.info('Writing pods.tsv..')
    with (output_path / 'pods.tsv').open('w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        with (output_path / 'slack.tsv').open('w') as csvfile2:
            slackwriter = csv.writer(csvfile2, delimiter='\t')
            for cluster_id, summary in sorted(cluster_summaries.items()):
                cpu_slack = collections.Counter()
                memory_slack = collections.Counter()
                for k, pod in summary['pods'].items():
                    namespace, name = k
                    requests = pod['requests']
                    application = pod['application'] or name.rsplit('-', 1)[0]
                    usage = pod.get('usage', collections.defaultdict(float))
                    cpu_slack[(namespace, application)] += requests['cpu'] - usage['cpu']
                    memory_slack[(namespace, application)] += requests['memory'] - usage['memory']
                    writer.writerow([cluster_id, summary['cluster'].api_server_url, namespace, name, pod['application'],
                                     requests['cpu'], requests['memory'], usage['cpu'], usage['memory']])
                cost_per_cpu = summary['cost'] / summary['allocatable']['cpu']
                cost_per_memory = summary['cost'] / summary['allocatable']['memory']
                for namespace_name, slack in cpu_slack.most_common(20):
                    namespace, name = namespace_name
                    slackwriter.writerow([cluster_id, summary['cluster'].api_server_url, namespace, name,
                                          'cpu', '{:3.2f}'.format(slack), '${:.2f} potential monthly savings'.format(slack * cost_per_cpu)])
                for namespace_name, slack in memory_slack.most_common(20):
                    namespace, name = namespace_name
                    slackwriter.writerow([cluster_id, summary['cluster'].api_server_url, namespace, name,
                                          'memory', '{:6.0f}Mi'.format(slack / ONE_MEBI), '${:.2f} potential monthly savings'.format(slack * cost_per_memory)])

    templates_path = Path(__file__).parent / 'templates'
    env = Environment(
        loader=FileSystemLoader(str(templates_path)),
        autoescape=select_autoescape(['html', 'xml'])
    )
    context = {
        'notifications': notifications,
        'cluster_summaries': cluster_summaries,
        'teams': teams,
        'applications': applications,
        'total_worker_nodes': sum([s['worker_nodes'] for s in cluster_summaries.values()]),
        'total_allocatable': total_allocatable,
        'total_requests': total_requests,
        'total_pods': sum([len(s['pods']) for s in cluster_summaries.values()]),
        'total_cost': sum([s['cost'] for s in cluster_summaries.values()]),
        'total_slack_cost': sum([a['slack_cost'] for a in applications.values()]),
        'now': datetime.datetime.utcnow(),
        'version': VERSION
    }
    for page in ['index', 'clusters', 'ingresses', 'teams', 'applications', 'pods']:
        file_name = '{}.html'.format(page)
        logger.info('Generating {}..'.format(file_name))
        template = env.get_template(file_name)
        context['page'] = page
        template.stream(**context).dump(str(output_path / file_name))

    for cluster_id, summary in cluster_summaries.items():
        page = 'clusters'
        file_name = 'cluster-{}.html'.format(cluster_id)
        logger.info('Generating {}..'.format(file_name))
        template = env.get_template('cluster.html')
        context['page'] = page
        context['cluster_id'] = cluster_id
        context['summary'] = summary
        template.stream(**context).dump(str(output_path / file_name))

    for team_id, team in teams.items():
        page = 'teams'
        file_name = 'team-{}.html'.format(team_id)
        logger.info('Generating {}..'.format(file_name))
        template = env.get_template('team.html')
        context['page'] = page
        context['team_id'] = team_id
        context['team'] = team
        template.stream(**context).dump(str(output_path / file_name))

    for path in templates_path.iterdir():
        if path.match('*.js') or path.match('*.css') or path.match('*.png'):
            shutil.copy(str(path), str(output_path / path.name))

    return cluster_summaries


if __name__ == '__main__':
    main()
