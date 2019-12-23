==========================
Kubernetes Resource Report
==========================

.. image:: https://travis-ci.org/hjacobs/kube-resource-report.svg?branch=master
   :target: https://travis-ci.org/hjacobs/kube-resource-report
   :alt: Travis CI Build Status

.. image:: https://coveralls.io/repos/github/hjacobs/kube-resource-report/badge.svg?branch=master;_=1
   :target: https://coveralls.io/github/hjacobs/kube-resource-report?branch=master
   :alt: Code Coverage

.. image:: 	https://img.shields.io/docker/pulls/hjacobs/kube-resource-report.svg
   :target: https://hub.docker.com/r/hjacobs/kube-resource-report
   :alt: Docker pulls

.. image:: https://img.shields.io/badge/calver-YY.MM.MICRO-22bfda.svg
   :target: http://calver.org
   :alt: Calendar Versioning

**This version only supports node costs for AWS EC2 (all regions, On Demand, Linux) and GKE/GCP machine types (all regions, On Demand, without sustained discount)**

Script to generate a HTML report of CPU/memory requests vs. usage (collected via Metrics API/Heapster) for one or more Kubernetes clusters.

Want to see how the report looks? Check out the `sample HTML report <https://hjacobs.github.io/kube-resource-report/sample-report/output/index.html>`_ and the `demo deployment <https://kube-resource-report.demo.j-serv.de/>`_!

What the script does:

* Discover all clusters (either via ``~/.kube/config``, via in-cluster serviceAccount, or via custom Cluster Registry REST endpoint)
* Collect all cluster nodes and their estimated costs (AWS and GCP only)
* Collect all pods and use the ``application`` or ``app`` label as application ID
* Get additional information for each app from the application registry (``team_id`` and ``active`` field) OR use the ``team`` label on the pod
* Group and aggregate resource usage and slack costs per cluster, team and application
* Allow custom links to existing systems (e.g. link to a monitoring dashboard for each cluster)


-----
Usage
-----

The usage requires `Poetry <https://python-poetry.org/>`_ (see below for alternative with Docker):

.. code-block::

    $ poetry install && poetry shell
    $ mkdir output
    $ python3 -m kube_resource_report output/ # uses clusters defined in ~/.kube/config
    $ OAUTH2_ACCESS_TOKENS=read-only=mytok python3 -m kube_resource_report --cluster-registry=https://cluster-registry.example.org output/ # discover clusters via registry
    $ OAUTH2_ACCESS_TOKENS=read-only=mytok python3 -m kube_resource_report --cluster-registry=https://cluster-registry.example.org output/ --application-registry=https://app-registry.example.org # get team information

The output will be HTML files plus multiple tab-separated files:

``output/index.html``
    Main HTML overview page, links to all other HTML pages.
``output/clusters.tsv``
    List of cluster summaries with number of nodes and overall costs.
``output/slack.tsv``
    List of potential savings (CPU/memory slack).
``output/ingresses.tsv``
    List of ingress host rules (informational).
``output/pods.tsv``
    List of all pods and their CPU/memory requests and usages.


---------------------
Deploying to Minikube
---------------------

This will deploy a single pod with kube-resource-report and nginx (to serve the static HTML):

.. code-block::

    $ minikube start
    $ kubectl apply -f deploy/
    $ kubectl port-forward service/kube-resource-report 8080:80

Now open http://localhost:8080/ in your browser.


-----------------------
Deploy using Helm Chart
-----------------------

IMPORTANT: Helm is not used by the maintainer of kube-resource-report - the Helm Chart was contributed by `Eriks Zelenka <https://github.com/ezelenka>`_ and is not officially tested or supported!

Assuming that you have already helm properly configured (refer to helm docs), below command will install chart in the
currently active Kubernetes cluster context.

This will deploy a single pod with kube-resource-report and nginx (to serve the static HTML):

.. code-block::

    $ git clone https://github.com/hjacobs/kube-resource-report
    $ cd kube-resource-report
    $ helm install --name kube-resource-report ./chart/kube-resource-report
    $ helm status kube-resource-report

If you want to do upgrade, try something like:

.. code-block::

    $ cd kube-resource-report
    $ git fetch --all
    $ git checkout master & git pull
    $ helm upgrade kube-resource-report ./chart/kube-resource-report
    $ helm status kube-resource-report

Use ``helm status`` command to verify deployment and obtain instructions to access ``kube-resource-report``.


---------------------------
Running as Docker container
---------------------------

.. code-block::

    $ kubectl proxy & # start proxy to your cluster (e.g. Minikube)
    $ # run kube-resource-report and generate static HTML to ./output
    $ docker run --rm -it --user=$(id -u) --net=host -v $(pwd)/output:/output hjacobs/kube-resource-report:19.12.2 /output

**For macOS**:

.. code-block::

    $ kubectl proxy --accept-hosts '.*' & # start proxy to your cluster (e.g. Minikube)
    $ # run kube-resource-report and generate static HTML to ./output
    $ docker run --rm -it -e CLUSTERS=http://docker.for.mac.localhost:8001 --user=$(id -u) -v $(pwd)/output:/output hjacobs/kube-resource-report:19.12.2 /output

--------------------
Application Registry
--------------------

The optional application registry can provide information per application ID, it needs to have a REST API like:

.. code-block::

    $ curl -H 'Authorization: Bearer <mytok>' https://app-registry.example.org/apps/<application-id>
    {
    "team_id": "<team-id>",
    "active": true
    }

See the ``application-registry.py`` script in the ``sample-report`` folder for an example implementation.


------------
Custom Links
------------

The generated report can be enhanced with custom links to existing systems, e.g. to link to monitoring dashboards or similar.
This currently works for clusters, teams, and applications. Custom links can be specified by providing the ``--links-file`` option which must point to a YAML file
with the links per entity. Example file:

.. code-block:: yaml

    cluster:
    - href: "https://mymonitoringsystem.example.org/dashboard?cluster={name}"
      title: "Grafana dashboard for cluster {name}"
      icon: chart-area
    application:
    - href: "https://mymonitoringsystem.example.org/dashboard?application={id}"
      title: "Grafana dashboard for application {id}"
      icon: chart-area
    - href: "https://apps.mycorp.example.org/apps/{id}"
      title: "Go to detail page of application {id}"
      icon: search
    team:
    - href: "https://people.mycorp.example.org/search?q=team:{id}"
      title: "Search team {id} on people.mycorp"
      icon: search
    ingress:
    - href: "https://kube-web-view.mycorp.example.org/clusters/{cluster}/namespaces/{namespace}/ingresses/{name}"
      title: "View ingress {name} in Kubernetes Web View"
      icon: external-link-alt
    node:
    - href: "https://kube-web-view.mycorp.example.org/clusters/{cluster}/nodes/{name}"
      title: "View node {name} in Kubernetes Web View"
      icon: external-link-alt
    namespace:
    - href: "https://kube-web-view.mycorp.example.org/clusters/{cluster}/namespaces/{name}"
      title: "View namespace {name} in Kubernetes Web View"
      icon: external-link-alt
    pod:
    - href: "https://kube-web-view.mycorp.example.org/clusters/{cluster}/namespaces/{namespace}/pods/{name}"
      title: "View pod {name} in Kubernetes Web View"
      icon: external-link-alt

For available icon names, see the `Font Awesome gallery with free icons <https://fontawesome.com/icons?d=gallery&m=free>`_.


--------
Settings
--------

You can run ``docker run --rm hjacobs/kube-resource-report:19.12.2 --help`` to find out information.

Besides this, you can also pass environment variables:

- ``NODE_LABEL_SPOT`` (default: ``"aws.amazon.com/spot"``)
- ``NODE_LABEL_ROLE`` (default: ``"kubernetes.io/role"``)
- ``NODE_LABEL_REGION`` (default: ``"failure-domain.beta.kubernetes.io/region"``)
- ``NODE_LABEL_INSTANCE_TYPE`` (default: ``"beta.kubernetes.io/instance-type"``)
- ``OBJECT_LABEL_APPLICATION`` (default: ``"application,app,app.kubernetes.io/name"``)
- ``OBJECT_LABEL_COMPONENT`` (default: ``"component,app.kubernetes.io/component"``)
- ``OBJECT_LABEL_TEAM`` (default: ``"team,owner"``)
