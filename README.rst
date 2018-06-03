==========================
Kubernetes Resource Report
==========================

**This early version only supports the AWS eu-central-1 region for price/cost information**

Script to generate a HTML report of CPU/memory requests vs. usage (collected via Heapster) for one or more Kubernetes clusters.

Want to see how the report looks? `Check out the sample HTML report! <https://hjacobs.github.io/kube-resource-report/sample-report/output/index.html>`_)

* Collect all cluster nodes and their estimated costs (AWS only)
* Collect all pods and use the ``application`` or ``app`` label as application ID
* Get additional information for each app from the application registry (``team_id`` and ``active`` field)
* Group and aggregate resource usage and slack costs per cluster, team and application

Usage (requires `Pipenv <https://docs.pipenv.org/>`_):

.. code-block::

    $ pipenv install && pipenv shell
    $ mkdir output
    $ ./report.py output/ # uses clusters defined in ~/.kube/config
    $ OAUTH2_ACCESS_TOKENS=read-only=mytok ./report.py --cluster-registry=https://cluster-registry.example.org output/ # discover clusters via registry
    $ OAUTH2_ACCESS_TOKENS=read-only=mytok ./report.py --cluster-registry=https://cluster-registry.example.org output/ --application-registry=https://app-registry.example.org # get team information

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
