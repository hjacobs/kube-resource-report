====================
Kubernetes Resources
====================

Script to generate a report of CPU/memory requests vs. usage (collected via Heapster).

Usage:

.. code-block::

    $ pipenv install && pipenv shell
    $ mkdir output
    $ ./report.py output/ # uses clusters defined in ~/.kube/config
    $ OAUTH2_ACCESS_TOKENS=read-only=mytok ./report.py --cluster-registry=https://cluster-registry.example.org output/ # discover clusters via registry

The output will be multiple tab-separated files:

``output/clusters.tsv``
    List of cluster summaries with number of nodes and overall costs.
``output/slack.tsv``
    List of potential savings (CPU/memory slack).

