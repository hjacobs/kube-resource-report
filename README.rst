====================
Kubernetes Resources
====================

Script to generate a report of CPU/memory requests vs. usage (collected via Heapster).

Usage:

.. code-block::

    $ pipenv install && pipenv shell
    $ ./report.py  # uses clusters defined in ~/.kube/config
    $ OAUTH2_ACCESS_TOKENS=read-only=mytok ./report.py --cluster-registry=https://cluster-registry.example.org  # discover clusters via registry

The output will be a tab-separted table with one row per cluster.

