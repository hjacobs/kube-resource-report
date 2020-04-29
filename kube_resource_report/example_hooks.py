"""Define example hook functions for Kubernetes Resource Report."""
import random

from pykube import Node
from pykube import Pod


def map_node(node: Node, mapped_node: dict):
    """
    Set a random cost for each node (as example).

    Example hook function to modify the mapped Kubernetes Node.
    """

    # random price between $10 and $500
    mapped_node["cost"] = random.uniform(10, 500)


def map_pod(pod: Pod, mapped_pod: dict):
    """
    Set a custom aggregation key for resource recommendations: aggregate by controller name (e.g. ReplicaSet name).

    Example hook function to modify the mapped Kubernetes Pod.
    """
    owner_names = []
    for ref in pod.metadata.get("ownerReferences", []):
        owner_names.append(ref["name"])

    # note that the aggregation MUST be a tuple of 4 strings
    mapped_pod["aggregation_key"] = (
        pod.namespace,
        mapped_pod["application"],
        ",".join(sorted(owner_names)),
        ",".join(sorted(mapped_pod["container_names"])),
    )


def prerender(template_name: str, context: dict):
    """
    Add some random external link for each application.

    Example hook function to modify context for HTML page rendering.
    """

    application_links = context["links"].get("application", [])
    # only add the link once
    if not application_links:
        application_links.append(
            {
                "href": "https://srcco.de/",
                "class": "is-link",
                "title": "Some example link!",
                "icon": "external-link-alt",
            }
        )

    context["links"]["application"] = application_links
