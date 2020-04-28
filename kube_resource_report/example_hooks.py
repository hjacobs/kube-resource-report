"""Define example hook functions for Kubernetes Resource Report."""


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
