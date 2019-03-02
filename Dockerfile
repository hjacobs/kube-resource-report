FROM python:3.7-alpine3.9

WORKDIR /

COPY Pipfile.lock /
COPY pipenv-install.py /

RUN /pipenv-install.py && \
    rm -fr /usr/local/lib/python3.7/site-packages/pip && \
    rm -fr /usr/local/lib/python3.7/site-packages/setuptools

FROM python:3.7-alpine3.9

WORKDIR /

COPY --from=0 /usr/local/lib/python3.7/site-packages /usr/local/lib/python3.7/site-packages

COPY kube_resource_report /kube_resource_report

ARG VERSION=dev
RUN sed -i "s/__version__ = .*/__version__ = '${VERSION}'/" /kube_resource_report/__init__.py

ENTRYPOINT ["python3", "-m", "kube_resource_report"]
