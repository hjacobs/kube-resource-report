FROM python:3.8-slim

WORKDIR /

RUN pip3 install poetry

COPY poetry.lock /
COPY pyproject.toml /

RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-dev --no-ansi

FROM python:3.8-slim

WORKDIR /

COPY --from=0 /usr/local/lib/python3.8/site-packages /usr/local/lib/python3.8/site-packages

COPY kube_resource_report /kube_resource_report

ARG VERSION=dev
RUN sed -i "s/__version__ = .*/__version__ = '${VERSION}'/" /kube_resource_report/__init__.py

ENTRYPOINT ["python3", "-m", "kube_resource_report"]
