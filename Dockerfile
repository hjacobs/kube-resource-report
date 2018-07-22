FROM python:3.7-alpine3.8

RUN pip3 install pipenv

COPY Pipfile /
COPY Pipfile.lock /

WORKDIR /
RUN pipenv install --system --deploy --ignore-pipfile

COPY kube_resource_report /kube_resource_report

ARG VERSION=dev
RUN sed -i "s/__version__ = .*/__version__ = '${VERSION}'/" /kube_resource_report/__init__.py

ENTRYPOINT ["python3", "-m", "kube_resource_report"]
