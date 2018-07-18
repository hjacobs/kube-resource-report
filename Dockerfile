FROM python:3.7-alpine3.8

RUN pip3 install pipenv

COPY Pipfile /
COPY Pipfile.lock /

WORKDIR /
RUN pipenv install --system --deploy --ignore-pipfile

COPY *.py /
COPY templates /templates

ENTRYPOINT ["/report.py"]
