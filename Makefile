.PHONY: test docker push

IMAGE            ?= hjacobs/kube-resource-report
VERSION          ?= $(shell git describe --tags --always --dirty)
TAG              ?= $(VERSION)

default: docker

.PHONY:
install:
	poetry install

.PHONY:
lint: install
	poetry run black --check kube_resource_report tests
	poetry run flake8
	poetry run mypy --ignore-missing-imports kube_resource_report/

.PHONY:
test: install lint
	poetry run coverage run --source=kube_resource_report -m py.test
	poetry run coverage report

docker: 
	docker build --build-arg "VERSION=$(VERSION)" -t "$(IMAGE):$(TAG)" .
	@echo 'Docker image $(IMAGE):$(TAG) can now be used.'

push: docker
	docker push "$(IMAGE):$(TAG)"
	docker tag "$(IMAGE):$(TAG)" "$(IMAGE):latest"
	docker push "$(IMAGE):latest"

.PHONY: version
version:
	poetry version $(VERSION)
	sed -i 's,$(IMAGE):[0-9.]*,$(IMAGE):$(TAG),g' README.rst deploy/*.yaml
	sed -i 's,version: v[0-9.]*,version: v$(VERSION),g' deploy/*.yaml
	sed -i 's,tag: "[0-9.]*",tag: "$(VERSION)",g' chart/*/values.yaml
	sed -i 's,appVersion: "[0-9.]*",appVersion: "$(VERSION)",g' chart/*/Chart.yaml

.PHONY: release
release: push version
