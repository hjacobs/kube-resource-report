.PHONY: docker push

IMAGE            ?= hjacobs/application-registry-example
VERSION          ?= latest
TAG              ?= $(VERSION)

default: docker

docker: 
	docker build --build-arg "VERSION=$(VERSION)" -t "$(IMAGE):$(TAG)" .
	@echo 'Docker image $(IMAGE):$(TAG) can now be used.'

push: docker
	docker push "$(IMAGE):$(TAG)"
