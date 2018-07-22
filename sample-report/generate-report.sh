#!/bin/bash

echo 'Please enable the Heapster addon for Minikube: minikube addons enable heapster'
minikube status

echo 'Deploying Application Registry..'
kubectl apply -f deploy/

echo 'Deploying Kubernetes Resource Report..'
kubectl apply -f ../deploy/

VERSION=$(git describe --tags --always --dirty)
echo "Updating version to $VERSION.."
sed -i "s/__version__ = .*/__version__ = '${VERSION}'/" ../kube_resource_report/__init__.py

sleep 10

(cd ../ && OAUTH2_ACCESS_TOKENS=read-only=mytok pipenv run python3 -m kube_resource_report sample-report/output/ --application-registry $(minikube service application-registry --url) \
    --additional-cost-per-cluster=30 --kubeconfig-contexts=minikube)
