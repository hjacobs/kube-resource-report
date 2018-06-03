#!/bin/bash

echo 'Please enable the Heapster addon for Minikube: minikube addons enable heapster'
minikube status

./application-registry.py &
APPLICATION_REGISTRY_PID=$!
(cd ../ && OAUTH2_ACCESS_TOKENS=read-only=mytok ./report.py sample-report/output/ --application-registry http://localhost:8080)
kill $APPLICATION_REGISTRY_PID
