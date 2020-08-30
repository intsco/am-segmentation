#!/usr/bin/env bash

image="intsco/am-segm-batch"

docker build -t "${image}" -f ecs/Dockerfile .
docker push "${image}"
