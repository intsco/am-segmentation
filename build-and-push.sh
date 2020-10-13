#!/usr/bin/env bash

# This script shows how to build the Docker image and push it to ECR to be ready for use
# by SageMaker.

# The argument to this script is the image name. This will be used as the image on the local
# machine and combined with the account and region to form the repository name for ECR.

# Validate script arguments
image_type=$1

if [ "$image_type" == "" ]; then
    echo "Usage: $0 <image-type: train|predict>"
    exit 1
fi

export AWS_PROFILE=am-segm

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text)
if [ $? -ne 0 ]
then
    exit 255
fi

# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)

# Choose type of model image
if [ "$image_type" == "train" ]; then
  image="am-segm/sagemaker-pytorch-train"

  # Choose base PyTorch image version
  pytorch_image_version=1.5.1-gpu-py36-cu101-ubuntu16.04
  echo "Using PyTorch image version: $pytorch_image_version"

  build_cmd="docker build -t ${image}
    --build-arg REGION=${region} --build-arg VERSION=${pytorch_image_version}
    -f sagemaker/train/Dockerfile ."
else
  image="am-segm/ecs-pytorch-predict"
  build_cmd="docker build -t ${image} -f ecs/Dockerfile ."
fi

# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --repository-names "${image}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${image}" > /dev/null
fi

# AWS public DL images repository
# https://github.com/aws/deep-learning-containers/blob/master/available_images.md
aws ecr get-login-password --region "${region}" | docker login \
    --username AWS \
    --password-stdin \
    763104351884.dkr.ecr."${region}".amazonaws.com

# am-segm images repository
aws ecr get-login-password --region "${region}" | docker login \
    --username AWS \
    --password-stdin \
    "${account}".dkr.ecr."${region}".amazonaws.com

# Build the docker image locally with the image name and then push it to ECR with the full name
eval $build_cmd
if [ $? -ne 0 ]
then
    exit 255
fi

# Tag image with full name and push to registry
fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"
docker tag "${image}" "${fullname}"
docker push "${fullname}"
