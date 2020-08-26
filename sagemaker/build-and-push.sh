#!/usr/bin/env bash

# This script shows how to build the Docker image and push it to ECR to be ready for use
# by SageMaker.

# The argument to this script is the image name. This will be used as the image on the local
# machine and combined with the account and region to form the repository name for ECR.
image=$1
processor_type=${2:-cpu}

if [ "$image" == "" ]; then
    echo "Usage: $0 <image-name> [<processor_type>]"
    exit 1
fi

# Choose base PyTorch image version
if [ "$processor_type" == "cpu" ]; then
    pytorch_image_version=1.5.1-cpu-py36-ubuntu16.04
else
    pytorch_image_version=1.5.1-gpu-py36-cu101-ubuntu16.04
fi
echo "Using PyTorch image version: $pytorch_image_version"

export AWS_PROFILE=am-segm

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text)

if [ $? -ne 0 ]
then
    exit 255
fi

# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"

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

# Build the docker image locally with the image name and then push it to ECR
# with the full name.

docker build  -t "${image}" \
  --build-arg REGION="${region}" --build-arg VERSION="${pytorch_image_version}" \
  -f sagemaker/train/Dockerfile .

docker tag "${image}" "${fullname}"

docker push "${fullname}"
