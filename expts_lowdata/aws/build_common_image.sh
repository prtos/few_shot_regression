#!/bin/bash
# This script uses a Dockerfile in the current workdir to create
# a docker image which is stored on aws ECR and can be used again later.
# IMPORTANT: This script requires to add ssh keys to the bitbucket account.

imagename=invivobase

account=$(aws sts get-caller-identity --query Account --output text)
# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)
region=${region:-us-east-1}
fullname="${account}.dkr.ecr.${region}.amazonaws.com/${imagename}:latest"
# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --repository-names "${imagename}" > /dev/null 2>&1
if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${imagename}" > /dev/null
fi
# Get the login command from ECR and execute it directly
$(aws ecr get-login --region ${region} --no-include-email)


# Build the docker image locally with the image name
docker build  \
--build-arg img_workdir=/opt/program/ \
--build-arg ssh_prv_key="$GIT_ROBOT_USERNAME" \
--build-arg ssh_pub_key="$GIT_ROBOT_PASSWD" \
-t ${imagename} .

# tag and push the image to aws ECR
docker tag ${imagename} ${fullname}
docker push ${fullname}
