#!/bin/bash

default_region=us-east-1
COMMON_IMAGENAME=invivobase

f_ecr_repo () { 
    account=$(aws sts get-caller-identity --query Account --output text)
    # Get the region defined in the current configuration (default to us-east-1 if none defined)
    region=$(aws configure get region)
    region=${region:-default_region}
    repo="${account}.dkr.ecr.${region}.amazonaws.com/$1:latest"
}

f_create_repo () { 
    # Get the region defined in the current configuration (default to us-east-1 if none defined)
    region=$(aws configure get region)
    region=${region:-default_region}
    # If the repository doesn't exist in ECR, create it.
    aws ecr describe-repositories --repository-names "$1" > /dev/null 2>&1
    if [ $? -ne 0 ]
    then
        aws ecr create-repository --repository-name "$1" > /dev/null
    fi
    # Get the login command from ECR and execute it directly
    $(aws ecr get-login --region ${region} --no-include-email)
}