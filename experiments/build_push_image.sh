#!/bin/bash
# This script uses a Dockerfile in the current workdir to create
# a docker image which name is given in the script options.
# You will need to change below what is copied in your container.

while [[ $# -gt 0 ]]
do
	case "$1" in
	    -n|--imagename)
	    input_imagename="$2"
	    shift # past argument
	    shift # past value
	    ;;
	    *)
		shift
		echo "Invalid option -$1" >&2
	    ;;
	esac
done

account=$(aws sts get-caller-identity --query Account --output text)

# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)
region=${region:-us-east-1}

fullname="${account}.dkr.ecr.${region}.amazonaws.com/${input_imagename}:latest"

# If the repository doesn't exist in ECR, create it.

aws ecr describe-repositories --repository-names "${input_imagename}" > /dev/null 2>&1

if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${input_imagename}" > /dev/null
fi

# Get the login command from ECR and execute it directly
$(aws ecr get-login --region ${region} --no-include-email)

# populate the folder content
# todo: change the lines below to copy what is important to you in the docker image
# todo: A script named 'train' must always be copied in the container because
# todo: that Sagemaker will run the following command : docker run ${input_imagename} train
# todo: The first line of this script should inform you about what interpreter
# todo: is required to execute the script unless you have an entry point in the Docker image
content_dir=./content_${input_imagename}/
mkdir ${content_dir}
cp -r ../metalearn ${content_dir}
cp ../install.sh ${content_dir}
cp ../setup.py ${content_dir}
cp expts_utils.py ${content_dir}
cp train ${content_dir}

# Build the docker image locally with the image name
docker build  \
--build-arg img_workdir=/opt/program/ \
--build-arg content_dir=${content_dir} \
--build-arg pyenv_name=${input_imagename} \
-t ${input_imagename} \
-f Dockerfile .

rm -rf $(basename ${content_dir})

# tag and push the image to aws ECR
docker tag ${input_imagename} ${fullname}
docker push ${fullname}