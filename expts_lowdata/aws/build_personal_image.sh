#!/bin/bash
# This script uses a Dockerfile in the current workdir to create
# a docker image which name is given in the script options.
# You will need to change below what is copied in your container.

# Example of usage:
# bash build_personal_image.sh -n imageXYZ -i content.txt
while [[ $# -gt 0 ]]
do
	case "$1" in
	    -n|--imagename)
	    input_imagename="$2"
	    shift # past argument
	    shift # past value
	    ;;
        -i|--inputfile)
	    inputfile="$2"
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
startpoint="${account}.dkr.ecr.${region}.amazonaws.com/invivobase:latest"
# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --repository-names "${input_imagename}" > /dev/null 2>&1
if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${input_imagename}" > /dev/null
fi
# Get the login command from ECR and execute it directly
$(aws ecr get-login --region ${region} --no-include-email)

# create the build context and it populate it with files/folders listed in the inputfile
# todo: A script named 'train' must always be copied in the container because
# todo: that Sagemaker will run the following command : docker run ${input_imagename} train
# todo: The first line of this script should inform you about what interpreter
# todo: is required to execute the script unless you have an entry point in the Docker image
content_dir=./content_${input_imagename}
dockerfile=${content_dir}/Dockerfile
mkdir -p ${content_dir}
cat ${inputfile} | while read f; do
    cp -r $f ${content_dir}
done
chmod +x ${content_dir}/train

# Build the docker image locally with the image name
echo -e "FROM ${startpoint}\nCOPY $content_dir ./" > ${dockerfile}
echo -e "\nRUN conda env create --name fewshot --file env.yml " >> ${dockerfile}
echo -e "\nRUN sed -i '/conda/d' ~/.bashrc" >> ${dockerfile}
echo -e '\nRUN echo ". ~/miniconda3/etc/profile.d/conda.sh;" >> ~/.bashrc' >> ${dockerfile}
echo -e '\nENV PATH=/root/miniconda3/envs/fewshot/bin:$PATH' >> ${dockerfile}
echo -e '\nRUN echo "conda activate fewshot"  >> ~/.bashrc && source ~/.bashrc' >> ${dockerfile}
echo -e "\nRUN pip install -e ." >> ${dockerfile}

docker build \
-t ${input_imagename} \
-f ${dockerfile} .

rm -rf ${content_dir}

# tag and push the image to aws ECR
docker tag ${input_imagename} ${fullname}
docker push ${fullname}
