#! /bin/bash

image_name=activeml
config_file=config.json
# instance_type=local
instance_type=ml.c4.2xlarge
nb_instances=20
if [ "$instance_type" = "local" ]
then
    out=./results
else
    out=s3://invivoai-sagemaker-artifacts/${image_name}
fi

bash  ../shared/build_personal_image.sh -n ${image_name} -i content.txt
dispatcher -n ${image_name} -e ${image_name} -t ${instance_type} -c ${nb_instances} -x 72000  -o ${out} -p ${config_file} -I 155
# dispatcher -n ${image_name} -e ${image_name} -t ml.c5.2xlarge -c 1 -x 72000  -o s3://invivoai-sagemaker-artifacts/${image_name} -p ${config_file}
