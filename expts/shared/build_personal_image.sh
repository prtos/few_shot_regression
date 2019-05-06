#!/bin/bash
# Example of usage:
# bash build_personal_image.sh -n imageXYZ -i content.txt
usage () {
    echo
    printf "
    This script allow to customize the common docker image which is stored on AWS ECR by
    copying the files you need for your experimentation in it. The intention being to
    use your new image later to do your experimentation on Sagemaker, we expect a file
    named *train* to be copied in your image.

    Usage: %s [OPTIONS] <ARGS>\n
    [OPTIONS] and corresponding <ARGS> are:\n
    [-n] name of the image name that will be created
    [-i] name of a file which contains all the files/dirs you want to copy \n
    in the common image before making your own. One file or dir per line is expected.
    [-h] # show this help screen then exit

    Example of input file content:
    ***
    ~/workspace/code/invivoai/invivoprojects/merck_challenge/config_baseline.json
    ~/workspace/code/invivoai/invivoprojects/merck_challenge/config_multitask.json
    ~/workspace/code/invivoai/invivoprojects/merck_challenge/data_utils.py
    ~/workspace/code/invivoai/invivoprojects/merck_challenge/merck.py
    ~/workspace/code/invivoai/invivoprojects/merck_challenge/models.py
    ~/workspace/code/invivoai/invivoprojects/merck_challenge/train

    ***
    An empty line must be left at the end of the content file.
    \n" "$0" 1>&2; exit 1;
}

# parse the options from the command line
while getopts ':h:n:i:' OPTION
do
    case $OPTION in
        n)  fc "$OPTARG"
            perso_imagename="$OPTARG"
            ;;
        i)  fc "$OPTARG"
            inputfile="$OPTARG"
            ;;
        h)  usage;;
        \?)  echo
            echo "Invalid option: -$OPTARG" >&2
            usage;;
        :)  echo
            echo "Option -$OPTARG requires an argument." >&2
            usage;;
    esac
done
shift $((OPTIND - 1))

SRCDIR=$(dirname $0)
source $SRCDIR/ecr_repository.sh

f_ecr_repo $perso_imagename
perso_repo=$repo

f_ecr_repo $COMMON_IMAGENAME
common_repo=$repo

f_create_repo $perso_imagename

# create the build context and it populate it with files/folders listed in the inputfile
# todo: A script named 'train' must always be copied in the container because
# todo: that Sagemaker will run the following command : docker run ${perso_imagename} train
# todo: The first line of this script should inform you about what interpreter
# todo: is required to execute the script unless you have an entry point in the Docker image
content_dir=./content_${perso_imagename}/
dockerfile=${content_dir}/Dockerfile
mkdir ${content_dir}
cat ${inputfile} | while read f; do
    cp -r $f ${content_dir}
done
ls ${content_dir}
chmod +x ${content_dir}train

# Build the docker image locally with the image name
printf "FROM ${common_repo}" > ${dockerfile}
printf "\nRUN sed -i '/conda activate/d' ~/.bashrc" >> ${dockerfile}
printf '\nRUN printf "\\n. /root/miniconda3/etc/profile.d/conda.sh;" >> ~/.bashrc' >> ${dockerfile}
printf "\nRUN echo \"conda activate invivo\"  >> ~/.bashrc && source ~/.bashrc" >> ${dockerfile}
printf "\nRUN pip install modAL" >> ${dockerfile}
printf "\nRUN pip install simdna" >> ${dockerfile}
printf "\nRUN pip install graphviz" >> ${dockerfile}
printf "\nRUN conda install -c conda-forge ffmpeg" >> ${dockerfile}
printf "\nCOPY $content_dir ./" >> ${dockerfile}
printf "\nRUN pip install -e ." >> ${dockerfile}

docker build --pull \
-t ${perso_imagename} \
-f ${dockerfile} .

rm -rf ${content_dir}

# tag and push the image to aws ECR
docker tag ${perso_imagename} ${perso_repo}
docker push ${perso_repo}
