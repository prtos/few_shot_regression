# Build an image that can do training and inference in SageMaker
# This is a Python 2 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

# FROM ubuntu:16.04 # for cpu images uncomment this line
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
MAINTAINER InvivoAI <prudencio@invivoai.com>
SHELL [ "/bin/bash", "-c" ]
ARG ssh_prv_key
ARG ssh_pub_key

RUN apt-get -y update \
    && apt-get install -y \
         gcc \
         wget \
         nginx \
         bzip2 \
         git \
         ssh \
         nano \
         locales \
         ca-certificates \
         libxrender-dev \
         libxext6 \
         libsm6 \
         openssh-server \
         libmysqlclient-dev \
    && echo "export LC_ALL=en_US.UTF-8" >> ~/.bashrc  \
    && locale-gen "en_US.UTF-8" \
    && update-locale LC_ALL="en_US.UTF-8" \
    && rm -rf /var/lib/apt/lists/*
# useful option # --no-install-recommends

ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
ENV PATH="$img_workdir:${PATH}"
ENV PATH=/root/miniconda3/bin:$PATH

# get Anaconda installer and install anaconda
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -f && \
    rm ~/anaconda.sh && \
    echo ". ~/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base"  >> ~/.bashrc

WORKDIR /app/
# Authorize SSH Host
# Add the keys and set permissions
# clone the git repo in /app/
# and remove any trace of the ssh keys
RUN mkdir -p /root/.ssh && \
    chmod 0700 /root/.ssh && \
    ssh-keyscan bitbucket.org > /root/.ssh/known_hosts && \
    echo "$ssh_prv_key" > ~/.ssh/id_rsa && \
    echo "$ssh_pub_key" > ~/.ssh/id_rsa.pub && \
    chmod 600 ~/.ssh/id_rsa && \
    chmod 600 ~/.ssh/id_rsa.pub && \
	git clone https://$ssh_prv_key:$ssh_pub_key@bitbucket.org/invivoai/invivobase.git && \
    rm -rf ~/.ssh/

# install the codebase and all requirements
RUN . ~/.bashrc && \
    cd invivobase && git pull origin && git checkout integration && \
    bash scripts/install.sh -c -n
# change the defaut conda env. This is a bad idea
ENV PATH=/root/miniconda3/envs/invivo/bin:$PATH

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

# Set up the workdir of the program in the image
WORKDIR /opt/program/
