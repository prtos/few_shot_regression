# Build an image that can do training and inference in SageMaker
# This is a Python 2 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM ubuntu:16.04
# for gpu images use the following
# FROM nvidia/cuda:9.0-base-ubuntu16.04
MAINTAINER InvivoAI <prudencio@invivoai.com>
SHELL [ "/bin/bash", "-c" ]
ARG img_workdir
ARG content_dir
ARG pyenv_name

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
    && echo "export LC_ALL=en_US.UTF-8" >> ~/.bashrc  \
    && locale-gen "en_US.UTF-8" && update-locale LC_ALL="en_US.UTF-8" \
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

# copy the files in the $content_dir into the docker
COPY $content_dir $img_workdir

# This is optional and means that there should be an install.sh and train in your depend on your goals
RUN . ~/.bashrc && \
    cd $img_workdir && \
    pwd && \
    bash install.sh -n $pyenv_name && \
    chmod +x train
# change the defaut conda env
ENV PATH=/root/miniconda3/envs/$pyenv_name/bin:$PATH

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

# Set up the workdir of the program in the image
WORKDIR $img_workdir
