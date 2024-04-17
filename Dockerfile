# Base image
#FROM nvcr.io/nvidia/pytorch:23.07-py3
FROM pytorch/pytorch:latest

# Maintainer
LABEL maintainer="Adrien joliat <adrien.joliat@epfl.ch>"

# Pre-configure tzdata package + install packages
ENV DEBIAN_FRONTEND=noninteractive
RUN echo "Etc/UTC" > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        pkg-config \
        tzdata \
        inkscape \
        texlive-latex-extra \
        dvipng \
        texlive-full \
        jed \
        libsm6 \
        libxext-dev \
        libxrender1 \
        lmodern \
        libcurl3-dev \
        libfreetype6-dev \
        libzmq3-dev \
        libcupti-dev \
        pkg-config \
        libjpeg-dev \
        libpng-dev \
        zlib1g-dev \
        locales \
        rsync \
        cmake \
        g++ \
        swig \
        vim \
        git \
        curl \
        wget \
        unzip \
        zsh \
        git \
        screen \
        tmux \
        openssh-server \
     && rm -rf /var/lib/apt/lists/*

# Install good vim
RUN curl http://j.mp/spf13-vim3 -L -o - | sh

# Configure environments
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && locale-gen

# Configure user
ARG NB_USER=ajoliat
ARG NB_UID=217440
ARG NB_GROUP=MLO-unit
ARG NB_GID=30200
ENV SHELL=/bin/bash \
    HOME=/home/$NB_USER

RUN groupadd $NB_GROUP -g $NB_GID && \
    useradd -m -s /bin/bash -N -u $NB_UID -g $NB_GID $NB_USER && \
    echo "${NB_USER}:${NB_USER}" | chpasswd && \
    usermod -aG sudo,adm,root ${NB_USER} && \
    chown -R ${NB_USER}:${NB_GROUP} ${HOME} && \
    echo "${NB_USER}   ALL = NOPASSWD: ALL" > /etc/sudoers

# Install Visual Studio Code Server
RUN curl -fsSL https://code-server.dev/install.sh | sh

# Switch to user by numeric ID
USER $NB_UID:$NB_GID

# Start code-server in the background
CMD code-server --bind-addr 0.0.0.0:8080 &

# Expose ports for ssh, notebook, tensorboard, and code-server
EXPOSE 22 8888 6666 8080