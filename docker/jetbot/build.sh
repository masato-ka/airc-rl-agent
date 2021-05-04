#!/bin/bash


L4T_VERSION_STRING=$(head -n 1 /etc/nv_tegra_release)
L4T_RELEASE=$(echo $L4T_VERSION_STRING | cut -f 2 -d ' ' | grep -Po '(?<=R)[^;]+')
L4T_REVISION=$(echo $L4T_VERSION_STRING | cut -f 2 -d ',' | grep -Po '(?<=REVISION: )[^;]+')

export L4T_VERSION="$L4T_RELEASE.$L4T_REVISION"
export JETBOT_DOCKER_REMOTE=jetbot

sudo docker build \
    --build-arg BASE_IMAGE=$JETBOT_DOCKER_REMOTE/jetbot:jupyter-$JETBOT_VERSION-$L4T_VERSION \
    -t learning_racer:$L4T_VERSION \
    -f Dockerfile \
    ../.. #learning_racer repo roos as context(airc-rl-agent/docker/jetbot)