#!/bin/bash

# Build the docker image that is contained learning_lacer from jetbot:juputer-[JETBOT_VERSION]-[L4T_VERSION]
JUPYTER_TAG_NAME=$(sudo docker images jetbot/jetbot | grep -P jupyter)
export JETBOT_VERSION=$(echo "$JUPYTER_TAG_NAME" | cut -f 8 -d ' ' | cut -f 2 -d '-')

L4T_VERSION_STRING=$(head -n 1 /etc/nv_tegra_release)
L4T_RELEASE=$(echo "$L4T_VERSION_STRING" | cut -f 2 -d ' ' | grep -Po '(?<=R)[^;]+')
L4T_REVISION=$(echo "$L4T_VERSION_STRING" | cut -f 2 -d ',' | grep -Po '(?<=REVISION: )[^;]+')

export JETBOT_DOCKER_REMOTE=jetbot
export L4T_VERSION="$L4T_RELEASE.$L4T_REVISION"

echo "Building docker image for jetbot:jupyter-$JETBOT_VERSION-$L4T_VERSION"

sudo docker build \
    --build-arg BASE_IMAGE=$JETBOT_DOCKER_REMOTE/jetbot:jupyter-"$JETBOT_VERSION"-"$L4T_VERSION" \
    -t learning_racer:"$L4T_VERSION" \
    -f Dockerfile \
    ../.. #learning_racer repo root as context(airc-rl-agent/docker/jetbot)