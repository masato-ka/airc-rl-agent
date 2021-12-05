ARG BASE_IMAGE=jetbot/jetbot:jupyter-0.4.2-32.4.4
FROM ${BASE_IMAGE}

ENV AI_RC_CAR_HOME=/opt/ai-rc-car
COPY . ${AI_RC_CAR_HOME}
RUN cd ${AI_RC_CAR_HOME} && ls && sh install_jetpack_docker.sh
