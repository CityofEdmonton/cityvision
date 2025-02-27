# Description: Dockerfile to build the flex template image

FROM gcr.io/dataflow-templates-base/flex-template-launcher-image:latest as template_launcher


FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /opt

ENV TZ=US/Arizona\
    DEBIAN_FRONTEND=noninteractive

COPY . /opt
RUN \
    # Add Deadsnakes repository that has a variety of Python packages for Ubuntu.
    # See: https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa
    apt-get update \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get install -y curl \
        python3.11 \
        python3.11-venv \
        python3-venv \
        ffmpeg \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 10 \
    && curl https://bootstrap.pypa.io/get-pip.py | python \
    && pip install --upgrade pip \
    && pip install apache-beam[gcp]==2.59.0\
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
    && pip install -e .
# Since we already downloaded all the dependencies, there's no need to rebuild everything.
ENV PIP_NO_DEPS=True

ENV PYTHONPATH "${PYTHONPATH}:/cityvision/"

RUN rm -fr /opt/cityvision/__pycache__

ENV FLEX_TEMPLATE_PYTHON_PY_FILE="main.py"

# Copy the Dataflow Template launcher
COPY --from=template_launcher /opt/google/dataflow/python_template_launcher /opt/google/dataflow/python_template_launcher

COPY --from=apache/beam_python3.11_sdk:2.59.0 /opt/apache/beam /opt/apache/beam

ENTRYPOINT ["/opt/apache/beam/boot"]



