# Description: Dockerfile to build the flex template image

FROM gcr.io/dataflow-templates-base/python311-template-launcher-base

# Location to store the pipeline artifacts.

# Set the working directory
ARG WORKDIR=/opt/dataflow_cityvision
RUN mkdir -p ${WORKDIR}
WORKDIR ${WORKDIR}

# Copy the files to the working directory
COPY . ${WORKDIR}/


# Set the environment variables
ENV FLEX_TEMPLATE_PYTHON_PY_FILE=${WORKDIR}/main.py
ENV FLEX_TEMPLATE_PYTHON_REQUIREMENTS_FILE=${WORKDIR}/requirements.txt


# Install the dependencies
RUN apt-get update \
    # Install any apt packages if required by your template pipeline.
    && apt-get install libffi-dev git ffmpeg libsm6 -y \
    && rm -rf /var/lib/apt/lists/* 

# Upgrade pip and install the requirements.
RUN pip install --no-cache-dir --upgrade pip
RUN pip install apache-beam[gcp]==2.54.0 
# Install dependencies from requirements file in the launch environment.
RUN pip install -U -r $FLEX_TEMPLATE_PYTHON_REQUIREMENTS_FILE
# When FLEX_TEMPLATE_PYTHON_REQUIREMENTS_FILE  option is used,
# then during Template launch Beam downloads dependencies
# into a local requirements cache folder and stages the cache to workers.
# To speed up Flex Template launch, pre-download the requirements cache
# when creating the Template.
RUN pip download --no-cache-dir --dest /tmp/dataflow-requirements-cache -r $FLEX_TEMPLATE_PYTHON_REQUIREMENTS_FILE

ENV PIP_NO_DEPS=True

RUN pip check

ENTRYPOINT ["/opt/google/dataflow/python_template_launcher"]
