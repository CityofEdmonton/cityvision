# Description: Dockerfile to build the flex template image

FROM gcr.io/dataflow-templates-base/python311-template-launcher-base:latest

# Location to store the pipeline artifacts.

# Set the working directory
ARG WORKDIR=/opt/dataflow_cityvision
RUN mkdir -p ${WORKDIR}
WORKDIR ${WORKDIR}

# Copy the files to the working directory
COPY . ${WORKDIR}/


# Set the environment variables
ENV FLEX_TEMPLATE_PYTHON_PY_FILE=${WORKDIR}/main.py
ENV FLEX_TEMPLATE_PYTHON_SETUP_FILE=${WORKDIR}/setup.py


# Install the dependencies
RUN pip install apache-beam[gcp]==2.54.0

RUN pip install -U -r ${WORKDIR}/requirements.txt
RUN pip install -e .

ENV PIP_NO_DEPS=True

RUN pip check

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

#RUN pip freeze > requirements.txt


#ENTRYPOINT ["/opt/google/dataflow/python_template_launcher"]