# Description: Dockerfile to build the flex template image

FROM python:3.11

WORKDIR /cityvision

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .



