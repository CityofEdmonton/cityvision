# Description: Dockerfile to build the flex template image

FROM python:3.11

# get essential packages for the image
RUN apt-get update && apt-get install -y \
	ffmpeg libsm6 libxext6 && \
	apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /cityvision

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .



