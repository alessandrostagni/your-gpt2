FROM tensorflow/tensorflow:latest-gpu

# System dependencies
RUN apt update && \
    apt install -y git

WORKDIR /home

# Python dependencies
COPY requirements.txt $WORKDIR
RUN pip install -r requirements.txt
