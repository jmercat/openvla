# ===
# OpenVLA Sagemaker Dockerfile
#   => Base Image :: Python 3.10 & Pytorch 2.2.0
# ===
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.2.0-gpu-py310-cu121-ubuntu20.04-sagemaker

# Sane Defaults
RUN apt-get update
RUN apt-get update && apt-get install -y \
    cmake \
    curl \
    docker.io \
    ffmpeg \
    git \
    htop \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    libsm6 \
    libxrender-dev \
    libxext6 \
    sudo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    libgl1 \
    libopenexr-dev \
    mesa-utils \
    freeglut3-dev \
    libsdl2-2.0-0 \
    python-pygame


# Install Prismatic + VLA Python Dependencies (`pip`) + Sagemaker
RUN pip install --upgrade pip
RUN pip install \
    accelerate>=0.25.0 \
    draccus@git+https://github.com/dlwh/draccus \
    einops \
    jsonlines \
    matplotlib \
    rich \
    sentencepiece \
    timm>=0.9.10 \
    transformers>=4.38.1 \
    wandb \
    tensorflow==2.15.0 \
    tensorflow_datasets==4.9.3 \
    tensorflow_graphics==2021.12.3 \
    dlimp@git+https://github.com/kvablack/dlimp@ad72ce3a9b414db2185bc0b38461d4101a65477a \
    sagemaker-training

# Flash Attention 2 Installation
RUN pip install packaging ninja
RUN pip install flash-attn==2.5.5 --no-build-isolation


# Set Sagemaker Environment Variables =>> Define `vla-scripts/train.py` as entrypoint!
ENV PATH="/opt/ml/code:${PATH}"

ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code
ENV SAGEMAKER_PROGRAM=/opt/ml/code/vla-scripts/train.py

# Copy Working Directory to `/opt/ml/code`
COPY . /opt/ml/code/
