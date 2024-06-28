# ===
# Prismatic VLM Sagemaker Dockerfile
#   => Base Image :: Python 3.10 & Pytorch 2.2.0
# ===
# FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.2.0-gpu-py310-cu121-ubuntu20.04-sagemaker
FROM nvcr.io/nvidia/pytorch:23.06-py3

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

RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio

# Install Prismatic Python Dependencies (`pip`) + Sagemaker
RUN pip install \
    accelerate>=0.25.0 \
    draccus@git+https://github.com/dlwh/draccus \
    git+https://github.com/mlfoundations/open_lm.git \
    git+https://github.com/kvablack/dlimp.git \
    composer \
    einops \
    huggingface_hub \
    jsonlines \
    matplotlib \
    pyyaml-include==1.4.1 \
    rich \
    sentencepiece \
    timm>=0.9.10 \
    transformers>=4.38.1 \
    tensorflow-graphics \
    tensorflow-datasets \
    wandb
    # sagemaker-training

RUN pip install packaging ninja
RUN pip install transformers --force-reinstall
RUN pip install transformer-engine --force-reinstall
RUN pip install --force-reinstall torch torchvision tensorflow
RUN pip install protobuf==3.20.1
RUN pip install --upgrade flash-attn --no-build-isolation
RUN pip install tensorflow-datasets


# Set Sagemaker Environment Variables =>> Define `pretrain.py` as entrypoint!
ENV PATH="/opt/ml/code:${PATH}"
ENV PYTHONPATH="/opt/ml/code:${PYTHONPATH}"

ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code
ENV SAGEMAKER_PROGRAM=/opt/ml/code/scripts/pretrain.py

# Copy Working Directory to `/opt/ml/code`
COPY . /opt/ml/code/