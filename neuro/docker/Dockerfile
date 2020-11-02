FROM ubuntu:20.04

# Download the model weights
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        apt-utils \
        cmake \
        ca-certificates \
        wget \
        git \
        libssl-dev \
        curl \
        unzip \
        software-properties-common \
        python3-distutils-extra \
        && \
    ln -s -T /usr/bin/python3 /usr/local/bin/python && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python ~/get-pip.py && \
    pip install -U pip && \
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

RUN wget -P /root/.cache/torch/hub/checkpoints/ http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth

COPY . /cfg
RUN pip install --progress-bar=off -U --no-cache-dir -r /cfg/requirements.txt
