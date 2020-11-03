FROM pytorch/pytorch

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        wget \
        libglib2.0-0

RUN wget -P /root/.cache/torch/hub/checkpoints/ http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth

COPY . /cfg
RUN pip install --progress-bar=off -U --no-cache-dir -r /cfg/requirements.txt
