# hash:sha256:aa77e5b538063a3363603a9e69166724c073f479d0c31e8b5fd04768c07e8cca
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    h5py==3.11.0 \
    pydantic==2.8.2 \
    scanimage-tiff-reader==1.4.1.4
