# hash:sha256:eb293166728a4a240deaca9ae18465560a3fd00fd71d7e198f6ace4f9834ac4c
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        vim=2:8.1.2269-1ubuntu5.23 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    h5py==3.11.0 \
    pydantic==2.8.2 \
    scanimage-tiff-reader==1.4.1.4

ADD "https://github.com/coder/code-server/releases/download/v4.9.0/code-server-4.9.0-linux-amd64.tar.gz" /.code-server/code-server.tar.gz
	
RUN cd /.code-server \
	&& tar -xvf code-server.tar.gz \
	&& rm code-server.tar.gz \
	&& ln -s /.code-server/code-server-4.9.0-linux-amd64/bin/code-server  /usr/bin/code-server
