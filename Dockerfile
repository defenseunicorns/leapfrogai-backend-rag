ARG ARCH=amd64

FROM ghcr.io/defenseunicorns/leapfrogai/python:3.11-dev-${ARCH} as builder

WORKDIR /leapfrogai

COPY requirements.txt .

RUN pip install -r requirements.txt --user

ARG CHROMADB_PORT=8000
ENV CHROMADB_PORT=${CHROMADB_PORT}

ARG CHROMADB_DATA_PATH=/leapfrogai/chromadb
ENV CHROMADB_DATA_PATH=${CHROMADB_DATA_PATH}

USER nonroot

RUN mkdir -p ${CHROMADB_DATA_PATH}

ENTRYPOINT /home/nonroot/.local/bin/chroma run --path ${CHROMADB_DATA_PATH} --host 0.0.0.0 --port ${CHROMADB_PORT}