# Dockerfile for biopb.image gRPC services
# Build context: repo root
# Requires: wheels/biopb-*.whl and wheels/biopb_tensor_server-*.whl
#   Build them with: biopb-image-runtime/scripts/build.sh (builds from latest git tags by default)
#
# The image is a base image for derived biopb.image services.
# It does not define a default entrypoint; derived images must provide one.
#
# Usage:
#   docker build -t biopb-image-base -f biopb-image-runtime/Dockerfile .
#   docker compose -f biopb-image-runtime/docker-compose.yaml up

FROM python:3.10-slim-bookworm

WORKDIR /app

# Install system deps and grpc_health_probe
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    procps \
    && curl -L -o /usr/local/bin/grpc_health_probe \
       https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/v0.4.24/grpc_health_probe-linux-amd64 \
    && chmod +x /usr/local/bin/grpc_health_probe \
    && rm -rf /var/lib/apt/lists/*

ENV HOME=/home/biopb
ENV PATH="$PATH:$HOME/.local/bin"

# Install Python dependencies (includes tensor server deps)
COPY biopb-image-runtime/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && rm requirements.txt

# Install biopb and biopb-tensor-server from pre-built wheels
COPY wheels/ /tmp/wheels/
RUN pip install --no-cache-dir \
        "$(ls /tmp/wheels/biopb-*.whl)[tensor]" \
        "$(ls /tmp/wheels/biopb_tensor_server-*.whl)[tensor]" \
    && rm -rf /tmp/wheels

WORKDIR /app

# Copy base utilities (includes mock_servicer for tests and explicit dev use)
COPY biopb-image-runtime/src/biopb_image_base /opt/biopb/biopb_image_base
ENV PYTHONPATH="/opt/biopb"

# Copy test client (for testing real servicers within Docker)
COPY biopb-image-runtime/tests/client.py /opt/biopb/tests/client.py
COPY biopb-image-runtime/tests/test_image.png /opt/biopb/tests/test_image.png

WORKDIR $HOME

# Create non-root user with writable home directory
RUN useradd -m biopb \
    && mkdir -p /home/biopb/.local/bin \
              /home/biopb/.cache \
              /data/cache \
    && chown -R biopb:biopb /home/biopb /opt/biopb /data/cache \
    && chmod 777 /data/cache
USER biopb

VOLUME /data/cache

EXPOSE 50051
EXPOSE 8817

# Derived images should provide an explicit entrypoint/command.