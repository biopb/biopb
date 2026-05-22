# Dockerfile for biopb.image gRPC services
# Build context: repo root
# Requires: wheels/biopb-*.whl (build wheel first with: pip wheel . --no-deps -w wheels/)
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

# Install biopb wheel with tensor extras (must be pre-built in wheels/ directory)
COPY wheels/biopb-*.whl /tmp/
RUN WHL_PATH=$(ls /tmp/biopb-*.whl) && \
    pip install --no-cache-dir "${WHL_PATH}[tensor]" && rm /tmp/*.whl

# Install tensor-server (for embedded cache support)
COPY biopb-tensor-server/pyproject.toml ./tensor-server/pyproject.toml
COPY biopb-tensor-server/VERSION ./tensor-server/VERSION
COPY biopb-tensor-server/build_version.py ./tensor-server/build_version.py
COPY biopb-tensor-server/biopb_tensor_server ./tensor-server/biopb_tensor_server
WORKDIR /app/tensor-server
RUN pip install --no-cache-dir "."
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