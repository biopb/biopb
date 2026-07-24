# Stage 1: Python build
FROM python:3.11-slim AS python-builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built wheels (built locally with: pip wheel . --no-deps -w wheels/ &&
# pip wheel ./biopb-tensor-server --no-deps -w wheels/)
#
# This image is a headless data plane: the Arrow Flight server + its FastAPI HTTP
# sidecar, launched directly via `biopb-tensor-server launch`. There is no control
# plane and no bundled webapp — a browser UI is served elsewhere (dev server or a
# separate control deployment) and points at this server's published ports.
COPY wheels/biopb-*.whl /app/wheels/
COPY wheels/biopb_tensor_server-*.whl /app/wheels/

# Install the wheels with extras. Dependency bounds (zarr<3, the tifffile
# Zarr-2 window) live in each wheel's own metadata, so no manual pre-pin is
# needed. The tensor server depends on core `biopb`, so install that wheel first
# (its own deps resolve from PyPI). The `web` extra pulls the sidecar's ASGI
# stack (fastapi/uvicorn); `bioformats` pulls bioio-bioformats + scyjava for the
# Java Bio-Formats fallback (ZVI, ...), and the runtime image ships a JDK below.
WORKDIR /app
RUN TENSOR_WHEEL=$(ls /app/wheels/biopb_tensor_server-*.whl) \
    && pip install --no-cache-dir /app/wheels/biopb-*.whl \
    && pip install --no-cache-dir "$TENSOR_WHEEL[web,aics,czi,bioformats,medical,ndtiff]"

# Stage 2: Runtime (python only)
FROM python:3.11-slim
WORKDIR /app

# Install Java (for bioio-bioformats) and runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    default-jdk-headless \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages
COPY --from=python-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=python-builder /usr/local/bin /usr/local/bin

# Copy entrypoint
COPY biopb-tensor-server/entrypoint.sh /usr/local/bin/

ENV JAVA_HOME=/usr/lib/jvm/default-java

# Configure scyjava to use system JDK instead of downloading via cjdk
RUN echo 'import scyjava.config; scyjava.config.set_java_constraints(fetch="never")' \
    > /usr/local/lib/python3.11/site-packages/sitecustomize.py

# Create data directory and set permissions
RUN mkdir -p /data \
    && chmod +x /usr/local/bin/entrypoint.sh

# Expose the HTTP sidecar (8814, the data-plane API) + Flight gRPC (8815). Both
# are token-authenticated on a public bind. There is no control web origin.
EXPOSE 8814 8815

# The entrypoint runs `biopb-tensor-server launch` in the foreground (PID 1).
ENTRYPOINT ["entrypoint.sh"]
