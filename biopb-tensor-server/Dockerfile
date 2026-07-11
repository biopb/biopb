# Stage 1: Webapp build (Node.js)
FROM node:20-slim AS webapp-builder
WORKDIR /app

# Copy pre-built webapp. Built for the CONTROL FRONT (this image runs the control
# as its web origin), so it is namespaced under /data_plane/viewer:
#   VITE_BASE_PATH="/data_plane/viewer/" VITE_TENSOR_API="/data_plane" pnpm --filter @biopb/web build
COPY biopb-tensor-server/packages/web/dist ./biopb-tensor-server/packages/web/dist

# Stage 2: Python build
FROM python:3.11-slim AS python-builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built wheels (built locally with: pip wheel . --no-deps -w wheels/ &&
# pip wheel ./biopb-tensor-server --no-deps -w wheels/ &&
# pip wheel ./biopb-control --no-deps -w wheels/)
COPY wheels/biopb-*.whl /app/wheels/
COPY wheels/biopb_tensor_server-*.whl /app/wheels/
COPY wheels/biopb_control-*.whl /app/wheels/

# Install the wheels with extras. Dependency bounds (zarr<3, the tifffile
# Zarr-2 window) live in each wheel's own metadata, so no manual pre-pin is
# needed. The `bioformats` extra pulls bioio-bioformats + scyjava for the Java
# Bio-Formats fallback (ZVI, ...); the runtime image ships a JDK below.
# biopb-control is the single web origin the entrypoint runs; it pulls its light
# ASGI stack (starlette/uvicorn/httpx/websockets) from PyPI.
WORKDIR /app
RUN TENSOR_WHEEL=$(ls /app/wheels/biopb_tensor_server-*.whl) \
    && pip install --no-cache-dir /app/wheels/biopb-*.whl \
    && pip install --no-cache-dir /app/wheels/biopb_control-*.whl \
    && pip install --no-cache-dir "$TENSOR_WHEEL[web,aics,czi,bioformats,medical,ndtiff,hdf5]"

# Stage 3: Runtime (nginx + python)
FROM python:3.11-slim
WORKDIR /app

# Install Java (for bioio-bioformats) and runtime deps - NO nginx
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    default-jdk-headless \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages
COPY --from=python-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=python-builder /usr/local/bin /usr/local/bin

# Copy webapp build output (for FastAPI StaticFiles)
COPY --from=webapp-builder /app/biopb-tensor-server/packages/web/dist /app/webapp

# Copy entrypoint
COPY biopb-tensor-server/entrypoint.sh /usr/local/bin/

ENV JAVA_HOME=/usr/lib/jvm/default-java

# Configure scyjava to use system JDK instead of downloading via cjdk
RUN echo 'import scyjava.config; scyjava.config.set_java_constraints(fetch="never")' \
    > /usr/local/lib/python3.11/site-packages/sitecustomize.py

# Create data directory and set permissions
RUN mkdir -p /data \
    && chmod +x /usr/local/bin/entrypoint.sh

# Expose the control web origin (8813) + Flight gRPC (8815). The tensor HTTP
# sidecar (8814) is now PRIVATE behind the control (loopback bind) and not exposed.
# Flight gRPC binds directly (no nginx proxy) for SDK clients.
EXPOSE 8813 8815

# The entrypoint runs the control plane in the foreground (no subcommand arg).
ENTRYPOINT ["entrypoint.sh"]
