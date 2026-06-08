# Stage 1: Webapp build (Node.js)
FROM node:20-slim AS webapp-builder
WORKDIR /app

# Copy pre-built webapp (built locally with: VITE_TENSOR_API="" pnpm --filter @biopb/web build)
COPY biopb-tensor-server/packages/web/dist ./biopb-tensor-server/packages/web/dist

# Stage 2: Python build
FROM python:3.11-slim AS python-builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built wheels (built locally with: pip wheel . --no-deps -w wheels/ && pip wheel ./biopb-tensor-server --no-deps -w wheels/)
COPY wheels/biopb-*.whl /app/wheels/
COPY wheels/biopb_tensor_server-*.whl /app/wheels/

# Install with correct order:
# 1. First pin zarr/numcodecs for compatibility
# 2. Then install both wheels with extras
# 3. Finally add bioformats-jar for LSM/OIF/OIB (BioFormats fallback for CZI/LIF)
WORKDIR /app
RUN TENSOR_WHEEL=$(ls /app/wheels/biopb_tensor_server-*.whl) \
    && pip install --no-cache-dir "zarr==2.15.0" "numcodecs==0.12.1" \
    && pip install --no-cache-dir /app/wheels/biopb-*.whl \
    && pip install --no-cache-dir "$TENSOR_WHEEL[web,aics,medical,ndtiff,hdf5]" \
    && pip install --no-cache-dir bioformats-jar

# Stage 3: Runtime (nginx + python)
FROM python:3.11-slim
WORKDIR /app

# Install Java (for bioformats-jar) and runtime deps - NO nginx
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

# Expose ports: HTTP (8814), gRPC (8815)
# Flight server gRPC binds directly (no nginx proxy)
EXPOSE 8814 8815

ENTRYPOINT ["entrypoint.sh"]
CMD ["launch"]