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

# Copy pre-built biopb wheel (built locally with: pip wheel . --no-deps -w wheels/)
COPY wheels/biopb-*.whl /app/wheels/

# Copy tensor server source
COPY biopb-tensor-server/pyproject.toml ./tensor-server/
COPY biopb-tensor-server/biopb_tensor_server ./tensor-server/biopb_tensor_server

# Install with correct order:
# 1. First pin zarr/numcodecs for compatibility
# 2. Then install biopb + tensor server with all extras (web, ome-zarr, aics, medical)
# 3. Finally add bioformats-jar for LSM/OIF/OIB (BioFormats fallback for CZI/LIF)
WORKDIR /app/tensor-server
RUN pip install --no-cache-dir "zarr==2.15.0" "numcodecs==0.12.1" \
    && pip install --no-cache-dir /app/wheels/biopb-*.whl \
    && pip install --no-cache-dir ".[web,ome-zarr,aics,medical,ndtiff]" \
    && pip install --no-cache-dir bioformats-jar

# Stage 3: Runtime (nginx + python)
FROM python:3.11-slim
WORKDIR /app

# Install nginx, Java (for bioformats-jar), and runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    libgl1 \
    default-jdk-headless \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages
COPY --from=python-builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=python-builder /usr/local/bin /usr/local/bin

# Copy webapp build output
COPY --from=webapp-builder /app/biopb-tensor-server/packages/web/dist /var/www/html

# Copy entrypoint, config, and nginx config
COPY biopb-tensor-server/entrypoint.sh /usr/local/bin/
COPY biopb-tensor-server/nginx.conf /etc/nginx/nginx.conf
COPY biopb-tensor-server/default-config.toml /app/config/

# Set default config file (can be overridden at runtime)
ENV CONFIG_FILE=/app/config/default-config.toml
ENV JAVA_HOME=/usr/lib/jvm/default-java

# Configure scyjava to use system JDK instead of downloading via cjdk
RUN echo 'import scyjava.config; scyjava.config.set_java_constraints(fetch="never")' \
    > /usr/local/lib/python3.11/site-packages/sitecustomize.py

# Create data directory and set permissions
RUN mkdir -p /data \
    && chmod +x /usr/local/bin/entrypoint.sh

# Expose ports: nginx HTTP (8814), nginx gRPC (8815)
# Internal services (Flight 8817, sidecar 8816) bind to localhost only
EXPOSE 8814 8815

ENTRYPOINT ["entrypoint.sh"]
CMD ["launch"]