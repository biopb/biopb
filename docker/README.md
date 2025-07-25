## Docker Images

The folder contains code for building the docker images of supported servers.

### Build

```
cd <build-dir>
docker buildx build --build-context base=../base -t <image-name> .
```

### Pre-built public images

  - jiyuuchc/cellpose
  - jiyuuchc/lacss
  - jiyuuchc/osilab


### Run

Requirements:
  - NVIDIA kernel driver (>=525)
  - [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)


``` sh
docker run --gpus=all -p 50051:50051 <image-name>

# running on a different port
docker run --gpus=all -p 8888:50051 <image-name>

# only accepting local RPC call
docker run --gpus=all -p 50051:50051 <image-name> --local

# debug mode
docker run --gpus=all -p 50051:50051 <image-name> --no-token --debug

# require a token to access
docker run --gpus=all -p 50051:50051 <image-name> --token
```

Note: Default transport is HTTP (no encryption). To use HTTPS, run container in "--local" mode and setup a reverse proxy server, e.g., Nginx, to forward RPC calls.

