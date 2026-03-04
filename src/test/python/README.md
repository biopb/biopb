## Functional tests for a running endpoint

A small pytest module exercises the **live** service listening on the
machine (by default `127.0.0.1:50051`).  It does *not* spin up any dummy
servers; instead it connects to whatever implementation is currently
responding, issuing a couple of requests and optionally checking the
standard gRPC health service.

### how to run

```sh
pytest tests/python/grpc_services_test.py
```

### configuration

- `SERVICE_SERVER` – address or URL of the combined
  ProcessImage/ObjectDetection endpoint.  Examples:

  ```sh
  127.0.0.1:50051             # insecure local port
  http://proxy.example.com     # equivalent to above
  https://proxy.example.com    # TLS-protected via nginx
  https://proxy.example.com:443
  ```

  If the scheme is `https` the test fixture will create a **secure channel**
  using `grpc.secure_channel`.  By default the system's root certificate store
  is used to verify the proxy certificate; to supply a custom CA bundle set
  `SSL_CA_CERT` to the path of a PEM file containing one or more root certs.

- (Legacy) `PROCESS_SERVER`/`DETECTION_SERVER` are still recognised but
  unnecessary when the service is co‑hosted.
- Tests automatically skip unreachable hosts or unimplemented RPCs; this
  behaviour keeps the suite safe for CI and intermittent service availability.

### what is covered

1.  **Connectivity** – a channel-ready check ensures the target host is
    reachable.
2.  **Health check** – if the server implements the standard
    `grpc.health.v1.Health` service and the `grpcio-health-checking`
    package is installed, the test will validate that `SERVING` is
    returned.
3.  **ProcessImage.Run** – send a tiny dummy image, expect a
    `ProcessResponse` with a serialisable payload.
4.  **ObjectDetection.RunDetection** – similar lightweight call.
5.  **ObjectDetection.RunDetectionStream** – exercise the streaming API.

The module serves as a foundation; add further functional and regression
checks as the service evolves.
