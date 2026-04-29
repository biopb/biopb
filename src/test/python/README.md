## Test Structure

### Python tests

**Server-side tests** (`biopb-tensor-server/tests/`):
- `adapter_unit_test.py` - Adapter unit tests (ZarrAdapter, OmeZarrAdapter, config, compute backend)
- `adapter_integration_test.py` - Integration tests with server/client
- `multifield_test.py` - Multifield source tests (multiple tensors per source)
- `tensor_extended_test.py` - Extended adapter tests (MultiFileOmeTiff, OmeZarr, HDF5)
- `cache_test.py` - Cache module tests

**Client tests** (`src/test/python/`):
- `tensor_test.py` - TensorFlightClient tests using new multifield API

### Java tests (`src/test/java/`)

- `TensorFlightClientTest.java` - Java client tests using new multifield API
- `image/UtilsTest.java` - Image utility tests
- `image/Imglib2OrderTest.java` - Imglib2 dimension ordering tests

## Running tests

### Python

```sh
# Run all tests
pytest biopb-tensor-server/tests/ src/test/python/tensor_test.py

# Run client tests only
pytest src/test/python/tensor_test.py

# Run server-side adapter tests only
pytest biopb-tensor-server/tests/
```

### Java

```sh
mvn -B test
```

## Test coverage

### Client tests
- `list_sources()` returns DataSourceDescriptor with tensor metadata
- `get_tensor(source_id, tensor_id)` returns lazy array
- Chunk loading, caching, scaled reads
- Error handling for invalid source/tensor IDs

### Server-side tests
- Adapter descriptor generation, chunk endpoints
- ZarrAdapter, OmeZarrAdapter, Hdf5Adapter, OmeTiffAdapter
- Multifield sources with varying tensor shapes
- Compute backend selection (CPU/GPU heuristics)
- OME-Zarr precomputed pyramid levels
- Cache module internals