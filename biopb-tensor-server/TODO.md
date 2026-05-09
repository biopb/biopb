# TODO - biopb/tensor-server Roadmap

## Version History

| Version | Status | Key Milestones |
|---------|--------|----------------|
| v0.1 | Complete | Core adapter architecture, Flight server, basic caching |
| v0.2 | Complete | Multi-scale pyramid routing, virtual scaling, HTTP sidecar |
| v0.3 | Planning | HCS plate support, metadata database, Sieve-K cache |

---

## Remaining Feature: OME-Zarr HCS Plate Support

### Overview

High-Content Screening (HCS) plate datasets are stored as OME-Zarr NGFF with a hierarchical structure:
- Plate level: metadata describing wells and fields
- Well level: groups of field images per well
- Field level: actual multiscale image data

Current `OmeZarrAdapter` only handles single multiscale arrays. HCS support requires:
1. Parsing HCS plate/well metadata schema
2. Hierarchical tensor enumeration (plate → wells → fields)
3. Plate-wide metadata queries (well names, field positions)
4. Remote storage support (S3, HTTP) via fsspec

### Prerequisite: Remote Storage Abstraction

**Problem**: Both HCS datasets and aicsimageio vendor formats increasingly use remote storage (S3, HTTP). Currently, all adapters assume local filesystem paths.

**Solution**: Design a `RemoteStore` abstraction layer that:
- Wraps fsspec for S3, HTTP, GCS, Azure storage
- Provides unified path handling for local and remote
- Integrates with zarr, tifffile, aicsimageio backends
- Supports credential injection (AWS profiles, signed URLs)

**Integration points**:
- `ZarrAdapter` / `OmeZarrAdapter`: Use fsspec-compatible zarr stores
- `AicsImageIoAdapter`: Pass fsspec URL to aicsimageio's remote reader
- Discovery: Extend claim protocol to recognize remote URLs
- Config: New `remote_store` section for credentials, cache settings

### HCS Implementation Scope

| Component | Description |
|-----------|-------------|
| `OmeZarrHcsAdapter` | New adapter for HCS plate datasets |
| Plate parsing | Parse plate/well metadata from `.zattrs` |
| Tensor hierarchy | Enumerate wells as sub-sources, fields as tensors |
| Navigation API | HTTP endpoints for plate/well/field traversal |
| Query integration | DuckDB metadata queries for HCS metadata |

### Key Files

| File | Changes |
|------|---------|
| `adapters/ome_zarr.py` | Add HCS parsing, well/field enumeration |
| `adapters/zarr.py` | Support fsspec stores |
| `discovery.py` | Claim protocol for remote URLs |
| `config.py` | Remote store configuration |
| `http_server.py` | HCS navigation endpoints |

---

## New Feature 1: DuckDB Metadata Database

### Overview

**Core Purpose**: Source filtering mechanism for catalogs with **>100k sources**.

As deployments scale to large microscopy facilities (thousands of plates, petabytes of data), the in-memory source catalog becomes a bottleneck. DuckDB provides:
- Indexed source filtering by metadata attributes
- Efficient source enumeration at scale (avoid O(n) scans for 100k+ sources)
- Source-centric query protocol (return filtered source_ids, not tensor details)

**Not intended for**: General OME metadata exploration, biological queries, FTS for arbitrary metadata fields. Focus is source selection efficiency.

### Use-Case

**Scenario: Large Facility Source Selection**
- Facility has 10,000 plates, 500TB of microscopy data
- User wants to find "plates acquired in Q1 2024 with >3 channels"
- Server enumerates sources via ListFlights → 10k entries (slow)
- With DuckDB: indexed query returns 50 matching source_ids → filtered ListFlights

**Scenario: HCS Plate Filtering**
- 100 HCS plates, each with 384 wells × 10 fields = 3840 tensors per plate
- User wants "plates with DAPI channel in row A wells"
- Direct enumeration would return 384,000 tensor entries
- DuckDB filters plates first → 5 matching plates → enumerate only those

### Feature Set

| Feature | Description |
|---------|-------------|
| In-memory DuckDB | No persistence file, faster startup, cleared on server restart |
| Source metadata table | Hybrid schema: typed columns + full JSON (see below) |
| GetFlightInfo cmd interface | New FlightDescriptor.cmd for metadata queries (not HTTP endpoint) |
| Auto-sync | Update database on source add/remove events |

**Hybrid Schema Design** (balance flexibility vs performance):

```sql
CREATE TABLE sources (
    source_id TEXT PRIMARY KEY,
    source_url TEXT,
    source_type TEXT,           -- Typed, indexed (ome-zarr, zarr, ome-tiff, etc.)
    dtype TEXT,                 -- Typed, indexed (uint8, uint16, float32, etc.)
    indexed_at TIMESTAMP,
    
    -- Full metadata as JSON (all fields accessible via DuckDB JSON operators)
    metadata_json TEXT,         -- Complete .zattrs / OME-XML / vendor metadata
    
    -- Optional: shape summary for quick size estimates
    shape_summary TEXT          -- JSON array or "N/A"
);

-- Indexed columns for predictable/common filters
CREATE INDEX idx_source_type ON sources(source_type);
CREATE INDEX idx_dtype ON sources(dtype);
```

**Query patterns**:

| Query Type | Syntax | Performance |
|------------|--------|-------------|
| Indexed filter | `WHERE source_type = 'ome-zarr'` | Fast (B-tree scan) |
| JSON field filter | `WHERE metadata_json->>'n_channels' = '3'` | Fast enough (in-memory, no disk I/O) |
| Combined | `WHERE source_type = 'ome-zarr' AND metadata_json->>'plate_format' LIKE '384%'` | Indexed first reduces JSON scan scope |

**Note**: For in-memory DuckDB, JSON queries are reasonably fast (microseconds per row). Typed columns still useful for very large catalogs (100k+) where JSON scan adds measurable latency, but not critical for smaller deployments.

**Query Protocol Design** (Snowflake-style SQL interface):

```
# FlightDescriptor.cmd contains full SQL query
FlightDescriptor.cmd = b"metadata:SELECT source_id FROM sources WHERE source_type='ome-zarr' AND metadata_json->>'n_channels'>'2'"

# GetFlightInfo returns FlightInfo with schema matching query result
# DoGet(ticket) returns RecordBatch with full query results

FlightInfo:
  schema: {"source_id": "string"}
  endpoints: [{ticket: "query-result-batch-0", ...}]
  
DoGet(ticket) → RecordBatch:
  columns: [["plate-001", "plate-042", "plate-089", ...]]
  total_rows: 50
```

**Security model** (read-only, restricted scope):
- Only `sources` table accessible (other tables blocked)
- Forbidden keywords: INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, EXECUTE
- No subqueries referencing external tables
- Query timeout enforced (prevent runaway queries)

**Client usage**:
```python
# Python client
results = client.query_metadata("SELECT source_id, source_url FROM sources WHERE dtype='uint16'")
for row in results:
    print(row['source_id'])
```

### Implementation Concept

**Architecture**:
```
SourceManager.on_source_added()
    → DuckDB.insert_source(source_id, source_type, dtype, metadata_json)

SourceManager.on_source_removed()
    → DuckDB.delete_source(source_id)

Flight GetFlightInfo(cmd)
    if cmd.startswith("metadata:"):
        → DuckDB.execute_safe_query(sql)
        → Returns FlightInfo with result schema
    else:
        → Existing tensor descriptor flow

Flight DoGet(ticket)
    → Returns RecordBatch with query results
```

**Safe Query Execution**:
```python
FORBIDDEN_KEYWORDS = {'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER', 'TRUNCATE', 'EXECUTE'}
ALLOWED_TABLES = {'sources'}

def execute_safe_query(sql: str) -> pa.RecordBatch:
    # Validate no forbidden keywords
    # Validate table references only sources
    # Execute with timeout
    # Return as Arrow RecordBatch
```

**Scalability Design**:
- In-memory DuckDB: zero startup latency, cleared on restart (re-sync from SourceManager state)
- Full metadata_json stored: no extraction logic needed, clients query via JSON operators
- Index strategy: B-tree on source_type and dtype (only typed columns)

**Integration Points**:
- `source_manager.py`: Hook into source lifecycle events, initial sync
- `server.py`: Handle "metadata:" cmd in GetFlightInfo, safe query execution
- `adapters/*.py`: Each adapter provides metadata_json via get_metadata()

---

## New Feature 2: Sieve-K Caching Algorithm

### Overview

Sieve-K is a scan-resistant caching algorithm designed for **file-based persistent cache**:
- Sequential scans (don't pollute cache)
- Random access (favor frequently accessed items)
- Mixed patterns (balance recency vs frequency)

**Scope clarification**:
- **MemoryCacheBackend**: **Keep current LRU** - primary purpose is temporary storage of computed virtual chunks. Dominant eviction factor is **size**, not access pattern. Computation cost (GPU/CPU downsampling) outweighs access frequency concerns.
- **ArrowFileBackend**: **Apply Sieve-K** - persistent storage in a more efficient format (arrow) than raw data benefits from scan resistance for microscopy plate navigation workflows.

Current file cache implementation:
- **ArrowFileBackend**: Segment-level LRU (evicts oldest segment by last_access_time)
- **Pooling**: Entries grouped by `(schema_key, size_class)` into separate segment files

Sieve-K offers better hit rates for microscopy workflows where users:
1. Scan through entire plate sequentially (low value to cache)
2. Return to specific fields repeatedly (high value to cache)

### Use-Case

**Scenario: Plate Navigation Workflow**
- User loads 384-well plate (384 wells × 10 fields = 3840 positions)
- Sequential scan: View each well once (3840 chunks read)
- Cache should NOT retain scan data (pollutes cache)
- User zooms into well A01, field 3, repeatedly pans
- Cache SHOULD retain frequently accessed field A01-3 chunks

**Current LRU behavior**: Sequential scan fills cache with least-recently-viewed wells. When user returns to A01-3, cache has evicted it.

**Sieve-K behavior**: Scan-resistant. Marks items as "visited" during scan. Only promotes frequently-visited items to cache. A01-3 retained because user accessed repeatedly.

### Feature Set

| Feature | Description |
|---------|-------------|
| Per-pool Sieve-K queues | Each `(schema, size_class)` pool maintains its own Sieve-K queue with hand pointer |
| Pool-level frequency tracking | Track `visited` and `frequency` per segment within each pool |
| Pool selection for eviction | Select pool with lowest aggregate hit rate when `max_total_bytes` exceeded |
| K-factor tuning | Adjustable K parameter per pool (or global default) |
| Reference counting | Preserve ref_count safety for concurrent reads |
| Config option | `cache.backend = "sieve-k-file"` to enable |

**Note**: MemoryCacheBackend remains LRU with size-aware optimization. No Sieve-K implementation needed for memory cache - virtual chunk computation cost justifies size-based eviction.

### Implementation Concept

**Per-Pool Sieve-K Architecture** (for ArrowFileBackend):

The file cache pools entries by `(schema_key, size_class)` to maintain segment homogeneity. Sieve-K must preserve this pooling:

```
PoolRegistry:
  pool_key → PoolEntry
    - segment_id: current active segment
    - sieve_k_queue: OrderedDict[segment_id → SieveKSegmentInfo]
    - hand: current hand pointer for this pool
    - k_factor: per-pool tuning (optional, default 2)
```

Each pool maintains its own:
- `visited` and `frequency` tracking for segments in that pool
- Independent hand pointer sweeping within pool segments
- Per-pool eviction decisions

**Eviction Strategy** (when `max_total_bytes` exceeded):
1. Select pool with lowest aggregate hit rate (or oldest average `last_access_time`)
2. Run Sieve-K sweep within that pool's queue
3. Evict low-frequency segment from selected pool

This maintains:
- Schema homogeneity (segments contain same dtype entries)
- Size class organization (tiny/small/medium pools separate)
- Pool-level Sieve-K characteristics (scan resistance per pool)

**Sieve-K Segment Sweep**:
```python
class SieveKArrowFileBackend(ArrowFileBackend):
    def _evict_segment_sieve_k(self) -> bool:
        """Per-pool Sieve-K sweep."""
        # 1. Select target pool (lowest hit rate)
        target_pool = self._select_pool_for_eviction()
        
        # 2. Sweep within pool's queue
        pool_queue = self._pool_queues[target_pool]
        hand = pool_queue.hand
        
        segment = pool_queue.entries[hand]
        if segment.visited and self._segment_is_evictable(hand):
            segment.visited = False
            pool_queue.advance_hand()
        elif not segment.visited and self._segment_is_evictable(hand):
            self._do_evict_segment(hand)
            pool_queue.advance_hand()
        else:
            pool_queue.advance_hand()
```

**Key Design Decisions**:
1. **Per-pool queues**: Each pool has independent Sieve-K state (hand, visited, frequency)
2. **Pool selection**: Target pool by aggregate metrics, not global sweep
3. **Segment-level**: Apply Sieve-K to segment selection, not individual entries
4. **K-factor default 2**: Good balance for microscopy workloads
5. **Memory cache unchanged**: Keep LRU + size-aware for virtual chunk storage

**Mmap Handle Lifecycle** (for large file caches):

When file cache grows to many segment files, keeping all mmap handles open consumes memory. Sieve-K frequency tracking enables smart mmap management:

| Frequency State | Mmap Action |
|-----------------|-------------|
| Hot (`frequency >= K`) | Keep mmap open |
| Cold (`frequency < 1` for extended period) | Release mmap handle |
| Access to cold segment | Re-open mmap on-demand, increment frequency |

```python
def _maybe_release_cold_mmaps(self):
    """Release mmap handles for cold segments (configurable threshold)."""
    for pool_key, pool in self._pool_queues.items():
        for seg_id, seg_info in pool.entries.items():
            if seg_info.frequency < 1 and seg_info.last_access_age > COLD_THRESHOLD:
                mmap = self._segment_mmaps.pop(seg_id, None)
                if mmap:
                    mmap.close()
                seg_info.mmap_released = True

def _read_batch_from_segment(self, key: bytes) -> Optional[pa.RecordBatch]:
    """Read batch, reopening mmap if cold segment was released."""
    entry_info = self._metadata.get(key)
    seg_info = self._pool_queues[...].entries[entry_info.segment_id]
    
    if seg_info.mmap_released:
        # Re-open mmap for cold segment
        mmap = pa.memory_map(str(seg_info.path), 'r')
        self._segment_mmaps[entry_info.segment_id] = mmap
        seg_info.mmap_released = False
        seg_info.frequency += 1  # Promote to warmer
    
    # ... rest of read logic
```

Trade-off: mmap overhead accumulates with many segments; reopen cost is one-time per cold access.

**Double-Buffered Segment Rotation** (thread efficiency optimization):

Currently when a segment reaches `max_segment_bytes`, the write flow blocks to:
1. Close the segment writer (flush to disk)
2. Open mmap for reading
3. Create a new segment for continued writes

This blocks all concurrent writers during the rotation. Double-buffered rotation would pre-create the next segment and overlap these operations:

| State | Segment Role |
|-------|--------------|
| Active | Currently being written to |
| Next | Pre-opened writer, ready to swap |
| Closing | Background close/flush (old active) |

```python
def _rotate_segment_double_buffered(self, current_seg_id: int) -> int:
    """Swap to pre-created next segment, close old in background."""
    # 1. Immediate swap (lock only guards pointer swap)
    next_seg_id = self._next_segment_id
    self._active_segment = next_seg_id
    self._next_segment_id = self._create_pre_opened_segment()
    
    # 2. Background close (no lock, async)
    threading.Thread(target=self._close_segment_background, args=[current_seg_id]).start()
    
    return next_seg_id

def _create_pre_opened_segment(self) -> int:
    """Pre-create segment with open writer, ready for immediate use."""
    seg_id = self._next_segment_id_counter
    path = self._segments_dir / f"seg_{seg_id:04d}.arrow"
    writer = pa.RecordBatchStreamWriter(pa.OSFile(str(path), 'wb'), UNIFIED_SCHEMA)
    self._pool_writers[seg_id] = writer
    self._pool_paths[seg_id] = path
    return seg_id
```

**Benefits**:
- Rotation latency reduced from ~50-100ms (blocking) to <1ms (pointer swap)
- Concurrent writers only paused for swap, not full close+create
- Better throughput under heavy write loads (multiple request threads)

**Trade-offs**:
- Extra memory for pre-opened writer (one per pool)
- Complexity in error handling (background close failures)

---

## New Feature 3: Benchmark Suite

### Overview

Benchmarks run on **HPC Singularity container**, not CI. Focus on realistic deployment conditions with NFS storage and production-scale datasets.

### Use-Case

**Scenario 1: Cache Tuning Validation**
- Measure Sieve-K vs LRU hit rates on real plate navigation workflow
- Validate K-factor tuning for mixed workloads
- Compare cold vs warm cache latency

**Scenario 2: Remote Storage Performance**
- Benchmark tensor reads from NFS-mounted microscopy data
- Measure first-read vs cached-read latency delta
- Quantify network overhead vs compute overhead

**Scenario 3: Metadata Filter Performance**
- Benchmark source filtering at scale (10k, 50k, 100k sources)
- Measure indexed vs unindexed filter query latency
- Validate batch insert performance for initial catalog load

### Feature Set

| Category | Benchmarks |
|----------|------------|
| **Baseline comparison** | aicsimageio direct access (OS page cache only) vs tensor server (with caching) |
| **Remote storage latency** | NFS tensor read (cold/warm), S3 tensor read (cold/warm), first-read vs cached-read delta |
| **Cache performance** | Sieve-K hit rate (sequential scan + revisit), LRU hit rate comparison, warm-up time measurement |
| **Source filtering** | Filter query latency (indexed), Batch insert latency (10k sources), Source enumeration vs filtered enumeration |
| **Throughput** | Multi-client concurrent reads, Chunk coalescing efficiency |
| **Adapter-specific** | Zarr, OME-TIFF, HDF5, OME-Zarr pyramid routing (on NFS data) |

**Note**: GPU benchmarks deferred - focus on storage/caching/metadata performance.

### Baseline Benchmark Design

**Purpose**: Quantify tensor server value vs "raw" access patterns.

**Baseline**: aicsimageio direct file access using OS page cache only:
- No application-level caching (no cachey, no ArrowFileBackend)
- OS manages page cache (standard Linux behavior)
- Single-threaded reads (no Flight protocol overhead)

**Comparison metrics**:

| Metric | Baseline (aicsimageio) | Tensor Server | What it measures |
|--------|------------------------|---------------|-------------------|
| First read latency | X ms | Y ms | Flight protocol + Flight server overhead |
| Second read latency (OS page cache) | X ms (same) | Y' ms (lower if Sieve-K cached) | Application cache benefit |
| Sequential scan throughput | X MB/s | Y MB/s | Chunk coalescing, prefetch benefit |
| Random access latency | X ms | Y ms | Cache hit rate impact |
| Multi-client throughput | X MB/s (single client) | Y MB/s (multiple) | Flight server concurrency benefit |

**Benchmark implementation**:
```python
def benchmark_aicsimageio_baseline(data_path: Path):
    """Baseline: aicsimageio direct access, OS page cache only."""
    from aicsimageio import AICSImage
    
    img = AICSImage(data_path)
    
    # First read (cold OS page cache)
    data = img.get_image_data("ZYX", ...)  # Read slice
    first_read_ms = measure_time()
    
    # Second read (warm OS page cache)
    data = img.get_image_data("ZYX", ...)  # Same slice
    second_read_ms = measure_time()
    
    return {"first_read_ms": first_read_ms, "second_read_ms": second_read_ms}

def benchmark_tensor_server_same_data(data_path: Path):
    """Tensor server: application cache + Flight protocol."""
    # Start server, register source
    # TensorFlightClient read same slice
    # Measure first read, second read (with cache)
    ...
```

**Expected results**:
- First read: Tensor server slightly slower (Flight overhead)
- Second read: Tensor server faster (application cache hit vs OS page cache)
- Sequential scan: Similar (OS page cache efficient for streaming)
- Random access: Tensor server much faster (Sieve-K keeps hot data)

### Implementation Concept

**HPC/Singularity Architecture**:

```
benchmarks/
├── biopb-bench.def         # Singularity definition file
├── run_benchmarks.sh       # SLURM batch script
├── benchmarks/
│   ├── conftest.py         # Server/client setup within container
│   ├── remote_storage_test.py   # NFS/S3 latency benchmarks
│   ├── cache_test.py       # Sieve-K hit rate benchmarks
│   ├── metadata_filter_test.py  # Source filtering benchmarks
│   └── adapter_test.py     # Adapter-specific on NFS data
└── results/
    └── baselines/
```

**SLURM Job Example**:
```bash
#!/bin/bash
#SBATCH --job-name=biopb-bench
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

singularity exec --bind /nfs/microscopy:/data:ro \
    --bind /scratch/$USER:/scratch \
    biopb-bench.sif \
    python -m benchmarks.run \
        --dataset nfs-hcs-384 \
        --category cache \
        --output /scratch/results/cache-bench.json
```

**Key Design Decisions**:
1. **HPC Singularity**: Batch jobs on shared HPC infrastructure, not cloud VM or CI
2. **NFS data source**: Benchmark against real microscopy storage, not synthetic local fixtures
3. **Scratch filesystem**: Temporary cache/results cleared per job
4. **No GPU benchmarks**: Focus on storage/caching/metadata performance
5. **Source filtering scale tests**: 10k, 50k, 100k synthetic catalogs for metadata benchmarks

---

## Dependencies

| Component | Current | New |
|-----------|---------|-----|
| Core | pyarrow, grpcio, tifffile, zarr, dask, cachey | + duckdb |
| Optional | ome-zarr, h5py, aicsimageio | + fsspec (s3fs, gcsfs) |
| Benchmark | pytest | + pytest-benchmark (for HPC container) |

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `biopb_tensor_server/base.py` | BackendAdapter interface |
| `biopb_tensor_server/server.py` | Flight server |
| `biopb_tensor_server/http_server.py` | HTTP sidecar |
| `biopb_tensor_server/source_manager.py` | Source lifecycle management |
| `biopb_tensor_server/cache/` | Cache backend implementations |
| `biopb_tensor_server/adapters/ome_zarr.py` | OME-Zarr adapter |
| `biopb_tensor_server/config.py` | Server configuration |
| `biopb_tensor_server/discovery.py` | Claim-based discovery |
| `tests/` | Test suite |