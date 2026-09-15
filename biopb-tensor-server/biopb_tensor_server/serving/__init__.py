"""Serving runtime: the Arrow Flight server, the sidecar HTTP app, and the
upload / precache / render machinery that runs on top of the ``core`` layer,
together with what those servers own directly -- ``metadata_db`` (the DuckDB
store behind the catalog, ROI and decode-rate surfaces), ``tls`` (the Flight
listener's self-signed leaf) and ``activity`` (its in-flight read tracking).
"""
