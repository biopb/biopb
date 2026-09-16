"""Tests for biopb-tensor-server."""


def catalog_server(*args, **kwargs):
    """A ``TensorFlightServer`` with a catalog, wired the way ``cli.py`` wires one.

    ``TensorFlightServer(metadata_db=None)`` is catalog-less by design -- its
    sources are addressed by ``source_id`` and every catalog surface refuses --
    so a test that lists, queries, describes or resolves has to supply one. This
    supplies an in-memory ``MetadataDatabase``; reach it afterwards through
    ``server.metadata_db``.
    """
    from biopb_tensor_server.serving.metadata_db import MetadataDatabase
    from biopb_tensor_server.serving.server import TensorFlightServer

    kwargs.setdefault("metadata_db", MetadataDatabase())
    return TensorFlightServer(*args, **kwargs)


def register_and_catalog(server, source_id, adapter):
    """Register a source on a server *and* write its catalog row.

    ``TensorFlightServer.register_source`` is the registry alone; a real
    deployment's second step is the SourceManager reconciler's
    ``sync_source_added``. A test that builds its own server has no
    SourceManager, so it owns that step -- this is it, in the reconciler's order
    (registry first, catalog second) minus the rollback a test has no use for.

    The server must have a catalog (see :func:`catalog_server`). A test that
    only reads pixels through a ticket needs neither: plain ``register_source``
    on a catalog-less server is the embedded in-process cache's own shape.
    """
    registered = server.register_source(source_id, adapter)
    server.metadata_db.sync_source_added(source_id, registered)
    return registered
