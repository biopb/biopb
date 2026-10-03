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


def make_manager(
    *,
    monitored_dirs=(),
    cloud_roots=(),
    monitored_aliases=None,
    monitored_upstreams=(),
    scan_once_sources=(),
    **kwargs,
):
    """A ``SourceManager`` whose roots are spelled the way a test thinks of them.

    ``monitored_dirs`` are watched directories; ``monitored_aliases`` gives some
    of them a display root; ``cloud_roots`` marks paths cloud (a monitored
    directory becomes a cloud directory, any other path a consented, dropped cloud root of its own);
    ``monitored_upstreams`` and ``scan_once_sources`` are config entries.
    """
    from biopb_tensor_server.sources.roots import Root, RootKind, Roots
    from biopb_tensor_server.sources.source_manager import SourceManager

    aliases = {p.resolve(): a for p, a in (monitored_aliases or {}).items()}
    cloud = {p.resolve() for p in cloud_roots}
    roots = Roots()
    for path in monitored_dirs:
        roots.add(
            Root(
                RootKind.MONITORED,
                str(path),
                aliases.get(path.resolve()),
                path.resolve() in cloud,
            )
        )
    monitored = {p.resolve() for p in monitored_dirs}
    for path in cloud_roots:
        if path.resolve() not in monitored:
            roots.add(Root(RootKind.DROPPED, str(path), cloud=True, label=path.name))
    for source in scan_once_sources:
        roots.add(Root.from_config(source, RootKind.SCAN_ONCE))
    for source in monitored_upstreams:
        roots.add(Root.from_config(source, RootKind.UPSTREAM))
    return SourceManager(roots=roots, **kwargs)
