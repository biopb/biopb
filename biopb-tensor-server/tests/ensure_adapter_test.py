"""``Reconciler.ensure_adapter``: every state of a source, with and without a
client's consent. Real zarr sources against a real catalog."""

import pytest
from biopb_tensor_server.core.errors import (
    SourceRegistrationError,
    SourceUnresolvedError,
)

from tests.deferred_registration_test import (
    _first_scan,
    _make_zarr,
    _manager,
    _only_ids,
)

zarr = pytest.importorskip("zarr")


class _Mirrors:
    """Stands in for a ``MirrorSet`` that owns one source."""

    def __init__(self, error=None):
        self.error = error
        self.built = []

    def owns(self, source_id):
        return source_id == "m1"

    def materialize(self, source_id):
        if self.error:
            raise self.error
        self.built.append(source_id)


def _pending(tmp_path):
    path = _make_zarr(tmp_path, "a.zarr")
    manager, server = _manager(tmp_path)
    _first_scan(manager)
    (sid,) = _only_ids(server)
    return manager._reconciler, server, sid, path


def _attempts(reconciler):
    calls = []
    real = reconciler._register_source_claim
    reconciler._register_source_claim = lambda *a, **k: (
        calls.append(1),
        real(*a, **k),
    )[1]
    return calls


@pytest.mark.parametrize("consent", [True, False])
class TestEveryConsent:
    def test_unknown_returns_and_builds_nothing(self, tmp_path, consent):
        reconciler, server, _, _ = _pending(tmp_path)
        reconciler.ensure_adapter("nope", consent=consent)
        assert server.sources.get("nope") is None

    def test_a_source_that_lost_its_adapter_is_rebuilt(self, tmp_path, consent):
        reconciler, server, sid, _ = _pending(tmp_path)
        reconciler.ensure_registered(sid)
        server.unregister_source(sid)
        reconciler.ensure_adapter(sid, consent=consent)
        assert server.sources.get(sid) is not None

    def test_a_restored_source_is_registered(self, tmp_path, consent):
        reconciler, server, sid, _ = _pending(tmp_path)
        reconciler._restored.add(sid)
        reconciler.ensure_adapter(sid, consent=consent)
        assert server.sources.get(sid) is not None

    def test_a_mirror_is_built_from_its_row(self, tmp_path, consent):
        reconciler, _, _, _ = _pending(tmp_path)
        mirrors = _Mirrors()
        reconciler._mirrors = {"u": mirrors}
        reconciler.ensure_adapter("m1", consent=consent)
        assert mirrors.built == ["m1"]

    def test_an_unresolved_mirror_raises(self, tmp_path, consent):
        reconciler, _, _, _ = _pending(tmp_path)
        reconciler._mirrors = {"u": _Mirrors(SourceUnresolvedError("no"))}
        with pytest.raises(SourceUnresolvedError):
            reconciler.ensure_adapter("m1", consent=consent)


class TestNeedsConsent:
    def test_a_pending_source_is_registered_with_consent(self, tmp_path):
        reconciler, server, sid, _ = _pending(tmp_path)
        reconciler.ensure_adapter(sid, consent=True)
        assert server.sources.get(sid) is not None

    def test_a_pending_source_raises_without_it_and_is_not_attempted(self, tmp_path):
        reconciler, server, sid, _ = _pending(tmp_path)
        attempts = _attempts(reconciler)
        with pytest.raises(SourceUnresolvedError):
            reconciler.ensure_adapter(sid, consent=False)
        assert attempts == [] and server.sources.get(sid) is None

    def test_a_cloud_recall_names_the_download(self, tmp_path):
        reconciler, _, sid, _ = _pending(tmp_path)
        reconciler._recall.add(sid)
        with pytest.raises(SourceUnresolvedError, match="download"):
            reconciler.ensure_adapter(sid, consent=False)

    def test_a_failed_source_raises_and_is_not_retried_without_it(self, tmp_path):
        reconciler, _, sid, path = _pending(tmp_path)
        reconciler._pending_failed[sid] = "boom"
        attempts = _attempts(reconciler)
        with pytest.raises(SourceRegistrationError):
            reconciler.ensure_adapter(sid, consent=False)
        assert attempts == []

    def test_a_failing_registration_is_attempted_once(self, tmp_path):
        reconciler, _, sid, _ = _pending(tmp_path)
        reconciler._restored.add(sid)
        reconciler._register_source_claim = lambda *a, **k: False
        calls = []
        real = reconciler.ensure_registered
        reconciler.ensure_registered = lambda s: (calls.append(s), real(s))[1]
        with pytest.raises(SourceRegistrationError):
            reconciler.ensure_adapter(sid, consent=True)
        assert calls == [sid]
