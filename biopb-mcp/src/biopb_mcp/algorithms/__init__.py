"""biopb-mcp's bundled algorithm-plane ops.

Every other ``*.py`` file here is a self-contained ``Ops`` server script (PEP 723
header, ``@op``-decorated functions, ``serve()``) that the installer copies into
``~/.config/biopb/algorithms/`` (``biopb-mcp-seed-algorithms``), where the control
runs it with ``uv`` like any other algorithm-plane entry -- see
``docs/algorithm-plane.md``.

These are the correctness-critical measurements a procedure doc depends on and
does not want an agent re-deriving from scratch each time: matching, splitting
and Fourier arithmetic that is short but wrong in ways invisible in the output.
Anything an agent can reliably one-shot inline stays inline; these did not.

``__init__.py`` documents the package -- it is not itself seeded, and a stem
starting with ``_`` is skipped the same way in the algorithm registry.
"""
