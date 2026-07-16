"""Built-in kernel-namespace plugins shipped with biopb-mcp (#92).

Each module here is registered as a ``biopb_mcp.namespace`` entry point in
``pyproject.toml`` and loaded into the agent kernel's namespace at bootstrap, the
same path a lab's own plugin would take. They double as **reference examples** of
the "bring your own tool" surface: a plain module exporting a few callables, no
biopb-internal imports, discoverable by the agent via ``dir()`` / ``inspect_object``.
"""
