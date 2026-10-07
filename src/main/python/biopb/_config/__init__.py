"""Where biopb config and runtime files live, and how config is validated.

Private and stdlib-only, shared by the control, the tensor server and
biopb-mcp, none of which can import another. Modules are imported by path
(``biopb._config.locations``); nothing is re-exported here.
"""
