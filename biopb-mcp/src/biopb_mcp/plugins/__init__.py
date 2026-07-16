"""biopb-mcp kernel plugins — extend the agent's Python namespace (#92).

Every ``*.py`` file in this directory has its top-level definitions loaded into
the biopb-mcp agent kernel's namespace at startup, right beside the built-in
handles ``viewer`` / ``client`` / ``np`` / ``da`` / ``ops`` — so a function you
define here is immediately callable by the agent and shows up in its ``dir()`` /
``inspect_object``. This is the low-friction "bring your own tool" path: drop a
file, no packaging required. (A lab can also distribute plugins as a
``biopb_mcp.namespace`` entry-point package; see biopb-mcp's ARCHITECTURE.md.)

Conventions:

- A file whose name starts with ``_`` (like this ``__init__.py``) is **skipped**
  by the loader — use it for notes/helpers, not agent-visible tools.
- Don't shadow the built-in handles (``viewer``/``client``/``np``/``da``/``ops``);
  the loader restores them and warns.
- Keep the public surface small — a startup file exec's directly into the
  namespace, so anything at top level leaks in. Alias imports/helpers privately
  (``import scipy.ndimage as _ndi``) so only your intended callables appear.
- The module docstring's first line is the summary shown in the control
  dashboard's kernel-plugin panel.

See ``rolling_ball.py`` in this directory for a worked example — a fast ImageJ
port of rolling-ball background subtraction — that biopb-mcp ships as its
reference plugin.
"""
