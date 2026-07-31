"""biopb-mcp kernel plugins — extend the agent's Python namespace (#92).

Every ``*.py`` file in this directory is imported at kernel start and bound in the
agent's namespace **under its own name** — ``rolling_ball.py`` becomes
``rolling_ball``, and the agent calls ``rolling_ball.subtract_background(...)``.
This is the low-friction "bring your own tool" path: drop a file, no packaging
required. (A lab can also distribute plugins as a ``biopb_mcp.namespace``
entry-point package; see biopb-mcp's ARCHITECTURE.md.)

One file, one name (#664). Your helpers, constants and imports stay on the module
where they belong; only the module itself joins ``viewer`` / ``client`` / ``np`` /
``da`` / ``ops`` in the namespace, so a plugin cannot collide with a built-in
handle, with the agent's own variables, or with another plugin.

Conventions:

- A file whose name starts with ``_`` (like this ``__init__.py``) is **skipped**
  by the loader — use it for notes/helpers, not agent-visible tools.
- Name the file what you want the agent to type. A file that would shadow a
  built-in handle (``viewer.py``, ``client.py``, ``np.py``, ``da.py``, ``ops.py``)
  is skipped with a warning; nothing else is reserved. The stem is also the token
  a skill's ``plugin:<name>`` requirement names, and what ``server_status``
  reports.
- ``__all__`` is worth declaring: it is what ``from <plugin> import *`` and
  tooling read.
- The module docstring's first line is the summary shown in the control
  dashboard's kernel-plugin panel; ``inspect_object("<plugin>")`` shows the agent
  the whole docstring plus every public callable's signature.
- Plugin functions are registered for by-value pickling, so they work inside
  ``da.map_blocks`` on a dask worker that has never seen this directory.
- To reach the *live* handles from inside a plugin function, read them at call
  time -- ``client`` in particular is re-bound per job::

      def measure(source_id):
          from IPython import get_ipython
          client = get_ipython().user_ns["client"]

  (A small accessor API is planned to replace this; see #664.)

Two plugins ship here. ``rolling_ball.py`` is the worked example — a fast ImageJ
port of rolling-ball background subtraction — and ``segmentation_qc.py`` backs the
``segmentation-qc-metrics`` skill, whose body carries the call signature while the
matching itself stays here where it is unit-tested.
"""
