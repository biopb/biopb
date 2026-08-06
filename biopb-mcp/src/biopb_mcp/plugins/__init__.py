"""biopb-mcp kernel plugins — extend the agent's Python namespace.

Every ``*.py`` file in this directory is imported at kernel start and bound in the
agent's namespace **under its own name** — ``rolling_ball.py`` becomes
``rolling_ball``, and the agent calls ``rolling_ball.subtract_background(...)``.
This is the low-friction "bring your own tool" path: drop a file, no packaging
required. To ship plugins as an installable package instead, expose them on the
``biopb_mcp.namespace`` entry-point group: each entry point resolves to a module,
bound under the entry-point name exactly as a file is.

One file, one name. Your helpers, constants and imports stay on the module
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

  (A small accessor API is planned to replace this.)

What ships here:

===================== ==========================================================
Plugin                What it is
===================== ==========================================================
``rolling_ball``      Rolling-ball background subtraction (Sternberg 1983), the
                      fast ImageJ port. The worked "bring your own tool"
                      example — start here if you are writing one.
``segmentation_qc``   Instance-segmentation QC: IoU matching, F1 at threshold,
                      splits and merges. Backs the ``segmentation-qc-metrics``
                      skill, whose body carries the call signature while the
                      matching stays here where it is unit-tested.
``chunked_label``     Connected components on a chunked dask array, linked
                      across chunk boundaries — which per-block
                      ``scipy.ndimage.label`` silently does not do.
``image_resolution``  Resolution in physical units: Fourier ring correlation
                      for two independent images (or one localization list),
                      decorrelation analysis for a single one. Localization
                      precision is not resolution, and neither is a focus
                      metric.
===================== ==========================================================

All of them are yours to edit: the installer seeds them once and never overwrites
a file that already exists. The module docstring is the documentation, so
``inspect_object("<plugin>")`` is the fastest way to see what one offers.

Project and issue tracker: https://github.com/biopb/biopb
"""
