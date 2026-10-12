"""Where a source's url sits in the catalog tree.

The one place the rules live on the server. The catalog stores each row's folder
and name (``source_tree``), computed here when the row is written; the viewer's
tree routes read them back and never parse a url.

The rules are the browser's ``getPathParts``:

* ``dnd://rest``           -> rest, backslashes as ``/``
* ``scheme://host/path``   -> the endpoint ``scheme://host`` as one first node,
  then path; an empty host (``file:///p``) is just the path
* anything else            -> the string itself

then empty segments are dropped (``//``, leading and trailing ``/``).

A stored path is its segments joined with ``\\x01`` rather than ``/``: no segment
contains it, so byte order of the joined strings is component order (with ``/``,
``ord/a-b/q`` and ``ord/a/r`` sort against the components). The endpoint's
``://`` is ``\\x02`` so it stays one segment; :func:`dec` spells it back.
"""

from typing import List, Tuple

ENDPOINT_MARK = "\x02"
SEP = "\x01"


def enc(text: str) -> str:
    return text.replace("://", ENDPOINT_MARK)


def dec(text: str) -> str:
    return text.replace(ENDPOINT_MARK, "://").replace(SEP, "/")


def tree_parts(url: str) -> List[str]:
    """The segments of *url* in the tree, the endpoint node in its ``\\x02``
    spelling. A url with no path has none."""
    if url.startswith("file:///"):
        body = url[8:]
    elif url.startswith("dnd://"):
        body = url[6:].replace("\\", "/")
    else:
        at = url.find("://")
        if at > 0 and "/" not in url[:at]:
            rest = url[at + 3 :]
            body = (
                rest if rest.split("/", 1)[0] == "" else url[:at] + ENDPOINT_MARK + rest
            )
        else:
            body = url
    return [s for s in body.split("/") if s]


def catalog_url(root_url: str, rel: str) -> str:
    """A source's url: its root's, then its path beneath it (the ``sources`` view's
    ``source_url``)."""
    if not root_url:
        return rel
    return root_url if rel == "." else f"{root_url}/{rel}"


def tree_columns(root_url: str, rel: str) -> Tuple[str, str]:
    """``(dir, name)`` of the source at *rel* under *root_url*: its folder's
    segments joined, and its own last segment. Both ``""`` for a url with no path."""
    parts = tree_parts(catalog_url(root_url, rel))
    if not parts:
        return "", ""
    return SEP.join(parts[:-1]), parts[-1]
