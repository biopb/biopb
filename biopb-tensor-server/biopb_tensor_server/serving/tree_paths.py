"""Where a source's url sits in the catalog tree.

The one place the rules live on the server. The ``source_tree`` view computes each
row's folder and name from its url with :func:`parts_sql`; the viewer's tree routes
read them back and never parse a url. :func:`tree_parts` is the same rules in
Python, the reference the SQL is tested against.

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

from typing import List

ENDPOINT_MARK = "\x02"
SEP = "\x01"


def enc(text: str) -> str:
    """*text* (a folder key) in the stored spelling: only a leading ``scheme://`` is
    an endpoint; a ``://`` later in the path is part of its segment."""
    at = text.find("://")
    if at <= 0 or "/" in text[:at]:
        return text
    return text[:at] + ENDPOINT_MARK + text[at + 3 :]


def key_dir(key: str) -> str:
    """The ``dir`` of the folder *key* ("" is the top, whose ``dir`` is ``""``)."""
    return enc(key).replace("/", SEP) if key else ""


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


def parts_sql(url: str) -> str:
    """:func:`tree_parts` as a SQL expression over the url expression *url*: a
    ``VARCHAR[]`` of the segments. The ``source_tree`` view computes a row's place
    from this, so the tree follows a root's url with nothing to keep in step; the
    two spellings of the rules are checked against each other in the tests."""
    scheme = "'^[^/]+://'"
    body = (
        f"CASE WHEN starts_with({url}, 'file:///') THEN substr({url}, 9) "
        f"WHEN starts_with({url}, 'dnd://') "
        f"THEN replace(substr({url}, 7), chr(92), '/') "
        f"WHEN regexp_matches({url}, {scheme}) THEN CASE "
        f"WHEN regexp_matches({url}, '^[^/]+://(/|$)') "
        f"THEN regexp_replace({url}, {scheme}, '') "
        f"ELSE regexp_replace({url}, '^([^/]+)://', '\\1' || chr(2)) END "
        f"ELSE {url} END"
    )
    return f"list_filter(string_split({body}, '/'), x -> x <> '')"
