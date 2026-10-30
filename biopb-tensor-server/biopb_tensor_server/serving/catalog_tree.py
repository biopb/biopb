"""One level of the catalog tree: what the ``catalog_children`` action answers.

Read from ``source_tree`` (see ``tree_paths`` for the spelling of ``dir``), on the
catalog's own cursor: the folders directly under a folder, and the sources
sitting in it.
"""

from types import MappingProxyType
from typing import Any, Dict, List, Mapping

from biopb_tensor_server.serving.tree_paths import SEP, dec, enc, key_dir

DEFAULT_LIMIT = 2000
# What one level may ask for; a larger limit is this.
MAX_LIMIT = 100_000


def _chain(lo: str, hi: str) -> List[str]:
    """The folder chain a group of rows shares, from the least and greatest folder
    among them (the common prefix of a set of strings is that of its min and max).

    A source sitting in a folder of the chain is a row whose folder is that short,
    so the chain stops there.
    """
    chain = []
    for x, y in zip(lo.split(SEP), hi.split(SEP), strict=False):
        if x != y:
            break
        chain.append(x)
    return chain


def children(
    cursor: Any,
    key: str,
    limit: int = DEFAULT_LIMIT,
    dropped: Mapping[str, str] = MappingProxyType({}),
) -> Dict[str, Any]:
    """The direct children of folder *key*: ``{key, folders: [{name, key}],
    sources: [{name, source_id}], truncated}``.

    A chain of single-child folders is one folder, ending where a source sits or
    the chain forks. Folders come first and share *limit* with the sources; both
    are in byte order of name. A source with no path sits at the top, named by its
    id; a resolved source holding no tensor is not listed. A *limit* above
    ``MAX_LIMIT`` is that.

    *dropped* maps the top-level folders that hold a drag-dropped (``dnd://``)
    root to that root's url. At the top only, a folder starting in one carries
    ``"dropped": true`` and ``"root_url"``, the url to remove it by; nothing else
    carries either, so a level without one is unchanged.
    """
    limit = min(limit, MAX_LIMIT)
    here = key_dir(key)
    if key:
        # The folder itself, and everything below it: a range on ``dir``.
        near = "(dir = ? OR (dir >= ? || chr(1) AND dir < ? || chr(2)))"
        near_params = [here, here, here]
        rest = "CASE WHEN dir = ? THEN '' ELSE substr(dir, ?) END"
        rest_params = [here, len(here) + 2]
    else:
        near, near_params = "TRUE", []
        rest, rest_params = "dir", []
    # One pass answers both halves: a row per first segment that holds a folder,
    # and a grand-total row counting the sources sitting directly in this folder,
    # so the second pass (the sources) runs only where there are any. The total
    # sorts first so a limit cannot cut it off. The distinct folders are found
    # before anything is split, so the work follows the folders, not the sources.
    rows = cursor.execute(
        f"WITH d AS (SELECT dir, count(*) AS n FROM source_tree"
        f" WHERE visible AND {near} GROUP BY dir),"
        f" r AS (SELECT {rest} AS rest, n FROM d)"
        " SELECT seg, min(rest) FILTER (WHERE rest <> ''),"
        " max(rest) FILTER (WHERE rest <> ''),"
        " sum(n) FILTER (WHERE rest = ''), grouping(seg)"
        " FROM (SELECT split_part(rest, chr(1), 1) AS seg, rest, n FROM r)"
        " GROUP BY GROUPING SETS ((seg), ())"
        " HAVING grouping(seg) = 1"
        " OR count(*) FILTER (WHERE rest <> '') > 0"
        " ORDER BY 5 DESC, seg LIMIT ?",
        near_params + rest_params + [limit + 2],
    ).fetchall()
    direct = sum(row[3] or 0 for row in rows if row[4])
    folder_rows = [row for row in rows if not row[4]]
    truncated = len(folder_rows) > limit
    folders = []
    for seg, lo, hi, _, _ in folder_rows[:limit]:
        name = "/".join(_chain(lo, hi))
        folder = {"name": dec(name), "key": dec(f"{enc(key)}/{name}" if key else name)}
        if not key and (root_url := dropped.get(dec(seg))):
            folder["dropped"] = True
            folder["root_url"] = root_url
        folders.append(folder)
    folders.sort(key=lambda f: f["name"].encode("utf-8"))
    sources: List[Dict[str, str]] = []
    if not truncated and direct:
        budget = limit - len(folders)
        listed = cursor.execute(
            "SELECT CASE WHEN name = '' THEN source_id ELSE name END AS n, source_id"
            " FROM source_tree WHERE visible AND dir = ?"
            " ORDER BY n, source_id LIMIT ?",
            [here, budget + 1],
        ).fetchall()
        truncated = len(listed) > budget
        sources = [
            {"name": dec(name), "source_id": source_id}
            for name, source_id in listed[:budget]
        ]
    return {
        "key": key,
        "folders": folders,
        "sources": sources,
        "truncated": truncated,
    }
