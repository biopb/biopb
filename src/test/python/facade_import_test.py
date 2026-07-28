"""Package-façade imports must not depend on statement order in `__init__.py`.

A module doing `from <pkg> import X` at module scope, where `X` is a name that
`<pkg>/__init__.py` *binds* rather than a submodule of `<pkg>`, loads only while
`__init__` binds `X` before it imports that module. Swapping two statements in
`__init__` -- a plausible tidy-up -- turns it into an ImportError at import
time. `biopb/image/utils.py` sat in that state until #621, and
`biopb_control/_control.py` until #619.

`from <pkg> import <submodule>` is a different thing and is safe: importing a
submodule does not require the parent `__init__` to have finished. The scan
below tells the two apart, so `_lifecycle/owned_child.py`'s `from . import
winjob` is not a finding.

The scan covers all four workspace packages and needs nothing but the standard
library, so it has two entry points: pytest here, for anyone running the client
suite, and `python3 src/test/python/facade_import_test.py` from `lint.yaml`,
which is the only CI job that fires on a change to any of the four.
"""

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

PACKAGE_ROOTS = (
    "src/main/python/biopb",
    "biopb-control/src/biopb_control",
    "biopb-tensor-server/biopb_tensor_server",
    "biopb-mcp/src/biopb_mcp",
)

# buf-generated at build time and gitignored. They take part in name resolution,
# so that a real submodule import is not misread as a façade one, but are not
# themselves scanned -- nothing hand-edits them.
GENERATED_SUFFIXES = ("_pb2.py", "_pb2_grpc.py")


def scan(repo_root: Path) -> list[str]:
    """Findings across every package root that is present, one line each."""
    findings = []
    for relative in PACKAGE_ROOTS:
        root = repo_root / relative
        if not root.is_dir():
            continue  # a package may be absent from a partial checkout
        findings.extend(_scan_package_root(root, repo_root))
    return findings


def _scan_package_root(root: Path, repo_root: Path) -> list[str]:
    modules = _modules(root)
    edges = {name: _edges(path, name, modules) for name, path in modules.items()}

    findings = []
    for package in sorted(name for name, path in modules.items() if _is_init(path)):
        # Only modules the package's own __init__ pulls in can be caught
        # half-initialized by it; everything else imports the package after it
        # has finished, whatever order its statements are in.
        for module in sorted(_reachable(package, edges) - {package}):
            findings.extend(
                _facade_imports(modules[module], module, package, modules, repo_root)
            )
    return findings


def test_no_order_dependent_facade_imports():
    findings = scan(REPO_ROOT)
    assert not findings, "order-dependent façade imports:\n" + "\n".join(findings)


def test_image_utils_loads_against_a_half_initialized_package():
    """Reproduce #621's failure mode head-on.

    Stand a bare `biopb.image` in `sys.modules` -- a package with a `__path__`
    and no attributes, which is what the real one looks like partway through its
    own `__init__` -- then import `utils` against it. A module-scope read off the
    façade fails here and nowhere else: every other test imports `biopb.image`
    first, by which point the names are bound and the ordering bug is invisible.
    """
    script = textwrap.dedent("""
        import importlib
        import importlib.util
        import sys
        import types

        # find_spec imports the parent, `biopb`, but not `biopb.image` itself.
        spec = importlib.util.find_spec("biopb.image")
        stub = types.ModuleType("biopb.image")
        stub.__path__ = list(spec.submodule_search_locations)
        stub.__spec__ = spec
        sys.modules["biopb.image"] = stub

        importlib.import_module("biopb.image.utils")
        print("OK")
    """)

    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, (
        "biopb.image.utils reads a name off its own package at module scope, so "
        "it only imports after biopb/image/__init__.py has bound that name. "
        "Import from the defining module instead.\n\n" + proc.stderr
    )
    assert proc.stdout.strip() == "OK"


def _is_init(path: Path) -> bool:
    return path.name == "__init__.py"


def _modules(root: Path) -> dict[str, Path]:
    base = root.parent
    out = {}
    for path in sorted(root.rglob("*.py")):
        parts = list(path.relative_to(base).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        out[".".join(parts)] = path
    return out


def _is_type_checking(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = node.test
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _module_scope_imports(path: Path):
    """Yield the import statements that run when `path` is loaded.

    Imports inside a def or class body are deferred -- by the time they run both
    modules are fully loaded -- and a `TYPE_CHECKING` body never runs at all.
    Counting either reports load-order coupling that does not exist; on
    `biopb_tensor_server/core/` alone, counting TYPE_CHECKING blocks invents 27
    cycles.
    """
    stack = list(ast.parse(path.read_text(encoding="utf-8")).body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        if _is_type_checking(node):
            stack.extend(node.orelse)
            continue
        if isinstance(node, ast.Import | ast.ImportFrom):
            yield node
        else:
            stack.extend(ast.iter_child_nodes(node))


def _absolute(node: ast.ImportFrom, module: str, is_init: bool) -> str:
    """Resolve an ImportFrom's `from` target to an absolute module name."""
    if not node.level:
        return node.module or ""
    package = module if is_init else module.rpartition(".")[0]
    parts = package.split(".") if package else []
    if node.level > 1:
        parts = parts[: len(parts) - (node.level - 1)]
    if node.module:
        parts += node.module.split(".")
    return ".".join(parts)


def _edges(path: Path, module: str, modules: dict[str, Path]) -> set[str]:
    """In-package modules that loading `module` at module scope pulls in."""
    out = set()

    def add(name: str) -> None:
        # Importing a.b.c executes a, then a.b, then a.b.c -- so each prefix is
        # an edge, except an ancestor of `module` itself. Those are already in
        # sys.modules by the time `module`'s body runs: reaching a.b.d means a
        # and a.b started first. Counting them would let every submodule reach
        # the top-level package, and from there the whole tree.
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            candidate = ".".join(parts[:i])
            if candidate not in modules:
                continue
            if module == candidate or module.startswith(candidate + "."):
                continue
            out.add(candidate)

    for node in _module_scope_imports(path):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add(alias.name)
        else:
            target = _absolute(node, module, _is_init(path))
            if not target:
                continue
            add(target)
            for alias in node.names:
                add(f"{target}.{alias.name}")

    return out - {module}


def _reachable(start: str, edges: dict[str, set[str]]) -> set[str]:
    seen, stack = set(), [start]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(edges.get(node, ()))
    return seen


def _is_submodule(package: str, name: str, modules: dict[str, Path]) -> bool:
    if f"{package}.{name}" in modules:
        return True
    # Generated modules do not exist until buf runs, and the lint job never
    # builds. Honour the naming convention so the scan gives the same answer
    # against a built tree and a bare checkout.
    return name.endswith(("_pb2", "_pb2_grpc"))


def _facade_imports(
    path: Path,
    module: str,
    package: str,
    modules: dict[str, Path],
    repo_root: Path,
) -> list[str]:
    """Module-scope `from <package> import X` where X is not a submodule."""
    if path.name.endswith(GENERATED_SUFFIXES):
        return []

    where = path.relative_to(repo_root)
    out = []
    for node in _module_scope_imports(path):
        if not isinstance(node, ast.ImportFrom):
            continue
        if _absolute(node, module, _is_init(path)) != package:
            continue
        for alias in node.names:
            if not _is_submodule(package, alias.name, modules):
                out.append(
                    f"  {where}:{node.lineno}: `{alias.name}` is bound by "
                    f"{package}/__init__.py, which imports {module} -- import it "
                    f"from the module that defines it instead"
                )
    return out


if __name__ == "__main__":
    # Entry point for lint.yaml: stdlib only, no install, whole repo.
    results = scan(REPO_ROOT)
    for line in results:
        print(line)
    print(f"{len(results)} order-dependent façade import(s)")
    sys.exit(1 if results else 0)
