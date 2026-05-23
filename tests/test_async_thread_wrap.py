"""Structural tests: every sync deriva-ml call inside an async tool's
``with deriva_call():`` block must be wrapped in ``asyncio.to_thread``.

Without this gate, the bug PR #28 (and the ``fix/async-thread-wrap-tools``
work) addresses can be silently re-introduced. See ``CLAUDE.md`` →
"Tool Implementation Rules → asyncio.to_thread" + "Development Gotchas
→ Sync calls in async tools" for the rule and history.

The check is AST-only -- no runtime catalog and no actual deriva-ml
import is required. For every ``async def`` function defined in the
listed tool modules, this test:

1. Walks every ``with deriva_call():`` block inside the function.
2. For every ``Call`` node whose receiver is a known deriva-ml object
   (``ml``/``ds``/``exe``/``wf``/``record``/``feature``/``asset``/
   ``dataset``/``execution``/``workflow``/``parent``/``child``), or
   whose function is the imported ``get_ml`` constructor, asserts the
   call is reachable only inside a nested function that is itself the
   first argument to an ``asyncio.to_thread(...)`` call.

The "wrapped" form ``await asyncio.to_thread(ml.find_features)`` passes
the *Attribute* node ``ml.find_features`` (not a Call) into
``to_thread``; that case generates no Call node and is therefore not
flagged. The unwrapped form ``ml.find_features()`` does generate a Call
node and IS flagged unless protected by a threaded nested helper.
"""

from __future__ import annotations

import ast
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent / "src" / "deriva_ml_mcp"
TOOLS_ROOT = PLUGIN_ROOT / "tools"
RESOURCES_ROOT = PLUGIN_ROOT / "resources"

# Modules whose async tool/resource functions are required to thread-wrap
# every sync deriva-ml call inside ``with deriva_call():`` blocks.
#
# Resource handlers run on the same asyncio event loop as tool handlers
# and have the same blocking-call hazard. ``resources/rag.py`` does not
# use ``with deriva_call():`` (it manages its own retries) so it's not
# in this list; the rule only fires inside that context manager.
TOOL_MODULES: list[Path] = [
    TOOLS_ROOT / "asset.py",
    TOOLS_ROOT / "feature.py",
    TOOLS_ROOT / "workflow.py",
    TOOLS_ROOT / "maintenance.py",
    TOOLS_ROOT / "dataset" / "read.py",
    TOOLS_ROOT / "dataset" / "mutate.py",
    TOOLS_ROOT / "dataset" / "complex.py",
    TOOLS_ROOT / "execution" / "read.py",
    # NOTE: execution/mutate.py was removed in v4.0.0 -- the execution
    # lifecycle moved to user-local Python per the stateless rule.
    # See docs/audit-2026-05-23.md.
    RESOURCES_ROOT / "ml.py",
]

# Names whose method calls indicate sync deriva-ml I/O. Any
# ``<name>.<method>(...)`` Call where ``<name>`` is in this set is
# considered a sync deriva-ml callsite that must be thread-wrapped.
DERIVA_OBJECT_NAMES: frozenset[str] = frozenset(
    {
        "ml",
        "ds",
        "exe",
        "wf",
        "record",
        "feature",
        "asset",
        "dataset",
        "execution",
        "workflow",
        "parent",
        "child",
    }
)

# Bare-function names (imported directly from deriva-ml plumbing) that
# are themselves sync deriva-ml callsites.
DERIVA_FUNC_NAMES: frozenset[str] = frozenset({"get_ml"})

# Method names on deriva-ml objects that are documented pure-Python
# (no catalog I/O) and are deliberately allowed to remain unwrapped
# inside ``with deriva_call():`` blocks. Each entry must correspond to
# an in-tree comment justifying the exemption.
#
# - ``feature_record_class`` (Feature) -- class-factory, no I/O.
#   See ``tools/feature.py`` ``deriva_ml_add_feature_values`` comment.
# - ``get_chaise_url`` (Dataset/Asset/Execution) -- f-string URL
#   construction over ``host_name``/``catalog_id``/``rid``; no
#   network calls. See ``deriva_ml/dataset/dataset.py:get_chaise_url``.
PURE_PYTHON_METHODS: frozenset[str] = frozenset(
    {
        "feature_record_class",
        "get_chaise_url",
    }
)


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _is_deriva_io_call(node: ast.Call) -> bool:
    """Return True iff ``node`` is a sync deriva-ml call by allowlist.

    Args:
        node: An ``ast.Call`` node.

    Returns:
        True if the call's function is ``<name>.<method>`` with
        ``<name>`` in ``DERIVA_OBJECT_NAMES``, or a bare name in
        ``DERIVA_FUNC_NAMES``. False otherwise.
    """
    func = node.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        if func.value.id not in DERIVA_OBJECT_NAMES:
            return False
        # Documented pure-Python methods are exempt -- they don't do
        # catalog I/O so wrapping them in a thread is unnecessary.
        return func.attr not in PURE_PYTHON_METHODS
    if isinstance(func, ast.Name):
        return func.id in DERIVA_FUNC_NAMES
    return False


def _is_with_deriva_call(node: ast.With | ast.AsyncWith) -> bool:
    """Return True iff ``node`` is a ``with deriva_call():`` (or async-with) block.

    Matches both ``with deriva_call():`` and
    ``with deriva_call() as x:`` shapes; the attribute-access form
    (``mod.deriva_call()``) is also matched defensively.

    Args:
        node: A ``with``/``async with`` AST node.

    Returns:
        True if any of the context managers is a Call to
        ``deriva_call`` (Name) or ``<x>.deriva_call`` (Attribute).
    """
    for item in node.items:
        ctx = item.context_expr
        if not isinstance(ctx, ast.Call):
            continue
        called = ctx.func
        if isinstance(called, ast.Name) and called.id == "deriva_call":
            return True
        if isinstance(called, ast.Attribute) and called.attr == "deriva_call":
            return True
    return False


def _collect_threaded_nested_funcs(async_func: ast.AsyncFunctionDef) -> set[int]:
    """Return ids of nested function defs passed as the 1st arg to ``asyncio.to_thread``.

    A nested ``def`` or ``async def`` whose name appears as the first
    positional argument to ``asyncio.to_thread(<name>, ...)`` runs
    entirely inside the worker thread, so any sync deriva-ml calls
    inside its body are allowed unwrapped (the whole body executes off
    the event loop). This helper finds every such nested definition by
    matching name -> definition.

    Args:
        async_func: The outer ``async def`` whose body to scan.

    Returns:
        A set of ``id(node)`` values for FunctionDef / AsyncFunctionDef
        nodes that are reachable as ``asyncio.to_thread(<their_name>, ...)``
        somewhere in the outer function's body.
    """
    # 1. Map nested function names -> their AST nodes.
    nested_by_name: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in ast.walk(async_func):
        if node is async_func:
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nested_by_name[node.name] = node

    # 2. Find every ``asyncio.to_thread(<name>, ...)`` call and collect
    #    the matching nested definitions.
    threaded_ids: set[int] = set()
    for node in ast.walk(async_func):
        if not isinstance(node, ast.Call):
            continue
        if not _is_asyncio_to_thread(node.func):
            continue
        if not node.args:
            continue
        first_arg = node.args[0]
        if isinstance(first_arg, ast.Name) and first_arg.id in nested_by_name:
            threaded_ids.add(id(nested_by_name[first_arg.id]))
    return threaded_ids


def _is_asyncio_to_thread(func_node: ast.AST) -> bool:
    """Return True iff ``func_node`` is the attribute reference ``asyncio.to_thread``.

    Args:
        func_node: An expression that appears in ``Call.func`` position.

    Returns:
        True iff the expression is exactly ``asyncio.to_thread``.
    """
    return (
        isinstance(func_node, ast.Attribute)
        and isinstance(func_node.value, ast.Name)
        and func_node.value.id == "asyncio"
        and func_node.attr == "to_thread"
    )


def _walk_async_funcs(tree: ast.AST) -> list[ast.AsyncFunctionDef]:
    """Return every ``async def`` defined anywhere in ``tree``.

    Includes top-level definitions and ones nested inside other
    functions (e.g. tools registered inside a ``register(ctx)`` call).
    Nested ``async def``s defined *inside* an outer ``async def`` will
    be returned in their own right; that's fine -- the per-function
    check is scoped to each ``async def``'s own body and the inner
    one's ``with deriva_call():`` blocks (if any) get checked twice,
    which is harmless and covers the case where a tool is genuinely
    nested.

    Args:
        tree: A parsed AST module.

    Returns:
        List of every ``ast.AsyncFunctionDef`` in document order.
    """
    return [node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef)]


def _calls_in_subtree(root: ast.AST) -> list[ast.Call]:
    """Return every ``ast.Call`` reachable from ``root``.

    Args:
        root: Any AST node.

    Returns:
        All Call descendants in document order.
    """
    return [node for node in ast.walk(root) if isinstance(node, ast.Call)]


def _node_within(node: ast.AST, ancestor: ast.AST) -> bool:
    """Return True iff ``node`` is the same as, or a descendant of, ``ancestor``.

    Implemented by walking ``ancestor`` once and checking object
    identity; the AST is not enormous and this runs once per (call,
    nested-helper) pair.

    Args:
        node: Candidate descendant.
        ancestor: Candidate ancestor.

    Returns:
        True iff ``node`` is reachable from ``ancestor`` via ``ast.walk``.
    """
    return any(child is node for child in ast.walk(ancestor))


# ---------------------------------------------------------------------------
# Core check
# ---------------------------------------------------------------------------


def _violations_in_module(path: Path) -> list[str]:
    """Return a list of violation strings for one tool module.

    Each violation string is a one-line summary identifying the file,
    enclosing async function, line number, and the offending source
    fragment. The list is empty when the module is clean.

    Args:
        path: Absolute path to the tool module.

    Returns:
        Sorted list of violation strings (empty when clean).
    """
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    violations: list[str] = []

    for async_func in _walk_async_funcs(tree):
        # Nested defs whose body runs in the worker thread -- calls
        # inside these are exempt from the wrap rule.
        threaded_ids = _collect_threaded_nested_funcs(async_func)
        threaded_nodes: list[ast.AST] = []
        for node in ast.walk(async_func):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and id(node) in threaded_ids
            ):
                threaded_nodes.append(node)

        # Find every ``with deriva_call():`` block in this async func,
        # including ``async with`` for completeness.
        with_blocks: list[ast.With | ast.AsyncWith] = []
        for node in ast.walk(async_func):
            if isinstance(node, (ast.With, ast.AsyncWith)) and _is_with_deriva_call(node):
                with_blocks.append(node)

        for with_block in with_blocks:
            for call in _calls_in_subtree(with_block):
                # Skip the ``deriva_call()`` context-manager call itself
                # (it appears in ``with_block.items[*].context_expr``).
                if any(call is item.context_expr for item in with_block.items):
                    continue
                # Skip ``asyncio.to_thread(...)`` calls themselves.
                if _is_asyncio_to_thread(call.func):
                    continue
                # Only deriva-IO calls are subject to the rule.
                if not _is_deriva_io_call(call):
                    continue
                # Allow calls inside a nested helper that's threaded.
                if any(_node_within(call, helper) for helper in threaded_nodes):
                    continue

                snippet = ast.unparse(call)
                violations.append(
                    f"{path}:{call.lineno}: in `async def {async_func.name}`: "
                    f"unwrapped sync deriva-ml call `{snippet}` inside "
                    f"`with deriva_call():` (must be `await asyncio.to_thread(...)`)"
                )

    return sorted(violations)


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


def test_async_tools_thread_wrap_sync_deriva_calls() -> None:
    """Every sync deriva-ml call in an async tool must be thread-wrapped.

    Iterates ``TOOL_MODULES``, parses each, and asserts no module
    contains an unwrapped sync deriva-ml call inside an async tool's
    ``with deriva_call():`` block. A failure means a contributor (or
    a future refactor) has reintroduced the regression class that
    PR #28 / ``fix/async-thread-wrap-tools`` is fixing.
    """
    all_violations: list[str] = []
    for module_path in TOOL_MODULES:
        assert module_path.is_file(), f"tool module not found: {module_path}"
        all_violations.extend(_violations_in_module(module_path))

    if all_violations:
        msg = "\n".join(["Unwrapped sync deriva-ml calls found:", *all_violations])
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Self-test: prove the checker actually catches violations.
#
# This guards against the checker silently degrading -- e.g. a refactor
# that broadens an exemption such that the rule no longer fires. It
# parses an in-memory module containing an unwrapped ``ml.find_datasets()``
# inside ``with deriva_call():`` and asserts ``_violations_in_module``
# (via the per-tree helper) reports it.
# ---------------------------------------------------------------------------


_BAD_SOURCE = '''
"""Synthetic broken module to verify the checker detects regressions."""
from __future__ import annotations

import asyncio  # noqa: F401 -- referenced by the wrap pattern in good code
from deriva_mcp_core import deriva_call


async def deriva_ml_intentionally_broken(hostname: str, catalog_id: str) -> str:
    with deriva_call():
        ml.find_datasets()  # unwrapped sync call -- MUST be flagged
    return "x"
'''

_GOOD_SOURCE = '''
"""Synthetic correct module: every sync call thread-wrapped."""
from __future__ import annotations

import asyncio
from deriva_mcp_core import deriva_call


async def deriva_ml_correctly_wrapped(hostname: str, catalog_id: str) -> str:
    with deriva_call():
        ml = await asyncio.to_thread(get_ml, hostname, catalog_id)
        result = await asyncio.to_thread(ml.find_datasets)
    return str(result)


async def deriva_ml_nested_helper(hostname: str, catalog_id: str) -> str:
    with deriva_call():
        ml = await asyncio.to_thread(get_ml, hostname, catalog_id)

        def _drain() -> list:
            # Inside a threaded nested helper -- exempt.
            return list(ml.find_datasets())

        rows = await asyncio.to_thread(_drain)
    return str(rows)
'''


def _violations_in_source(source: str) -> list[str]:
    """Run the checker over an in-memory source string. Test-only helper.

    Args:
        source: Python source text.

    Returns:
        Violation strings, same shape as ``_violations_in_module``.
    """
    tree = ast.parse(source)
    violations: list[str] = []

    for async_func in _walk_async_funcs(tree):
        threaded_ids = _collect_threaded_nested_funcs(async_func)
        threaded_nodes: list[ast.AST] = []
        for node in ast.walk(async_func):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and id(node) in threaded_ids
            ):
                threaded_nodes.append(node)

        with_blocks: list[ast.With | ast.AsyncWith] = []
        for node in ast.walk(async_func):
            if isinstance(node, (ast.With, ast.AsyncWith)) and _is_with_deriva_call(node):
                with_blocks.append(node)

        for with_block in with_blocks:
            for call in _calls_in_subtree(with_block):
                if any(call is item.context_expr for item in with_block.items):
                    continue
                if _is_asyncio_to_thread(call.func):
                    continue
                if not _is_deriva_io_call(call):
                    continue
                if any(_node_within(call, helper) for helper in threaded_nodes):
                    continue
                violations.append(f"line {call.lineno}: {ast.unparse(call)}")

    return violations


def test_self_check_flags_unwrapped_call() -> None:
    """Sanity: the checker DOES flag a deliberately unwrapped sync call.

    Without this guard, a refactor could neutralize the main test
    (e.g. by accidentally exempting all calls) and the suite would
    still appear green.
    """
    violations = _violations_in_source(_BAD_SOURCE)
    assert any("ml.find_datasets()" in v for v in violations), (
        f"checker failed to flag the synthetic violation; got: {violations!r}"
    )


def test_self_check_passes_on_correctly_wrapped_code() -> None:
    """Sanity: the checker does NOT flag the canonical wrapped patterns.

    Confirms the two valid shapes -- direct ``await asyncio.to_thread(...)``
    and threaded-nested-helper -- are accepted.
    """
    violations = _violations_in_source(_GOOD_SOURCE)
    assert violations == [], f"checker false-positived on good code: {violations!r}"
