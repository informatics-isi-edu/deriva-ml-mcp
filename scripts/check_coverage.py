"""Coverage validation script.

Walks ``../deriva-mcp/src/deriva_mcp/tools/*.py`` and the resource
module, extracts every ``@mcp.tool()`` function name and ``@mcp.resource``
URI, and confirms each appears at least once in ``docs/coverage.md``.

Usage:
    Run from the deriva-ml-mcp repo root::

        $ uv run python scripts/check_coverage.py
        Coverage OK: 98 tools, 59 resources

Exits non-zero on any missing entry. Prints a sorted list of missing
names so a human can add them.

Note on duplicates:
    The plan's original spec asked for "exactly once" per name. We
    relaxed that to "at least once" because some old tools split into
    multiple new tools (e.g. ``update_execution_status`` -> three
    state-machine transitions), and recording the split as multiple
    rows with the same ``old_name`` is the clearest disposition. The
    "exactly once" check is still doable -- pass ``--strict`` -- but
    by default we just check for completeness.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OLD_REPO = REPO_ROOT.parent / "deriva-mcp" / "src" / "deriva_mcp"
COVERAGE = REPO_ROOT / "docs" / "coverage.md"

# Old tools use ``@mcp.tool()`` immediately followed by ``async def name(``.
TOOL_PATTERN = re.compile(r"@mcp\.tool\([^)]*\)\s*\n\s*async def (\w+)\(")

# Old resources use ``@mcp.resource("deriva://...")``. Some are registered in a
# loop with an f-string ``f"deriva://docs/{suffix}"``; we collect the literal
# string ``"deriva://docs/{suffix}"`` from coverage and report the loop in
# coverage.md as a single row (matching how Phase 6.1 catalogued it).
RESOURCE_PATTERN = re.compile(r"@mcp\.resource\(\s*[fr]?\"(deriva://[^\"]+)\"")


def find_old_tools() -> set[str]:
    """Collect every ``@mcp.tool()`` async function name from the old repo."""
    tools: set[str] = set()
    for path in sorted((OLD_REPO / "tools").glob("*.py")):
        tools.update(TOOL_PATTERN.findall(path.read_text()))
    return tools


def find_old_resources() -> set[str]:
    """Collect every ``@mcp.resource(uri)`` URI from the old repo."""
    text = (OLD_REPO / "resources.py").read_text()
    return set(RESOURCE_PATTERN.findall(text))


def covered_tools() -> Counter[str]:
    """Count occurrences of each tool name in the coverage table."""
    text = COVERAGE.read_text()
    # crude row parse: leading pipe, whitespace, name, pipe, whitespace, file.py, pipe
    return Counter(re.findall(r"^\|\s*(\w+)\s*\|\s*\w+\.py\s*\|", text, re.MULTILINE))


def covered_resources() -> Counter[str]:
    """Count occurrences of each resource URI in the coverage table.

    Coverage URIs may be wrapped in backticks for readability; strip them.
    """
    text = COVERAGE.read_text()
    raw = re.findall(r"^\|\s*`?(deriva://[^\s|`]+)", text, re.MULTILINE)
    return Counter(raw)


def main() -> int:
    """Validate coverage; return 0 on success, 1 on missing rows."""
    strict = "--strict" in sys.argv

    old_tools = find_old_tools()
    old_resources = find_old_resources()
    seen_tools = covered_tools()
    seen_resources = covered_resources()

    missing_tools = old_tools - set(seen_tools)
    missing_resources = old_resources - set(seen_resources)

    ok = True
    if missing_tools:
        print(f"Missing from coverage.md ({len(missing_tools)} tools):")
        for name in sorted(missing_tools):
            print(f"  - {name}")
        ok = False
    if missing_resources:
        print(f"Missing from coverage.md ({len(missing_resources)} resources):")
        for uri in sorted(missing_resources):
            print(f"  - {uri}")
        ok = False

    if strict:
        dup_tools = {name: n for name, n in seen_tools.items() if n > 1 and name in old_tools}
        dup_resources = {
            uri: n for uri, n in seen_resources.items() if n > 1 and uri in old_resources
        }
        if dup_tools:
            print(f"Duplicate tool rows in coverage.md ({len(dup_tools)}):")
            for name in sorted(dup_tools):
                print(f"  - {name} ({dup_tools[name]}x)")
            ok = False
        if dup_resources:
            print(f"Duplicate resource rows in coverage.md ({len(dup_resources)}):")
            for uri in sorted(dup_resources):
                print(f"  - {uri} ({dup_resources[uri]}x)")
            ok = False

    if ok:
        print(f"Coverage OK: {len(old_tools)} tools, {len(old_resources)} resources")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
