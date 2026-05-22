"""Smoke tests for the deriva-ml-mcp plugin entry point."""

from __future__ import annotations

from importlib import metadata

from deriva_ml_mcp.plugin import register

# Per-domain frozensets of registered tools. Each new domain phase
# (workflow, execution) adds one of these and OR-unions it
# into ``_ALL_REGISTERED_TOOLS`` below. The exact-equality check in
# ``test_all_registered_tools_exact`` then catches both:
#
#   - tools added without an entry in their per-domain frozenset, AND
#   - tools registered without an updated ``coverage.md`` row.
#
# Update protocol when adding a new domain:
#   1. Add ``_WORKFLOW_TOOLS = frozenset({...})`` (etc.) here.
#   2. Add it to the ``_ALL_REGISTERED_TOOLS`` union below.
#   3. ``test_all_registered_tools_exact`` keeps its shape; no rename.

_DATASET_TOOLS = frozenset(
    {
        # Read-only tools
        "deriva_ml_list_datasets",
        "deriva_ml_get_dataset",
        "deriva_ml_list_dataset_members",
        "deriva_ml_list_dataset_relations",
        "deriva_ml_list_dataset_element_types",
        "deriva_ml_bag_info",
        "deriva_ml_get_dataset_spec",
        # v3.3: cheap metadata-only pre-flight validators (see deriva-ml ADR-0002).
        "deriva_ml_validate_dataset_specs",
        "deriva_ml_validate_execution_configuration",
        # v3.5: AST-based whole-file validator + catalog-driven bootstrap
        # (the read-side of the config <-> catalog reconciliation; the
        # write-side stays in skill prose).
        "deriva_ml_validate_config_file",
        "deriva_ml_bootstrap_config",
        # Mutation tools
        "deriva_ml_create_dataset",
        "deriva_ml_delete_dataset",
        "deriva_ml_add_dataset_members",
        "deriva_ml_delete_dataset_members",
        # v1.2: renamed (was deriva_ml_update_dataset_types) and
        # widened to take both `dataset_types` and `description`.
        "deriva_ml_update_dataset",
        "deriva_ml_add_dataset_element_type",
        "deriva_ml_release",
        # Complex / local-FS tools
        "deriva_ml_cache_dataset",
        "deriva_ml_denormalize_dataset",
    }
)

_FEATURE_TOOLS = frozenset(
    {
        # Read-only tools
        "deriva_ml_list_features",
        "deriva_ml_get_feature",
        "deriva_ml_list_feature_values",
        # Mutation tools
        "deriva_ml_create_feature",
        "deriva_ml_delete_feature",
        "deriva_ml_add_feature_values",
    }
)

_WORKFLOW_TOOLS = frozenset(
    {
        # Read-only tools
        "deriva_ml_list_workflows",
        "deriva_ml_get_workflow",
        "deriva_ml_find_workflow_by_url",
        # Mutation tools
        "deriva_ml_create_workflow",
        "deriva_ml_update_workflow",
    }
)

_EXECUTION_TOOLS = frozenset(
    {
        # Read-only tools
        "deriva_ml_list_executions",
        "deriva_ml_get_execution",
        "deriva_ml_find_workflow_executions",
        "deriva_ml_list_execution_children",
        "deriva_ml_list_execution_parents",
        # v3.3: provenance traversal (data-flow walk; see deriva-ml ADR-0001).
        "deriva_ml_get_lineage",
        # Mutation tools
        "deriva_ml_create_execution",
        "deriva_ml_start_execution",
        "deriva_ml_commit_execution",
        # v1.2: net-new curation tool (description-only).
        "deriva_ml_update_execution",
        "deriva_ml_abort_execution",
        "deriva_ml_create_execution_dataset",
        "deriva_ml_add_nested_execution",
    }
)

# Asset domain -- 3 tools (2 read + 1 metadata-mutation). File I/O is
# intentionally NOT covered here; that lives in the deriva-skills
# `work-with-assets` skill which generates Python the user runs locally.
# Asset table discovery moved to the ``ml/assets/{schema}`` resource in
# v3.4; the ``deriva_ml_list_asset_tables`` tool was retired alongside
# the ``ml/asset-tables`` resource.
_ASSET_TOOLS = frozenset(
    {
        "deriva_ml_list_assets",
        "deriva_ml_lookup_asset",
        "deriva_ml_update_asset",
    }
)

# v1.1 vocabulary domain -- one tool: a manual RAG re-index trigger.
# It mutates the vector store but not catalog state, so mutates=False
# (audit log treats cache refreshes as reads).
_VOCABULARY_TOOLS = frozenset(
    {
        "deriva_ml_create_vocabulary",
        "deriva_ml_reindex_vocabularies",
        "deriva_ml_resync_indexes",
    }
)

# Union of all per-domain tool sets. Phase 5 registered dataset + feature
# + workflow + execution; v1.1 adds vocabulary; v1.2 adds asset.
_ALL_REGISTERED_TOOLS = (
    _DATASET_TOOLS
    | _FEATURE_TOOLS
    | _WORKFLOW_TOOLS
    | _EXECUTION_TOOLS
    | _VOCABULARY_TOOLS
    | _ASSET_TOOLS
)

# All ML-domain MCP resource URIs registered by ``resources/ml.py``.
# Mirrors the per-domain tool frozensets above so the same exact-equality
# assertion pattern catches both missing and unexpected resource
# registrations -- forcing any new resource to be added here AND to
# ``coverage.md`` in lockstep with ``resources/ml.py``.
_ML_RESOURCE_URIS = frozenset(
    {
        "deriva://catalog/{hostname}/{catalog_id}/ml/datasets",
        "deriva://catalog/{hostname}/{catalog_id}/ml/dataset/{dataset_rid}",
        "deriva://catalog/{hostname}/{catalog_id}/ml/dataset/{dataset_rid}/members",
        "deriva://catalog/{hostname}/{catalog_id}/ml/workflows",
        "deriva://catalog/{hostname}/{catalog_id}/ml/workflow/{workflow_rid}",
        "deriva://catalog/{hostname}/{catalog_id}/ml/executions",
        "deriva://catalog/{hostname}/{catalog_id}/ml/execution/{execution_rid}",
        "deriva://catalog/{hostname}/{catalog_id}/ml/features/{table_name}",
        # Asset resources. Single-asset detail by RID is unchanged;
        # ``ml/asset-tables`` was retired in v3.4 in favor of the
        # schema-scoped ``ml/assets/{schema}`` discovery surface.
        "deriva://catalog/{hostname}/{catalog_id}/ml/asset/{asset_rid}",
        "deriva://catalog/{hostname}/{catalog_id}/ml/assets/{schema}",
        "deriva://catalog/{hostname}/{catalog_id}/ml/assets/{schema}/{asset_table}",
        # v3.4 vocabulary discovery resources. Replaced the
        # ``ml/registries`` curated-four bundle with a schema-scoped
        # hierarchy that surfaces user-defined vocabularies too.
        "deriva://catalog/{hostname}/{catalog_id}/ml/vocabularies/{schema}",
        "deriva://catalog/{hostname}/{catalog_id}/ml/vocabularies/{schema}/{vocab_name}",
        # v3.3 lineage resource (mirrors deriva_ml_get_lineage tool).
        "deriva://catalog/{hostname}/{catalog_id}/ml/lineage/{rid}",
        # v3.3 dataset-spec resource (mirrors deriva_ml_get_dataset_spec tool).
        "deriva://catalog/{hostname}/{catalog_id}/ml/dataset/{dataset_rid}/spec",
        # bag-preview resource (mirrors deriva_ml_bag_info tool, current
        # version, no exclusions). Closes a documented-but-missing URI
        # flagged by the debug-bag-contents skill (2026-05-13 e2e findings).
        "deriva://catalog/{hostname}/{catalog_id}/ml/dataset/{dataset_rid}/bag-preview",
        # Static cold-start orientation resources. Same content as the
        # matching MCP prompts (``deriva_ml_getting_started`` /
        # ``deriva_ml_concepts``); exposed as resources so resource-walking
        # clients discover the orientation without going through the prompt
        # subsystem (closes Section A / C1c of the 2026-05-16 findings).
        "deriva://deriva-ml/getting-started",
        "deriva://deriva-ml/concepts",
    }
)

# Names of the four MCP prompts registered by ``prompts.py``. The
# plugin-level test below uses ``>=`` (superset) rather than equality
# so a future fully-loaded core that registers additional prompts via
# the same context wouldn't break this assertion -- the contract here
# is "the ML prompts landed", not "ONLY the ML prompts are present".
# Per-prompt name exactness is pinned in ``test_prompts.py``.
_ML_PROMPT_NAMES = frozenset(
    {
        "deriva_ml_concepts",
        "deriva_ml_getting_started",
    }
)


def test_register_runs_without_error(ctx):
    """``register(ctx)`` must succeed end-to-end."""
    register(ctx)


def test_entry_point_resolves_to_register():
    """The 'deriva-ml-mcp' entry point must resolve to our register function.

    Name matches the PyPI package name (set in pyproject.toml under
    ``[project.entry-points."deriva_mcp.plugins"]``) so the deriva-docker
    default ``DERIVA_MCP_PLUGIN_ALLOWLIST=facebase,deriva-ml-mcp`` loads
    the plugin without override. The earlier name was ``deriva-ml`` --
    see commit edd2aae for the alignment rationale.
    """
    eps = metadata.entry_points(group="deriva_mcp.plugins")
    matching = [ep for ep in eps if ep.name == "deriva-ml-mcp"]
    assert matching, "entry point 'deriva-ml-mcp' not declared"
    assert matching[0].load() is register


def test_all_registered_tools_exact(ctx, capturing_mcp):
    """The set of registered tools must equal ``_ALL_REGISTERED_TOOLS`` exactly.

    Exact-equality (rather than ``issubset`` + ``not extra``) catches
    BOTH directions in one assertion:

    - missing: a tool listed in a per-domain frozenset but not yet
      registered (regression).
    - unexpected: a tool registered without being added to its
      per-domain frozenset (forces a ``coverage.md`` and frozenset
      update on every new tool).

    When Phase 3+ adds a new domain, define ``_FEATURE_TOOLS`` (etc.)
    and OR-it into ``_ALL_REGISTERED_TOOLS`` above; this test keeps
    its shape.
    """
    register(ctx)
    actual = frozenset(capturing_mcp.tools.keys())
    assert actual == _ALL_REGISTERED_TOOLS, (
        f"missing: {sorted(_ALL_REGISTERED_TOOLS - actual)}; "
        f"unexpected (update per-domain frozenset and coverage.md): "
        f"{sorted(actual - _ALL_REGISTERED_TOOLS)}"
    )


def test_all_registered_tools_have_explicit_mutates(ctx, capturing_mcp):
    """Every registered tool must go through ``ctx.tool(mutates=...)``.

    Note on what this catches vs. what core catches:

    - ``deriva-mcp-core``'s ``PluginContext.tool`` already raises
      ``TypeError`` at registration time if ``mutates=`` is omitted
      from a ``@ctx.tool(...)`` call. So the "I forgot mutates="
      bug is caught by ``test_register_runs_without_error`` (the
      whole register call would fail).
    - This test catches a different bug: a tool author who BYPASSES
      ``ctx.tool`` entirely and uses ``@some_mcp.tool()`` directly,
      which would skip core's mutates validation. Such a tool
      registers successfully but never appears in our wrapped-call
      record below, so the ``missing`` check flags it.

    Implementation: we wrap ``ctx.tool`` for the duration of
    ``register(ctx)`` and record the ``mutates`` value alongside each
    tool's name. ``PluginContext.tool`` strips ``mutates`` before
    forwarding to FastMCP, so the capturing fixture's ``tool_kwargs``
    dict never sees the value — hence the wrapper.
    """
    seen_mutates: dict[str, bool] = {}
    original_tool = ctx.tool

    def capturing_tool(*args, mutates, **kwargs):
        decorator = original_tool(*args, mutates=mutates, **kwargs)

        def wrapper(fn):
            seen_mutates[fn.__name__] = mutates
            return decorator(fn)

        return wrapper

    ctx.tool = capturing_tool  # type: ignore[method-assign]
    try:
        register(ctx)
    finally:
        ctx.tool = original_tool  # type: ignore[method-assign]

    missing = [name for name in capturing_mcp.tools if name not in seen_mutates]
    assert not missing, f"tools missing explicit mutates=: {missing}"
    assert all(isinstance(v, bool) for v in seen_mutates.values()), (
        f"mutates= must be bool; got: {seen_mutates}"
    )


def test_all_registered_resources_exact(ctx, capturing_mcp):
    """The set of registered resource URIs must equal ``_ML_RESOURCE_URIS`` exactly.

    Same drift-detector pattern as ``test_all_registered_tools_exact``:
    exact-equality catches BOTH directions in one assertion:

    - missing: a URI listed in ``_ML_RESOURCE_URIS`` but not registered
      by ``resources/ml.py`` (regression).
    - unexpected: a resource registered without being added to
      ``_ML_RESOURCE_URIS`` (forces a ``coverage.md`` and frozenset
      update on every new resource).
    """
    register(ctx)
    actual = frozenset(capturing_mcp.resources.keys())
    assert actual == _ML_RESOURCE_URIS, (
        f"missing: {sorted(_ML_RESOURCE_URIS - actual)}; "
        f"unexpected (update _ML_RESOURCE_URIS and coverage.md): "
        f"{sorted(actual - _ML_RESOURCE_URIS)}"
    )


def test_register_wires_rag_github_sources(ctx):
    """``register(ctx)`` must declare two GitHub doc sources.

    Pins the wiring of ``resources.rag.register_rag_sources`` into the
    plugin entry point so a future regression that drops one of the
    sources is caught here as well as in ``tests/test_rag.py``.

    The two sources:
      - 'deriva-ml-docs' -- the deriva-ml repo's user-guide / api-
        reference / etc. (v3.x: prefix widened from "docs/" to ""
        to also pick up the top-level README.md and CHANGELOG.md).
      - 'deriva-ml-mcp-docs' -- the deriva-ml-mcp repo's own README
        (v3.x: new source so MCP-server documentation is searchable
        via rag_search alongside the library docs).
    """
    register(ctx)
    assert len(ctx._rag_sources) == 2
    names = {decl.name for decl in ctx._rag_sources}
    assert names == {"deriva-ml-docs", "deriva-ml-mcp-docs"}


def test_register_wires_four_catalog_connect_hooks(ctx):
    """``register(ctx)`` must wire the four RAG catalog hooks.

    Three per-user row indexers (Dataset, Workflow, Execution) plus
    one catalog-public vocab indexer (added in v1.1). Hook identity
    is already covered by ``tests/test_rag.py``; this test only pins
    the count so that dropping one hook from ``register_rag_sources``
    is also caught at the plugin level.
    """
    register(ctx)
    assert len(ctx._catalog_connect_hooks) == 4


def test_register_wires_ml_prompts(ctx, capturing_mcp):
    """``register(ctx)`` must wire all ML prompts via ``prompts.register``.

    Uses ``>=`` rather than ``==`` so a future fully-loaded core that
    happens to register prompts via the same ``ctx`` wouldn't break
    this check -- the contract is "the ML prompts landed", not "ONLY
    the ML prompts are present". Per-prompt exact-set assertions live
    in ``test_prompts.py``.
    """
    register(ctx)
    assert set(capturing_mcp.prompts.keys()) >= _ML_PROMPT_NAMES


def test_register_does_not_use_rag_dataset_indexer(ctx):
    """Per-user-safety pin: the global enriched-source API must stay unused.

    ``ctx.rag_dataset_indexer`` produces a single global enriched source
    shared across all users (the enricher fires under whichever
    credential happens to connect first), which would leak per-user ACL
    rows across users. Phase 6.3 deliberately rejected this API in
    favor of per-user ``on_catalog_connect`` hooks. v1.1 vocab indexing
    likewise rejects it -- vocab chunks land via direct store writes
    under a custom ``vocab:`` prefix, not via ``rag_dataset_indexer``.
    If a future commit accidentally introduces a call to
    ``ctx.rag_dataset_indexer(...)`` anywhere in the plugin (in
    ``tools/``, ``resources/``, or any new file), this test catches it
    at the plugin level.
    """
    register(ctx)
    assert len(ctx._rag_dataset_indexers) == 0


def test_vocab_hook_writes_to_vocab_prefix_not_data_prefix(ctx):
    """Architectural carve-out pin: vocab chunks land under ``vocab:``, never ``data:``.

    The vocab hook is the fourth ``on_catalog_connect`` hook registered
    by ``register_rag_sources``. It MUST write to the vector store
    under a ``vocab:`` source prefix (catalog-public, no per-user
    partitioning) and MUST NOT route through ``index_table_data``
    (whose source naming is hardcoded to ``data:{host}:{cat}:{user_id}``
    -- which would (a) over-partition vocabularies per user, and (b)
    get caught by upstream's ``data:`` user-id filter for any caller
    whose user_id differs from the indexer's).

    The test fires the registered vocab hook with a mocked ``ml`` and
    ``store``, then asserts every chunk passed to ``store.add`` carries
    a source starting with ``vocab:``. If a future refactor pushes
    vocab indexing through ``index_table_data`` instead, the chunks
    would carry ``data:`` and this test would fail.
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    register(ctx)
    vocab_hook = ctx._catalog_connect_hooks[3]

    # Mock the vocab table the hook iterates over.
    vocab_table = MagicMock()
    vocab_table.name = "Dataset_Type"
    vocab_table.schema.name = "deriva-ml"
    term = MagicMock()
    term.name = "Training"
    term.description = "training-data partition"
    term.synonyms = []
    term.rid = "1-TRAIN"

    fake_ml = MagicMock()
    fake_ml.find_vocabularies.return_value = [vocab_table]
    fake_ml.list_vocabulary_terms.return_value = [term]

    fake_store = MagicMock()
    fake_store.delete_source = AsyncMock()
    fake_store.add = AsyncMock()

    from deriva_ml_mcp.resources import rag as rag_module

    with (
        patch.object(rag_module, "get_rag_store", return_value=fake_store),
        patch.object(rag_module, "get_ml", return_value=fake_ml),
    ):
        import asyncio

        asyncio.run(vocab_hook("h.example", "1", "hash", {}))

    # All chunks land under vocab:..., never data:... .
    add_calls = fake_store.add.await_args_list
    assert add_calls, "vocab hook should have written at least one chunk batch"
    for call in add_calls:
        chunks = call.args[0]
        for chunk in chunks:
            assert chunk.source.startswith("vocab:"), (
                f"vocab hook produced non-vocab source: {chunk.source!r}"
            )
            assert not chunk.source.startswith("data:"), (
                f"vocab hook accidentally routed through data: prefix: {chunk.source!r}"
            )


def test_data_sources_use_per_rid_naming(ctx):
    """v1.3 pin: per-user trio chunks land under ``data:{h}:{c}:{user_id}:{table}:{rid}``.

    The pre-v1.3 source-name shape was ``data:{host}:{cat}:{user_id}``
    (one source per user, all rows lumped together). v1.3's surgical
    re-index requires per-RID source naming so a single mutating tool
    can refresh just one source via ``_reindex_<entity>``. This test
    pins that source-naming shape end-to-end at the plugin level: fire
    each hook with mocked fetchers + a captured ``_write_row_chunk``,
    then assert every captured source name matches the v1.3 shape AND
    starts with the user-id prefix that upstream's filter gates on.
    """
    from unittest.mock import MagicMock, patch

    captured: list[tuple[str, str]] = []

    async def fake_write(store, source, table_name, row, serializer):
        captured.append((table_name, source))
        return 1

    from deriva_ml_mcp.resources import rag as rag_module

    user_id = "https://auth.example.org/sub/u1"
    user_prefix = f"data:h.example:1:{user_id}"

    # Patch the fetchers BEFORE register(ctx) -- the hook factory
    # captures the fetch fn at registration time, so any post-register
    # patch.object on the module-level fetch name has no effect on the
    # already-captured closure reference.
    with (
        patch.object(rag_module, "_fetch_dataset_rows", return_value=[{"rid": "1-DSAA"}]),
        patch.object(rag_module, "_fetch_workflow_rows", return_value=[{"rid": "1-WFAA"}]),
        patch.object(rag_module, "_fetch_execution_rows", return_value=[{"rid": "1-EXAA"}]),
        patch.object(rag_module, "resolve_user_identity", return_value=user_id),
        patch.object(rag_module, "get_rag_store", return_value=MagicMock()),
        patch.object(rag_module, "_write_row_chunk", side_effect=fake_write),
    ):
        register(ctx)
        dataset_hook = ctx._catalog_connect_hooks[0]
        workflow_hook = ctx._catalog_connect_hooks[1]
        execution_hook = ctx._catalog_connect_hooks[2]

        import asyncio

        asyncio.run(dataset_hook("h.example", "1", "hash", {}))
        asyncio.run(workflow_hook("h.example", "1", "hash", {}))
        asyncio.run(execution_hook("h.example", "1", "hash", {}))

    # Every source carries the user-id prefix (so upstream's user-id
    # filter still gates correctly) AND the per-RID shape.
    assert captured == [
        ("Dataset", f"{user_prefix}:dataset:1-DSAA"),
        ("Workflow", f"{user_prefix}:workflow:1-WFAA"),
        ("Execution", f"{user_prefix}:execution:1-EXAA"),
    ]
    for _, source in captured:
        assert source.startswith(user_prefix + ":"), (
            f"per-RID source must extend user-id prefix; got {source!r}"
        )
