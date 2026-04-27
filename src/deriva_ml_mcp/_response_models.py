"""Pydantic response models for deriva-ml-mcp tools and resources.

Every list / detail / summary response shape returned by an ``_*_impl``
helper is declared here as a ``pydantic.BaseModel`` so the resource
and tool wrappers share an explicit, validated contract instead of an
implicit "whatever the dict happens to look like."

## Why this module exists

The v1.0 design had each ``_<verb>_impl`` helper return ``dict[str,
Any]`` -- the resource handler in ``resources/ml.py`` and the tool
wrapper in ``tools/<domain>.py`` both consumed the same dict. This was
a deliberate "single source of truth" choice (no drift between the
two surfaces) but the contract was implicit:

- A field rename in a helper silently changed both the tool and the
  resource shape with no signal.
- Refactoring a list response shape was a "find every consumer"
  search with no model to anchor on.
- Bad data flowing in from a deriva-ml call (e.g. ``None`` where the
  caller expected ``str``) propagated silently to the wire as JSON.

Naming the response shapes via ``pydantic.BaseModel`` makes the
contract explicit AND validated:

1. **Named contract**: helpers declare ``-> DatasetSummary`` instead
   of ``-> dict[str, Any]``. Refactor target is a single grep on the
   class name.
2. **Runtime validation**: constructing ``DatasetSummary(rid=...,
   description=...)`` validates each field against its declared type.
   If a deriva-ml object suddenly returns the wrong type for a field
   we expose, ``ValidationError`` raises at construction time --
   inside the helper, with a useful field-level error message --
   instead of crossing the wire as malformed JSON.
3. **JSON serialization**: ``model.model_dump_json()`` replaces
   ``json.dumps(payload)`` with no manual conversion needed.
4. **IDE tooling**: editors with Pydantic support give autocomplete,
   jump-to-definition, and field-level hover docs.
5. **Reuse from deriva-ml**: where deriva-ml itself exposes a
   Pydantic model with fields we'd duplicate (e.g. ``DatasetHistory``
   in ``deriva_ml.dataset.aux_classes``), we import and embed it
   directly rather than re-declaring the shape.

## Conventions

- ``*Summary`` types are the JSON-friendly shape produced by the
  ``_summarize_*`` helpers. Used as elements in list responses and
  as the base for detail responses.
- ``*Detail`` types extend the relevant Summary with extra keys
  (chaise URL, history, members, etc.) returned by the
  ``_get_*_detail_impl`` helpers. Resources use the detail shape;
  tools usually use the summary shape.
- ``*ListResponse`` types are the paginated list-page shape:
  ``{"<plural>": list[Summary], "count", "truncated", "next_after_rid"}``.
- ``model_config = ConfigDict(extra="forbid")`` on every class so
  unexpected fields surface as a ``ValidationError`` instead of
  silently passing through. This catches helper-side typos and
  upstream-shape drift.
- Optional fields use ``= None`` defaults; their type annotation
  includes ``| None``. Pydantic treats this as a nullable field.

## Stability

These models are part of the wire contract -- tools and resources
are versioned, and these shapes mirror what crosses the MCP wire as
JSON. Changing a model is a contract change; bump the plugin's minor
or major version accordingly per the deriva-ml-mcp release rules in
``CLAUDE.md``.

## Behavior change (v2.0.0)

Prior to v2.0, helpers returned plain ``dict[str, Any]`` and bad data
silently passed through as malformed JSON. Starting v2.0, helpers
construct Pydantic models which validate at construction time --
malformed inputs now surface as ``ValidationError`` (caught by the
tool wrapper's ``_error_envelope`` and returned as
``{"error": "..."}`` on the wire). Tools that previously appeared
"successful" but returned garbage now correctly report failure.

## Reuse map (deriva-ml Pydantic models)

deriva-ml exposes several Pydantic models we COULD embed directly
(``DatasetHistory``, ``Workflow``, ``ExecutionRecord``, ``AssetRecord``).
PR 1 (v1.6) deliberately does NOT embed them -- instead each of our
response models declares its own field set that mirrors the v1.x wire
shape. This keeps the wire output identical to v1.5 (no breaking
changes for skill consumers).

PR 2 (v2.0) is expected to switch to embedding the deriva-ml models
directly where their field set is the right contract for the wire,
which will be a wire-shape change (extra fields surfaced from the
deriva-ml model) signaled by the major version bump.

The list of deriva-ml models we'd consider embedding in PR 2:
- ``DatasetHistory`` -> would replace ``DatasetVersionEntry``
- ``Workflow`` -> would replace ``WorkflowSummary`` (catalog-binding
  state would need to be excluded via ``model_dump(exclude=...)``)
- ``ExecutionRecord`` -> would replace ``ExecutionSummary`` (similar
  catalog-binding state to exclude)
- ``AssetRecord`` -> would inform ``AssetSummary`` (the dynamic
  per-table subclass adds metadata columns we'd want to surface)
"""

from __future__ import annotations

from datetime import datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Common pagination envelope (mixed into every *ListResponse)
# ---------------------------------------------------------------------------


class _PaginationFields(BaseModel):
    """Shared pagination fields present on every list response.

    Subclassed by ``*ListResponse`` models; not used directly. The
    ``count`` field is the number of rows in THIS page (not total).
    Use ``preflight_count=True`` on the underlying tool to get the
    catalog-wide count.
    """

    model_config = ConfigDict(extra="forbid")

    count: int = Field(description="Number of rows on THIS page (not total).")
    truncated: bool = Field(
        description="True when more pages exist beyond this one.",
    )
    next_after_rid: str | None = Field(
        description=(
            "Cursor token for the next page. Pass back as ``after_rid=`` on "
            "the next call. Opaque -- do not parse."
        ),
    )


# ---------------------------------------------------------------------------
# Dataset response models
# ---------------------------------------------------------------------------


class DatasetSummary(BaseModel):
    """One Dataset in a list response. Produced by ``_summarize_dataset``.

    The deriva-ml ``Dataset`` class is NOT a Pydantic model (it's a
    custom class with ``dataset_rid`` / ``description`` /
    ``dataset_types`` / ``current_version`` properties), so we mirror
    its readable attributes here rather than embedding directly.
    """

    model_config = ConfigDict(extra="forbid")

    rid: str
    description: str | None
    dataset_types: list[str]
    current_version: str | None


class DatasetVersionEntry(BaseModel):
    """One row in a Dataset's version history (v1.x wire shape).

    The v1.x wire shape has four keys: ``version`` (semver string),
    ``snapshot`` (catalog snapshot ID), ``description`` (free-text),
    and ``execution_rid``. PR 2 (v2.0) is expected to switch to
    embedding deriva-ml's full ``DatasetHistory`` Pydantic model
    directly, which adds extra fields (``dataset_rid``, ``version_rid``,
    ``minid``, ``spec_hash``); v1.6 preserves the legacy 4-key shape
    for backward compatibility.
    """

    model_config = ConfigDict(extra="forbid")

    version: str
    snapshot: str | None
    description: str | None
    execution_rid: str | None


class DatasetDetail(DatasetSummary):
    """Full Dataset detail: summary + chaise URL + version history.

    Produced by ``_get_dataset_detail_impl``. Used by the
    ``deriva://catalog/{h}/{c}/ml/dataset/{rid}`` resource.
    """

    chaise_url: str
    version_history: list[DatasetVersionEntry]


class DatasetListResponse(_PaginationFields):
    """Paginated list of Datasets. Produced by ``_list_datasets_impl``."""

    datasets: list[DatasetSummary]


class DatasetMemberRef(BaseModel):
    """One dataset member: ``{"table": str, "rid": str}`` flat shape."""

    model_config = ConfigDict(extra="forbid")

    table: str
    rid: str


class DatasetMembersSummaryResponse(BaseModel):
    """Members-summary payload for a Dataset.

    Produced by ``_list_dataset_members_summary_impl``. Used by the
    ``deriva://catalog/{h}/{c}/ml/dataset/{rid}/members`` resource.
    The ``members`` list is capped at ``_MAX_LIMIT`` rows; for the
    full paginated member list, use the
    ``deriva_ml_list_dataset_members`` tool instead.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_rid: str
    summary: dict[str, int]
    total_count: int
    members: list[DatasetMemberRef]
    truncated: bool
    tables: list[str]


# ---------------------------------------------------------------------------
# Workflow response models
# ---------------------------------------------------------------------------


class WorkflowSummary(BaseModel):
    """One Workflow in a list response. Produced by ``_summarize_workflow``.

    Mirrors the user-facing fields of deriva-ml's ``Workflow`` model
    (which we cannot embed directly because it carries catalog-
    binding state and Pydantic ``PrivateAttr``s we don't want on the
    wire). The field set here is exactly what crosses the MCP wire.
    """

    model_config = ConfigDict(extra="forbid")

    rid: str | None
    name: str | None
    url: str | None
    checksum: str | None
    version: str | None
    workflow_type: list[str]
    description: str | None


class WorkflowDetail(WorkflowSummary):
    """Full Workflow detail. Produced by ``_get_workflow_impl``.

    Currently shape-identical to ``WorkflowSummary`` (Workflows have
    no extra detail-only fields today). Kept as a separate type so
    detail-only fields can be added in future without touching the
    summary shape.
    """


class WorkflowListResponse(_PaginationFields):
    """Paginated list of Workflows. Produced by ``_list_workflows_impl``."""

    workflows: list[WorkflowSummary]


# ---------------------------------------------------------------------------
# Execution response models
# ---------------------------------------------------------------------------


class ExecutionSummary(BaseModel):
    """One Execution in a list response. Produced by ``_summarize_execution``.

    Mirrors the user-facing fields of deriva-ml's ``ExecutionRecord``
    (which uses ``PrivateAttr`` for ``status`` / ``description`` /
    ``workflow`` so a direct ``model_dump`` would omit them).

    Datetime fields are typed as ``datetime | str | None`` so the
    helper can pass through deriva-ml's ``datetime`` objects (Pydantic
    auto-serializes them as ISO 8601 in JSON output) or string
    representations from mocks. ``duration`` is similarly tolerant
    -- deriva-ml may return a ``timedelta`` or a string.
    """

    model_config = ConfigDict(extra="forbid")

    rid: str | None
    workflow_rid: str | None
    status: str
    description: str | None
    start_time: datetime | str | None
    stop_time: datetime | str | None
    duration: timedelta | str | None


class ExecutionInputDatasetRef(BaseModel):
    """One input dataset on an Execution detail's ``inputs.datasets`` list."""

    model_config = ConfigDict(extra="forbid")

    rid: str
    version: str | None


class ExecutionAssetRef(BaseModel):
    """One asset on an Execution detail's ``inputs.assets`` / ``outputs.assets``."""

    model_config = ConfigDict(extra="forbid")

    rid: str | None
    filename: str | None


class ExecutionInputs(BaseModel):
    """Inputs grouping on an Execution detail.

    Datasets and assets are intentionally separate lists -- a dataset
    is a typed bundle of catalog rows (with a version), an asset is a
    single file. Treating them uniformly would lose type information
    that consumers care about (e.g. version pinning).
    """

    model_config = ConfigDict(extra="forbid")

    datasets: list[ExecutionInputDatasetRef] = Field(default_factory=list)
    assets: list[ExecutionAssetRef] = Field(default_factory=list)


class ExecutionOutputs(BaseModel):
    """Outputs grouping on an Execution detail.

    Outputs are asset-only today -- DerivaML doesn't currently model
    "this execution produced this dataset" as a first-class output
    relationship (datasets are produced by features that link back to
    their producing execution; the relationship is reachable but not
    direct). Kept as a single-key dict for forward compatibility with
    a possible future ``outputs.datasets`` field.
    """

    model_config = ConfigDict(extra="forbid")

    assets: list[ExecutionAssetRef] = Field(default_factory=list)


class ExecutionExperiment(BaseModel):
    """Experiment association on an Execution detail.

    Present only when the execution is part of a Hydra-driven
    experiment run. The ``model_config`` field name conflicts with
    Pydantic's reserved attribute, so it's aliased to ``model_cfg``
    in Python while the wire JSON key remains ``"model_config"``.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True, protected_namespaces=())

    name: str | None
    config_choices: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    # Aliased: Python attribute is ``model_cfg``; wire key is ``"model_config"``
    # to preserve the v1.x wire shape. Pydantic's protected_namespaces=() above
    # is needed because ``model_*`` is a reserved prefix.
    model_cfg: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict, alias="model_config"
    )


class ExecutionDetail(ExecutionSummary):
    """Full Execution detail: summary + inputs + outputs + optional experiment.

    Produced by ``_get_execution_detail_impl``. Used by the
    ``deriva://catalog/{h}/{c}/ml/execution/{rid}`` resource.

    The ``experiment`` key is None when the execution is not a
    Hydra-driven experiment (the common case). The ``inputs.datasets``
    list is best-effort -- if ``record.list_input_datasets()`` raises
    (e.g. record not bound to a catalog), the list comes back empty
    rather than aborting the whole detail payload. Same best-effort
    semantics for ``inputs.assets`` and ``outputs.assets``.

    The ``metadata`` key from the v1.x design (Deriva_Config /
    Execution_Config / Hydra_Config / Runtime_Env) is intentionally
    OMITTED until upstream provides a generic enumerator -- see the
    TODO comment in ``_get_execution_detail_impl``.

    v2.0 wire shape: identical to v1.x (this PR Pydantic-izes the
    contract; there's no field rename or restructure on this model).
    """

    inputs: ExecutionInputs = Field(default_factory=ExecutionInputs)
    outputs: ExecutionOutputs = Field(default_factory=ExecutionOutputs)
    experiment: ExecutionExperiment | None = None


class ExecutionListResponse(_PaginationFields):
    """Paginated list of Executions. Produced by ``_list_executions_impl``."""

    executions: list[ExecutionSummary]


# ---------------------------------------------------------------------------
# Feature response models
# ---------------------------------------------------------------------------


class FeatureSummary(BaseModel):
    """One Feature in a list response. Produced by ``_summarize_feature``.

    The ``*_columns`` fields are lists of column names (strings); the
    full column metadata (types, vocabulary references, required
    flags) lives on ``FeatureDetail``.
    """

    model_config = ConfigDict(extra="forbid")

    feature_name: str
    target_table: str
    feature_table: str
    term_columns: list[str]
    asset_columns: list[str]
    value_columns: list[str]


class FeatureColumnSpec(BaseModel):
    """One column on a Feature's detail view."""

    model_config = ConfigDict(extra="forbid")

    name: str
    type: str
    nullok: bool
    vocabulary: str | None = None
    asset_table: str | None = None


class FeatureDetail(BaseModel):
    """Full Feature detail with column specs. Produced by feature get tools.

    Note: this is NOT a subclass of FeatureSummary because the
    ``*_columns`` fields change type (list[str] in Summary,
    list[FeatureColumnSpec] in Detail).
    """

    model_config = ConfigDict(extra="forbid")

    feature_name: str
    target_table: str
    feature_table: str
    term_columns: list[FeatureColumnSpec]
    asset_columns: list[FeatureColumnSpec]
    value_columns: list[FeatureColumnSpec]
    required_fields: list[str]


class FeatureListResponse(_PaginationFields):
    """Paginated list of Features. Produced by ``_list_features_impl``."""

    features: list[FeatureSummary]


# ---------------------------------------------------------------------------
# Asset response models
# ---------------------------------------------------------------------------


class AssetSummary(BaseModel):
    """One Asset in a list response. Produced by ``_summarize_asset``.

    deriva-ml's ``AssetRecord`` is dynamically generated per asset
    table (each subclass has different fields), so we mirror the
    common subset that's stable across all asset tables.
    """

    model_config = ConfigDict(extra="forbid")

    rid: str
    filename: str | None
    length: int | None
    md5: str | None
    asset_table: str
    asset_types: list[str]


class AssetExecutionRef(BaseModel):
    """One execution association on an asset detail.

    ``asset_role`` is ``"Input"`` / ``"Output"`` / ``None``. The role
    is best-effort: when the deriva-ml ``ExecutionRecord`` doesn't
    carry a per-asset role attribute (the common case), the role
    surfaces as ``None``.
    """

    model_config = ConfigDict(extra="forbid")

    rid: str | None
    asset_role: str | None


class AssetDetail(AssetSummary):
    """Full Asset detail with associated-execution provenance.

    Produced by ``_get_asset_detail_impl``. Used by the
    ``deriva://catalog/{h}/{c}/ml/asset/{rid}`` resource.

    The ``executions`` list is best-effort: if
    ``Asset.list_executions()`` raises (e.g. malformed catalog
    metadata), it comes back empty rather than aborting the whole
    detail payload.
    """

    url: str | None
    description: str | None
    executions: list[AssetExecutionRef] = Field(default_factory=list)


class AssetListResponse(_PaginationFields):
    """Paginated list of Assets."""

    assets: list[AssetSummary]


class AssetTableRef(BaseModel):
    """One asset table reference: ``{"name": str, "schema": str}``.

    The ``schema`` field is aliased -- the Python attribute is
    ``schema_`` (Pydantic v2 issues a warning for ``schema`` as it
    shadows ``BaseModel.model_json_schema``), but the wire JSON key
    is ``"schema"``. Construct via ``AssetTableRef(name=..., schema=...)``
    or ``AssetTableRef.model_validate({"name": ..., "schema": ...})``;
    both work because ``populate_by_name=True``.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str
    schema_: str = Field(alias="schema")


class AssetTablesResponse(BaseModel):
    """All asset tables in a catalog (no pagination -- typically <20).

    Produced by ``_list_asset_tables_impl``. Used by the
    ``deriva://catalog/{h}/{c}/ml/asset-tables`` resource.

    No pagination -- ``Table`` objects don't carry a stable RID for
    the cursor protocol and catalogs typically have a handful of
    asset tables.
    """

    model_config = ConfigDict(extra="forbid")

    asset_tables: list[AssetTableRef]
    count: int


# ---------------------------------------------------------------------------
# Maintenance / cache response models (v1.1, v1.4)
# ---------------------------------------------------------------------------


class ReindexVocabulariesResponse(BaseModel):
    """Per-vocab term counts after a reindex pass.

    Produced by ``deriva_ml_reindex_vocabularies``. The dict maps each
    vocab qname (``"schema.table"``) to the count of terms re-indexed.
    """

    model_config = ConfigDict(extra="forbid")

    reindexed: dict[str, int]


class ResyncIndexesResponse(BaseModel):
    """Per-table source counts after a resync pass.

    Produced by ``deriva_ml_resync_indexes``. The dict maps each ML
    domain (``"dataset"``, ``"workflow"``, ``"execution"``) to the
    count of per-user RAG sources successfully refreshed.
    """

    model_config = ConfigDict(extra="forbid")

    resynced: dict[str, int]


# ---------------------------------------------------------------------------
# Error envelope (returned by every mutating tool on failure)
# ---------------------------------------------------------------------------


class ErrorEnvelope(BaseModel):
    """The ``{"error": "..."}`` shape returned by tools on failure.

    Documented once here so failure-path tests can assert against the
    typed shape instead of duck-typing ``response.get("error")``. The
    ``deriva_ml_getting_started`` prompt's ERROR ENVELOPE section is
    the user-facing contract; this Pydantic model is the
    implementation-side contract.
    """

    model_config = ConfigDict(extra="forbid")

    error: str
