"""Pydantic response models for deriva-ml-mcp-plugin tools and resources.

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
or major version accordingly per the deriva-ml-mcp-plugin release rules in
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
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny

# ---------------------------------------------------------------------------
# Preflight-count response (shared by every paginated tool's
# preflight_count=True branch)
# ---------------------------------------------------------------------------


class PreflightCountResponse(BaseModel):
    """Response shape for the ``preflight_count=True`` branch of any
    paginated MCP tool.

    Documented once in the ``deriva_ml_getting_started`` prompt's
    PAGINATION CONTRACT section so individual ``list_*`` tool
    docstrings don't redocument it. Returned as JSON
    ``{"total_count": N, "entities_fetched": false,
    "action_required": "..."}``, optionally with extra context fields
    like ``element_table`` (added by ``deriva_ml_list_dataset_members``
    to indicate which element table the preflight is counting).

    ``extra="allow"`` lets call sites attach context fields without
    needing a separate model per tool. The three required fields
    (``total_count``, ``entities_fetched``, ``action_required``) are
    always present.
    """

    model_config = ConfigDict(extra="allow")

    total_count: int | None = Field(
        description=(
            "Total count of catalog rows matching the (post-filter) selection. "
            "May be ``None`` for tools that delegate counting to upstream APIs "
            "that don't always provide an estimate "
            "(e.g. ``deriva_ml_denormalize_dataset``'s describe step)."
        ),
    )
    entities_fetched: bool = Field(
        default=False,
        description=(
            "Always False for preflight responses. The flag exists so a single "
            "consumer-side dispatch on the response shape can distinguish "
            "preflight from a real page."
        ),
    )
    action_required: str = Field(
        description=(
            "Human-readable hint telling the LLM what to do next: choose a "
            "limit and call the tool again with ``preflight_count=False``."
        ),
    )


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
    """Full Dataset detail: summary + cite URL + version history.

    Produced by ``_get_dataset_detail_impl``. Used by the
    ``deriva://catalog/{h}/{c}/deriva-ml/dataset/{rid}`` resource.

    The ``cite_url`` field replaces the previous ``chaise_url`` field
    (renamed in v3.4 with no compat shim). The new URL is the
    ``/id/{cat}/{rid}[@snaptime]`` resolver form -- table-agnostic,
    routable to whatever current Chaise UI is hosted, and (for
    released dataset versions) snaptime-pinned. For dev versions
    (``current_version.is_devrelease``) the URL has no ``@snaptime``
    component, per ADR-0003. Returns ``None`` if URL construction
    failed (degraded but never blocking).
    """

    cite_url: str | None = None
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
    ``deriva://catalog/{h}/{c}/deriva-ml/dataset/{rid}/members`` resource.
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


class DatasetRelationsResponse(BaseModel):
    """Response shape for ``deriva_ml_list_dataset_relations``.

    The wrapper supports three directions: ``parents``, ``children``,
    or ``both``. Each direction populates the corresponding
    ``parents`` / ``children`` list and its ``parents_truncated`` /
    ``children_truncated`` flag. When the caller requested only one
    direction, the other side's fields are absent (None on the wire).

    The ``warning`` field is set when the caller passed ``after_rid``
    with ``direction="both"`` -- the single cursor is incoherent
    because parents and children RIDs aren't synchronized, so we
    drop the cursor and warn rather than silently mis-paginating.

    All fields use Optional shapes (``= None``) to mirror the v1.x
    "field omitted when not relevant" wire shape. Pydantic's
    ``exclude_none=True`` on serialization could drop the absent
    fields, but the v1.x wire used dict literal omission -- we
    replicate that with ``model_dump_json(exclude_none=True,
    by_alias=True)`` at the call site to preserve the v1.x wire
    shape exactly.
    """

    model_config = ConfigDict(extra="forbid")

    parents: list[DatasetSummary] | None = None
    parents_truncated: bool | None = None
    children: list[DatasetSummary] | None = None
    children_truncated: bool | None = None
    warning: str | None = None


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

    ``duration`` is the algorithm-phase measurement (catalog column
    ``Execution_Duration``); ``download_duration`` and
    ``upload_duration`` are the two companion phases added 2026-05-19.
    All three are ``None`` for catalogs that predate the schema bump
    (forward-only migration -- see
    docs/bugs/2026-05-19-execution-phase-durations-design.md).
    """

    model_config = ConfigDict(extra="forbid")

    rid: str | None
    workflow_rid: str | None
    status: str
    description: str | None
    start_time: datetime | str | None
    stop_time: datetime | str | None
    duration: timedelta | str | None
    download_duration: timedelta | str | None = None
    upload_duration: timedelta | str | None = None


class DatasetExecutionSummary(ExecutionSummary):
    """ExecutionSummary plus the dataset-edge role/version for dataset-scoped queries.

    Returned (in place of plain ``ExecutionSummary``) by
    ``_list_executions_impl`` whenever a dataset filter is active --
    i.e. by ``deriva_ml_find_dataset_executions`` and by
    ``deriva_ml_list_executions(dataset=...)``. The two extra fields
    answer "how is this execution related to the dataset I asked
    about?":

    - ``dataset_role`` -- ``"input"`` when the execution *consumed* the
      dataset (a ``Dataset_Execution`` edge), ``"output"`` when it
      *produced* the dataset version (authored a ``Dataset_Version``).
      The role is derived relationally per row, never read from a
      config file.
    - ``dataset_version`` -- the version string of the dataset edge for
      that execution, or ``None`` when no version is pinned/resolvable
      (e.g. an input edge consumed at the live/unpinned state).

    Serialization note: ``ExecutionListResponse.executions`` is typed
    ``list[SerializeAsAny[ExecutionSummary]]`` so a list carrying these
    subclass instances keeps ``dataset_role`` / ``dataset_version`` on
    the wire (plain Pydantic v2 base-typed serialization would drop
    subclass-only fields). A list of plain ``ExecutionSummary`` rows
    serializes byte-identically to before, so the no-dataset path is
    unaffected.
    """

    dataset_role: str  # "input" | "output"
    dataset_version: str | None = None


class ExecutionInputDatasetRef(BaseModel):
    """One input dataset on an Execution detail's ``inputs.datasets`` list.

    Carries the dataset RID, the version that was consumed, and a cite
    URL pinned to that version. The cite URL routing matches ADR-0003:

    - For a released ``version`` (PEP 440 not-devrelease), the URL
      includes the ``@snaptime`` for that release.
    - For a dev ``version`` (PEP 440 devrelease, e.g.
      ``"0.4.0.post1.dev3"``), the URL has no ``@snaptime`` -- it
      resolves to the live catalog state, per ADR-0003's "dev rows
      have no snapshot" rule.
    - When ``version`` is ``None`` (rare; consumed-without-pinned-version),
      the URL resolves to the live state.
    """

    model_config = ConfigDict(extra="forbid")

    rid: str
    version: str | None
    cite_url: str | None = None


class ExecutionAssetRef(BaseModel):
    """One asset on an Execution detail's ``inputs.assets`` / ``outputs.assets`` / ``outputs.metadata``.

    Carries enough per-asset detail to answer "what is this?" without a
    follow-up ``ml/asset/{rid}`` fetch:

    - ``rid`` / ``filename`` -- the navigation handle and the
      human-readable name.
    - ``description`` -- the asset row's free-text description column,
      ideally a one-liner explaining what the file is for. May be
      ``None`` if the producing code (or the framework upload step)
      didn't set one.
    - ``asset_types`` -- controlled-vocab tags from the asset table's
      ``Asset_Type`` join (e.g. ``["Model_File"]``,
      ``["Hydra_Config"]``, ``["Output_File"]``). Empty list when the
      asset has no Asset_Type associations.
    - ``asset_table`` -- the underlying asset table name (e.g.
      ``"Execution_Asset"``, ``"Execution_Metadata"``, ``"Image"``).
      Used by the detail payload's outputs categorization (Execution_Asset
      vs Execution_Metadata) and useful to downstream consumers that
      want to know "is this an Image vs a generic Execution_Asset?".
    - ``cite_url`` -- the ``/id/{cat}/{rid}`` resolver URL for this
      asset. Always the live (no-snaptime) form -- assets are not
      versioned the way datasets are, so there is no historical
      snapshot to pin to. ``None`` when the rid is missing.
    """

    model_config = ConfigDict(extra="forbid")

    rid: str | None
    filename: str | None
    description: str | None = None
    asset_types: list[str] = Field(default_factory=list)
    asset_table: str | None = None
    cite_url: str | None = None


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

    Outputs are split into two buckets:

    - ``assets`` -- "real" ``Execution_Asset`` outputs the run produced
      (model weights, prediction CSVs, training logs, plots).
    - ``metadata`` -- ``Execution_Metadata`` outputs (configuration.json,
      hydra-*.yaml, uv.lock, environment snapshots). These are
      provenance-of-the-run rather than products-of-the-run, and most
      reader use cases want them separated from the real outputs.

    Categorization is by the asset row's ``asset_table`` attribute,
    which deriva-ml's ``ExecutionRecord.list_assets`` already exposes
    (no upstream change needed -- the underlying enumerator walks both
    ``Execution_Asset_Execution`` and ``Execution_Metadata_Execution``
    association tables).

    DerivaML doesn't currently model "this execution produced this
    dataset" as a first-class output relationship (datasets are
    produced by features that link back to their producing execution;
    the relationship is reachable but not direct), so an
    ``outputs.datasets`` field is intentionally absent. The dict stays
    extensible for that future.
    """

    model_config = ConfigDict(extra="forbid")

    assets: list[ExecutionAssetRef] = Field(default_factory=list)
    metadata: list[ExecutionAssetRef] = Field(default_factory=list)


class ExecutionExperiment(BaseModel):
    """Experiment association on an Execution detail.

    Present only when the execution is part of a Hydra-driven
    experiment run. The ``model_config`` field name conflicts with
    Pydantic's reserved attribute, so it's aliased to ``model_cfg``
    in Python while the wire JSON key remains ``"model_config"``.

    The ``config_choices`` and ``model_config`` dicts are typed as
    ``dict[str, Any]`` because real Hydra-zen configs include
    non-primitive values — lists (e.g. ``_zen_exclude: ["description"]``),
    nested dicts (e.g. nested optimizer config), and class-path
    strings under reserved keys (``_target_``, ``_zen_target``). An
    earlier typing of ``dict[str, str | int | float | bool | None]``
    rejected those values at Pydantic validation time, the bare
    ``except Exception`` in ``_get_execution_detail_impl`` silently
    swallowed the failure, and ``experiment: null`` came back for
    every real Hydra-driven execution. Broadening to ``Any`` is the
    right contract: this is the cheap-accessor surface, not a
    schema-validated one.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True, protected_namespaces=())

    name: str | None
    config_choices: dict[str, Any] = Field(default_factory=dict)
    # Aliased: Python attribute is ``model_cfg``; wire key is ``"model_config"``
    # to preserve the v1.x wire shape. Pydantic's protected_namespaces=() above
    # is needed because ``model_*`` is a reserved prefix.
    model_cfg: dict[str, Any] = Field(default_factory=dict, alias="model_config")


class ExecutionDetail(ExecutionSummary):
    """Full Execution detail: summary + inputs + outputs + optional experiment.

    Produced by ``_get_execution_detail_impl``. Used by the
    ``deriva://catalog/{h}/{c}/deriva-ml/execution/{rid}`` resource.

    The ``experiment`` key is None when the execution is not a
    Hydra-driven experiment (the common case). The ``inputs.datasets``
    list is best-effort -- if ``record.list_input_datasets()`` raises
    (e.g. record not bound to a catalog), the list comes back empty
    rather than aborting the whole detail payload. Same best-effort
    semantics for ``inputs.assets`` and ``outputs.assets``.

    Outputs are split between ``outputs.assets`` (real
    ``Execution_Asset`` files like model weights and prediction CSVs)
    and ``outputs.metadata`` (``Execution_Metadata`` files like
    ``configuration.json``, ``hydra-*.yaml``, ``uv.lock``, and the
    environment snapshot). See ``ExecutionOutputs`` for the
    categorization rationale.

    v2.0 wire shape: identical to v1.x (this PR Pydantic-izes the
    contract; there's no field rename or restructure on this model).
    """

    inputs: ExecutionInputs = Field(default_factory=ExecutionInputs)
    outputs: ExecutionOutputs = Field(default_factory=ExecutionOutputs)
    experiment: ExecutionExperiment | None = None


class ExecutionListResponse(_PaginationFields):
    """Paginated list of Executions. Produced by ``_list_executions_impl``.

    Also returned by ``deriva_ml_find_workflow_executions`` (since v2.3),
    which reuses this shape -- it's a paginated list of executions
    filtered by workflow.

    The ``executions`` field is wrapped in ``SerializeAsAny`` so that
    when ``_list_executions_impl`` runs with a dataset filter and
    populates the list with ``DatasetExecutionSummary`` subclass
    instances, the subclass-only fields (``dataset_role`` /
    ``dataset_version``) survive ``model_dump_json``. Without
    ``SerializeAsAny``, Pydantic v2 serializes each row against the
    declared base type (``ExecutionSummary``) and silently drops the
    subclass fields. A list of plain ``ExecutionSummary`` rows is
    unaffected -- it serializes byte-identically to before, so the
    no-dataset path keeps its exact wire shape.
    """

    executions: list[SerializeAsAny[ExecutionSummary]]


class ExecutionChildrenResponse(BaseModel):
    """Response shape for ``deriva_ml_list_execution_children``.

    Returns descendants of an execution (children, optionally
    recursively all descendants). Distinct from
    ``ExecutionListResponse`` because:

    - Not paginated -- the relation graph is bounded per execution.
    - Carries the parent's RID + the recurse flag as context so the
      caller can introspect the query parameters from the response.
    """

    model_config = ConfigDict(extra="forbid")

    parent_rid: str
    recurse: bool
    count: int
    children: list[ExecutionSummary]


class ExecutionParentsResponse(BaseModel):
    """Response shape for ``deriva_ml_list_execution_parents``.

    Returns ancestors of an execution. Symmetric with
    ``ExecutionChildrenResponse`` but with ``child_rid`` and
    ``parents`` instead of ``parent_rid`` and ``children``.
    """

    model_config = ConfigDict(extra="forbid")

    child_rid: str
    recurse: bool
    count: int
    parents: list[ExecutionSummary]


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


class FeatureListResponse(_PaginationFields):
    """Paginated list of Features. Produced by ``_list_features_impl``."""

    features: list[FeatureSummary]


class FeatureValueRecord(BaseModel):
    """One feature-value row on a ``deriva_ml_list_feature_values`` page.

    Mirrors the wire shape produced by ``FeatureRecord.model_dump()``
    in ``deriva-ml`` (see ``deriva_ml.feature.FeatureRecord``). The
    three stable provenance/identity fields are declared explicitly;
    the remaining fields are the feature's term / asset / value
    columns, whose names are dynamic per feature definition and so
    flow through under ``extra="allow"``.

    Stable fields (always present after deriva-ml's ``model_dump()``):

    - ``RID`` -- RID of the feature-value row in the feature table.
      Used as the pagination cursor by ``deriva_ml_list_feature_values``.
    - ``Execution`` -- RID of the execution that wrote this record.
      Provenance handle; ``None`` for legacy rows that predate
      execution tracking.
    - ``Feature_Name`` -- Name of the feature this record belongs to
      (denormalized from the ``Feature_Name`` vocabulary join for
      convenience).
    - ``RCT`` -- Row Creation Time as an ISO 8601 string. Populated by
      the catalog; used by ``select_newest`` for recency ordering.

    Dynamic fields (vary per feature):

    - The target column (e.g. ``Image``, ``Subject``) is the RID of
      the target entity the feature attaches to.
    - Term columns carry vocabulary-term RIDs.
    - Asset columns carry asset-row RIDs.
    - Value columns carry the raw column value (typed per the feature
      definition).

    ``extra="allow"`` is intentional: the alternative would be a
    Pydantic class per feature definition (impractical at runtime). The
    contract this model pins is the four stable fields + the fact that
    a dynamic column set crosses the wire as a flat dict.
    """

    model_config = ConfigDict(extra="allow")

    RID: str | None = Field(
        default=None,
        description="RID of the feature-value row in the feature table.",
    )
    Execution: str | None = Field(
        default=None,
        description=(
            "RID of the execution that wrote this record. ``None`` for "
            "legacy rows that predate execution tracking."
        ),
    )
    Feature_Name: str | None = Field(
        default=None,
        description="Name of the feature this record belongs to.",
    )
    RCT: str | None = Field(
        default=None,
        description=(
            "Row Creation Time as an ISO 8601 string. Used by "
            "``select_newest`` for recency ordering."
        ),
    )


class FeatureValueListResponse(_PaginationFields):
    """Paginated list of feature-value rows. Produced by
    ``deriva_ml_list_feature_values``.

    Each entry is a ``FeatureValueRecord`` -- four stable provenance
    fields plus the feature's dynamic value / term / asset columns
    flowing through under ``extra="allow"``.
    """

    records: list[FeatureValueRecord]


# ---------------------------------------------------------------------------
# Denormalize response models (v0.5.x sweep)
# ---------------------------------------------------------------------------
#
# ``deriva_ml_denormalize_dataset`` has four legitimate response shapes
# discriminated by the ``mode`` field, mirroring the underlying split
# between catalog-wide estimation (``ml.estimate_denormalized_size``)
# and dataset-scoped describe / fetch (``Dataset.describe_denormalized``,
# ``Dataset.get_denormalized_as_dict``):
#
# - ``catalog_shape`` — catalog-wide size estimate; no rows.
# - ``dataset_shape`` — dataset-scoped describe plan; no rows.
# - ``dataset_rows`` — dataset-scoped describe + paginated rows.
# - ``dataset_preflight_required`` — 10x guard short-circuit when the
#   estimated row count is wildly larger than the requested limit.
#
# The ``dataset_preflight`` mode (returned when the caller passes
# ``preflight_count=True``) reuses ``PreflightCountResponse`` with the
# extra ``mode`` and ``dataset_rid`` context fields, so it is not
# modeled separately here.
#
# Both ``dataset_shape`` and ``dataset_rows`` embed the 12-key plan
# returned by ``Denormalizer.describe`` (see
# ``deriva_ml/local_db/denormalizer.py``). The plan's inner dicts
# (``estimated_row_count``, ``anchors``, per-table info, ambiguity
# entries) are typed as ``dict[str, Any]`` because the denormalizer
# layer documents them as best-effort dry-run output that may carry
# extra contextual keys (e.g., ``reason`` on ``estimated_row_count``
# when downstream-anchor cardinality is unknown).


class DenormalizeTableStats(BaseModel):
    """Per-table stats in the catalog-shape response's ``tables`` map.

    Produced by ``ml.estimate_denormalized_size`` for each table in
    the catalog-wide join. ``asset_bytes`` is 0 for non-asset tables.
    """

    model_config = ConfigDict(extra="forbid")

    row_count: int
    is_asset: bool
    asset_bytes: int


class DenormalizeCatalogShapeResponse(BaseModel):
    """Response shape for the catalog-wide size-estimate branch.

    Produced when ``deriva_ml_denormalize_dataset`` is called without
    a ``dataset_rid`` -- the tool wraps ``ml.estimate_denormalized_size``
    and tags the result with ``mode="catalog_shape"`` + the original
    ``include_tables`` echo.

    ``columns`` is a list of ``(column_name, column_type)`` 2-element
    lists -- JSON has no tuples, so the wire emits a list. The upstream
    method returns tuples; the JSON serializer flattens them.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["catalog_shape"]
    include_tables: list[str]
    columns: list[list[str]]
    join_path: list[str]
    tables: dict[str, DenormalizeTableStats]
    total_rows: int
    total_asset_bytes: int
    total_asset_size: str


class _DenormalizeDatasetShapeFields(BaseModel):
    """Shared fields for the two dataset-scoped describe responses.

    ``dataset_shape`` and ``dataset_rows`` both embed the 12-key plan
    returned by ``Denormalizer.describe``. The plan keys are declared
    here so the two response models share them without inheritance
    overhead at the discriminator level.

    The inner-dict fields (``estimated_row_count``, ``anchors``,
    ambiguity entries) are typed as ``dict[str, Any]`` rather than
    sub-models because the denormalizer is explicit that they may
    carry contextual extra keys (e.g., a ``reason`` field on
    ``estimated_row_count`` when downstream-anchor cardinality is
    unknown). Pinning those shapes here would force a contract change
    every time the denormalizer adds explanatory keys.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_rid: str
    version: str | None
    # The 12 keys spread from Denormalizer.describe():
    row_per: str | None
    row_per_source: str
    row_per_candidates: list[str]
    columns: list[list[str]]
    include_tables: list[str]
    via: list[str]
    join_path: list[str]
    transparent_intermediates: list[str]
    ambiguities: list[dict[str, Any]]
    estimated_row_count: dict[str, Any]
    anchors: dict[str, Any]
    source: str


class DenormalizeDatasetShapeResponse(_DenormalizeDatasetShapeFields):
    """Response shape for the dataset-scoped describe-only branch.

    Returned when ``deriva_ml_denormalize_dataset`` is called with a
    ``dataset_rid`` and ``limit=0`` (default). Carries the full 12-key
    describe plan + the dataset RID and (optional) version. No rows.
    """

    mode: Literal["dataset_shape"]


class DenormalizeDatasetPageResponse(_DenormalizeDatasetShapeFields):
    """Response shape for the dataset-scoped describe + rows branch.

    Returned when ``deriva_ml_denormalize_dataset`` is called with a
    ``dataset_rid`` and ``limit > 0`` and the estimated row count
    passes the 10x guard. Carries the full 12-key describe plan, the
    dataset RID and (optional) version, AND a paginated row page.

    The ``rows`` field is intentionally ``list[dict[str, Any]]``: each
    row is a denormalized join across user-defined tables, so the
    column schema is user-defined and cannot be modeled at the
    response-shape layer. The ``columns`` field on the describe plan
    documents the schema for any given call.
    """

    mode: Literal["dataset_rows"]
    rows: list[dict[str, Any]]
    returned_count: int
    truncated: bool
    next_after_rid: str | None


class DenormalizePreflightRequiredResponse(BaseModel):
    """10x-guard short-circuit response.

    Returned when ``deriva_ml_denormalize_dataset`` would refuse to
    drain the generator because the estimated row count exceeds 10x
    the requested limit. The caller is routed back through
    ``preflight_count=True`` to confirm the cost before retrying with
    a larger limit (or accepting the OOM risk).
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["dataset_preflight_required"]
    dataset_rid: str
    estimated_row_count: int
    requested_limit: int
    entities_fetched: bool = False
    action_required: str


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
    ``deriva://catalog/{h}/{c}/deriva-ml/asset/{rid}`` resource.

    Executions semantics. Two distinct empty-list cases are
    distinguishable on the wire so callers can tell "no executions
    are linked to this asset" apart from "the executions lookup
    failed" -- the latter previously surfaced as an indistinguishable
    silent ``[]`` (issue #41 / B18):

    - ``executions == []`` and ``executions_error is None``: the asset
      genuinely has no execution associations, or its asset table has
      no Execution association at all (e.g. a domain asset table that
      doesn't track executions). This is the "no data" answer.
    - ``executions == []`` and ``executions_error`` is a string: the
      executions lookup raised. The asset detail (filename, length,
      url, types, ...) still comes through, but the executions
      sub-field could not be populated. The string is
      ``f"{exc.__class__.__name__}: {exc}"`` -- enough for a caller to
      diagnose without re-running.
    """

    url: str | None
    description: str | None
    executions: list[AssetExecutionRef] = Field(default_factory=list)
    executions_error: str | None = None


class AssetListResponse(_PaginationFields):
    """Paginated list of Assets."""

    assets: list[AssetSummary]


class AssetTableNameRef(BaseModel):
    """One asset table reference inside a schema-scoped list.

    Used by ``ml/assets/{schema}``: the URI carries the schema, so the
    per-entry shape elides it and surfaces only ``name`` + ``rid``.
    Distinct from ``AssetSummary`` (which describes one asset row, not
    a table).
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    rid: str


class AssetTablesInSchemaResponse(BaseModel):
    """All asset tables in one schema.

    Produced by the ``ml/assets/{schema}`` resource's inline helper.
    No pagination -- catalogs typically carry a handful of asset
    tables per schema, and ``Table`` objects have no stable RID for
    the cursor protocol.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_: str = Field(alias="schema")
    count: int
    asset_tables: list[AssetTableNameRef]


class AssetTableContentsResponse(_PaginationFields):
    """One page of asset rows inside one asset table.

    Produced by the ``ml/assets/{schema}/{asset_table}`` resource.
    Reuses ``AssetSummary`` so the snapshot and the paginated
    ``deriva_ml_list_assets`` tool cannot drift in per-row shape.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_: str = Field(alias="schema")
    asset_table: str
    assets: list[AssetSummary]


# ---------------------------------------------------------------------------
# Vocabulary discovery response models (v3.4)
# ---------------------------------------------------------------------------


class VocabularyTableSummary(BaseModel):
    """One vocabulary table in a schema-scoped list.

    Used by ``ml/vocabularies/{schema}``. ``term_count`` is computed
    via ``len(ml.list_vocabulary_terms(table))``; on a per-vocab fetch
    failure (transient catalog hiccup, malformed table) it surfaces as
    ``None`` so one bad vocab doesn't abort the whole snapshot --
    mirrors the pre-v3.4 ``_vocab_terms`` best-effort pattern.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    rid: str
    term_count: int | None = None


class VocabularyTablesResponse(BaseModel):
    """All vocabulary tables in one schema.

    Produced by the ``ml/vocabularies/{schema}`` resource's inline
    helper. No pagination -- a single schema typically holds a
    handful of vocabularies and ``Table`` objects have no stable RID
    for the cursor protocol.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_: str = Field(alias="schema")
    count: int
    vocabularies: list[VocabularyTableSummary]


class VocabularyTermDetail(BaseModel):
    """Full vocabulary term contents.

    Surfaces all six fields from deriva-ml's ``VocabularyTerm`` Pydantic
    class (``name``, ``rid``, ``description``, ``synonyms``, ``id`` /
    CURIE, ``uri``). The class name is ``*Detail`` rather than
    ``VocabularyTerm`` to avoid visual collision with the deriva-ml
    class on grep / IDE hover.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    rid: str
    description: str | None = None
    synonyms: list[str]
    id: str | None = None
    uri: str | None = None


class VocabularyTermsResponse(BaseModel):
    """All terms in one vocabulary table (capped at ``_MAX_LIMIT``).

    Produced by the ``ml/vocabularies/{schema}/{vocab_name}`` resource.
    Vocabularies almost always fit under the 1000-row cap; if
    truncation ever fires, callers fall back to core's
    ``list_vocabulary_terms`` tool for paginated access.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_: str = Field(alias="schema")
    vocabulary: str
    count: int
    terms: list[VocabularyTermDetail]


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
# Vocabulary response models
# ---------------------------------------------------------------------------


class CreateVocabularyResponse(BaseModel):
    """Response from ``deriva_ml_create_vocabulary``.

    Mirrors the canonical DERIVA vocabulary shape produced by
    ``ml.create_vocabulary``: the table is created in the requested
    schema (or domain schema when omitted) with the standard
    columns (Name, ID, URI, Description, Synonyms) typed via
    deriva-py's ``VocabularyTableDef`` -- the ``ID`` and ``URI``
    columns use the ``ermrest_curie`` / ``ermrest_uri`` pseudo-types
    so subsequent ``add_term`` calls succeed without ID / URI
    supplied at insert time.

    Distinct from the generic ``create_vocabulary`` in deriva-mcp-core:
    this tool uses ``ml.create_vocabulary``, which scopes the curie
    template to the deriva-ml project name and updates the
    navigation bar by default. Prefer this tool on any catalog that
    has the deriva-ml schema installed.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["created"]
    schema_name: str = Field(..., alias="schema")
    vocab_name: str
    columns: list[str]


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


# ---------------------------------------------------------------------------
# Mutating-tool response models (v3.0)
#
# Every ``mutates=True`` tool's success-path response is a Pydantic model
# in this section. Each model has a ``Literal[...]`` ``status`` field that
# acts as the discriminator -- consumers can ``match payload["status"]:``
# without checking which tool produced the response.
#
# v3.0 wire-break notes (see ``CLAUDE.md`` for full migration guide):
#
# 1. Six tools' ``status`` values were renamed away from the generic
#    ``"success"`` to operation-specific verbs (``created``, ``added``,
#    ``removed``, ``incremented``, ``cached``, ``split``).
# 2. ``start_execution`` and ``abort_execution`` no-op branches now emit
#    ``status="already_running"`` / ``status="already_aborted"`` instead
#    of carrying an optional ``note`` string.
# 3. Conditional-key fields on ``update_dataset`` are now always present
#    (``null`` when not meaningful) instead of being omitted.
# 4. ``cache_dataset`` nests upstream bag-info keys under a ``bag_info``
#    field instead of spreading them top-level.
# ---------------------------------------------------------------------------


class UpdateAssetResponse(BaseModel):
    """Response from ``deriva_ml_update_asset``."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["updated"]
    asset_rid: str
    updated_fields: list[str]


class CreateWorkflowResponse(BaseModel):
    """Response from ``deriva_ml_create_workflow``.

    ``status="created"`` for new workflows, ``status="exists"`` for the
    idempotent dedup branch (URL or checksum already registered).
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["created", "exists"]
    workflow_rid: str
    name: str
    workflow_type: list[str]
    url: str
    checksum: str | None = None
    version: str | None = None


class UpdateWorkflowResponse(BaseModel):
    """Response from ``deriva_ml_update_workflow``."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["updated"]
    workflow_rid: str
    updated_fields: list[str]


class CreateFeatureResponse(FeatureSummary):
    """Response from ``deriva_ml_create_feature``.

    Wraps ``FeatureSummary`` (which the wrapper already builds) and
    adds the ``status`` discriminator. Pydantic inherits
    ``model_config`` from the parent so ``extra="forbid"`` is in
    effect; the discriminator is added on top.
    """

    status: Literal["created"]


class DeleteFeatureResponse(BaseModel):
    """Response from ``deriva_ml_delete_feature``.

    ``status="deleted"`` when the feature was removed,
    ``status="not_found"`` on the idempotent no-op when the feature
    didn't exist.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["deleted", "not_found"]
    feature_name: str
    table: str


# NOTE: the nine execution-mutation / feature-staging response models
# (AddFeatureValuesResponse, CommitExecutionReport, CreateExecutionResponse,
# StartExecutionResponse, CommitExecutionResponse, UpdateExecutionResponse,
# AbortExecutionResponse, CreateExecutionDatasetResponse,
# AddNestedExecutionResponse) were removed in v0.5.0 when the execution
# lifecycle moved out of the MCP surface per the stateless rule
# (CLAUDE.md, docs/audit-2026-05-23.md). The corresponding tools and
# their response shapes are now owned by the caller's local Python.


class _DatasetVersionBumpMixin(BaseModel):
    """Common fields for tools that change a dataset's version.

    Three tools share this shape: ``add_dataset_members``,
    ``delete_dataset_members``, and ``release``. They all return the
    dataset RID and the post-call version string. Subclasses add the
    ``status`` discriminator and any tool-specific delta fields
    (e.g. ``added_count``).

    Per ADR-0003 (deriva-ml 1.34+), ``add_dataset_members`` and
    ``delete_dataset_members`` flip the dataset to a *dev* version
    (a label of the form ``<last_release>.post1.devN``); the
    returned ``new_version`` may therefore be a dev label, not a
    released label. ``release`` is the only operation that produces
    a released version.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_rid: str
    new_version: str


class AddDatasetMembersResponse(_DatasetVersionBumpMixin):
    """Response from ``deriva_ml_add_dataset_members``.

    Per ADR-0003 (deriva-ml 1.34+), the returned ``new_version`` is
    a dev label (``<last_release>.post1.devN``) — adding members
    flips the dataset to dev, not to a released version. Call
    ``deriva_ml_release`` afterward to mint a released version.

    v3.0: ``status`` was renamed from ``"success"`` to ``"added"`` to
    match the v3.0 status vocabulary (operation-specific verbs).
    """

    status: Literal["added"]
    added_count: int


class DeleteDatasetMembersResponse(_DatasetVersionBumpMixin):
    """Response from ``deriva_ml_delete_dataset_members``.

    Per ADR-0003 (deriva-ml 1.34+), the returned ``new_version`` is
    a dev label (``<last_release>.post1.devN``) — deleting members
    flips the dataset to dev, not to a released version. Call
    ``deriva_ml_release`` afterward to mint a released version.

    v3.0: ``status`` was renamed from ``"success"`` to ``"removed"``.
    """

    status: Literal["removed"]
    removed_count: int


class ReleaseDatasetResponse(_DatasetVersionBumpMixin):
    """Response from ``deriva_ml_release``.

    Per ADR-0003 (deriva-ml 1.34+), ``release`` promotes a dev
    period to a released version. The returned ``new_version`` is
    a released label (e.g. ``"0.5.0"``); ``previous_version`` is
    the dev label (e.g. ``"0.4.0.post1.dev3"``) that was promoted.
    The ``bump`` field records which release segment was advanced
    (``major`` / ``minor`` / ``patch``).
    """

    status: Literal["released"]
    previous_version: str
    bump: str


class CreateDatasetResponse(DatasetSummary):
    """Response from ``deriva_ml_create_dataset``.

    Extends ``DatasetSummary`` with the v3.0 status discriminator and
    the linking ``execution_rid``. ``model_config`` inherits from
    ``DatasetSummary`` so ``extra="forbid"`` is in effect; the new
    fields are added on top.
    """

    status: Literal["created"]
    execution_rid: str


class DeleteDatasetResponse(BaseModel):
    """Response from ``deriva_ml_delete_dataset``."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["deleted"]
    dataset_rid: str
    recursive: bool


class UpdateDatasetResponse(BaseModel):
    """Response from ``deriva_ml_update_dataset``.

    v3.0: the ``dataset_types``, ``added``, and ``removed`` fields
    are now ALWAYS present in the response (``null`` when the
    description-only branch ran). v2.x omitted these keys when the
    description-only branch ran.
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["updated"]
    dataset_rid: str
    updated_fields: list[str]
    new_version: str
    dataset_types: list[str] | None = None
    added: list[str] | None = None
    removed: list[str] | None = None


class AddDatasetElementTypeResponse(BaseModel):
    """Response from ``deriva_ml_add_dataset_element_type``.

    v3.0: ``status`` was renamed from ``"success"`` to ``"created"``
    -- registering an element type IS a creation event (a new
    association table row).
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["created"]
    table_name: str
    association_table: str


# NOTE: CacheDatasetBagInfo and CacheDatasetResponse were removed
# in v0.5.0 when the deriva_ml_cache_dataset tool was retired per the
# stateless rule (see CLAUDE.md, docs/audit-2026-05-23.md, and
# docs/superpowers/notes/2026-05-23-cache-denormalize-deprecation-design.md).
# Bag materialization is now a user-local Python operation; the
# response models are no longer needed.


class DescribeRidResponse(BaseModel):
    """Response shape for ``deriva_ml_describe_rid``.

    The "step 0" answer for a bare RID of unknown type: where it lives
    (schema, table), which DerivaML abstraction it is, and what to call
    next. ``resource_uri`` carries the matching ``ml/<kind>/{rid}``
    resource for the kinds that have one (dataset / workflow /
    execution / asset) and is ``None`` otherwise.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    rid: str
    schema_: str = Field(alias="schema")
    table: str
    kind: Literal[
        "dataset",
        "workflow",
        "execution",
        "vocabulary_term",
        "asset",
        "association",
        "entity",
    ]
    suggestion: str
    resource_uri: str | None = None
