"""Built-in MCP prompts for deriva-ml-mcp.

These are MCP prompts (registered via ``@ctx.prompt(...)``), not Python
docstrings. FastMCP surfaces them through the MCP ``prompts/list`` and
``prompts/get`` endpoints so an LLM client can pull them up by name at
the start of a conversation -- they are the cold-start anchor for the
plugin's 39 tools and 9 resources.

The three prompts complement the four built-in core prompts shipped by
``deriva-mcp-core`` (``query_guide``, ``entity_guide``,
``annotation_guide``, ``catalog_guide``) -- they do not replace them.
Core's prompts cover generic ERMrest / annotation / catalog primitives;
these cover the ML-domain layered on top: dataset / feature / workflow
/ execution objects, the execution state machine, and the workflow
dedup contract.

Prompts registered here:

    deriva_ml_getting_started     -- read first; orients the LLM
    deriva_ml_execution_lifecycle -- state machine + commit semantics
    deriva_ml_workflow_dedup      -- deriva_ml_create_workflow is idempotent

All prompts are static strings (no f-strings, no tool calls, no catalog
access required to render). Plain ASCII only (workspace convention).

Example:
    Manual invocation (for testing only -- normally invoked by core's
    plugin loader through ``register(ctx)``)::

        from deriva_mcp_core.plugin.api import PluginContext
        from deriva_ml_mcp.prompts import register

        ctx = PluginContext(some_mcp_server)
        register(ctx)
        # prompts now discoverable via FastMCP's prompts/list endpoint
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deriva_mcp_core.plugin.api import PluginContext


# -- Prompt content ----------------------------------------------------------
#
# Each constant is the full text returned as a single user message.
# Plain ASCII only -- no en-dashes, smart quotes, or other Unicode.


_GETTING_STARTED_GUIDE = """\
DERIVA-ML GETTING STARTED -- read this before using any deriva-ml-mcp tool \
or resource.

This plugin exposes DerivaML's ML-workflow surface (datasets, features,
workflows, executions) on top of any Deriva catalog. It is layered above
the generic catalog primitives in ``deriva-mcp-core`` -- if a question
is about plain ERMrest queries, entity CRUD, annotations, or catalog
management, see core's ``query_guide`` / ``entity_guide`` /
``annotation_guide`` / ``catalog_guide`` prompts instead.

THE (HOSTNAME, CATALOG_ID) RULE
-------------------------------
Every tool and every resource in this plugin takes a ``hostname`` and a
``catalog_id`` argument. There is NO implicit "current catalog" -- the
plugin holds no per-session state.

Pass the same pair on every call. The catalog connection is opened on
demand and cached behind the scenes; passing different ``catalog_id``
values just talks to different catalogs.

THE FOUR ML DOMAINS
-------------------
The 39 tools are organized into four domain modules. Pick the domain
first, then the verb:

    dataset    -- 17 tools. Curated bundles of catalog rows (image
                  collections, training subsets, splits). Verbs:
                  list / get / create / add_members / delete_members /
                  update_types / increment_version / cache /
                  denormalize / split / deriva_ml_get_dataset_spec / deriva_ml_bag_info.

    feature    -- 6 tools. Per-row labels, scores, and asset attachments
                  attached to a target table. Verbs: list / get /
                  deriva_ml_list_feature_values / create / delete /
                  deriva_ml_add_feature_values.

    workflow   -- 5 tools. Registered runnable artefacts (script + Git
                  URL + checksum + workflow_type). Verbs: list / get /
                  deriva_ml_find_workflow_by_url / create / update.

    execution  -- 11 tools. A single run of a workflow against datasets
                  and assets. Carries the lifecycle state machine.
                  Verbs: list / get / deriva_ml_find_workflow_executions /
                  deriva_ml_list_execution_children / deriva_ml_list_execution_parents /
                  create / start / commit / abort /
                  deriva_ml_create_execution_dataset / deriva_ml_add_nested_execution.

DISCOVERY: PREFER RESOURCES AND RAG OVER PAGINATED TOOL SCANS
-------------------------------------------------------------
For browsing and "what is in this catalog" questions, the read-only
resources are usually the right starting point -- they bundle data that
would otherwise be several tool calls into one URI fetch:

    deriva://catalog/{h}/{c}/ml/datasets         -- all datasets, capped at 1000
    deriva://catalog/{h}/{c}/ml/dataset/{rid}    -- one dataset + version_history
    deriva://catalog/{h}/{c}/ml/dataset/{rid}/members  -- members grouped by table
    deriva://catalog/{h}/{c}/ml/workflows        -- all workflows
    deriva://catalog/{h}/{c}/ml/workflow/{rid}   -- one workflow
    deriva://catalog/{h}/{c}/ml/executions       -- all executions
    deriva://catalog/{h}/{c}/ml/execution/{rid}  -- one execution + inputs/outputs/metadata
    deriva://catalog/{h}/{c}/ml/features/{table} -- features defined on one table
    deriva://catalog/{h}/{c}/ml/registries       -- the four ML vocabularies bundled

For semantic discovery ("which workflows train CNN models", "find
executions that produced quality scores"), prefer ``rag_search`` over
listing tools:

    rag_search(query="...", doc_type="ml-docs")      -- search DerivaML docs
    rag_search(query="...", doc_type="catalog-data") -- search per-user RAG
                                                        index of Dataset /
                                                        Workflow / Execution rows

Fall back to the paginated list tools (``deriva_ml_list_datasets``,
``deriva_ml_list_workflows``, ``deriva_ml_list_executions``, ``deriva_ml_list_features``) only when
you need filtered scans (e.g. "executions with status=Failed for
workflow 1-WF") that resources cannot express.

MUTATION: WORKFLOW -> EXECUTION -> OUTPUTS
------------------------------------------
The canonical mutation chain is:

    1. ``deriva_ml_create_workflow(...)`` (or reuse an existing one -- it dedups;
       see the ``deriva_ml_workflow_dedup`` prompt).
    2. ``deriva_ml_create_execution(workflow_rid=...)`` to register a new run.
    3. ``deriva_ml_start_execution(...)`` to advance Created -> Running.
    4. Write outputs (``deriva_ml_add_feature_values``, ``deriva_ml_create_execution_dataset``,
       etc.) -- they all attribute provenance to the running execution.
    5. ``deriva_ml_commit_execution(...)`` to drain staged outputs and transition
       to Uploaded.

ALWAYS create artefacts inside an execution context. Bare insertions
(via core's ``insert_records`` etc.) bypass provenance tracking and
should only be used when no domain-specific tool exists.

For the lifecycle state machine details, see the
``deriva_ml_execution_lifecycle`` prompt.

THE MENU
--------
Quick orientation: 39 tools across 4 domains (dataset, feature, workflow,
execution) + 9 read-only resources under the
``deriva://catalog/{h}/{c}/ml/...`` URI prefix + 1 GitHub doc source
indexed for RAG (``deriva-ml-docs``) + 3 per-user RAG indexes that
ingest Dataset / Workflow / Execution rows on first connect to a
catalog. Plus 4 built-in core prompts and 3 ML prompts (this one,
``deriva_ml_execution_lifecycle``, ``deriva_ml_workflow_dedup``).
"""


_EXECUTION_LIFECYCLE_GUIDE = """\
DERIVA-ML EXECUTION LIFECYCLE -- read this before using deriva_ml_create_execution, \
deriva_ml_start_execution, deriva_ml_commit_execution, deriva_ml_abort_execution, or deriva_ml_add_feature_values.

An ``Execution`` row tracks one run of a registered workflow against a
set of dataset and asset inputs. It carries a status field that the
plugin's lifecycle tools advance through a small state machine.

THE STATE MACHINE
-----------------

    Created  --deriva_ml_start_execution-->  Running  --deriva_ml_commit_execution-->  Pending_Upload  -->  Uploaded
        \\                            \\
         \\--deriva_ml_abort_execution-->        \\--deriva_ml_abort_execution-->  (Aborted)
          (Aborted)                                          OR  (Failed)

Concrete state values (from ``deriva_ml.execution.execution.ExecutionStatus``):

    Created          -- newly registered, no work has started
    Running          -- deriva_ml_start_execution has been called
    Stopped          -- the execute() context manager exited cleanly
    Pending_Upload   -- commit drained from Running/Stopped, awaiting upload finish
    Uploaded         -- terminal: outputs persisted, provenance frozen
    Failed           -- terminal: execution raised before commit
    Aborted          -- terminal: deriva_ml_abort_execution called explicitly

Two state sets are load-bearing for the lifecycle tools:

    _START_REJECT_STATES = {Stopped, Failed, Pending_Upload, Uploaded, Aborted}
        -- deriva_ml_start_execution refuses to advance from any of these. Stopped
           and Pending_Upload are past the algorithmic phase; Failed /
           Uploaded / Aborted are terminal.

    _COMMIT_ALLOWED_STATES = {Created, Running, Stopped, Pending_Upload, Uploaded}
        -- deriva_ml_commit_execution accepts these. (Pending_Upload is included
           because commit's whole purpose is to drain it. Uploaded is
           the additive-upload entry point: calling deriva_ml_commit_execution on
           an Uploaded execution that has new pending entries cycles
           Uploaded -> Pending_Upload -> Uploaded; with no pending
           entries it is a clean no-op.)

THE FIVE LIFECYCLE TOOLS
------------------------

1. ``deriva_ml_create_execution(workflow_rid=..., dataset_rids=[...], asset_rids=[...])``
   Returns a row in ``Created`` state. Inputs and the parent workflow
   are bound in this call; you cannot change them after.

2. ``deriva_ml_start_execution(execution_rid=...)``
   Required before any feature or output write that goes through the
   ``Running`` path. Advances Created -> Running.

3. ``deriva_ml_commit_execution(execution_rid=...)``
   Drains staged outputs via the upstream ``upload_execution_outputs``
   step and transitions to ``Uploaded``. REQUIRED to make staged feature
   values, datasets, and assets actually persist in queries -- a
   forgotten commit leaves outputs invisible to downstream consumers.

4. ``deriva_ml_abort_execution(execution_rid=..., reason=...)``
   Escape hatch. Terminates a non-terminal execution with optional
   ``reason`` text. Use when a run cannot continue and you want the
   provenance row to record that fact.

5. ``deriva_ml_add_feature_values(execution_rid=..., ...)``
   Hybrid dispatch (Q1) -- the wrapping depends on the current state:

   - ``Created``  -> the call auto-wraps in ``with execution.execute():``
     so Created -> Running on enter and Running -> Stopped on exit.
     Suits one-shot scripts that just want to flush some values.
   - ``Running``  -> the LLM has explicitly called ``deriva_ml_start_execution``
     and is mid-pipeline; the call goes through directly. The eventual
     ``deriva_ml_commit_execution`` closes the lifecycle.
   - Other states -> arg-validation error. ``add_features`` on a
     Stopped or terminal execution has no defined behaviour.

   In other words: you can either let ``deriva_ml_add_feature_values`` drive the
   whole lifecycle (Created -> auto-execute -> Stopped) or drive it
   yourself (deriva_ml_start_execution -> deriva_ml_add_feature_values N times ->
   deriva_ml_commit_execution). Pick one and stick with it.

TWO PITFALLS TO AVOID
---------------------

1. Do NOT call ``update_record`` to flip ``Status`` manually. The state
   machine is enforced by the lifecycle tools (and by upstream
   ``deriva_ml.Execution`` itself). A direct ``Status`` update bypasses
   the upload-outputs side effect of ``deriva_ml_commit_execution`` and can leave
   the execution in an inconsistent state. Always use
   ``deriva_ml_start_execution`` / ``deriva_ml_commit_execution`` / ``deriva_ml_abort_execution``.

2. Do NOT forget ``deriva_ml_commit_execution`` after ``deriva_ml_add_feature_values``
   (when you drove the lifecycle yourself with ``deriva_ml_start_execution``).
   Feature values written during ``Running`` are staged -- they only
   become visible to downstream queries once commit drains them and
   transitions the execution to ``Uploaded``. If you let
   ``deriva_ml_add_feature_values`` auto-wrap (Created path), you do NOT need a
   separate commit -- the auto-execute closes the loop on exit.

INSPECTING STATE
----------------
For a one-shot snapshot of any execution -- status, inputs, outputs,
metadata -- read the resource:

    deriva://catalog/{h}/{c}/ml/execution/{execution_rid}

For filtered scans (e.g. "all Failed executions for workflow 1-WF"),
use the tool with cursor pagination:

    deriva_ml_list_executions(workflow_rid="<workflow_rid>", status="Failed")
"""


_WORKFLOW_DEDUP_GUIDE = """\
DERIVA-ML WORKFLOW DEDUP -- read this before using deriva_ml_create_workflow or \
deriva_ml_find_workflow_by_url.

Workflows are deduplicated by ``(URL, checksum)`` at insert time. This
matters because ``deriva_ml_create_workflow`` is the right tool to call BOTH when
the workflow is new and when it already exists -- you do not need to
preflight.

CREATE_WORKFLOW IS IDEMPOTENT
-----------------------------
Calling ``deriva_ml_create_workflow(url=X, checksum=Y, ...)`` twice with the same
``(X, Y)`` returns the SAME RID both times. The second call carries
``status="exists"`` in the response; the first carries
``status="created"``.

Example response shapes:

    First call:
      {"status": "created", "workflow_rid": "<workflow_rid>",
       "name": "MyPipeline", "url": "https://...", "checksum": "abc123",
       "version": "1.0.0", ...}

    Second call (same url + checksum):
      {"status": "exists",  "workflow_rid": "<workflow_rid>",
       "name": "MyPipeline", "url": "https://...", "checksum": "abc123",
       "version": "1.0.0", ...}

Both responses point to the same ``workflow_rid`` -- the LLM can
unconditionally use the returned RID without branching on ``status``.

DO NOT PREFLIGHT WITH FIND_WORKFLOW_BY_URL
------------------------------------------
The wrong pattern:

    # ANTI-PATTERN -- do not do this
    existing = deriva_ml_find_workflow_by_url(url=X)
    if existing is None:
        deriva_ml_create_workflow(url=X, checksum=Y, ...)
    else:
        workflow_rid = existing["workflow_rid"]

This wastes a round-trip. The dedup is in ``deriva_ml_create_workflow`` itself;
the preflight just duplicates work the server already does.

The right pattern:

    # CORRECT
    result = deriva_ml_create_workflow(url=X, checksum=Y, name="MyPipeline",
                             workflow_type="Model_Training")
    workflow_rid = result["workflow_rid"]  # works whether new or existing

WHEN FIND_WORKFLOW_BY_URL IS THE RIGHT TOOL
-------------------------------------------
``deriva_ml_find_workflow_by_url(url=X)`` is the right tool when you have a URL
but you DO NOT intend to create the workflow if it is missing -- e.g.
you are answering "is this workflow already registered?" or you are
linking against an existing workflow whose presence you want to verify
before doing other work.

If your intent is "make sure this workflow exists, creating it if not",
go straight to ``deriva_ml_create_workflow``.

WHAT THE URL AND CHECKSUM SHOULD POINT AT
-----------------------------------------
The workflow's job is to identify the runnable artefact -- the script
or notebook that will actually execute. So:

    url       -- Git URL pinned to a specific commit, pointing at the
                 file that runs (e.g. ``train.py`` at SHA ``abc123def``).
                 NOT the URL of the registration script that calls
                 ``deriva_ml_create_workflow``.

    checksum  -- Git object hash of that runnable file. Optional but
                 strongly recommended; serves as the secondary dedup key.

The boundary rule (see ``tools/workflow.py``): the caller computes URL,
checksum, and version locally and passes them in. The MCP server never
runs git introspection itself.

INSPECTING REGISTERED WORKFLOWS
-------------------------------
For a snapshot of all workflows in the catalog:

    deriva://catalog/{h}/{c}/ml/workflows

For one workflow's full detail (name, type, url, checksum, version, etc.):

    deriva://catalog/{h}/{c}/ml/workflow/{workflow_rid}
"""


# -- Registration ------------------------------------------------------------


def register(ctx: PluginContext) -> None:
    """Register the three deriva-ml MCP prompts with the plugin context.

    Args:
        ctx: PluginContext supplied by deriva-mcp-core at startup.

    Returns:
        None.

    Example:
        >>> from deriva_mcp_core.plugin.api import PluginContext
        >>> # ctx provided by the framework
        >>> register(ctx)  # doctest: +SKIP
    """

    @ctx.prompt(
        "deriva_ml_getting_started",
        description=(
            "Cold-start orientation for deriva-ml-mcp: the (hostname, catalog_id) rule, "
            "the four ML domains, discovery via resources/RAG, the workflow->execution->outputs chain"
        ),
    )
    def deriva_ml_getting_started() -> str:
        return _GETTING_STARTED_GUIDE

    @ctx.prompt(
        "deriva_ml_execution_lifecycle",
        description=(
            "Execution state machine (Created/Running/Stopped/Pending_Upload/Uploaded), "
            "the five lifecycle tools, deriva_ml_add_feature_values hybrid dispatch, and commit pitfalls"
        ),
    )
    def deriva_ml_execution_lifecycle() -> str:
        return _EXECUTION_LIFECYCLE_GUIDE

    @ctx.prompt(
        "deriva_ml_workflow_dedup",
        description=(
            "deriva_ml_create_workflow is idempotent on (URL, checksum); skip the deriva_ml_find_workflow_by_url preflight "
            "and use deriva_ml_find_workflow_by_url only when you do not intend to create"
        ),
    )
    def deriva_ml_workflow_dedup() -> str:
        return _WORKFLOW_DEDUP_GUIDE
