"""Built-in MCP prompts for deriva-ml-mcp-plugin.

These are MCP prompts (registered via ``@ctx.prompt(...)``), not Python
docstrings. FastMCP surfaces them through the MCP ``prompts/list`` and
``prompts/get`` endpoints so an LLM client can pull them up by name at
the start of a conversation -- they are the cold-start anchor for the
plugin's 44 tools and 18 resources.

The two prompts complement the four built-in core prompts shipped by
``deriva-mcp-core`` (``query_guide``, ``entity_guide``,
``annotation_guide``, ``catalog_guide``) -- they do not replace them.
Core's prompts cover generic ERMrest / annotation / catalog primitives;
these cover the ML-domain layered on top: the five abstractions
(dataset / feature / workflow / execution / asset), the provenance
principle, and the operational conventions (the (hostname, catalog_id)
rule, pagination, error envelopes, resource/RAG discovery).

Prompts registered here:

    deriva_ml_concepts            -- conceptual frame; what DerivaML is,
                                     the five abstractions, the provenance
                                     principle. Read first if cold-starting.
    deriva_ml_getting_started     -- operational orientation; (hostname,
                                     catalog_id) rule, pagination, error
                                     envelope, tool domains, discovery

(Two earlier prompts were removed in v3.x: ``deriva_ml_workflow_dedup``
whose content moved to the ``deriva_ml_create_workflow`` and
``deriva_ml_find_workflow_by_url`` tool docstrings, and
``deriva_ml_execution_lifecycle`` whose content -- the execution state
machine -- moved to the ``user-guide/executions.md`` doc in the
deriva-ml repo (RAG-indexed). v0.5.0 then removed the eight
execution-lifecycle TOOLS themselves: per the stateless rule, executions
originate in the caller's local Python environment, not in the MCP
server. The authoritative how-to lives in ``user-guide/executions.md``
(retrieve it over this MCP via ``rag_search(query="execution lifecycle",
doc_type="ml-docs")``); Claude Code clients also have the deriva-ml-skills
``execution-lifecycle`` skill.)

All prompts are static strings (no f-strings, no tool calls, no catalog
access required to render). Plain ASCII only (workspace convention).

Example:
    Manual invocation (for testing only -- normally invoked by core's
    plugin loader through ``register(ctx)``)::

        from deriva_mcp_core.plugin.api import PluginContext
        from deriva_ml_mcp_plugin.prompts import register

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
#
# SYNC NOTE -- KEEP `_CONCEPTS_GUIDE` IN LOCKSTEP WITH THE
# `deriva-ml-context` SKILL.
#
# `_CONCEPTS_GUIDE` deliberately mirrors the conceptual sections (what
# DerivaML is, the five abstractions, the provenance principle / steering
# principle, the vocabulary-extension pattern) of the `deriva-ml-context`
# always-on skill in the companion `deriva-ml-skills` Claude Code plugin
# (`skills/deriva-ml-context/SKILL.md`).
#
# The duplication is intentional:
#   - Claude Code clients with the `deriva-ml-skills` plugin loaded get
#     the conceptual frame pushed into context proactively via the
#     always-on skill (the audit calls this the "load-bearing" path).
#   - Non-Claude-Code clients (Cursor, SDK-based agents, raw FastMCP
#     clients, etc.) pull the same frame in via the `deriva_ml_concepts`
#     prompt over the MCP wire.
#
# The skill is RICHER than this prompt -- it adds tool-selection
# guidance, cross-references to other skills, and the worked
# "when to reach back to the raw catalog surface" table. This prompt is
# the conceptual FLOOR; the skill is floor + Claude-Code value-add.
#
# When the abstractions evolve (rare -- they're fundamental), update BOTH:
#   1. `_CONCEPTS_GUIDE` below
#   2. `../deriva-ml-skills/skills/deriva-ml-context/SKILL.md`
#      (the skill side carries a matching comment block at the top).


_CONCEPTS_GUIDE = """\
DERIVA-ML CONCEPTS -- read this if you do not already have a mental model of \
DerivaML's domain. The deriva_ml_getting_started prompt assumes this conceptual \
frame; if you arrived here cold, read this first.

WHAT IS DERIVA-ML
-----------------
DerivaML is a reproducible-ML layer built on top of Deriva catalogs. It
records the full provenance of every ML run -- inputs, code versions,
configurations, outputs, and intermediate artifacts -- as first-class
catalog entities so that experiments can be reproduced, audited,
compared across users, and resumed across sessions.

The stack:

  - ``deriva-ml`` is the Python library; provides the ``DerivaML`` class,
    ``Workflow``, ``ExecutionConfiguration``, dataset / feature / asset
    APIs, and the ``with ml.create_execution(config) as exe:`` context
    manager pattern.
  - ``deriva-ml-mcp-plugin`` is THIS plugin (loaded by ``deriva-mcp-core``);
    exposes the ``deriva_ml_*`` MCP tools and the
    ``deriva://catalog/{h}/{c}/deriva-ml/...`` resource family.
  - ``deriva-ml-skills`` is a companion Claude Code skill plugin that
    layers workflow guidance on top -- only relevant when the LLM is
    running inside Claude Code with that plugin loaded.

THE FIVE CORE ABSTRACTIONS
--------------------------
These are the surface DerivaML adds on top of plain Deriva. Each is
stored as one or more Deriva tables underneath, but TREAT THEM AS
DERIVA-ML DOMAIN OBJECTS, NOT AS RAW TABLES.

  Dataset    A versioned collection of catalog rows that an execution
             consumed or produced. Datasets carry a type
             (``Dataset_Type`` vocabulary), an element-type spec, a
             version history, and can be downloaded as bags. Versions
             are two-state per ADR-0003: every member-mutation flips
             the dataset to a *dev* version
             (``<last_release>.post1.devN``); ``deriva_ml_release`` is
             the only operation that produces a released version.
             Use ``deriva_ml_create_dataset``,
             ``deriva_ml_add_dataset_members``, ``deriva_ml_release``.
             For local bag materialization, run ``ml.cache_dataset(spec)``
             in your own Python -- the MCP server has no per-user
             filesystem (v0.5.0 stateless rule). Use the
             ``deriva://catalog/.../deriva-ml/dataset/{rid}/bag-preview``
             resource (or ``deriva_ml_bag_info`` tool) to size a bag
             before deciding to download it locally.

  Workflow   A versioned reference to the code (URL + git commit hash)
             that knows how to do a thing. A Workflow is content-
             addressed: same URL + same commit = same Workflow row.
             Workflows are typed (``Workflow_Type`` vocabulary). Use
             ``deriva_ml_create_workflow``,
             ``deriva_ml_find_workflow_by_url``. See those tools'
             docstrings for the idempotency contract (create_workflow
             dedups internally on URL+checksum; do not preflight).

  Execution  One run of a Workflow against specific input Datasets,
             producing output Datasets / Features / Assets. Executions
             have a status (the ``ExecutionStatus`` enum: Created,
             Running, Stopped, Failed, Pending_Upload, Uploaded,
             Aborted), inputs / outputs links, and a state machine that
             advances Created -> Running -> {Stopped, Failed} ->
             Pending_Upload -> Uploaded (with Created/Running -> Aborted
             as the cancel path). **Executions originate in your
             local Python environment, not in the MCP server.** The
             MCP surface for executions is observational only: query
             with ``deriva_ml_list_executions`` /
             ``deriva_ml_get_execution`` /
             ``deriva_ml_find_workflow_executions`` /
             ``deriva_ml_find_dataset_executions`` /
             ``deriva_ml_get_lineage`` after the run lands. The full
             lifecycle (create / start / commit / abort / update /
             add_feature_values / create_execution_dataset /
             add_nested_execution) lives in user-local Python via the
             DerivaML ``with Execution(...) as exe:`` context manager.
             Cross-cutting state-machine depth lives in the
             ``user-guide/executions.md`` doc on the deriva-ml repo
             (RAG-indexed; ``rag_search`` for "execution state machine"
             surfaces it).

  Feature    A typed value attached to a row of some target table (e.g.
             a per-image classification label produced by a run, or a
             scalar metric like ``f1_score`` per execution). Features
             link the value back to the producing Execution for
             provenance. The DEFINITION of a feature (its schema) is
             created via ``deriva_ml_create_feature``; the VALUES are
             written from user-local Python during an execution (via
             ``exe.add_features(...)``), not through the MCP wire,
             because feature staging writes to a per-process SQLite
             manifest that only the originating process can commit.
             Query values with ``deriva_ml_list_feature_values``.

  Asset      A file uploaded to hatrac and recorded in the catalog with
             an ``Asset_Type`` and provenance link to its producing
             Execution. The MCP surface covers asset metadata only;
             file bytes are out of scope (the MCP server has no general
             access to the user's local filesystem). Use
             ``deriva_ml_list_assets``, ``deriva_ml_lookup_asset``,
             ``deriva_ml_update_asset``. Browse asset tables by schema
             via the ``ml/assets/{schema}`` resource. For asset bytes, hand off to
             local Python via the user's environment (``ml.download_asset``,
             ``exe.asset_file_path()``).

USE DIRECT ATTRIBUTE ACCESS ON THESE DOMAIN OBJECTS
---------------------------------------------------
The five abstractions above are returned as TYPED records (Pydantic
models / DerivaML domain objects), not as raw dicts. Read their
fields with the dot operator:

  - ``Dataset``         -> ``.dataset_rid``, ``.current_version``,
                           ``.dataset_types``, ``.description``
  - ``Execution``       -> ``.execution_rid``, ``.status``,
                           ``.workflow``, ``.description``
  - ``Workflow``        -> ``.workflow_rid``, ``.workflow_type``,
                           ``.url``, ``.checksum``
  - ``FeatureRecord``   -> one named attribute per feature column
                           (``.Diagnosis_Type``, ``.Confidence``, ...)
                           plus ``.Execution``, ``.Feature_Name``,
                           ``.RCT``
  - ``Asset``           -> ``.asset_rid``, ``.filename``, ``.length``,
                           ``.md5``, ``.asset_types``

The ``repr()`` of any domain object spells out its field names --
``print(obj)`` is faster than guessing.

DO NOT use ``getattr(obj, "name", default)`` on these objects. The
attributes either exist by contract or they don't; a wrong name is
an ``AttributeError`` you WANT to see. ``getattr``-with-default
converts API ignorance into a silent fallback --
``getattr(d, "version", "?")`` quietly returns ``"?"`` when the real
field is ``current_version``, and the bug ships. Dot directly into
the object, let ``AttributeError`` tell you what's wrong, and fix
the call. Reach for ``getattr`` only when reflecting over genuinely
unknown attribute names (e.g. iterating user-supplied column names,
debug-printing arbitrary records). That is rare; the normal case is
``obj.field_name``.

CARRY STRUCTURE, DON'T DECAY IT
-------------------------------
The dot-access rule is the local case. The broader principle: when
data has a known shape, carry it in a container that knows the shape,
end-to-end through the analysis. Match the container to the shape:

  - One record (a row, a config, a ranking entry): ``@dataclass`` or
    Pydantic ``BaseModel`` -- named fields, typo-checked at definition.
  - Many records of the same shape (a query result, a denormalized
    dataset, a per-image prediction table): ``pandas.DataFrame``.
    A DataFrame returned by ``Dataset.denormalize_dataset()`` is
    exactly this -- keep it as a DataFrame.
  - A few labeled tables that belong together: a ``@dataclass`` whose
    fields are DataFrames, or ``dict[str, DataFrame]`` if names are
    dynamic.
  - Genuinely schemaless mappings (config blobs from arbitrary YAML,
    MCP responses you're inspecting interactively): ``dict``, and
    ``.get(key, default)`` is correct here -- but ONLY here.

The rule is not "avoid DataFrames" -- DataFrames ARE the typed
container for table-of-records. The rule is don't downgrade: don't
turn a DataFrame into a list-of-dicts, don't turn a Pydantic record
into a dict, don't turn a ``@dataclass`` into a tuple. Each downgrade
trades a real schema for stringly-typed lookups, and every downstream
call site pays the cost. ``getattr(d, "version", "?")`` returning
``"?"`` is the symptom; the disease is that ``d`` should never have
been treated as if its fields were optional.

THE ASSET_ROLE CONTRACT
-----------------------
Every execution-linked asset carries TWO pieces of role metadata
that deriva-ml assigns automatically -- never the caller:

  1. An ``Asset_Role`` of ``Input`` or ``Output`` on its
     ``{Asset}_Execution`` association row. ``Input`` is assigned
     to assets materialized at execution start (consumed by the
     run); ``Output`` is assigned to assets uploaded via the
     execution's commit path.
  2. A directional ``Input_File`` or ``Output_File`` tag added to
     the asset's ``Asset_Type`` list alongside any caller-supplied
     content tags (e.g. ``Model_File``, ``Training_Image``).

Implications for MCP consumers:

  - When filtering ``deriva_ml_list_assets`` results by
    ``asset_types``, expect ``Input_File`` / ``Output_File`` to
    appear in every execution-linked asset's type list. Filtering
    on the exact set ``["Model_File"]`` will miss anything that's
    also been wired into an execution -- filter on subset / any-of
    semantics instead.
  - Use ``deriva_ml_get_lineage`` to walk asset provenance; the
    ``Input``/``Output`` distinction surfaces there as edge
    direction, not as a separate field to query.
  - ``deriva_ml_update_asset(asset_types=...)`` is set-style. If
    you pass a list that drops ``Input_File`` / ``Output_File``,
    the diff will try to REMOVE the role tag -- which breaks
    downstream provenance queries. Always read the current types
    first and preserve the directional tags when updating.

Source of truth: deriva-ml CLAUDE.md "Asset_Role contract" section
and the worked example in ``docs/user-guide/executions.md`` (under
"How execution-asset roles work"); both RAG-indexed via
``ml-docs``.

THE PROVENANCE PRINCIPLE
------------------------
Every artifact (Dataset, Feature value, output Asset) MUST be linked to
the Execution that produced it. This is why:

  - You create an Execution BEFORE writing outputs, not after. From
    user-local Python: ``with ml.create_execution(workflow=..., ...) as
    exe:`` opens the Created -> Running transition.
  - You write outputs INSIDE the context manager:
    ``exe.add_features(records)`` to stage feature values, and
    ``exe.asset_file_path(...)`` to register asset files for upload.
  - The context manager closes the lifecycle on exit (Running ->
    Stopped, or Running -> Failed if the block raised); the separate
    ``exe.commit_output_assets()`` call then drives Pending_Upload ->
    Uploaded. The staged values become visible to downstream queries
    (including the MCP ``deriva_ml_list_*`` / ``deriva_ml_get_*`` read
    tools) only after the upload completes.
  - The state machine is enforced by the local DerivaML library; the
    MCP server cannot mutate Execution status. A would-be "manual
    status flip" via raw entity CRUD on the underlying Deriva tables
    would let an LLM put the catalog into a state where the
    upload-outputs side effect never ran -- which is exactly why
    write access to execution rows is intentionally absent from the
    MCP surface.

The provenance principle is what makes ML runs reproducible across
users, sessions, and time. Bypassing it (e.g. via raw entity CRUD on
the underlying Deriva tables) breaks reproducibility silently.

THE RULE: INHERITANCE WITH OVERRIDE
-----------------------------------
The deriva-ml-mcp-plugin plugin EXTENDS ``deriva-mcp-core``. Everything that
applies in a Deriva catalog applies in a deriva-ml catalog by default.
OVERRIDE: if a deriva-ml surface exists for an operation, prefer it
over the equivalent deriva surface. This applies symmetrically on all
three planes:

  - MCP: prefer ``deriva_ml_*`` MCP tools, prompts, and resources over
    the equivalent ``deriva-mcp-core`` tool / prompt / resource.
  - Python API: prefer ``deriva-ml`` objects and methods (``DerivaML``,
    ``Dataset``, ``Workflow``, ``Execution``, ``Feature``, the
    ``with ml.create_execution(config) as exe:`` context manager,
    ``exe.asset_file_path()``, etc.) over the equivalent ``deriva-py``
    calls (``ErmrestCatalog``, ``PathBuilder``, raw entity resource
    access).
  - Skills (Claude Code clients only): prefer ``/deriva-ml:<skill>``
    over ``/deriva:<skill>`` when both exist.

The override boundary is mechanical: "is there a deriva-ml <thing>
for this?" If yes, use it. If no, the deriva default applies and you
should reach for the corresponding ``deriva-mcp-core`` tool,
``deriva-py`` call, or (in Claude Code) ``/deriva:<skill>``.

The five abstractions above are where the override mostly lands.
Going around them -- using ``insert_records`` / ``update_record`` /
``delete_record`` to mutate Datasets, Workflows, Executions, Features,
or Asset rows -- bypasses real machinery:

  - Business logic (e.g. ``deriva_ml_add_dataset_members`` validates RIDs
    against the dataset's element-type spec; raw inserts will let you
    add wrong-table rows that break the dataset on materialization).
  - FK validation across the Dataset / Workflow / Execution graph (every
    Execution links to a Workflow; every output Dataset links to its
    producing Execution; raw inserts can create dangling references).
  - Provenance tracking (each mutation links back to the active
    Execution; raw inserts have no Execution context).
  - Version management (``deriva_ml_add_dataset_members`` and
    siblings flip the dataset to a dev version;
    ``deriva_ml_release`` promotes that dev period to a released
    snapshot. Raw inserts skip the version flip entirely and leave
    consumers pointed at stale data).
  - RAG re-indexing (the ``deriva_ml_*`` tools fire surgical re-index
    hooks so freshly mutated rows are searchable on the next
    ``rag_search``; raw inserts do not).
  - Audit emission (every ``deriva_ml_*`` mutation emits an audit event
    with the operation name; raw inserts use the generic core audit
    which lacks DerivaML-specific context).

WHAT DERIVA-ML ADDS ON TOP
--------------------------
Deriva's design is about DATA DESIGN -- how to model your data so it's
findable, accessible, interoperable, and reusable. DerivaML adds
PROCESS DESIGN -- how to run an ML pipeline against that data so the
run itself is reproducible. The two are orthogonal: a Deriva catalog
with no DerivaML use can be FAIR-by-construction; a DerivaML catalog
adds reproducibility-by-construction on top. The mechanism is three
abstractions doing complementary jobs: Datasets PIN which rows the
run consumed; Workflows PIN which code (URL + git commit) ran them;
Executions LINK the two so any output Feature or Asset traces back
to (specific code) x (specific inputs).

VOCABULARIES AND THE EXTENSION PATTERN
--------------------------------------
DerivaML ships four built-in controlled vocabularies:

  Dataset_Type            -- tag a Dataset's purpose (e.g. ``Training``,
                             ``Test``, ``Validation``)
  Workflow_Type           -- tag a Workflow's role (e.g. ``Model_Training``,
                             ``Inference``, ``Data_Curation``)
  Asset_Type              -- tag an Asset's file kind (e.g. ``Metrics_File``,
                             ``Model_Weights``, ``Image``)
  Execution_Status        -- managed automatically by the state machine;
                             do NOT extend manually

These are CONTROLLED enumerations: a Dataset's type must be a registered
term, not free-form text. Pass term values that already exist; if the
right term doesn't exist, EXTEND the vocabulary first using core's
generic ``add_term`` tool with ``schema="deriva-ml"``:

    add_term(hostname=..., catalog_id=..., schema="deriva-ml",
             table="Dataset_Type", name="My_New_Type",
             description="...")

The generic ``add_term`` from ``deriva-mcp-core`` handles all four
DerivaML vocabularies; pass ``schema="deriva-ml"`` and the
appropriate ``table=``.

Before adding a new term to any of these four ML vocabularies, check
whether it duplicates an existing term on the same conceptual dimension
(e.g. ``Training`` vs a proposed ``trainings``, or ``Labeled`` vs a
proposed ``Annotated``). Apply the substitution test: "would you swap
one for the other in any context where the existing term applies?" If
yes, the two terms collide on a single dimension and the right action
is to add the new name as a synonym of the existing term rather than
create a parallel term that splits future queries.

For YOUR OWN domain vocabularies (``Sample_Type``, ``Tissue_Type``,
``Image_Quality``, etc.), use the same generic ``add_term`` with your
domain schema name instead of ``"deriva-ml"``.

THE ENTITY RESOLUTION WORKFLOW
------------------------------
This applies to ANY catalog entity referenced by name -- tables,
columns, schemas, vocabulary terms, datasets, workflows, executions,
features, assets, or anything else the catalog tracks. ML-domain or
generic, the workflow is the same.

When the user mentions an entity by name, OR when the user asks to
create a new one, follow these steps:

  1. EXACT MATCH FIRST. If the user-supplied string matches a known
     canonical name exactly (case-sensitive), use it. Don't search,
     don't ask. Catalog names are case-sensitive: ``"Training"`` is
     the ``Dataset_Type`` term; ``"training"`` is not.

  2. SEMANTIC SEARCH IF AMBIGUOUS, FUZZY, OR DESCRIPTIVE. If the
     user's phrasing doesn't match a canonical name exactly -- it's
     descriptive (``"the training data type"``), abbreviated
     (``"DR"``), misspelled (``"Diagnossis"``), or just unfamiliar --
     call ``rag_search`` with their phrase. Use the appropriate
     ``doc_type``:

       - ``catalog-schema`` for tables, columns, features,
         vocabulary terms
       - ``catalog-data`` for datasets, workflows, executions
       - ``ml-docs`` / ``user-guide`` for documentation references

  3. PRESENT A PICKER WHEN MULTIPLE OPTIONS APPEAR. If RAG returns
     more than one plausible candidate, list 3-5 of them with their
     canonical name + one-line description + RID (or
     ``table.column`` for column hits) and ask the user to pick.
     Don't choose blindly when reasonable people might disagree. If
     RAG returns ONE clear top hit (significantly above runners-up),
     use it but tell the user what you resolved it to in one
     sentence (``"I'm using the Training Dataset_Type."``). If RAG
     returns NO useful hits, ask a clarifying question. DO NOT
     fabricate a name; DO NOT call ``create_*`` with a guessed
     identifier.

  4a. LOOKUP PATH ENDS HERE. With the canonical entity in hand, you
      can call the relevant ``lookup_*`` / ``get_*`` / ``find_*``
      tool, or pass the canonical name / RID to whatever operation
      the user requested.

  4b. CREATE PATH HAS ONE MORE STEP. If you arrived here because
      the user asked to CREATE a new entity, before actually
      calling ``create_*``, surface the candidates from step 3 to
      the user explicitly:

         "I found these similar existing entities: <list>. Would
          modifying or reusing one of these work, or do you want
          to create a new one?"

      If the user picks an existing one, switch to the lookup path
      (4a). If the user confirms a new one is needed, proceed to
      step 5.

  5. DESCRIPTION HANDLING ON CREATE. Every ``create_*`` /
     ``add_*`` tool that accepts a ``description`` (or ``comment``)
     argument SHOULD receive a non-empty one. Descriptions become
     part of the catalog's RAG index and are visible to every
     future user; an empty description means future LLMs and
     humans cannot tell what the entity was for.

     If the user did not supply a description, GENERATE A
     SUGGESTION from the conversation context (what was the user
     trying to accomplish? what role does this entity play in
     their workflow?), THEN SHOW IT TO THE USER for
     confirmation or edit:

         "I'm going to create the <entity> with this description:
          '<generated suggestion>'. OK to proceed, or would you
          like to edit it?"

     Pass the confirmed text (or the user's edit) to the tool.
     Don't pass an empty string. Don't pass placeholder text like
     ``"TODO"`` or ``"(no description)"``. Don't fabricate a
     description without showing the user.

     If you're operating autonomously with no human in the loop
     (an unattended agent script), fall back to your best
     generated suggestion and add a note in your response so a
     future audit can see which descriptions were
     auto-generated without confirmation.

WHY THIS WORKFLOW MATTERS
-------------------------
The cost of getting it wrong:

  - Fabricating a name leads to FK-violation errors at best, or
    silent data corruption at worst (e.g. a typo'd ``"Trianing"``
    Dataset_Type that creates a duplicate vocab term).
  - Skipping the picker when there are multiple matches lets the
    LLM commit the user to an entity they didn't intend.
  - Empty descriptions destroy catalog discoverability -- a
    catalog with 500 datasets all described as ``""`` is
    indistinguishable from a catalog with 500 datasets nobody
    can find.

The cost of doing it right is one or two extra round-trips per
operation. Always prefer the round-trips.

WHERE TO GO NEXT
----------------
Now that you have the conceptual frame, read these in this order:

  1. ``deriva_ml_getting_started``  -- operational orientation:
     stateless model, pagination, error envelope, the five tool
     domains, discovery via resources / RAG, the mutation chain.
     Also reachable as a resource at
     ``deriva://deriva-ml/getting-started`` for clients that walk
     resources instead of prompts.

  2. The deriva-ml ``user-guide/executions.md`` doc (RAG-indexed,
     fetched via ``rag_search(query="execution state machine",
     doc_type="ml-docs")``) -- before doing any execution work in your
     local Python. This is the cross-cutting state-machine reference;
     it covers the Created -> Running -> {Stopped, Failed} ->
     Pending_Upload -> Uploaded transitions (and the Aborted cancel
     path), the staged-vs-flushed semantics, and the
     ``with Execution(...) as exe:`` context-manager idiom that
     anchors the lifecycle. The MCP plugin has no execution-mutation
     tools as of v0.5.0; the local Python pattern is the only path.

  (Earlier prompts removed: ``deriva_ml_workflow_dedup`` (v3.x) ->
  ``deriva_ml_create_workflow`` and ``deriva_ml_find_workflow_by_url``
  docstrings; ``deriva_ml_execution_lifecycle`` (v3.x) -> the
  ``user-guide/executions.md`` doc. The execution-lifecycle TOOLS
  themselves were removed in v0.5.0 per the stateless rule.)
"""


_GETTING_STARTED_GUIDE = """\
DERIVA-ML GETTING STARTED -- read this before using any deriva-ml-mcp-plugin tool \
or resource.

If you do not already have a mental model of DerivaML's domain (what a
Dataset / Workflow / Execution / Feature / Asset is, why every artifact
is linked to its producing Execution), read the ``deriva_ml_concepts``
prompt FIRST (also reachable as resource ``deriva://deriva-ml/concepts``).
This prompt assumes that conceptual frame.

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

These two arguments mean the same thing in every tool, so individual
tool docstrings DO NOT redocument them in their ``Args:`` blocks. If
you see a tool whose ``Args:`` starts on its third parameter, this is
why -- ``hostname`` is a Deriva server hostname (e.g.
``"www.example.org"``) and ``catalog_id`` is the catalog ID as a
string (e.g. ``"1"`` or an alias like ``"production"``).

PAGINATION CONTRACT
-------------------
Every tool whose name is ``deriva_ml_list_*`` follows the same
two-step pagination contract; individual ``list_*`` tool docstrings
do NOT redocument it.

Step 1 -- preflight: call with ``preflight_count=True`` to learn the
total count without fetching rows. The response is
``{"total_count": N, "entities_fetched": false, "action_required":
"Found N <items>. Choose a limit and call again with
preflight_count=False."}``. Present the count to the user and choose a
``limit`` (or accept the default).

Step 2 -- fetch a page: call with ``preflight_count=False`` (the
default) and the chosen ``limit`` (default 100, max 1000). The
response is ``{"<items>": [...], "count": N, "truncated": bool,
"next_after_rid": <last-rid> | null}``.

Step 3 -- advance: when ``truncated`` is true, pass the response's
``next_after_rid`` value as the ``after_rid`` argument on the next
call. ``after_rid`` is opaque -- treat it as a cursor token; do not
parse it.

Use the preflight step whenever the count could be surprising
(e.g. you're about to fetch all executions for a workflow). For
fixed-shape "give me the latest 10" queries, skip preflight and call
directly with ``limit=10``.

ERROR ENVELOPE
--------------
Every tool that can fail returns ``{"error": "<message>"}`` as its
JSON payload on failure (instead of the success-shape payload). The
error message is a short string; sometimes it includes the failing
RID or qname. ``mutates=True`` tools also emit an audit-log row with
the same operation name suffixed ``_failed`` (e.g.
``deriva_ml_create_dataset_failed``).

Individual tool ``Returns:`` blocks document only the success shape;
the failure shape is implicit. Individual tool docstrings DO NOT
redocument the error envelope.

VERB CONVENTIONS ON THE WIRE
----------------------------
All tools share the ``deriva_ml_`` prefix; the verb that follows
predicts the call shape and the response shape. Knowing the rule
means you don't have to guess from the tool catalog which call to
reach for:

    list_*    Paginated browse of entities of one kind, optionally
              filtered. PAGINATION CONTRACT applies (preflight ->
              page -> advance). Response carries ``count``,
              ``truncated``, ``next_after_rid``.

    get_*     Single-RID detail read. Takes the RID as a required
              parameter; returns one bundled detail payload. No
              pagination.

    find_*    Catalog-wide search by a non-RID identifier (a URL,
              an external name, a workflow's executions). Use when
              you HAVE the identifying information but DON'T have
              the RID yet. Response shape mirrors the matching
              ``list_*`` (paginated) or ``get_*`` (single hit)
              depending on the search's arity.

    create_ / update_ / delete_ / add_
              Mutating operations. Every one emits an audit row
              (success or ``_failed``). Mutating tools have NO
              resource counterpart -- resources are read-only by
              MCP convention. (Execution lifecycle verbs such as
              ``start_`` / ``commit_`` / ``abort_`` are not MCP tools at
              all -- they run in the caller's local Python; see the
              MUTATION section and ``rag_search(query="execution
              lifecycle", doc_type="ml-docs")``.)

    lookup_*  Specialized RID-or-name resolution (currently only
              ``lookup_asset``). Treat as a sibling of ``get_*``.

This convention applies to the MCP wire surface only. The Python
``deriva-ml`` library uses a slightly different ``find_*`` vs
``list_*`` rule (catalog-wide vs parent-scoped) -- see the
``DerivaML`` class docstring or the ``deriva-ml-context`` skill if
you're driving the library directly from a notebook or skill.

THE FIVE ML DOMAINS
-------------------
Forty-one of the 44 tools are organized into five domain modules (the
other three are the catalog-maintenance tools
``deriva_ml_create_vocabulary``, ``deriva_ml_reindex_vocabularies``,
and ``deriva_ml_resync_indexes``). Pick the domain first, then the
verb. All actual tool names are prefixed
``deriva_ml_<verb>`` (e.g. the ``create`` verb under ``dataset`` is
the ``deriva_ml_create_dataset`` tool). The bare verbs below name the
concept; prepend ``deriva_ml_`` (and append the noun where natural)
for the wire name.

    dataset    -- 19 tools. Curated bundles of catalog rows (image
                  collections, training subsets, splits). Verbs:
                  list / get / list_members / list_relations /
                  list_element_types / create / delete / add_members /
                  delete_members / update / add_element_type / release /
                  get_dataset_spec / bag_info / denormalize /
                  validate_dataset_specs / validate_execution_configuration /
                  validate_config_file / bootstrap_config.

    feature    -- 5 tools. Per-row labels, scores, and asset attachments
                  attached to a target table. Verbs: list / get /
                  list_feature_values / create / delete.
                  Writing feature VALUES (add_feature_values) is a
                  user-local Python operation -- see the execution
                  domain note below.

    workflow   -- 5 tools. Registered runnable artefacts (script + Git
                  URL + checksum + workflow_type). Verbs: list / get /
                  find_workflow_by_url / create / update.

    execution  -- 8 tools (read-only). A single run of a workflow
                  against datasets and assets. The MCP surface for
                  executions is observational only: list / get /
                  find_workflow_executions / find_dataset_executions /
                  list_execution_children /
                  list_execution_parents / find_experiments /
                  get_lineage. ``find_dataset_executions`` answers
                  "which runs used this dataset?" -- role is derived
                  relationally (input = a Dataset_Execution edge,
                  output = Dataset_Version authorship), never from a
                  config file. The full
                  lifecycle (create / start / commit / abort / update /
                  add_feature_values / create_execution_dataset /
                  add_nested_execution) is owned by the caller's local
                  Python and lives in skills that generate user-side
                  code. Executions originate in your environment, not
                  the MCP server -- the workflow code, the feature-
                  staging SQLite manifest, the asset bytes, and the
                  ``with Execution(...) as exe:`` context-manager
                  semantics all live there. For the local-Python
                  pattern, ``rag_search(query="execution lifecycle",
                  doc_type="ml-docs")`` (user-guide/executions.md);
                  Claude Code clients also have the deriva-ml-skills
                  ``execution-lifecycle`` skill.

    asset      -- 4 tools. File-backed catalog rows (images, model
                  weights, etc.) -- catalog-state operations only.
                  File I/O (upload/download) runs in local Python; for
                  the how-to, ``rag_search(query="asset upload download",
                  doc_type="ml-docs")`` (user-guide/assets.md). Verbs:
                  list_assets / find_assets / lookup / update. For "what
                  asset tables exist in a schema?" use the
                  ml/assets/{schema} resource (no dedicated tool).

READ-SIDE QUESTIONS: FETCH THE RESOURCE FIRST
---------------------------------------------
For READ-SIDE QUESTIONS ABOUT AN EXISTING ENTITY -- "show me X by RID",
"what's in Y", "what did Z produce / consume", "what's the current
version of W", "have we tried <thing>", "why was <choice> made" -- fetch
the matching ``deriva://catalog/{h}/{c}/deriva-ml/...`` resource BEFORE
reaching for ``deriva_ml_*`` tools or generic catalog CRUD
(``get_entities``, ``query_attribute``, ``list_foreign_keys``).

The URI is constructable from the catalog hostname + catalog id + entity
RID -- no tool search needed. The resources bundle the entity's summary
plus its associated children (a dataset's members and version, an
execution's inputs / outputs / metadata, an asset's producing-execution
chain) in a stable shape, while the equivalent tool path typically
takes 2-7 round trips of fetch + filter + join.

ANTI-PATTERN (resource-capable clients): do NOT call
``deriva_ml_get_execution`` (or ``deriva_ml_get_dataset``, etc.) when
you already have the RID, want the entity's full state, and your client
can fetch MCP resources. Those tools return a thinner record and force
follow-up calls; the ``ml/<kind>/{rid}`` resource is the one-fetch
answer. If your client cannot read resources, the tools ARE the right
surface -- follow the ORDERED STRATEGY ladder below (typed deriva-ml
tools, chained; generic catalog tools only as the gated fallback).

PRESENTING & STORING RIDs
-------------------------
Resource and tool payloads include a ``cite_url`` field alongside each
RID. Use it:

  - PRESENTING to a user: render an RID as a Markdown link to its
    ``cite_url`` -- ``[2-ABCD](https://host/id/cat/2-ABCD)`` -- not a bare
    string. A bare RID is not clickable or citable.
  - The ``cite_url`` of a RELEASED dataset version is snapshot-pinned
    (``/id/cat/rid@snaptime``) -- a permanent, reproducible reference.
    Assets, executions, workflows, and dev-version datasets have LIVE
    cite_urls (no snaptime); they track current state.
  - STORING an RID reference INSIDE the catalog (in a description,
    annotation, or markdown field): always use the ``/id/`` resolver
    form (``/id/<cat>/<rid>``), never a ``/chaise/...`` UI URL. The
    ``/id/`` form survives UI/deployment changes; ``/chaise/`` URLs do
    not. When a payload gives you a ``cite_url``, use it verbatim.

The full read-only resource family:

    deriva://catalog/{h}/{c}/deriva-ml/datasets         -- all datasets, capped at 1000
    deriva://catalog/{h}/{c}/deriva-ml/dataset/{rid}    -- one dataset + version_history + members
    deriva://catalog/{h}/{c}/deriva-ml/dataset/{rid}/members  -- members grouped by table
    deriva://catalog/{h}/{c}/deriva-ml/dataset/{rid}/spec     -- DatasetSpecConfig snippet for hydra-zen configs
    deriva://catalog/{h}/{c}/deriva-ml/dataset/{rid}/bag-preview
                                                 -- bag size + table counts BEFORE downloading
                                                    (use to size a bag before deciding to
                                                    run ml.cache_dataset(spec) locally)
    deriva://catalog/{h}/{c}/deriva-ml/workflows        -- all workflows
    deriva://catalog/{h}/{c}/deriva-ml/workflow/{rid}   -- one workflow
    deriva://catalog/{h}/{c}/deriva-ml/executions       -- all executions
    deriva://catalog/{h}/{c}/deriva-ml/execution/{rid}  -- one execution: summary + inputs +
                                                    outputs (split into ``assets`` and
                                                    ``metadata``) + experiment
    deriva://catalog/{h}/{c}/deriva-ml/lineage/{rid}    -- provenance chain for any artifact
                                                    (Dataset, Asset, Feature value, Execution)
    deriva://catalog/{h}/{c}/deriva-ml/features/{table} -- features defined on one table
    deriva://catalog/{h}/{c}/deriva-ml/asset/{rid}      -- one asset + bundled executions
    deriva://catalog/{h}/{c}/deriva-ml/assets/{schema}  -- asset tables in one schema
    deriva://catalog/{h}/{c}/deriva-ml/assets/{schema}/{asset_table}
                                                 -- contents of one asset table
    deriva://catalog/{h}/{c}/deriva-ml/vocabularies/{schema}
                                                 -- vocabulary tables in one schema
    deriva://catalog/{h}/{c}/deriva-ml/vocabularies/{schema}/{vocab_name}
                                                 -- terms in one vocabulary table

IMPORTANT -- these catalog-scoped resources are URI TEMPLATES, not static
resources. ``ListMcpResourcesTool`` enumerates only the handful of static
resources (the orientation guides); it does NOT list the
``deriva://catalog/.../deriva-ml/...`` templates above. If ListMcpResources
returns only 2-3 entries, that is expected -- it does NOT mean the catalog
resources are missing. Construct the URI from (hostname, catalog_id, rid)
yourself and read it directly with ``ReadMcpResourceTool``.

RESOURCE-TOOL PAIRINGS (when in doubt, prefer the resource)
-----------------------------------------------------------
Every browseable group below has both a snapshot resource and a
paginated list tool. The resource is one read (up to 1000 rows + a
``truncated`` hint); the tool is the cursor-paginated alternative for
filtered queries or for drilling past the snapshot bound.

Default: prefer the resource. Switch to the tool when EITHER
``truncated`` came back true on the snapshot OR you need a filter the
resource cannot express (status, workflow, deleted-only, etc.).

    Snapshot resource                                      Paginated tool
    --------------------------------------------------     -------------------------------
    deriva://catalog/{h}/{c}/deriva-ml/datasets                   deriva_ml_list_datasets
    deriva://catalog/{h}/{c}/deriva-ml/dataset/{rid}              deriva_ml_get_dataset
    deriva://catalog/{h}/{c}/deriva-ml/dataset/{rid}/members      deriva_ml_list_dataset_members
    deriva://catalog/{h}/{c}/deriva-ml/dataset/{rid}/spec         deriva_ml_get_dataset_spec
    deriva://catalog/{h}/{c}/deriva-ml/dataset/{rid}/bag-preview  deriva_ml_bag_info
    deriva://catalog/{h}/{c}/deriva-ml/workflows                  deriva_ml_list_workflows
    deriva://catalog/{h}/{c}/deriva-ml/workflow/{rid}             deriva_ml_get_workflow
    deriva://catalog/{h}/{c}/deriva-ml/executions                 deriva_ml_list_executions
    deriva://catalog/{h}/{c}/deriva-ml/execution/{rid}            deriva_ml_get_execution
    deriva://catalog/{h}/{c}/deriva-ml/features/{table}           deriva_ml_list_features (table-scoped)
    deriva://catalog/{h}/{c}/deriva-ml/assets/{schema}            deriva_ml_list_assets (schema-scoped)
    deriva://catalog/{h}/{c}/deriva-ml/asset/{rid}                (no list tool -- one-RID only)
    deriva://catalog/{h}/{c}/deriva-ml/lineage/{rid}              deriva_ml_get_lineage
    deriva://catalog/{h}/{c}/deriva-ml/vocabularies/{schema}      list_vocabulary_terms (core; per-table)

Write tools (create / update / commit / abort / start / add_members /
delete_members / ...) have NO resource counterpart -- resources are
read-only by MCP convention.

For SEMANTIC DISCOVERY ("which workflows train CNN models", "find
executions that produced quality scores"), prefer ``rag_search`` over
the paginated list tools:

    rag_search(query="...", doc_type="ml-docs")      -- search DerivaML docs
    rag_search(query="...", doc_type="catalog-data") -- search per-user RAG
                                                        index of Dataset /
                                                        Workflow / Execution rows

Reach for the paginated list tools (``deriva_ml_list_datasets``,
``deriva_ml_list_workflows``, ``deriva_ml_list_executions``,
``deriva_ml_list_features``, ``deriva_ml_list_assets``) only when you
need filtered scans (e.g. "executions with status=Failed for workflow
1-WF") that resources cannot express, or for paginated browse of large
asset tables. Reach for ``deriva_ml_*`` mutation tools (create / update
/ commit / abort / start) only for write-side operations -- those have
no resource counterpart.

NO SKILLS? USE THE DOCS (rag_search is your how-to source)
----------------------------------------------------------
Some clients (Claude Code) load companion "skills" with step-by-step
how-to guidance; many clients (chatbots, SDK agents, raw MCP clients)
do NOT. If you need workflow/how-to guidance that is not already in a
tool docstring or this prompt -- how to run an execution, the
``with ml.create_execution(...) as exe:`` pattern, recording feature
values, uploading/downloading asset bytes, the dataset version
lifecycle, writing a hydra-zen config, working offline, denormalizing,
sharing/reproducibility -- do NOT dead-end on a "see the X skill"
pointer. Instead retrieve the doc over this MCP:

    rag_search(query="<your how-to question>", doc_type="ml-docs")

The deriva-ml repo's user-guide is fully indexed under
``doc_type="ml-docs"``; these pages contain the runnable Python
patterns (not just concepts):

    executions.md      run an execution; with-context pattern;
                       add_features; commit_output_assets; asset I/O;
                       state machine; crash/resume
    datasets.md        create / version / release / split / download a dataset
    features.md        create a feature; record + read feature values; selectors
    assets.md          upload (asset_file_path), download (download_asset /
                       asset.download), metadata reads
    denormalization.md wide-table joins; multi-value feature row semantics
    offline.md         materialize for offline; read from a bag; reconnect
    reproducibility.md version pinning; workflow checksums; dirty-tree rules
    hydra-zen.md       config composition; CLI overrides
    sharing.md         MINID / BDBag / catalog clone
    exploring.md       find_* vs list_*; pathBuilder queries; RID resolution

COMMON TASKS -> WHERE TO LOOK
-----------------------------
    Intent                         Tools / route
    ------------------------------ --------------------------------------------
    "what's in this catalog?"      rag_search(doc_type="catalog-schema" |
                                   "catalog-data"); ml/datasets, ml/workflows
                                   resources
    "set up / run an experiment"   create_workflow (MCP) -> local Python
                                   execution; rag_search "execution lifecycle"
    "organize data for ML"         create_dataset, add_dataset_members,
                                   split_dataset, release; rag_search "datasets"
    "label / score records"        create_feature (MCP) + exe.add_features
                                   (local); rag_search "features"
    "upload / fetch a file"        exe.asset_file_path / exe.download_asset
                                   (local); rag_search "asset upload download"
    "which runs used dataset X?"   find_dataset_executions / get_lineage
    "why is my run failing?"       rag_search "troubleshoot execution",
                                   doc_type="ml-docs"

PROVENANCE & RELATIONSHIP QUERIES -> ORDERED STRATEGY
-----------------------------------------------------
When the user asks how artifacts relate -- execution detail (status /
duration / inputs / outputs), which datasets a run consumed or
produced, asset provenance, a lineage walk, dataset membership --
recognize the pattern EARLY and run this escalation ladder. Do not
start at step 3.

1. TYPED DERIVA-ML SURFACE FIRST. If your client reads MCP resources,
   the ``ml/<kind>/{rid}`` resource is the preferred one-fetch form for
   a known-RID detail question (it bundles the entity plus its children
   -- see READ-SIDE QUESTIONS above); the tools below are the same data
   for tool-only clients and for filtered / paginated questions the
   resources don't express. Either way, the typed deriva-ml surface
   comes first:

    Question                              Tool (start here)
    ------------------------------------- -------------------------------
    execution status/duration/detail      deriva_ml_get_execution(rid)
    full provenance chain for any RID      deriva_ml_get_lineage(rid)
    dataset summary + version history      deriva_ml_get_dataset(rid)
    what's in a dataset                    deriva_ml_list_dataset_members(rid)
    parent/child dataset relations         deriva_ml_list_dataset_relations(rid)
    which runs used / produced a dataset   deriva_ml_find_dataset_executions /
                                           deriva_ml_get_lineage
    executions of a workflow               deriva_ml_find_workflow_executions
    feature values a run produced          deriva_ml_list_feature_values(
                                             ..., execution_rids=[rid])
                                           -- NOT a raw query on the feature
                                           table, even though it carries
                                           Execution provenance columns
    nested execution tree                  deriva_ml_list_execution_children /
                                           _list_execution_parents

   No RID in hand yet? Use the deriva-ml DISCOVERY tools to find it,
   then return to this table: ``deriva_ml_list_executions`` (filter by
   status / workflow / type), ``deriva_ml_find_workflow_executions``,
   ``deriva_ml_find_dataset_executions``, ``deriva_ml_list_datasets``.
   Discovery is still deriva-ml territory -- don't drop to generic
   catalog tools just because the RID is unknown.

2. CHAIN INTO LINEAGE. If ``deriva_ml_get_execution`` succeeds and the
   question implicates the run's inputs, outputs, or related artifacts
   (not just its status), chain directly into
   ``deriva_ml_get_lineage(rid)`` -- one call returns the consumed
   datasets/assets and the producing-execution tree, replacing 5-15
   raw-table round-trips. These tools encode the traversal a hand-built
   join gets subtly wrong: which FK to follow, the dataset VERSION per
   edge, the input-vs-output ROLE per execution, nested-execution
   recursion. (This is the INHERITANCE WITH OVERRIDE rule applied to
   reads, not just mutations.)

3. FALL BACK TO RAW QUERIES when the typed tools don't cover the
   question. Inferring relationships from association / linking tables
   with the generic catalog tools (``query_attribute`` /
   ``query_aggregate`` / ``get_entities``) is legitimate -- as the
   fallback, not the opening move. Do NOT skip steps 1-2 and jump
   straight here. Reach for it when:
   - the relationship is DOMAIN-specific, outside the five ML
     abstractions (e.g. an ``Image_Diagnosis`` image-to-diagnosis link
     or any domain-schema association table -- the ML tools don't model
     those edges at all);
   - a typed tool returned an error or its shape can't express the
     question (an aggregate over edges, a filter the tool doesn't
     expose);
   - you need to verify or debug what a typed tool reported.
   When you do go raw, check the table's row count or a small page
   first; don't assume the table is populated or empty -- both occur
   (some subsystem tables are empty, others hold 100K+ rows).

   WHEN A QUERY COMES BACK EMPTY: report it plainly ("0 rows in X for
   this RID") -- then escalate, don't iterate. If you haven't run
   deriva_ml_get_lineage yet, that is the next move, not another
   table. If lineage already returned null, you are done: the answer
   is "no RECORDED provenance / no REGISTERED assets" -- never "no
   assets exist", and never a "would have produced ..." guess.
   Unregistered outputs are simply out of reach: do NOT go looking
   for them in Hatrac (no namespace browsing / object-store fishing),
   and do not fan out table-by-table hoping one is populated. Report
   what is recorded and stop.

ANTI-PATTERNS (observed failures -- do not repeat)

  WRONG: "Let me query Model_Artifact_Execution for 6-05FW."
    Opening with a single association table. A per-RID slice of one
    table can be EMPTY even when the run has full provenance -- the
    edge may live in a different table (Execution_Asset,
    Dataset_Execution, ...). And ``{Asset}_Execution`` association
    rows only exist at all for runs that REGISTERED their outputs
    through the execution machinery (``exe.commit_output_assets()``);
    a run that wrote files to Hatrac directly leaves no association
    row, by design. One empty slice, no fallback plan, wasted
    round-trip. get_lineage walks every recorded edge in one call.

  WRONG: "Image_Diagnosis came back empty, so this run *would have*
    produced ..." NEVER speculate from an empty slice. An empty
    per-RID result does not mean no data exists, and fabricated
    "would have" provenance is worse than reporting none. If
    deriva_ml_get_lineage returns "lineage": null, the honest answer
    is: this artifact has no recorded provenance link (often a run
    that never registered its outputs). Note lineage walks the same
    recorded catalog edges -- it cannot recover unregistered outputs
    either; nothing can. Its value is that it checks EVERY edge and
    reports absence explicitly instead of leaving you to misread one
    table's empty slice.

  WRONG: "Let me join Dataset_Image / Subject_Dataset to see what's in
    dataset X." Dataset membership is VERSION-PINNED and may RECURSE:
    a raw association join returns only the current unversioned rows of
    one element table -- it cannot answer "members of v0.3.0" (the
    released version the user usually means) and it is blind to members
    contributed by nested child datasets.
    deriva_ml_list_dataset_members(version=..., recurse=...) handles
    both; call it in summary mode first (element_table=None) to see the
    shape before paging rows.

  WRONG: "Let me read the Image_Diagnosis table directly to get the
    labels." A feature table interleaves ground truth AND every model's
    predictions -- one row per (record, execution) for every run that
    ever wrote the feature. A raw read returns them indistinguishably
    mixed. Use deriva_ml_list_feature_values with its filters
    (execution_rids=..., etc.) to select which annotation layer you
    mean.

  WRONG: "Let me explore the schema first to understand the
    structure." For provenance questions the typed tools already
    encode the graph -- schema spelunking before
    get_execution/get_lineage burns round-trips without changing the
    answer. Explore schema when the question is about the SCHEMA, or
    at step 3 when you genuinely need a domain table's shape.

  CORRECT: deriva_ml_get_execution(rid)
           -> (inputs/outputs implicated?) deriva_ml_get_lineage(rid)
           -> deriva_ml_get_dataset(...) on the consumed dataset RIDs
              lineage returned, if the user needs dataset detail
           -> done. Raw queries only via step 3 of the ladder above.

DISCOVERY: RESOLVING USER-MENTIONED NAMES TO CATALOG IDENTIFIERS
----------------------------------------------------------------
When the user mentions a vocabulary term, table or column, workflow,
dataset, or feature by name, resolve it to the canonical catalog
identifier in this order:

1. EXACT MATCH FIRST. If the user-supplied string matches a known
   canonical name exactly (case-sensitive), use it. Don't search,
   don't ask. DerivaML names are case-sensitive: "Training" is the
   Dataset_Type term; "training" is not. If the user says
   "Training", they mean Training.

2. SEMANTIC SEARCH WHEN AMBIGUOUS OR DESCRIPTIVE. If the user's
   phrasing is descriptive ("the training data type", "labels for
   image quality"), or if their string doesn't match any canonical
   name, call rag_search with their phrase. Examine the results:

   - Single clear top hit (score significantly above runners-up):
     use it, but tell the user what you resolved it to in one
     sentence. ("I'm using the Training Dataset_Type.") Fast for
     normal cases, traceable if wrong.

   - Multiple close hits or genuinely ambiguous: present a picker.
     List 3-5 candidates with their canonical name + one-line
     description + RID (or table.column for column hits). Ask
     the user to pick. Don't choose blindly when reasonable
     people might disagree.

   - No useful hits: ask a clarifying question. Do not fabricate
     a name; do not call create_* with a guessed identifier.

3. EXPLICIT LIST REQUEST GETS A LIST. If the user says "show me
   all dataset types" or "list workflows", use the appropriate
   list endpoint (ml/vocabularies/{schema}/Dataset_Type,
   deriva_ml_list_workflows, deriva://catalog/{h}/{c}/deriva-ml/datasets).
   Don't run rag_search when they explicitly want enumeration.

INDEX COVERAGE BY CATEGORY
--------------------------
The picker pattern works for these categories. Freshness varies:

  Vocabulary terms (built-in + user-defined)
    Indexed by this plugin per-vocab. Fresh on first connect to
    the catalog. To re-index after adding a term via core's
    add_term, call deriva_ml_reindex_vocabularies first.

  Tables and columns (any schema, ML or domain)
    Indexed by the framework's schema RAG, per-user. Fresh on
    first connect; refreshed automatically when any tool mutates
    the schema. A column hit lands inside its surrounding table
    chunk -- when the user says "the rid column" you'll see RID
    exists in many tables and need to ask which one.

  Workflows, datasets, executions
    Indexed by this plugin per-user-per-RID, read-through (NOT
    bulk-indexed on connect). A row becomes rag_search-able once
    you've listed or fetched it -- each deriva_ml_list_* /
    deriva_ml_get_* tool warms the index for the rows it just
    returned -- AND it's surgically refreshed on mutation: each
    deriva_ml_create_* / deriva_ml_update_* / deriva_ml_delete_*
    MCP tool refreshes just the affected source(s) before
    returning. Note that EXECUTION row creation/mutation does NOT
    happen through MCP (executions originate in user-local Python
    per the stateless rule) -- newly-created execution rows
    surface into RAG once you list/get them, or via an explicit
    ``deriva_ml_resync_indexes(target="execution:<rid>")`` call.

    Cross-user freshness is best-effort. User A's mutation does
    NOT refresh user B's per-user sources -- B sees A's change only
    once B next lists/fetches that row (which warms it) or resyncs.
    Same gap applies to mutations from non-MCP clients (Chaise UI,
    ERMrest direct, other deriva-ml scripts): they don't propagate
    to your per-user sources until you next read those rows.

    When working on multi-user collaborative catalogs, treat RAG
    hits for shared-visible rows as hints, NOT as ground truth. The
    recovery pattern: when an LLM hit feels stale, verify with the
    corresponding deriva_ml_get_<entity> tool, which always reads
    live catalog state. If you have reason to believe your view is
    significantly stale (e.g., a colleague just released a dataset),
    call deriva_ml_resync_indexes(hostname, catalog_id) first to
    refresh your per-user sources -- or pass target="<table>:<rid>"
    to refresh just one source surgically.

MUTATION: WORKFLOW (MCP) -> EXECUTION (LOCAL PYTHON) -> OUTPUTS
---------------------------------------------------------------
The canonical chain crosses the MCP <-> local-Python boundary:

    On MCP:
    1. ``deriva_ml_create_workflow(...)`` -- registers the workflow
       (it dedups internally on URL+checksum; see that tool's
       docstring for the idempotency contract).

    In your local Python:
    2. ``with ml.create_execution(workflow=wf_rid, ...) as exe:``
       opens the execution context (Created -> Running on enter,
       Running -> Stopped on clean exit, Running -> Failed if the block
       raises, Aborted on cancel).
    3. Inside the context: ``exe.add_features(records)`` to stage
       feature values, ``exe.asset_file_path(...)`` to register asset
       files, ``exe.create_dataset(...)`` for output datasets.
    4. On exit the execution moves to Stopped; then call
       ``exe.commit_output_assets()`` to drive Pending_Upload ->
       Uploaded and flush all staged outputs to the catalog.

    Back on MCP:
    5. Query the freshly-landed catalog state via the read tools:
       ``deriva_ml_get_execution(execution_rid)``,
       ``deriva_ml_list_feature_values(...)``, etc.

Why the split: executions inherit per-process state (workflow code in
your git checkout, feature-staging SQLite manifest, asset bytes on
your disk, ``with`` context-manager semantics). An MCP server cannot
participate in that state -- so execution-mutation tools were removed
in v0.5.0. The MCP surface for executions is observational only.

ALWAYS create artefacts inside an execution context. Bare insertions
(via core's ``insert_records`` etc.) bypass provenance tracking and
should only be used when no domain-specific path exists.

For the lifecycle state machine details AND the runnable how-to (the
``with ml.create_execution(...) as exe:`` pattern, add_features,
commit_output_assets), call ``rag_search(query="execution lifecycle",
doc_type="ml-docs")`` -- the ``user-guide/executions.md`` doc in the
deriva-ml repo carries the full Python pattern and the state-machine
reference at depth, and is retrievable over this MCP. (Claude Code
clients also have the deriva-ml-skills ``execution-lifecycle`` skill;
clients WITHOUT skills should use the rag_search route above -- the
doc contains the same code patterns.)

ASSETS: METADATA HERE, FILE I/O IN LOCAL PYTHON
-----------------------------------------------
The asset surface covers the catalog-state half of the asset
lifecycle:

    deriva_ml_list_assets        -- rows in one asset table (paginated)
    deriva_ml_lookup_asset       -- bundled detail for one asset RID
                                    (filename, length, md5, url,
                                     description, asset_types, executions)
    deriva_ml_update_asset       -- mutate asset_type tags + description

Plus three matching resources:

    deriva://catalog/{h}/{c}/deriva-ml/asset/{rid}      -- bundled per-asset detail
    deriva://catalog/{h}/{c}/deriva-ml/assets/{schema}  -- asset tables in one schema
    deriva://catalog/{h}/{c}/deriva-ml/assets/{schema}/{asset_table}
                                                 -- snapshot of one asset table

What these tools do NOT do: register a new asset from a local file, or
download asset bytes to a local path. The MCP server has no general way
to access the user's local filesystem, so file I/O runs in the user's
local Python, not over this MCP. The runnable patterns are:

    upload:   inside ``with ml.create_execution(...) as exe:`` call
              ``exe.asset_file_path(asset_name, file_name, ...)``,
              write the file, then ``exe.commit_output_assets()``
              after the block (asset tagged ``Output_File``).
    download: ``exe.download_asset(asset_rid, dest_dir)`` for a
              provenance-tracked input (tagged ``Input_File``), or the
              bare ``asset.download(dest_dir)`` for a throwaway read
              with no execution edge.

For the full how-to with code, call ``rag_search(query="asset upload
download", doc_type="ml-docs")`` -- the ``user-guide/assets.md`` doc in
the deriva-ml repo carries upload, download, and metadata-read patterns
and is retrievable over this MCP. (Claude Code clients also have the
deriva-skills ``work-with-assets`` skill; clients WITHOUT skills should
use the rag_search route -- the doc has the same patterns.)

The round trip: do the local-Python file I/O (above), then come back to
MCP for ``deriva_ml_lookup_asset`` / ``deriva_ml_update_asset`` /
``deriva_ml_list_assets`` to confirm and curate the landed rows.

CURATION PATTERN: ONE update_<entity> PER TYPED ENTITY
------------------------------------------------------
Every catalog-mutable typed entity has exactly one
``deriva_ml_update_<entity>(rid, *fields)`` tool. Pass only the kwargs
you want to change; leave others as ``None``. At least one field must
be non-None or the tool returns ``{"error": ...}``.

    deriva_ml_update_dataset    (dataset_types? + description?)
    deriva_ml_update_workflow   (workflow_type? + description?)
    deriva_ml_update_asset      (asset_types? + description?)

For the type-list fields (``dataset_types``, ``workflow_type``,
``asset_types``): SET-STYLE. Pass the desired final list. The tool
fetches the current types and computes the diff -- terms in the new
list that aren't in the current set get added; terms in the current
set that aren't in the new list get removed; terms in both are left
alone.

For ``description``: free-form text overwrite of the catalog row's
``Description`` column.

There is NO ``deriva_ml_update_execution`` tool. As of v0.5.0,
execution rows are only mutated from the caller's local Python
context manager (where description, status, and all other fields are
set via the ``Execution`` API). The reasons are identical to the
ones in the MUTATION section: per-process state ownership, the
``with`` lifecycle, the SQLite manifest. Description-only edits via
MCP would also have invited "free-form status flip" requests that
the state-machine design forbids.

v1.2 breaking rename. The pre-v1.2 ``deriva_ml_update_dataset_types``
tool (with ``add`` / ``remove`` kwargs) was renamed to
``deriva_ml_update_dataset`` and widened to take both ``dataset_types``
and ``description``. There is no compat shim; update any references.

THE MENU
--------
Quick orientation: 44 tools -- 41 across the 5 ML domains (dataset,
feature, workflow, execution, asset) plus 3 catalog-maintenance tools
(create_vocabulary, reindex_vocabularies, resync_indexes) -- + 18
read-only resources under the
``deriva://catalog/{h}/{c}/deriva-ml/...`` URI prefix + 1 GitHub doc source
indexed for RAG (``deriva-ml-docs``) + per-user RAG indexes that
ingest Dataset / Workflow / Execution rows read-through (warmed when
you list/get them, surgically refreshed on mutation -- not on
connect). Plus 4 built-in core prompts and 2 ML prompts: this one
(``deriva_ml_getting_started``) and ``deriva_ml_concepts``. (v3.x
removed two earlier prompts, ``deriva_ml_execution_lifecycle`` and
``deriva_ml_workflow_dedup``, and redistributed their content to the
relevant tool docstrings and RAG-indexed docs.)
"""


# Manifest of guides advertised by the primer (manifest-as-data). Each entry
# is (name, source, summary). source is "deriva-ml" for guides owned by this
# plugin (fetchable via get_guide) or "core" for deriva-mcp-core tier-1 guides
# (fetchable only via the /<server>:<name> slash-command prompt).
#
# SYNC: the "core" rows mirror prompt names registered in
# deriva-mcp-core/src/deriva_mcp_core/tools/prompts.py. If a core guide is
# renamed there, update the matching row here. We cannot enumerate core
# prompts at runtime without reaching into core internals; the names are
# stable public API. The drift-guard test lives in test_prompts.py.
_GUIDE_MANIFEST: list[tuple[str, str, str]] = [
    ("query_guide", "core", "ERMrest query and path syntax, pagination, result interpretation"),
    ("entity_guide", "core", "entity CRUD, preflight count rule, display rules"),
    ("annotation_guide", "core", "Chaise annotation patterns, context names, templates"),
    ("catalog_guide", "core", "catalog create/clone/alias, snaptime format, history"),
]


# Bodies of guides this plugin owns and can return directly via get_guide.
# Core guides are intentionally absent -- their bodies live in
# deriva-mcp-core and are reachable only via their slash-command prompts.
_PLUGIN_GUIDE_BODIES: dict[str, str] = {
    "deriva_ml_concepts": _CONCEPTS_GUIDE,
    "deriva_ml_getting_started": _GETTING_STARTED_GUIDE,
}


def _render_primer() -> str:
    """Render the DerivaML startup primer body.

    Composes three blocks: (1) the mandatory-core guide bodies
    (concepts + getting-started), (2) a one-line manifest of on-demand
    guides grouped by source, and (3) a closing directive on when to
    fetch a guide and to prefer resources for read-side questions.

    The body is phrased neutrally ("DERIVA-ML AGENT GUIDELINES") rather
    than "you MUST", so it reads correctly whether it lands in a system
    prompt or arrives as tool/prompt output in an agent conversation.

    Returns:
        The full primer text as a single plain-ASCII string.

    Example:
        >>> text = _render_primer()
        >>> "DERIVA-ML AGENT GUIDELINES" in text
        True
    """
    ml_guides = [(n, s) for (n, src, s) in _GUIDE_MANIFEST if src == "deriva-ml"]
    core_guides = [(n, s) for (n, src, s) in _GUIDE_MANIFEST if src == "core"]

    parts: list[str] = []

    # Block 1 -- mandatory core (full bodies).
    parts.append("=== DERIVA-ML AGENT GUIDELINES ===\n")
    parts.append(_CONCEPTS_GUIDE)
    parts.append(_GETTING_STARTED_GUIDE)

    # Block 2 -- manifest of on-demand guides.
    parts.append("=== ON-DEMAND GUIDES ===")
    if ml_guides:  # empty today; activates when _GUIDE_MANIFEST gains "deriva-ml" rows
        parts.append(
            "DerivaML domain guides (this plugin) -- fetch with "
            "get_guide(name) when you first need them:"
        )
        for name, summary in ml_guides:
            parts.append(f"  - {name}: {summary}")
    parts.append(
        "Generic catalog guides (deriva-mcp-core) -- fetch the matching "
        "/<server>:<name> prompt when you first use that tool group:"
    )
    for name, summary in core_guides:
        parts.append(f"  - {name}: {summary}")

    # Block 3 -- closing directive.
    parts.append(
        "When you reach an unfamiliar tool covered by a guide above, fetch "
        "that guide once (get_guide(name) for the domain guides above, or "
        "the matching /<server>:<name> prompt for the core guides) and "
        "proceed; do not re-fetch a guide already loaded "
        "this conversation. For read-side questions about existing entities "
        "(show X, what is in Y, what did Z produce), prefer the "
        "deriva://catalog/<host>/<cat>/deriva-ml/... resources over the "
        "equivalent list/get tools -- they are cached, page-free, and emit "
        "no audit entries."
    )

    return "\n\n".join(parts)


# -- Registration ------------------------------------------------------------


def register(ctx: PluginContext) -> None:
    """Register the deriva-ml MCP prompts and the primer tool with the plugin context.

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
        "deriva_ml_concepts",
        description=(
            "Conceptual frame for any LLM client cold-starting on deriva-ml-mcp-plugin: "
            "what DerivaML is, the five core abstractions (Dataset, Workflow, "
            "Execution, Feature, Asset), the provenance principle, and the "
            "vocabulary-extension pattern. Read this BEFORE deriva_ml_getting_started "
            "if you do not already have a DerivaML mental model. Mirrors the "
            "deriva-ml-context skill in the deriva-ml-skills Claude Code plugin "
            "for non-Claude-Code clients (Cursor, SDK-based agents, etc.)."
        ),
    )
    def deriva_ml_concepts() -> str:
        return _CONCEPTS_GUIDE

    @ctx.prompt(
        "deriva_ml_getting_started",
        description=(
            "Cold-start orientation for deriva-ml-mcp-plugin: the (hostname, catalog_id) rule, "
            "the five ML domains, discovery via resources/RAG, the workflow->execution->outputs chain. "
            "Assumes the conceptual frame from deriva_ml_concepts."
        ),
    )
    def deriva_ml_getting_started() -> str:
        return _GETTING_STARTED_GUIDE

    @ctx.tool(mutates=False)
    def deriva_ml_primer(hostname: str = "", catalog_id: str = "") -> str:
        """Load DerivaML agent guidelines and the manifest of available guides.

        Call this FIRST when working with DerivaML in a catalog, before any
        ``deriva_ml_*`` tool or ``deriva://...deriva-ml/...`` resource. Returns
        the conceptual frame, the operating contract (hostname/catalog rule,
        pagination, error envelope), and a one-line manifest of on-demand
        guides. Call once per session; the content does not change.

        Args:
            hostname: Optional Deriva server hostname. Advisory only -- the
                primer content is static and does not vary by catalog.
            catalog_id: Optional catalog id. Advisory only, as above.

        Returns:
            The primer text (plain ASCII).

        Example:
            >>> deriva_ml_primer()  # doctest: +SKIP
            '=== DERIVA-ML AGENT GUIDELINES ===\\n...'
        """
        return _render_primer()

    @ctx.prompt(
        "deriva_ml_primer",
        description=(
            "DerivaML startup primer: agent guidelines (concepts + operating "
            "contract) plus a manifest of on-demand guides. Invoke first when "
            "working with DerivaML; the equivalent deriva_ml_primer tool is "
            "auto-callable by agents that do not surface prompts as commands."
        ),
    )
    def deriva_ml_primer_prompt() -> str:
        return _render_primer()

    @ctx.tool(mutates=False)
    def get_guide(name: str) -> str:
        """Fetch a DerivaML or generic-catalog guide by name.

        For a guide this plugin owns, returns its full body. For a
        deriva-mcp-core tier-1 guide, returns a short redirect pointing at
        the ``/<server>:<name>`` slash-command prompt (core guide bodies are
        not retrievable through this plugin). For an unknown name, returns a
        structured error listing valid names.

        Args:
            name: The guide name, as advertised in the primer manifest.

        Returns:
            The guide body, a redirect string, or a JSON ``{"error": ...}``
            payload for an unknown name.

        Example:
            >>> get_guide("deriva_ml_concepts")  # doctest: +SKIP
            '...the five core abstractions...'
        """
        import json

        if name in _PLUGIN_GUIDE_BODIES:
            return _PLUGIN_GUIDE_BODIES[name]

        core_names = {n for (n, src, _) in _GUIDE_MANIFEST if src == "core"}
        if name in core_names:
            return (
                f"Guide '{name}' is registered in deriva-mcp-core and is not "
                f"retrievable through this plugin. Fetch it via the "
                f"/<server>:{name} slash-command prompt."
            )

        valid = sorted(_PLUGIN_GUIDE_BODIES) + sorted(core_names)
        return json.dumps({"error": f"Unknown guide '{name}'. Valid guides: {', '.join(valid)}."})
