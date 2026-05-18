"""Built-in MCP prompts for deriva-ml-mcp.

These are MCP prompts (registered via ``@ctx.prompt(...)``), not Python
docstrings. FastMCP surfaces them through the MCP ``prompts/list`` and
``prompts/get`` endpoints so an LLM client can pull them up by name at
the start of a conversation -- they are the cold-start anchor for the
plugin's 45 tools and 11 resources.

The four prompts complement the four built-in core prompts shipped by
``deriva-mcp-core`` (``query_guide``, ``entity_guide``,
``annotation_guide``, ``catalog_guide``) -- they do not replace them.
Core's prompts cover generic ERMrest / annotation / catalog primitives;
these cover the ML-domain layered on top: dataset / feature / workflow
/ execution objects, the execution state machine, and the workflow
dedup contract.

Prompts registered here:

    deriva_ml_concepts            -- conceptual frame; what DerivaML is,
                                     the five abstractions, the provenance
                                     principle. Read first if cold-starting.
    deriva_ml_getting_started     -- operational orientation; (hostname,
                                     catalog_id) rule, pagination, error
                                     envelope, tool domains, discovery

(Two earlier prompts were removed in v3.x and their content moved to
the proper homes:

  - ``deriva_ml_workflow_dedup`` -> the ``deriva_ml_create_workflow``
    and ``deriva_ml_find_workflow_by_url`` tool docstrings. The LLM-trap
    warning belongs on the trap.
  - ``deriva_ml_execution_lifecycle`` -> the four lifecycle tool
    docstrings (``deriva_ml_start_execution``,
    ``deriva_ml_commit_execution``, ``deriva_ml_abort_execution``,
    ``deriva_ml_add_feature_values``) for the per-tool warnings, and
    the ``user-guide/executions.md`` doc in the deriva-ml repo
    (already RAG-indexed) for the cross-cutting state-machine depth.)

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
  - ``deriva-ml-mcp`` is THIS plugin (loaded by ``deriva-mcp-core``);
    exposes the ``deriva_ml_*`` MCP tools and the
    ``deriva://catalog/{h}/{c}/ml/...`` resource family.
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
             ``deriva_ml_add_dataset_members``, ``deriva_ml_release``,
             ``deriva_ml_cache_dataset``.

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
             have a status (``Execution_Status_Type``), inputs / outputs
             links, and a state machine that the lifecycle tools advance
             through. Use ``deriva_ml_create_execution``,
             ``deriva_ml_start_execution``, ``deriva_ml_commit_execution``,
             ``deriva_ml_abort_execution``, ``deriva_ml_update_execution``.
             Each lifecycle tool's docstring documents the per-tool
             state-machine constraints; the cross-cutting state-machine
             depth lives in the ``user-guide/executions.md`` doc on the
             deriva-ml repo (RAG-indexed; ``rag_search`` for "execution
             state machine" surfaces it).

  Feature    A typed value attached to a row of some target table (e.g.
             a per-image classification label produced by a run, or a
             scalar metric like ``f1_score`` per execution). Features
             link the value back to the producing Execution for
             provenance. Use ``deriva_ml_create_feature``,
             ``deriva_ml_add_feature_values``,
             ``deriva_ml_list_feature_values``.

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

THE PROVENANCE PRINCIPLE
------------------------
Every artifact (Dataset, Feature value, output Asset) MUST be linked to
the Execution that produced it. This is why:

  - You create an Execution BEFORE writing outputs, not after.
  - You ``deriva_ml_start_execution`` to advance the state machine
    BEFORE writing outputs (the alternative is the auto-wrap path of
    ``deriva_ml_add_feature_values``; see the lifecycle prompt).
  - You ``deriva_ml_commit_execution`` AFTER writing outputs, so the
    staged values become visible to downstream queries.
  - ``deriva_ml_update_execution`` deliberately does NOT accept a
    ``status=`` argument -- the state machine is enforced by the
    lifecycle tools, NOT by free-form status edits. Manual status
    flips would let an LLM put the catalog into a state where the
    upload-outputs side effect of commit never ran.

The provenance principle is what makes ML runs reproducible across
users, sessions, and time. Bypassing it (e.g. via raw entity CRUD on
the underlying Deriva tables) breaks reproducibility silently.

THE RULE: INHERITANCE WITH OVERRIDE
-----------------------------------
The deriva-ml-mcp plugin EXTENDS ``deriva-mcp-core``. Everything that
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
  Execution_Status_Type   -- managed automatically by the state machine;
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

  2. The lifecycle tool docstrings (``deriva_ml_start_execution``,
     ``deriva_ml_commit_execution``, ``deriva_ml_abort_execution``,
     ``deriva_ml_add_feature_values``)  -- before doing any execution
     work. Each tool's docstring carries the per-tool state-machine
     constraints (acceptance sets, terminal states, the staged-vs-
     flushed semantics that make a missing commit invisible).

  (Two earlier prompts were removed in v3.x: ``deriva_ml_workflow_dedup``
  whose content moved to the ``deriva_ml_create_workflow`` and
  ``deriva_ml_find_workflow_by_url`` docstrings, and
  ``deriva_ml_execution_lifecycle`` whose content moved to the four
  lifecycle tools' docstrings plus the RAG-indexed
  ``user-guide/executions.md`` doc on the deriva-ml repo. Per-tool
  warnings belong on the trap; cross-cutting depth belongs in
  searchable docs.)
"""


_GETTING_STARTED_GUIDE = """\
DERIVA-ML GETTING STARTED -- read this before using any deriva-ml-mcp tool \
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

    create_ / update_ / delete_ / add_ / start_ / commit_ / abort_
              Mutating operations. Every one emits an audit row
              (success or ``_failed``). Mutating tools have NO
              resource counterpart -- resources are read-only by
              MCP convention.

    lookup_*  Specialized RID-or-name resolution (currently only
              ``lookup_asset``). Treat as a sibling of ``get_*``.

This convention applies to the MCP wire surface only. The Python
``deriva-ml`` library uses a slightly different ``find_*`` vs
``list_*`` rule (catalog-wide vs parent-scoped) -- see the
``DerivaML`` class docstring or the ``deriva-ml-context`` skill if
you're driving the library directly from a notebook or skill.

THE FIVE ML DOMAINS
-------------------
The 45 tools are organized into five domain modules. Pick the domain
first, then the verb. All actual tool names are prefixed
``deriva_ml_<verb>`` (e.g. the ``create`` verb under ``dataset`` is
the ``deriva_ml_create_dataset`` tool). The bare verbs below name the
concept; prepend ``deriva_ml_`` (and append the noun where natural)
for the wire name.

    dataset    -- 17 tools. Curated bundles of catalog rows (image
                  collections, training subsets, splits). Verbs:
                  list / get / create / add_members / delete_members /
                  update / increment_version / cache /
                  denormalize / split / get_dataset_spec / bag_info.

    feature    -- 6 tools. Per-row labels, scores, and asset attachments
                  attached to a target table. Verbs: list / get /
                  list_feature_values / create / delete /
                  add_feature_values.

    workflow   -- 5 tools. Registered runnable artefacts (script + Git
                  URL + checksum + workflow_type). Verbs: list / get /
                  find_workflow_by_url / create / update.

    execution  -- 12 tools. A single run of a workflow against datasets
                  and assets. Carries the lifecycle state machine.
                  Verbs: list / get / find_workflow_executions /
                  list_execution_children / list_execution_parents /
                  create / start / commit / update / abort /
                  create_execution_dataset / add_nested_execution.

    asset      -- 3 tools. File-backed catalog rows (images, model
                  weights, etc.) -- catalog-state operations only.
                  File I/O lives in deriva-skills's work-with-assets
                  skill. Verbs: list_assets / lookup / update.
                  For "what asset tables exist in a schema?" use the
                  ml/assets/{schema} resource (no dedicated tool).

READ-SIDE QUESTIONS: FETCH THE RESOURCE FIRST
---------------------------------------------
For READ-SIDE QUESTIONS ABOUT AN EXISTING ENTITY -- "show me X by RID",
"what's in Y", "what did Z produce / consume", "what's the current
version of W", "have we tried <thing>", "why was <choice> made" -- fetch
the matching ``deriva://catalog/{h}/{c}/ml/...`` resource BEFORE
reaching for ``deriva_ml_*`` tools or generic catalog CRUD
(``get_entities``, ``query_attribute``, ``list_foreign_keys``).

The URI is constructable from the catalog hostname + catalog id + entity
RID -- no tool search needed. The resources bundle the entity's summary
plus its associated children (a dataset's members and version, an
execution's inputs / outputs / metadata, an asset's producing-execution
chain) in a stable shape, while the equivalent tool path typically
takes 2-7 round trips of fetch + filter + join.

ANTI-PATTERN: do NOT call ``deriva_ml_get_execution`` (or
``deriva_ml_get_dataset``, etc.) when you already have the RID and want
the entity's full state. Those tools return a thinner record and force
follow-up ``deriva_ml_list_assets`` / ``query_attribute`` calls. The
``ml/<kind>/{rid}`` resource is the one-fetch answer.

The full read-only resource family:

    deriva://catalog/{h}/{c}/ml/datasets         -- all datasets, capped at 1000
    deriva://catalog/{h}/{c}/ml/dataset/{rid}    -- one dataset + version_history + members
    deriva://catalog/{h}/{c}/ml/dataset/{rid}/members  -- members grouped by table
    deriva://catalog/{h}/{c}/ml/dataset/{rid}/spec     -- DatasetSpecConfig snippet for hydra-zen configs
    deriva://catalog/{h}/{c}/ml/dataset/{rid}/bag-preview
                                                 -- bag size + table counts BEFORE downloading
                                                    (use to size a download before cache_dataset)
    deriva://catalog/{h}/{c}/ml/workflows        -- all workflows
    deriva://catalog/{h}/{c}/ml/workflow/{rid}   -- one workflow
    deriva://catalog/{h}/{c}/ml/executions       -- all executions
    deriva://catalog/{h}/{c}/ml/execution/{rid}  -- one execution: summary + inputs +
                                                    outputs (split into ``assets`` and
                                                    ``metadata``) + experiment
    deriva://catalog/{h}/{c}/ml/lineage/{rid}    -- provenance chain for any artifact
                                                    (Dataset, Asset, Feature value, Execution)
    deriva://catalog/{h}/{c}/ml/features/{table} -- features defined on one table
    deriva://catalog/{h}/{c}/ml/asset/{rid}      -- one asset + bundled executions
    deriva://catalog/{h}/{c}/ml/assets/{schema}  -- asset tables in one schema
    deriva://catalog/{h}/{c}/ml/assets/{schema}/{asset_table}
                                                 -- contents of one asset table
    deriva://catalog/{h}/{c}/ml/vocabularies/{schema}
                                                 -- vocabulary tables in one schema
    deriva://catalog/{h}/{c}/ml/vocabularies/{schema}/{vocab_name}
                                                 -- terms in one vocabulary table

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
    deriva://catalog/{h}/{c}/ml/datasets                   deriva_ml_list_datasets
    deriva://catalog/{h}/{c}/ml/dataset/{rid}              deriva_ml_get_dataset
    deriva://catalog/{h}/{c}/ml/dataset/{rid}/members      deriva_ml_list_dataset_members
    deriva://catalog/{h}/{c}/ml/dataset/{rid}/spec         deriva_ml_get_dataset_spec
    deriva://catalog/{h}/{c}/ml/dataset/{rid}/bag-preview  deriva_ml_bag_info
    deriva://catalog/{h}/{c}/ml/workflows                  deriva_ml_list_workflows
    deriva://catalog/{h}/{c}/ml/workflow/{rid}             deriva_ml_get_workflow
    deriva://catalog/{h}/{c}/ml/executions                 deriva_ml_list_executions
    deriva://catalog/{h}/{c}/ml/execution/{rid}            deriva_ml_get_execution
    deriva://catalog/{h}/{c}/ml/features/{table}           deriva_ml_list_features (table-scoped)
    deriva://catalog/{h}/{c}/ml/assets/{schema}            deriva_ml_list_assets (schema-scoped)
    deriva://catalog/{h}/{c}/ml/asset/{rid}                (no list tool -- one-RID only)
    deriva://catalog/{h}/{c}/ml/lineage/{rid}              deriva_ml_get_lineage
    deriva://catalog/{h}/{c}/ml/vocabularies/{schema}      list_vocabulary_terms (core; per-table)

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
   deriva_ml_list_workflows, deriva://catalog/{h}/{c}/ml/datasets).
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
    Indexed by this plugin per-user-per-RID. Fresh on first
    connect AND surgically refreshed on every mutation: each
    deriva_ml_create_* / deriva_ml_update_* / deriva_ml_delete_*
    / deriva_ml_commit_execution / etc. tool refreshes just the
    affected source(s) before returning. So a freshly created or
    modified row is searchable via rag_search on the very next
    call from the same user.

    Cross-user freshness is best-effort. User A's mutation does
    NOT refresh user B's per-user sources -- B sees A's change only
    at B's next first-connect to the catalog. Same gap applies to
    mutations from non-MCP clients (Chaise UI, ERMrest direct,
    other deriva-ml scripts): they don't propagate to your per-user
    sources at all.

    When working on multi-user collaborative catalogs, treat RAG
    hits for shared-visible rows as hints, NOT as ground truth. The
    recovery pattern: when an LLM hit feels stale, verify with the
    corresponding deriva_ml_get_<entity> tool, which always reads
    live catalog state. If you have reason to believe your view is
    significantly stale (e.g., a colleague just released a dataset),
    call deriva_ml_resync_indexes(hostname, catalog_id) first to
    refresh your per-user sources -- or pass target="<table>:<rid>"
    to refresh just one source surgically.

MUTATION: WORKFLOW -> EXECUTION -> OUTPUTS
------------------------------------------
The canonical mutation chain is:

    1. ``deriva_ml_create_workflow(...)`` (or reuse an existing one -- it dedups
       internally on URL+checksum; see that tool's docstring for the
       idempotency contract).
    2. ``deriva_ml_create_execution(workflow_rid=...)`` to register a new run.
    3. ``deriva_ml_start_execution(...)`` to advance Created -> Running.
    4. Write outputs (``deriva_ml_add_feature_values``, ``deriva_ml_create_execution_dataset``,
       etc.) -- they all attribute provenance to the running execution.
    5. ``deriva_ml_commit_execution(...)`` to drain staged outputs and transition
       to Uploaded.

ALWAYS create artefacts inside an execution context. Bare insertions
(via core's ``insert_records`` etc.) bypass provenance tracking and
should only be used when no domain-specific tool exists.

For the lifecycle state machine details, see each lifecycle tool's
docstring (``deriva_ml_start_execution``, ``deriva_ml_commit_execution``,
``deriva_ml_abort_execution``, ``deriva_ml_add_feature_values``) and
``rag_search`` for "execution state machine" -- the
``user-guide/executions.md`` doc in the deriva-ml repo carries the
cross-cutting state-machine reference at depth.

ASSETS: METADATA HERE, FILE I/O IN THE SKILL
--------------------------------------------
The asset surface covers the catalog-state half of the asset
lifecycle:

    deriva_ml_list_assets        -- rows in one asset table (paginated)
    deriva_ml_lookup_asset       -- bundled detail for one asset RID
                                    (filename, length, md5, url,
                                     description, asset_types, executions)
    deriva_ml_update_asset       -- mutate asset_type tags + description

Plus three matching resources:

    deriva://catalog/{h}/{c}/ml/asset/{rid}      -- bundled per-asset detail
    deriva://catalog/{h}/{c}/ml/assets/{schema}  -- asset tables in one schema
    deriva://catalog/{h}/{c}/ml/assets/{schema}/{asset_table}
                                                 -- snapshot of one asset table

What these tools do NOT do: register a new asset from a local file, or
download asset bytes back to a local path. The MCP server has no
general way to access the user's local filesystem, so file I/O is
deliberately out of scope here. For those two flows, use the
``work-with-assets`` skill in the deriva-skills plugin -- it generates
the Python the user runs locally (which talks to the catalog directly
via deriva-ml's ``execution.asset_file_path()`` for upload, and
``asset.download()`` for fetch).

The MCP <-> skill round trip looks like this:

    1. Inside an execution, the user wants to register a new file as an
       asset. Use the ``work-with-assets`` skill -- it produces a small
       Python snippet calling ``execution.asset_file_path(...)``,
       writing the file, and tying it to the running execution.
    2. The user runs the snippet locally; the file is staged and the
       asset row is created.
    3. Come back here for ``deriva_ml_commit_execution`` (uploads the
       staged file) and any ``deriva_ml_lookup_asset`` /
       ``deriva_ml_update_asset`` follow-ups.

CURATION PATTERN: ONE update_<entity> PER TYPED ENTITY
------------------------------------------------------
Every typed entity (Dataset, Workflow, Asset, Execution) has exactly
one ``deriva_ml_update_<entity>(rid, *fields)`` tool. Pass only the
kwargs you want to change; leave others as ``None``. At least one
field must be non-None or the tool returns ``{"error": ...}``.

    deriva_ml_update_dataset    (dataset_types? + description?)
    deriva_ml_update_workflow   (workflow_type? + description?)
    deriva_ml_update_asset      (asset_types? + description?)
    deriva_ml_update_execution  (description?)  -- description-only

For the type-list fields (``dataset_types``, ``workflow_type``,
``asset_types``): SET-STYLE. Pass the desired final list. The tool
fetches the current types and computes the diff -- terms in the new
list that aren't in the current set get added; terms in the current
set that aren't in the new list get removed; terms in both are left
alone.

For ``description``: free-form text overwrite of the catalog row's
``Description`` column.

Execution is the asymmetric one. There is no ``Execution_Type``
vocabulary, so no type-list field appears. ``Status`` edits remain
forbidden -- they are state-machine territory driven by
``deriva_ml_start_execution`` / ``deriva_ml_commit_execution`` /
``deriva_ml_abort_execution``, NOT freely editable. Free-form status
writes were rejected in v1.0 (they let an LLM drive the lifecycle
into invalid states).

v1.2 breaking rename. The pre-v1.2 ``deriva_ml_update_dataset_types``
tool (with ``add`` / ``remove`` kwargs) was renamed to
``deriva_ml_update_dataset`` and widened to take both ``dataset_types``
and ``description``. There is no compat shim; update any references.

THE MENU
--------
Quick orientation: 45 tools across 5 domains (dataset, feature, workflow,
execution, asset) + 11 read-only resources under the
``deriva://catalog/{h}/{c}/ml/...`` URI prefix + 1 GitHub doc source
indexed for RAG (``deriva-ml-docs``) + 3 per-user RAG indexes that
ingest Dataset / Workflow / Execution rows on first connect to a
catalog. Plus 4 built-in core prompts and 1 ML prompt (this one --
v3.x removed ``deriva_ml_execution_lifecycle`` and
``deriva_ml_workflow_dedup`` and redistributed their content to the
relevant tool docstrings; ``deriva_ml_concepts`` is the second
remaining ML prompt).
"""




# -- Registration ------------------------------------------------------------


def register(ctx: PluginContext) -> None:
    """Register the four deriva-ml MCP prompts with the plugin context.

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
            "Conceptual frame for any LLM client cold-starting on deriva-ml-mcp: "
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
            "Cold-start orientation for deriva-ml-mcp: the (hostname, catalog_id) rule, "
            "the four ML domains, discovery via resources/RAG, the workflow->execution->outputs chain. "
            "Assumes the conceptual frame from deriva_ml_concepts."
        ),
    )
    def deriva_ml_getting_started() -> str:
        return _GETTING_STARTED_GUIDE


