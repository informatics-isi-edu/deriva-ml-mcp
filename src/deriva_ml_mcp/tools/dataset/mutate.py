"""Simple-mutation dataset tools.

This submodule houses the 7 simple-mutation dataset tools:
``deriva_ml_create_dataset``, ``deriva_ml_delete_dataset``,
``deriva_ml_add_dataset_members``, ``deriva_ml_delete_dataset_members``,
``deriva_ml_update_dataset``, ``deriva_ml_add_dataset_element_type``,
and ``deriva_ml_release``.

These are the "one operation, one catalog mutation" tools -- each is
short, body fits in a screen, no multi-step orchestration. The
substantial ``deriva_ml_denormalize_dataset`` lives in ``complex.py``
to give its longer body breathing room. (``cache_dataset`` retired
in v0.5.0 per the stateless rule; ``split_dataset`` retired earlier
because it required a live ``Execution`` context.)

Every tool wraps DERIVA I/O in ``with deriva_call():``, emits
``audit_event(...)`` on success, and routes failures through
``_error_envelope`` (which emits the ``deriva_ml_<op>_failed`` audit row).

Audit event lookup goes through ``_pkg.audit_event(...)`` (attribute
lookup on the parent package) rather than a direct import, so the
package's single ``audit_event`` binding (set in ``__init__.py``) is the
one canonical patch site for both ``mutate`` and ``complex`` -- preserving
the pre-split behaviour where ``patch("deriva_ml_mcp.tools.dataset.audit_event")``
captured every success-path emission in one shot.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Literal

from deriva_mcp_core import deriva_call
from deriva_ml.dataset.aux_classes import DatasetVersion, VersionPart

logger = logging.getLogger(__name__)

# Note on patchable names (``audit_event``, ``get_ml``, ``Dataset``):
# the package ``__init__.py`` imports each of these at module level and
# the test fixtures patch them at ``deriva_ml_mcp.tools.dataset.<name>``.
# We deliberately access them via attribute lookup on the package
# (``_pkg.<name>``) rather than ``from ... import <name>`` so a single
# ``patch(...)`` redirects every call across read / mutate / complex
# submodules. ``from ... import`` here would create per-submodule
# bindings the patch can't reach. The failure path goes through
# ``_error_envelope`` in ``_helpers`` and still needs the second patch
# on ``deriva_ml_mcp._helpers.audit_event`` -- see
# ``make_patch_audit("dataset")`` in ``tests/_helpers.py`` (the
# canonical factory) for the dual-patch context manager.
import deriva_ml_mcp.tools.dataset as _pkg  # noqa: E402  (intentional cycle)
from deriva_ml_mcp._helpers import _error_envelope
from deriva_ml_mcp._response_models import (
    AddDatasetElementTypeResponse,
    AddDatasetMembersResponse,
    CreateDatasetResponse,
    DeleteDatasetMembersResponse,
    DeleteDatasetResponse,
    ReleaseDatasetResponse,
    UpdateDatasetResponse,
)
from deriva_ml_mcp.tools.dataset.read import _summarize_dataset

if TYPE_CHECKING:
    from deriva_mcp_core.plugin.api import PluginContext


def register(ctx: PluginContext) -> None:
    """Register all simple-mutation dataset tools with the plugin context.

    Args:
        ctx: PluginContext supplied by deriva-mcp-core at startup.

    Returns:
        None.

    Example:
        >>> from deriva_mcp_core.plugin.api import PluginContext
        >>> # ctx provided by the framework
        >>> register(ctx)  # doctest: +SKIP
    """

    @ctx.tool(mutates=True)
    async def deriva_ml_create_dataset(
        hostname: str,
        catalog_id: str,
        execution_rid: str,
        dataset_types: list[str] | None = None,
        description: str = "",
        version: str | None = None,
    ) -> str:
        """Register a new dataset under an execution for provenance tracking.

        Args:
            execution_rid: RID of the parent Execution. Required -- the new
                dataset's provenance is rooted in this execution.
            dataset_types: Optional list of dataset-type term names to tag
                the new dataset with.
            description: Free-text description of the dataset.
            version: Optional initial semver string (e.g. ``"1.0.0"``).
                If None, DerivaML defaults to ``0.1.0``.

        Returns:
            JSON string ``{"status": "created", "rid", "description",
            "dataset_types", "current_version", "execution_rid"}``.

        Raises:
            RuntimeError: Wrapped, propagated from
                ``deriva_ml.dataset.dataset.Dataset.create_dataset`` (e.g.
                invalid execution_rid, unknown dataset_type term).

        Example:
            ``{"status": "created", "rid": "1-NEW", "description": "...",
            "dataset_types": ["Training"], "current_version": "0.1.0",
            "execution_rid": "1-EXEC"}``.
        """
        try:
            parsed_version = DatasetVersion.parse(version) if version else None
            with deriva_call():
                # Run the synchronous deriva-ml calls in a thread pool so
                # the event loop stays responsive. See deriva-mcp-core
                # plugin-authoring-guide.md §"Synchronous work in threads".
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                new_ds = await asyncio.to_thread(
                    _pkg.Dataset.create_dataset,
                    ml,
                    execution_rid=execution_rid,
                    dataset_types=dataset_types,
                    description=description,
                    version=parsed_version,
                )
                summary = _summarize_dataset(new_ds)
            _pkg.audit_event(
                "deriva_ml_create_dataset",
                hostname=hostname,
                catalog_id=catalog_id,
                execution_rid=execution_rid,
                dataset_rid=new_ds.dataset_rid,
                dataset_types=dataset_types or [],
                version=summary.current_version,
            )
            # v1.3 surgical re-index: refresh just the new dataset's
            # per-RID source so rag_search picks it up on the next call
            # from this user. Best-effort -- a re-index hiccup must NOT
            # propagate to the tool's success path (the catalog mutation
            # already succeeded). Lazy import to avoid the resources/rag
            # <-> tools/dataset module-load cycle.
            try:
                from deriva_ml_mcp.resources.rag import _reindex_dataset

                await _reindex_dataset(hostname, catalog_id, new_ds.dataset_rid)
            except Exception:  # noqa: BLE001 -- best-effort cache refresh
                logger.exception(
                    "re-index failed for dataset %s after create_dataset",
                    new_ds.dataset_rid,
                )
            return CreateDatasetResponse(
                status="created",
                **summary.model_dump(),
                execution_rid=execution_rid,
            ).model_dump_json(by_alias=True)
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="create_dataset",
                hostname=hostname,
                catalog_id=catalog_id,
                execution_rid=execution_rid,
                dataset_types=dataset_types or [],
            )

    @ctx.tool(mutates=True)
    async def deriva_ml_delete_dataset(
        hostname: str,
        catalog_id: str,
        dataset_rid: str,
        recurse: bool = False,
    ) -> str:
        """Soft-delete a dataset (sets ``Deleted=True`` on the catalog row).

        With ``recurse=True``, also soft-deletes the dataset's nested
        children. Without ``recurse``, a dataset that has parents (i.e. is
        nested in another dataset) cannot be deleted; the upstream raises.

        Args:
            dataset_rid: The RID of the dataset to delete.
            recurse: If True, also soft-delete nested child datasets.

        Returns:
            JSON string ``{"status": "deleted", "dataset_rid", "recursive"}``.

        Raises:
            RuntimeError: Wrapped, propagated from
                ``deriva_ml.DerivaML.delete_dataset`` (e.g. dataset not
                found, dataset is nested and ``recurse=False``).

        Example:
            ``{"status": "deleted", "dataset_rid": "1-AAAA",
            "recursive": false}``.
        """
        try:
            with deriva_call():
                # Run the synchronous deriva-ml calls in a thread pool so
                # the event loop stays responsive. See deriva-mcp-core
                # plugin-authoring-guide.md §"Synchronous work in threads".
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                ds = await asyncio.to_thread(ml.lookup_dataset, dataset_rid)
                await asyncio.to_thread(ml.delete_dataset, ds, recurse=recurse)
            _pkg.audit_event(
                "deriva_ml_delete_dataset",
                hostname=hostname,
                catalog_id=catalog_id,
                dataset_rid=dataset_rid,
                recursive=recurse,
            )
            # v1.3 surgical re-index: drop the dataset's per-RID source so
            # rag_search no longer surfaces a row that's been soft-deleted.
            # Best-effort -- failure to drop is logged but does not affect
            # the tool's success envelope.
            try:
                from deriva_ml_mcp.resources.rag import _delete_dataset_source

                await _delete_dataset_source(hostname, catalog_id, dataset_rid)
            except Exception:  # noqa: BLE001 -- best-effort cache refresh
                logger.exception(
                    "re-index drop failed for dataset %s after delete_dataset",
                    dataset_rid,
                )
            return DeleteDatasetResponse(
                status="deleted",
                dataset_rid=dataset_rid,
                recursive=recurse,
            ).model_dump_json(by_alias=True)
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="delete_dataset",
                hostname=hostname,
                catalog_id=catalog_id,
                dataset_rid=dataset_rid,
                recursive=recurse,
            )

    @ctx.tool(mutates=True)
    async def deriva_ml_add_dataset_members(
        hostname: str,
        catalog_id: str,
        dataset_rid: str,
        member_rids: list[str] | None = None,
        members_by_table: dict[str, list[str]] | None = None,
        description: str = "",
        execution_rid: str | None = None,
    ) -> str:
        """Add records (or whole datasets) as members of a dataset.

        Provide EITHER ``member_rids`` (a mixed-table list -- each RID is
        resolved to its table) OR ``members_by_table`` (a table-keyed dict
        -- faster, no resolution needed). Exactly one of the two must be
        non-empty.

        Per ADR-0003 (deriva-ml 1.34+), adding members flips the
        dataset to a *dev* version (``<last_release>.post1.devN``),
        not a released version. To mint a stable, citable released
        label, call ``deriva_ml_release`` after the mutation. The
        returned ``new_version`` will be a dev label.

        Member tables must already be registered as dataset element
        types (see ``deriva_ml_add_dataset_element_type``).

        Args:
            dataset_rid: The RID of the dataset to add members to.
            member_rids: Mixed-table list of RIDs to add.
            members_by_table: Map of table name to list of RIDs.
            description: Free-text description recorded with the dev
                row's ``Description`` column. **Replaces** any prior
                dev-period description rather than appending.
            execution_rid: Optional execution to attribute the change to.

        Returns:
            JSON string ``{"status": "added", "added_count",
            "dataset_rid", "new_version"}``. ``new_version`` is a
            dev label; call ``deriva_ml_release`` to promote to a
            released label.

        Raises:
            ValueError: If neither or both of ``member_rids``/
                ``members_by_table`` are supplied.
            RuntimeError: Wrapped, propagated from
                ``deriva_ml.dataset.dataset.Dataset.add_dataset_members``
                (e.g. table not a registered element type, would create a
                cycle, duplicate members).

        Example:
            ``{"status": "added", "added_count": 3, "dataset_rid":
            "1-AAAA", "new_version": "0.4.0.post1.dev1"}``.
        """
        has_list = bool(member_rids)
        has_dict = bool(members_by_table)
        if has_list == has_dict:
            return json.dumps(
                {
                    "error": (
                        "Provide exactly one of member_rids or "
                        "members_by_table (and it must be non-empty)."
                    )
                }
            )
        members: list[str] | dict[str, list[str]]
        if has_list:
            members = list(member_rids or [])
            added_count = len(members)
        else:
            members = dict(members_by_table or {})
            added_count = sum(len(v) for v in members.values())
        try:
            with deriva_call():
                # Run the synchronous deriva-ml calls in a thread pool so
                # the event loop stays responsive. See deriva-mcp-core
                # plugin-authoring-guide.md §"Synchronous work in threads".
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                ds = await asyncio.to_thread(ml.lookup_dataset, dataset_rid)
                await asyncio.to_thread(
                    ds.add_dataset_members,
                    members=members,
                    description=description,
                    execution_rid=execution_rid,
                )
                new_version = str(ds.current_version) if ds.current_version is not None else None
            _pkg.audit_event(
                "deriva_ml_add_dataset_members",
                hostname=hostname,
                catalog_id=catalog_id,
                dataset_rid=dataset_rid,
                added_count=added_count,
                new_version=new_version,
            )
            # v1.3 surgical re-index: member-count + version changed.
            try:
                from deriva_ml_mcp.resources.rag import _reindex_dataset

                await _reindex_dataset(hostname, catalog_id, dataset_rid)
            except Exception:  # noqa: BLE001 -- best-effort cache refresh
                logger.exception(
                    "re-index failed for dataset %s after add_dataset_members",
                    dataset_rid,
                )
            return AddDatasetMembersResponse(
                status="added",
                added_count=added_count,
                dataset_rid=dataset_rid,
                new_version=new_version,
            ).model_dump_json(by_alias=True)
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="add_dataset_members",
                hostname=hostname,
                catalog_id=catalog_id,
                dataset_rid=dataset_rid,
                attempted_count=added_count,
            )

    @ctx.tool(mutates=True)
    async def deriva_ml_delete_dataset_members(
        hostname: str,
        catalog_id: str,
        dataset_rid: str,
        member_rids: list[str],
        description: str = "",
        execution_rid: str | None = None,
    ) -> str:
        """Remove records from a dataset.

        The records themselves are NOT deleted -- only the association
        rows linking them to the dataset are removed.

        Per ADR-0003 (deriva-ml 1.34+), removing members flips the
        dataset to a *dev* version (``<last_release>.post1.devN``),
        not a released version. To mint a stable, citable released
        label, call ``deriva_ml_release`` after the mutation. The
        returned ``new_version`` will be a dev label.

        Args:
            dataset_rid: The RID of the dataset to remove members from.
            member_rids: List of member RIDs to remove.
            description: Free-text description recorded with the dev
                row's ``Description`` column. **Replaces** any prior
                dev-period description rather than appending.
            execution_rid: Optional execution to attribute the change to.

        Returns:
            JSON string ``{"status": "removed", "removed_count",
            "dataset_rid", "new_version"}``. ``new_version`` is a
            dev label; call ``deriva_ml_release`` to promote to a
            released label.

        Raises:
            RuntimeError: Wrapped, propagated from
                ``deriva_ml.dataset.dataset.Dataset.delete_dataset_members``
                (e.g. RID not part of dataset).

        Example:
            ``{"status": "removed", "removed_count": 2, "dataset_rid":
            "1-AAAA", "new_version": "0.4.0.post1.dev2"}``.
        """
        removed_count = len(member_rids)
        try:
            with deriva_call():
                # Run the synchronous deriva-ml calls in a thread pool so
                # the event loop stays responsive. See deriva-mcp-core
                # plugin-authoring-guide.md §"Synchronous work in threads".
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                ds = await asyncio.to_thread(ml.lookup_dataset, dataset_rid)
                await asyncio.to_thread(
                    ds.delete_dataset_members,
                    members=member_rids,
                    description=description,
                    execution_rid=execution_rid,
                )
                new_version = str(ds.current_version) if ds.current_version is not None else None
            _pkg.audit_event(
                "deriva_ml_delete_dataset_members",
                hostname=hostname,
                catalog_id=catalog_id,
                dataset_rid=dataset_rid,
                removed_count=removed_count,
                new_version=new_version,
            )
            # v1.3 surgical re-index: member-count + version changed.
            try:
                from deriva_ml_mcp.resources.rag import _reindex_dataset

                await _reindex_dataset(hostname, catalog_id, dataset_rid)
            except Exception:  # noqa: BLE001 -- best-effort cache refresh
                logger.exception(
                    "re-index failed for dataset %s after delete_dataset_members",
                    dataset_rid,
                )
            return DeleteDatasetMembersResponse(
                status="removed",
                removed_count=removed_count,
                dataset_rid=dataset_rid,
                new_version=new_version,
            ).model_dump_json(by_alias=True)
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="delete_dataset_members",
                hostname=hostname,
                catalog_id=catalog_id,
                dataset_rid=dataset_rid,
                attempted_count=removed_count,
            )

    @ctx.tool(mutates=True)
    async def deriva_ml_update_dataset(
        hostname: str,
        catalog_id: str,
        dataset_rid: str,
        dataset_types: list[str] | None = None,
        description: str | None = None,
    ) -> str:
        """Update dataset metadata (types and/or description).

        Curation tool: pass only the fields you want to change. Both
        arguments are optional; at least one must be non-None.

        For ``dataset_types``: set-style. Pass the desired final list
        of Dataset_Type term names; the tool fetches the current types
        and computes the diff (add the terms in the new list that
        aren't in the current set; remove the terms in the current set
        that aren't in the new list). Adds happen before removes.

        For ``description``: free-form text overwrite of the dataset
        row's ``Description`` column. As of deriva-ml v1.38.0, the
        ``Dataset.description`` setter is write-through (assigning the
        property performs the catalog write and the in-memory mirror
        in one call); this tool delegates to that setter.

        v1.2 breaking rename. This tool used to be
        ``deriva_ml_update_dataset_types`` with separate ``add`` /
        ``remove`` kwargs. The new shape mirrors the curation pattern
        shared with ``update_workflow`` / ``update_asset`` /
        ``update_execution``: a single per-entity update tool with
        only-the-changed-fields semantics. Skill consumers (the
        ``deriva-skills`` plugin) must update references to the new
        name and signature.

        Args:
            dataset_rid: The RID of the dataset to update.
            dataset_types: Desired final list of Dataset_Type term
                names. ``None`` leaves types unchanged.
            description: New description text. ``None`` leaves the
                description unchanged.

        Returns:
            JSON string ``{"status": "updated", "dataset_rid",
            "updated_fields": [...], "dataset_types", "added",
            "removed", "new_version"}``. ``dataset_types`` /
            ``added`` / ``removed`` are present only when
            ``dataset_types`` was actually edited. ``new_version`` is
            the dataset's version after the update (the underlying
            type-mutation APIs auto-bump the minor version).

        Raises:
            RuntimeError: Wrapped, propagated from
                ``deriva_ml.dataset.dataset.Dataset.add_dataset_types``
                / ``remove_dataset_type`` (e.g. unknown vocabulary
                term) or the Description pathBuilder write (e.g.
                read-only catalog).

        Example:
            ``{"status": "updated", "dataset_rid": "1-AAAA",
            "updated_fields": ["dataset_types", "description"],
            "dataset_types": ["Training", "Validation"],
            "added": ["Validation"], "removed": [],
            "new_version": "1.4.0"}``.
        """
        # Argument validation -- return errors directly without audit.
        if dataset_types is None and description is None:
            return json.dumps(
                {"error": "at least one of dataset_types or description must be provided"}
            )

        # Track partial progress for the failure path: dataset_types is a
        # per-term loop on the remove half, so an LLM caller benefits
        # from seeing which terms landed before the failure.
        adds_done: list[str] = []
        removes_done: list[str] = []
        adds_requested: list[str] = []
        removes_requested: list[str] = []
        updated_fields: list[str] = []
        updated_types: list[str] = []
        new_version: str | None = None
        try:
            with deriva_call():
                # Run the synchronous deriva-ml calls in a thread pool so
                # the event loop stays responsive. See deriva-mcp-core
                # plugin-authoring-guide.md §"Synchronous work in threads".
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                ds = await asyncio.to_thread(ml.lookup_dataset, dataset_rid)

                if dataset_types is not None:
                    desired = set(dataset_types)
                    current = set(ds.dataset_types or [])
                    adds_requested = sorted(desired - current)
                    removes_requested = sorted(current - desired)
                    if adds_requested:
                        await asyncio.to_thread(ds.add_dataset_types, adds_requested)
                        adds_done = list(adds_requested)
                    for term in removes_requested:
                        await asyncio.to_thread(ds.remove_dataset_type, term)
                        removes_done.append(term)
                    updated_types = list(ds.dataset_types) if ds.dataset_types else []
                    updated_fields.append("dataset_types")

                if description is not None:
                    # deriva-ml v1.38.0+ exposes a write-through setter:
                    # assigning ds.description performs the catalog write
                    # AND the in-memory mirror in one call. Pre-v1.38 the
                    # plugin had a _set_row_description pathBuilder
                    # workaround here; that helper was retired in v0.5.0.
                    # The setter is sync I/O -- thread it.
                    def _set_dataset_description() -> None:
                        ds.description = description

                    await asyncio.to_thread(_set_dataset_description)
                    updated_fields.append("description")

                new_version = str(ds.current_version) if ds.current_version is not None else None

            _pkg.audit_event(
                "deriva_ml_update_dataset",
                hostname=hostname,
                catalog_id=catalog_id,
                dataset_rid=dataset_rid,
                updated_fields=updated_fields,
                added=adds_requested,
                removed=removes_requested,
                new_version=new_version,
            )
            # v1.3 surgical re-index: types and/or description changed.
            try:
                from deriva_ml_mcp.resources.rag import _reindex_dataset

                await _reindex_dataset(hostname, catalog_id, dataset_rid)
            except Exception:  # noqa: BLE001 -- best-effort cache refresh
                logger.exception(
                    "re-index failed for dataset %s after update_dataset",
                    dataset_rid,
                )
            # v3.0: dataset_types / added / removed are ALWAYS in the response
            # (null when the description-only branch ran). Was conditionally
            # omitted in v2.x.
            if "dataset_types" in updated_fields:
                response = UpdateDatasetResponse(
                    status="updated",
                    dataset_rid=dataset_rid,
                    updated_fields=updated_fields,
                    new_version=new_version,
                    dataset_types=updated_types,
                    added=adds_requested,
                    removed=removes_requested,
                )
            else:
                response = UpdateDatasetResponse(
                    status="updated",
                    dataset_rid=dataset_rid,
                    updated_fields=updated_fields,
                    new_version=new_version,
                )
            return response.model_dump_json(by_alias=True)
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="update_dataset",
                hostname=hostname,
                catalog_id=catalog_id,
                dataset_rid=dataset_rid,
                updated_fields=updated_fields,
                added=adds_requested,
                removed=removes_requested,
                added_done=adds_done,
                removed_done=removes_done,
                response_fields={
                    "dataset_rid": dataset_rid,
                    "updated_fields": updated_fields,
                    "added_done": adds_done,
                    "removed_done": removes_done,
                    "added_requested": adds_requested,
                    "removed_requested": removes_requested,
                },
            )

    @ctx.tool(mutates=True)
    async def deriva_ml_add_dataset_element_type(
        hostname: str,
        catalog_id: str,
        table_name: str,
    ) -> str:
        """Register a domain table as a valid dataset member type.

        Creates the association table linking the dataset table to
        ``table_name``. The caller must have catalog admin privileges.

        Args:
            table_name: Name of the domain table to register.

        Returns:
            JSON string ``{"status": "created", "table_name",
            "association_table"}``.

        Raises:
            RuntimeError: Wrapped, propagated from
                ``deriva_ml.DerivaML.add_dataset_element_type`` (e.g.
                unknown table, table is system/ML and not eligible).

        Example:
            ``{"status": "created", "table_name": "Image",
            "association_table": "Dataset_Image"}``.
        """
        try:
            with deriva_call():
                # Run the synchronous deriva-ml calls in a thread pool so
                # the event loop stays responsive. See deriva-mcp-core
                # plugin-authoring-guide.md §"Synchronous work in threads".
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                assoc_table = await asyncio.to_thread(ml.add_dataset_element_type, table_name)
                association_name = assoc_table.name
            _pkg.audit_event(
                "deriva_ml_add_dataset_element_type",
                hostname=hostname,
                catalog_id=catalog_id,
                table_name=table_name,
                association_table=association_name,
            )
            return AddDatasetElementTypeResponse(
                status="created",
                table_name=table_name,
                association_table=association_name,
            ).model_dump_json(by_alias=True)
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="add_dataset_element_type",
                hostname=hostname,
                catalog_id=catalog_id,
                table_name=table_name,
            )

    @ctx.tool(mutates=True)
    async def deriva_ml_release(
        hostname: str,
        catalog_id: str,
        dataset_rid: str,
        bump: Literal["major", "minor", "patch"] = "minor",
        description: str = "",
        execution_rid: str | None = None,
    ) -> str:
        """Promote a dataset's dev period to a released version.

        Per ADR-0003 (deriva-ml 1.34+), ``release`` is the only operation
        that produces a released ``Dataset_Version`` row. Member-mutation
        operations (``deriva_ml_add_dataset_members``,
        ``deriva_ml_delete_dataset_members``) flip the dataset to a dev
        version (``<last_release>.post1.devN``); call this tool afterward
        to mint a stable, citable released version.

        Promotes the existing dev row in place: rewrites ``Version`` to
        the released label, stamps ``Snapshot`` with the catalog snapshot
        at release time, replaces ``Description`` with release notes, and
        sets the row's ``Execution`` link to the supplied execution RID
        (or ``NULL`` if none). Bumping a higher-order release segment
        resets lower-order segments to zero.

        Args:
            dataset_rid: The RID of the dataset to release.
            bump: Which release segment to advance from the last released
                version (``"minor"`` is the common case;
                ``"major"`` for schema-breaking changes;
                ``"patch"`` for small clean-ups).
            description: Release notes. **Replaces** the dev row's
                accumulated description, not appended.
            execution_rid: Optional execution RID to attribute the
                release to. Stored on the released row's ``Execution``
                link.

        Returns:
            JSON string ``{"status": "released", "dataset_rid",
            "previous_version", "new_version", "bump"}``.
            ``previous_version`` is the dev label that was promoted
            (e.g. ``"0.4.0.post1.dev3"``); ``new_version`` is the
            released label (e.g. ``"0.5.0"``).

        Raises:
            RuntimeError: Wrapped, propagated from
                ``deriva_ml.dataset.dataset.Dataset.release``. Most
                common: the dataset has no dev period to promote (call
                ``deriva_ml_add_dataset_members`` first, or call the
                ``deriva_ml.Dataset.mark_dev`` Python API to declare a
                dev period for a no-op release).

        Example:
            ``{"status": "released", "dataset_rid": "1-AAAA",
            "previous_version": "0.4.0.post1.dev3",
            "new_version": "0.5.0", "bump": "minor"}``.
        """
        try:
            version_part = VersionPart(bump)
            with deriva_call():
                # Run the synchronous deriva-ml calls in a thread pool so
                # the event loop stays responsive. See deriva-mcp-core
                # plugin-authoring-guide.md §"Synchronous work in threads".
                ml = await asyncio.to_thread(_pkg.get_ml, hostname, catalog_id)
                ds = await asyncio.to_thread(ml.lookup_dataset, dataset_rid)
                previous_version = (
                    str(ds.current_version) if ds.current_version is not None else None
                )

                # The deriva-ml Dataset.release(execution=...) takes an
                # Execution object, not a RID. Resolve to the typed
                # object only when an execution_rid was supplied.
                execution_obj = None
                if execution_rid is not None:
                    execution_obj = await asyncio.to_thread(
                        ml.lookup_execution, execution_rid
                    )

                new_version = await asyncio.to_thread(
                    ds.release,
                    bump=version_part,
                    description=description,
                    execution=execution_obj,
                )
                new_version_str = str(new_version)
            _pkg.audit_event(
                "deriva_ml_release",
                hostname=hostname,
                catalog_id=catalog_id,
                dataset_rid=dataset_rid,
                bump=bump,
                previous_version=previous_version,
                new_version=new_version_str,
            )
            # v1.3 surgical re-index: version field in the chunk changed.
            try:
                from deriva_ml_mcp.resources.rag import _reindex_dataset

                await _reindex_dataset(hostname, catalog_id, dataset_rid)
            except Exception:  # noqa: BLE001 -- best-effort cache refresh
                logger.exception(
                    "re-index failed for dataset %s after release",
                    dataset_rid,
                )
            return ReleaseDatasetResponse(
                status="released",
                dataset_rid=dataset_rid,
                previous_version=previous_version,
                new_version=new_version_str,
                bump=bump,
            ).model_dump_json(by_alias=True)
        except Exception as exc:
            return _error_envelope(
                exc,
                operation="release",
                hostname=hostname,
                catalog_id=catalog_id,
                dataset_rid=dataset_rid,
                bump=bump,
            )
