"""Unit tests for mutating dataset tools (mirrors ``tools/dataset/mutate.py``).

Covers ``deriva_ml_create_dataset``, ``deriva_ml_delete_dataset``,
``deriva_ml_add_dataset_members``, ``deriva_ml_delete_dataset_members``,
``deriva_ml_update_dataset``, ``deriva_ml_add_dataset_element_type``,
``deriva_ml_release``, plus the v1.3 surgical
RAG re-index wiring on the create / delete paths.

The split mirrors the source-side ``tools/dataset/{read,mutate,complex}.py``
package and lets the mutate tests stay scrolling-distance from the
mutate tools they exercise. Shared fixtures (``_make_dataset_mock``,
``_patch_audit``) live in ``tests/_dataset_helpers.py``.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from tests._dataset_helpers import _make_dataset_mock, _patch_audit
from tests._helpers import _success_calls


@pytest.fixture()
def dataset_ctx(ctx, mock_ml):
    """Register dataset tools with mock_ml as the DerivaML stand-in.

    Patches at the use-site (``deriva_ml_mcp.tools.dataset.get_ml``) and
    imports the tool module *inside* the patch block so registration
    sees the mock.
    """
    with patch("deriva_ml_mcp.tools.dataset.get_ml", return_value=mock_ml):
        from deriva_ml_mcp.tools import dataset as dataset_module

        dataset_module.register(ctx)
        yield ctx


# ---------------------------------------------------------------------------
# create_dataset
# ---------------------------------------------------------------------------


async def test_create_dataset_success(dataset_ctx, capturing_mcp, mock_ml):
    """Happy path: returns the new dataset summary and emits success audit."""
    new_ds = _make_dataset_mock("1-NEW", "desc", ["Training"], "1.0.0")
    with (
        patch("deriva_ml_mcp.tools.dataset.Dataset") as mock_dataset_cls,
        _patch_audit() as mock_audit,
    ):
        mock_dataset_cls.create_dataset.return_value = new_ds
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_create_dataset"](
                hostname="h",
                catalog_id="1",
                execution_rid="EXEC-1",
                dataset_types=["Training"],
                description="desc",
            )
        )
    assert out["status"] == "created"
    assert out["rid"] == "1-NEW"
    assert out["description"] == "desc"
    assert out["dataset_types"] == ["Training"]
    assert out["current_version"] == "1.0.0"
    assert out["execution_rid"] == "EXEC-1"

    success = _success_calls(mock_audit, "deriva_ml_create_dataset")
    assert success, "expected deriva_ml_create_dataset audit event"
    assert success[0].kwargs["execution_rid"] == "EXEC-1"
    assert success[0].kwargs["dataset_rid"] == "1-NEW"
    assert success[0].kwargs["dataset_types"] == ["Training"]


async def test_create_dataset_parses_explicit_version(dataset_ctx, capturing_mcp, mock_ml):
    """A version string is parsed into a DatasetVersion before being passed in."""
    new_ds = _make_dataset_mock("1-NEW", current_version="2.5.7")
    with (
        patch("deriva_ml_mcp.tools.dataset.Dataset") as mock_dataset_cls,
        patch("deriva_ml_mcp.tools.dataset.audit_event"),
    ):
        mock_dataset_cls.create_dataset.return_value = new_ds
        await capturing_mcp.tools["deriva_ml_create_dataset"](
            hostname="h",
            catalog_id="1",
            execution_rid="EXEC-1",
            version="2.5.7",
        )

    passed_version = mock_dataset_cls.create_dataset.call_args.kwargs["version"]
    # DatasetVersion subclasses semver.Version; verify the components.
    assert (passed_version.major, passed_version.minor, passed_version.patch) == (2, 5, 7)


async def test_create_dataset_failure_emits_failed_audit(dataset_ctx, capturing_mcp, mock_ml):
    """Error path: audit fires `_failed`, response has {error: ...}."""
    with (
        patch("deriva_ml_mcp.tools.dataset.Dataset") as mock_dataset_cls,
        _patch_audit() as mock_audit,
    ):
        mock_dataset_cls.create_dataset.side_effect = RuntimeError("invalid term")
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_create_dataset"](
                hostname="h",
                catalog_id="1",
                execution_rid="EXEC-1",
                dataset_types=["BadTerm"],
            )
        )
    assert out == {"error": "invalid term"}

    failed = _success_calls(mock_audit, "deriva_ml_create_dataset_failed")
    assert failed, "expected deriva_ml_create_dataset_failed audit event"
    assert failed[0].kwargs["error_type"] == "RuntimeError"
    assert failed[0].kwargs["execution_rid"] == "EXEC-1"
    assert failed[0].kwargs["dataset_types"] == ["BadTerm"]
    # No success event was emitted.
    assert not _success_calls(mock_audit, "deriva_ml_create_dataset")


# ---------------------------------------------------------------------------
# delete_dataset
# ---------------------------------------------------------------------------


async def test_delete_dataset_success(dataset_ctx, capturing_mcp, mock_ml):
    ds = _make_dataset_mock("1-AAAA")
    mock_ml.lookup_dataset.return_value = ds
    with _patch_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_delete_dataset"](
                hostname="h", catalog_id="1", dataset_rid="1-AAAA"
            )
        )
    assert out == {"status": "deleted", "dataset_rid": "1-AAAA", "recursive": False}
    mock_ml.delete_dataset.assert_called_once_with(ds, recurse=False)

    success = _success_calls(mock_audit, "deriva_ml_delete_dataset")
    assert success
    assert success[0].kwargs["dataset_rid"] == "1-AAAA"
    assert success[0].kwargs["recursive"] is False


async def test_delete_dataset_recursive(dataset_ctx, capturing_mcp, mock_ml):
    """recurse=True is forwarded to ml.delete_dataset and recorded in audit."""
    ds = _make_dataset_mock("1-AAAA")
    mock_ml.lookup_dataset.return_value = ds
    with _patch_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_delete_dataset"](
                hostname="h", catalog_id="1", dataset_rid="1-AAAA", recurse=True
            )
        )
    assert out["recursive"] is True
    mock_ml.delete_dataset.assert_called_once_with(ds, recurse=True)
    success = _success_calls(mock_audit, "deriva_ml_delete_dataset")
    assert success[0].kwargs["recursive"] is True


async def test_delete_dataset_error_path(dataset_ctx, capturing_mcp, mock_ml):
    mock_ml.lookup_dataset.side_effect = RuntimeError("not found")
    with _patch_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_delete_dataset"](
                hostname="h", catalog_id="1", dataset_rid="missing"
            )
        )
    assert out == {"error": "not found"}
    failed = _success_calls(mock_audit, "deriva_ml_delete_dataset_failed")
    assert failed
    assert failed[0].kwargs["error_type"] == "RuntimeError"
    assert failed[0].kwargs["dataset_rid"] == "missing"


# ---------------------------------------------------------------------------
# add_dataset_members
# ---------------------------------------------------------------------------


async def test_add_dataset_members_with_list(dataset_ctx, capturing_mcp, mock_ml):
    ds = _make_dataset_mock("1-AAAA", current_version="1.1.0")
    mock_ml.lookup_dataset.return_value = ds
    with _patch_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_add_dataset_members"](
                hostname="h",
                catalog_id="1",
                dataset_rid="1-AAAA",
                member_rids=["r1", "r2", "r3"],
                description="adding three",
                execution_rid="EXEC-1",
            )
        )
    assert out["status"] == "added"
    assert out["added_count"] == 3
    assert out["dataset_rid"] == "1-AAAA"
    assert out["new_version"] == "1.1.0"
    ds.add_dataset_members.assert_called_once_with(
        members=["r1", "r2", "r3"], description="adding three", execution_rid="EXEC-1"
    )
    success = _success_calls(mock_audit, "deriva_ml_add_dataset_members")
    assert success
    assert success[0].kwargs["added_count"] == 3
    assert success[0].kwargs["new_version"] == "1.1.0"


async def test_add_dataset_members_with_dict(dataset_ctx, capturing_mcp, mock_ml):
    ds = _make_dataset_mock("1-AAAA", current_version="1.2.0")
    mock_ml.lookup_dataset.return_value = ds
    with patch("deriva_ml_mcp.tools.dataset.audit_event"):
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_add_dataset_members"](
                hostname="h",
                catalog_id="1",
                dataset_rid="1-AAAA",
                members_by_table={"Image": ["i1", "i2"], "Subject": ["s1"]},
            )
        )
    assert out["added_count"] == 3
    forwarded = ds.add_dataset_members.call_args.kwargs["members"]
    assert forwarded == {"Image": ["i1", "i2"], "Subject": ["s1"]}


async def test_add_dataset_members_validates_inputs(dataset_ctx, capturing_mcp, mock_ml):
    """Both args missing -> early ValueError-style {error: ...}, no audit call."""
    with _patch_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_add_dataset_members"](
                hostname="h", catalog_id="1", dataset_rid="1-AAAA"
            )
        )
    assert "error" in out
    assert "exactly one" in out["error"]
    assert mock_audit.call_count == 0
    mock_ml.lookup_dataset.assert_not_called()


async def test_add_dataset_members_both_args_rejected(dataset_ctx, capturing_mcp, mock_ml):
    """Supplying both member_rids and members_by_table is rejected."""
    out = json.loads(
        await capturing_mcp.tools["deriva_ml_add_dataset_members"](
            hostname="h",
            catalog_id="1",
            dataset_rid="1-AAAA",
            member_rids=["r1"],
            members_by_table={"Image": ["i1"]},
        )
    )
    assert "error" in out


async def test_add_dataset_members_error_path(dataset_ctx, capturing_mcp, mock_ml):
    ds = _make_dataset_mock("1-AAAA")
    ds.add_dataset_members.side_effect = RuntimeError("cycle detected")
    mock_ml.lookup_dataset.return_value = ds
    with _patch_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_add_dataset_members"](
                hostname="h",
                catalog_id="1",
                dataset_rid="1-AAAA",
                member_rids=["r1"],
            )
        )
    assert out == {"error": "cycle detected"}
    failed = _success_calls(mock_audit, "deriva_ml_add_dataset_members_failed")
    assert failed
    assert failed[0].kwargs["attempted_count"] == 1


# ---------------------------------------------------------------------------
# delete_dataset_members
# ---------------------------------------------------------------------------


async def test_delete_dataset_members_success(dataset_ctx, capturing_mcp, mock_ml):
    ds = _make_dataset_mock("1-AAAA", current_version="1.3.0")
    mock_ml.lookup_dataset.return_value = ds
    with _patch_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_delete_dataset_members"](
                hostname="h",
                catalog_id="1",
                dataset_rid="1-AAAA",
                member_rids=["r1", "r2"],
                description="trimming",
                execution_rid="EXEC-2",
            )
        )
    assert out["status"] == "removed"
    assert out["removed_count"] == 2
    assert out["new_version"] == "1.3.0"
    ds.delete_dataset_members.assert_called_once_with(
        members=["r1", "r2"], description="trimming", execution_rid="EXEC-2"
    )
    success = _success_calls(mock_audit, "deriva_ml_delete_dataset_members")
    assert success
    assert success[0].kwargs["removed_count"] == 2


async def test_delete_dataset_members_error(dataset_ctx, capturing_mcp, mock_ml):
    ds = _make_dataset_mock("1-AAAA")
    ds.delete_dataset_members.side_effect = RuntimeError("rid not in dataset")
    mock_ml.lookup_dataset.return_value = ds
    with _patch_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_delete_dataset_members"](
                hostname="h",
                catalog_id="1",
                dataset_rid="1-AAAA",
                member_rids=["r1", "r2", "r3"],
            )
        )
    assert out == {"error": "rid not in dataset"}
    failed = _success_calls(mock_audit, "deriva_ml_delete_dataset_members_failed")
    assert failed
    assert failed[0].kwargs["attempted_count"] == 3


# ---------------------------------------------------------------------------
# update_dataset (v1.2 -- renamed from update_dataset_types and widened to
# take optional `dataset_types` (set-style diff) + optional `description`.)
# ---------------------------------------------------------------------------


async def test_update_dataset_set_diff_adds_and_removes(dataset_ctx, capturing_mcp, mock_ml):
    """`dataset_types` is set-style: diff against current types, add+remove."""
    ds = _make_dataset_mock("1-AAAA", dataset_types=["Training", "Stale"], current_version="1.4.0")
    mock_ml.lookup_dataset.return_value = ds
    with _patch_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_update_dataset"](
                hostname="h",
                catalog_id="1",
                dataset_rid="1-AAAA",
                dataset_types=["Training", "Validation"],
            )
        )
    assert out["status"] == "updated"
    assert out["updated_fields"] == ["dataset_types"]
    assert out["added"] == ["Validation"]
    assert out["removed"] == ["Stale"]
    assert out["new_version"] == "1.4.0"
    # add_dataset_types is called with the diff-add list (not the desired
    # final list) -- Training is already present and not re-added.
    ds.add_dataset_types.assert_called_once_with(["Validation"])
    ds.remove_dataset_type.assert_called_once_with("Stale")
    success = _success_calls(mock_audit, "deriva_ml_update_dataset")
    assert success
    assert success[0].kwargs["added"] == ["Validation"]
    assert success[0].kwargs["removed"] == ["Stale"]


async def test_update_dataset_no_diff_skips_calls(dataset_ctx, capturing_mcp, mock_ml):
    """If desired == current, neither add_dataset_types nor remove fire."""
    ds = _make_dataset_mock("1-AAAA", dataset_types=["Training"])
    mock_ml.lookup_dataset.return_value = ds
    with patch("deriva_ml_mcp.tools.dataset.audit_event"):
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_update_dataset"](
                hostname="h",
                catalog_id="1",
                dataset_rid="1-AAAA",
                dataset_types=["Training"],
            )
        )
    assert out["status"] == "updated"
    assert out["added"] == []
    assert out["removed"] == []
    ds.add_dataset_types.assert_not_called()
    ds.remove_dataset_type.assert_not_called()


async def test_update_dataset_remove_only_loops_per_term(dataset_ctx, capturing_mcp, mock_ml):
    """Diff-remove of multiple terms loops through remove_dataset_type."""
    ds = _make_dataset_mock("1-AAAA", dataset_types=["Keep", "A", "B", "C"])
    mock_ml.lookup_dataset.return_value = ds
    with patch("deriva_ml_mcp.tools.dataset.audit_event"):
        await capturing_mcp.tools["deriva_ml_update_dataset"](
            hostname="h",
            catalog_id="1",
            dataset_rid="1-AAAA",
            dataset_types=["Keep"],
        )
    # add_dataset_types should not be called when no diff-add.
    ds.add_dataset_types.assert_not_called()
    assert ds.remove_dataset_type.call_count == 3
    # Removes happen in sorted order (the diff is computed via set
    # arithmetic and then sorted for determinism).
    assert [c.args[0] for c in ds.remove_dataset_type.call_args_list] == ["A", "B", "C"]


async def test_update_dataset_validation_both_none(dataset_ctx, capturing_mcp, mock_ml):
    """update_dataset() with no fields returns the validation error envelope."""
    with _patch_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_update_dataset"](
                hostname="h", catalog_id="1", dataset_rid="1-AAAA"
            )
        )
    assert "error" in out
    assert "at least one" in out["error"]
    # No catalog work and no audit on validation failure.
    mock_ml.lookup_dataset.assert_not_called()
    assert mock_audit.call_count == 0


async def test_update_dataset_error(dataset_ctx, capturing_mcp, mock_ml):
    ds = _make_dataset_mock("1-AAAA")
    ds.add_dataset_types.side_effect = RuntimeError("unknown term")
    mock_ml.lookup_dataset.return_value = ds
    with _patch_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_update_dataset"](
                hostname="h",
                catalog_id="1",
                dataset_rid="1-AAAA",
                dataset_types=["BadTerm"],
            )
        )
    # Error response now includes partial-state fields so the LLM can
    # see which sub-operations completed before the failure.
    assert out["error"] == "unknown term"
    assert out["dataset_rid"] == "1-AAAA"
    assert out["added_done"] == []  # add_dataset_types raised; nothing committed
    assert out["removed_done"] == []
    assert out["added_requested"] == ["BadTerm"]
    failed = _success_calls(mock_audit, "deriva_ml_update_dataset_failed")
    assert failed
    assert failed[0].kwargs["added"] == ["BadTerm"]
    assert failed[0].kwargs["added_done"] == []
    assert failed[0].kwargs["removed_done"] == []


async def test_update_dataset_partial_remove_failure_surfaces_progress(
    dataset_ctx, capturing_mcp, mock_ml
):
    """If a remove mid-way through the loop fails, the error response and
    audit must surface which terms were already removed."""
    # Current types include the two we're going to remove (A, B) plus
    # one we're keeping ("Keep") -- since dataset_types is set-style,
    # passing ["Keep", "NewTag"] diffs to add=[NewTag], remove=[A, B].
    ds = _make_dataset_mock("1-AAAA", dataset_types=["Keep", "A", "B"])
    # add_dataset_types succeeds; first remove succeeds; second remove fails.
    call_count = {"n": 0}

    def remove_side_effect(term):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("term B not found")

    ds.remove_dataset_type.side_effect = remove_side_effect
    mock_ml.lookup_dataset.return_value = ds

    with _patch_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_update_dataset"](
                hostname="h",
                catalog_id="1",
                dataset_rid="1-AAAA",
                dataset_types=["Keep", "NewTag"],
            )
        )

    # Error response shows partial state: NewTag added, A removed, B failed.
    assert out["error"] == "term B not found"
    assert out["added_done"] == ["NewTag"]
    assert out["removed_done"] == ["A"]
    assert out["added_requested"] == ["NewTag"]
    assert out["removed_requested"] == ["A", "B"]

    # Audit also captures partial state.
    failed = _success_calls(mock_audit, "deriva_ml_update_dataset_failed")
    assert failed
    assert failed[0].kwargs["added_done"] == ["NewTag"]
    assert failed[0].kwargs["removed_done"] == ["A"]


async def test_update_dataset_description_only(dataset_ctx, capturing_mcp, mock_ml):
    """description-only update writes via the deriva-ml v1.38.0 setter; types untouched.

    v4.0.x: the pathBuilder workaround (_set_row_description) was retired
    when deriva-ml introduced the write-through Dataset.description
    setter. The test now uses PropertyMock to verify the setter fires
    instead of mocking pathBuilder.
    """
    from unittest.mock import PropertyMock

    ds = _make_dataset_mock("1-AAAA", dataset_types=["Training"], current_version="2.0.0")
    mock_ml.lookup_dataset.return_value = ds
    # Replace the plain attribute with a PropertyMock so we can detect
    # setter invocation. PropertyMock is attached to the TYPE, not the
    # instance, so build a fresh type that we mutate without touching
    # MagicMock's class globally.
    description_prop = PropertyMock(return_value="initial")
    type(ds).description = description_prop

    with _patch_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_update_dataset"](
                hostname="h",
                catalog_id="1",
                dataset_rid="1-AAAA",
                description="new desc",
            )
        )
    assert out["status"] == "updated"
    assert out["updated_fields"] == ["description"]
    # types fields are always present in v3.0 (null when description-only branch ran).
    assert out["dataset_types"] is None
    assert out["added"] is None
    assert out["removed"] is None
    # PropertyMock records every set; the setter was called with the new value.
    description_prop.assert_any_call("new desc")
    # No type-mutation API was touched.
    ds.add_dataset_types.assert_not_called()
    ds.remove_dataset_type.assert_not_called()
    success = _success_calls(mock_audit, "deriva_ml_update_dataset")
    assert success
    assert success[0].kwargs["updated_fields"] == ["description"]
    # Clean up the class-level PropertyMock so it doesn't leak to siblings.
    del type(ds).description


async def test_update_dataset_types_and_description_in_one_call(
    dataset_ctx, capturing_mcp, mock_ml
):
    """Passing both fields edits both; updated_fields lists both.

    Description-only branch uses the v1.38.0 write-through setter
    (verified via PropertyMock); the types branch uses add_dataset_types
    as before.
    """
    from unittest.mock import PropertyMock

    ds = _make_dataset_mock("1-AAAA", dataset_types=["Training"], current_version="1.0.0")
    mock_ml.lookup_dataset.return_value = ds
    description_prop = PropertyMock(return_value="initial")
    type(ds).description = description_prop

    with patch("deriva_ml_mcp.tools.dataset.audit_event"):
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_update_dataset"](
                hostname="h",
                catalog_id="1",
                dataset_rid="1-AAAA",
                dataset_types=["Training", "Validation"],
                description="combined edit",
            )
        )
    assert out["status"] == "updated"
    assert sorted(out["updated_fields"]) == ["dataset_types", "description"]
    # Diff add ran for the new term; description setter also fired.
    ds.add_dataset_types.assert_called_once_with(["Validation"])
    description_prop.assert_any_call("combined edit")
    del type(ds).description


# ---------------------------------------------------------------------------
# add_dataset_element_type
# ---------------------------------------------------------------------------


async def test_add_dataset_element_type_success(dataset_ctx, capturing_mcp, mock_ml):
    assoc = MagicMock()
    assoc.name = "Dataset_Image"
    mock_ml.add_dataset_element_type.return_value = assoc
    with _patch_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_add_dataset_element_type"](
                hostname="h", catalog_id="1", table_name="Image"
            )
        )
    assert out == {
        "status": "created",
        "table_name": "Image",
        "association_table": "Dataset_Image",
    }
    mock_ml.add_dataset_element_type.assert_called_once_with("Image")
    success = _success_calls(mock_audit, "deriva_ml_add_dataset_element_type")
    assert success
    assert success[0].kwargs["table_name"] == "Image"
    assert success[0].kwargs["association_table"] == "Dataset_Image"


async def test_add_dataset_element_type_error(dataset_ctx, capturing_mcp, mock_ml):
    mock_ml.add_dataset_element_type.side_effect = RuntimeError("not a domain table")
    with _patch_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_add_dataset_element_type"](
                hostname="h", catalog_id="1", table_name="Execution"
            )
        )
    assert out == {"error": "not a domain table"}
    failed = _success_calls(mock_audit, "deriva_ml_add_dataset_element_type_failed")
    assert failed
    assert failed[0].kwargs["table_name"] == "Execution"
    assert failed[0].kwargs["error_type"] == "RuntimeError"


# ---------------------------------------------------------------------------
# release (replaces the old increment_dataset_version per ADR-0003)
# ---------------------------------------------------------------------------


async def test_release_success(dataset_ctx, capturing_mcp, mock_ml):
    """Successful release: dev label -> released label, execution resolved to object."""
    ds = _make_dataset_mock("1-AAAA", current_version="0.4.0.post1.dev3")
    new_ver = MagicMock()
    new_ver.__str__ = lambda self: "0.5.0"
    ds.release.return_value = new_ver
    mock_ml.lookup_dataset.return_value = ds

    # The MCP tool resolves execution_rid -> Execution object via
    # ml.lookup_execution before calling Dataset.release(execution=...).
    exec_obj = MagicMock()
    exec_obj.execution_rid = "EXEC-3"
    mock_ml.lookup_execution.return_value = exec_obj

    with _patch_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_release"](
                hostname="h",
                catalog_id="1",
                dataset_rid="1-AAAA",
                bump="minor",
                description="Add training images for v0.5.0",
                execution_rid="EXEC-3",
            )
        )
    assert out["status"] == "released"
    assert out["dataset_rid"] == "1-AAAA"
    assert out["previous_version"] == "0.4.0.post1.dev3"
    assert out["new_version"] == "0.5.0"
    assert out["bump"] == "minor"

    # Check that VersionPart.minor was passed and the typed Execution
    # object was forwarded (not the bare RID).
    from deriva_ml.dataset.aux_classes import VersionPart

    forwarded = ds.release.call_args.kwargs
    assert forwarded["bump"] == VersionPart.minor
    assert forwarded["description"] == "Add training images for v0.5.0"
    assert forwarded["execution"] is exec_obj

    success = _success_calls(mock_audit, "deriva_ml_release")
    assert success
    assert success[0].kwargs["bump"] == "minor"
    assert success[0].kwargs["previous_version"] == "0.4.0.post1.dev3"
    assert success[0].kwargs["new_version"] == "0.5.0"


async def test_release_major(dataset_ctx, capturing_mcp, mock_ml):
    """bump='major' is forwarded as VersionPart.major."""
    ds = _make_dataset_mock("1-AAAA", current_version="1.2.3.post1.dev1")
    new_ver = MagicMock()
    new_ver.__str__ = lambda self: "2.0.0"
    ds.release.return_value = new_ver
    mock_ml.lookup_dataset.return_value = ds
    with patch("deriva_ml_mcp.tools.dataset.audit_event"):
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_release"](
                hostname="h", catalog_id="1", dataset_rid="1-AAAA", bump="major"
            )
        )
    assert out["new_version"] == "2.0.0"
    from deriva_ml.dataset.aux_classes import VersionPart

    forwarded = ds.release.call_args.kwargs
    assert forwarded["bump"] == VersionPart.major
    # No execution_rid given -- the typed execution argument should be None.
    assert forwarded["execution"] is None


async def test_release_error_no_dev_period(dataset_ctx, capturing_mcp, mock_ml):
    """Releasing a dataset with no dev period surfaces the deriva-ml error."""
    ds = _make_dataset_mock("1-AAAA")
    ds.release.side_effect = RuntimeError(
        "Dataset 1-AAAA has no dev period to release"
    )
    mock_ml.lookup_dataset.return_value = ds
    with _patch_audit() as mock_audit:
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_release"](
                hostname="h",
                catalog_id="1",
                dataset_rid="1-AAAA",
                bump="patch",
            )
        )
    assert "no dev period" in out["error"]
    failed = _success_calls(mock_audit, "deriva_ml_release_failed")
    assert failed
    assert failed[0].kwargs["bump"] == "patch"


# ---------------------------------------------------------------------------
# v1.3: surgical RAG re-index wired into mutating dataset tools
# ---------------------------------------------------------------------------


async def test_create_dataset_triggers_surgical_reindex(dataset_ctx, capturing_mcp, mock_ml):
    """``deriva_ml_create_dataset`` calls ``_reindex_dataset`` with the new RID."""
    from unittest.mock import AsyncMock

    new_ds = _make_dataset_mock("1-NEW", "desc", ["Training"], "1.0.0")
    fake_reindex = AsyncMock(return_value=1)
    with (
        patch("deriva_ml_mcp.tools.dataset.Dataset") as mock_dataset_cls,
        patch("deriva_ml_mcp.resources.rag._reindex_dataset", new=fake_reindex),
        _patch_audit(),
    ):
        mock_dataset_cls.create_dataset.return_value = new_ds
        await capturing_mcp.tools["deriva_ml_create_dataset"](
            hostname="h",
            catalog_id="1",
            execution_rid="EXEC-1",
            description="d",
        )
    fake_reindex.assert_awaited_once_with("h", "1", "1-NEW")


async def test_delete_dataset_triggers_surgical_drop(dataset_ctx, capturing_mcp, mock_ml):
    """``deriva_ml_delete_dataset`` calls ``_delete_dataset_source`` with the RID."""
    from unittest.mock import AsyncMock

    fake_drop = AsyncMock(return_value=True)
    with (
        patch("deriva_ml_mcp.resources.rag._delete_dataset_source", new=fake_drop),
        _patch_audit(),
    ):
        await capturing_mcp.tools["deriva_ml_delete_dataset"](
            hostname="h", catalog_id="1", dataset_rid="1-AAAA"
        )
    fake_drop.assert_awaited_once_with("h", "1", "1-AAAA")


async def test_create_dataset_reindex_failure_does_not_fail_tool(
    dataset_ctx, capturing_mcp, mock_ml
):
    """A re-index exception is logged but the success envelope is still returned."""
    from unittest.mock import AsyncMock

    new_ds = _make_dataset_mock("1-NEW", "desc", ["Training"], "1.0.0")
    fake_reindex = AsyncMock(side_effect=RuntimeError("rag boom"))
    with (
        patch("deriva_ml_mcp.tools.dataset.Dataset") as mock_dataset_cls,
        patch("deriva_ml_mcp.resources.rag._reindex_dataset", new=fake_reindex),
        _patch_audit() as mock_audit,
    ):
        mock_dataset_cls.create_dataset.return_value = new_ds
        out = json.loads(
            await capturing_mcp.tools["deriva_ml_create_dataset"](
                hostname="h",
                catalog_id="1",
                execution_rid="EXEC-1",
                description="d",
            )
        )
    # Catalog mutation succeeded -- response is the success envelope, not an error.
    assert out["status"] == "created"
    assert out["rid"] == "1-NEW"
    assert "error" not in out
    # The success audit fired despite the re-index failure.
    success = _success_calls(mock_audit, "deriva_ml_create_dataset")
    assert success
