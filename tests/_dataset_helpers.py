"""Shared helpers for the split dataset test suites.

The dataset tool surface is split into three source files
(``tools/dataset/read.py``, ``mutate.py``, ``complex.py``); the test
suite was split to mirror that layout
(``test_dataset_read.py``, ``test_dataset_mutate.py``,
``test_dataset_complex.py``). The three test modules share a small
amount of fixture wiring -- the dataset-mock factory and the
dual-patch context manager -- which lives here rather than being
duplicated three ways.

The ``dataset_ctx`` fixture itself is NOT exported from here -- pytest
fixtures must be defined at module level (in a test file or
``conftest.py``) to participate in collection. Each test file defines
``dataset_ctx`` locally, all three using the same body. Centralizing
the body here would only save a handful of lines while making the
fixture less greppable from the test sites.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from tests._helpers import make_patch_audit

# Dual-patch context manager for the dataset module. See
# ``make_patch_audit`` in ``tests/_helpers.py`` for the canonical
# explanation of why both bind sites are patched together.
_patch_audit = make_patch_audit("dataset")


def _make_dataset_mock(
    rid: str,
    description: str = "",
    dataset_types: list[str] | None = None,
    current_version: str = "0.1.0",
    chaise_url: str = "https://example.org/chaise",
) -> MagicMock:
    """Build a MagicMock that quacks like a Dataset object."""
    ds = MagicMock()
    ds.dataset_rid = rid
    ds.description = description
    ds.dataset_types = dataset_types or []
    ds.current_version = current_version
    ds.get_chaise_url.return_value = chaise_url
    return ds
