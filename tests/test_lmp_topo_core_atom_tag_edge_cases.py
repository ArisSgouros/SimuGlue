import pytest

from simuglue.topology.core import ensure_atom_tags_from_lmp_type_table


def test_ensure_atom_tags_missing_tag_field_raises_valueerror():
    """Bug: ensure_atom_tags_from_lmp_type_table() raises KeyError today.

    Expected: a user-facing ValueError with a clear message.
    """
    with pytest.raises(ValueError, match="tag"):
        ensure_atom_tags_from_lmp_type_table({})


def test_ensure_atom_tags_tag_none_raises_valueerror():
    """Bug: dict(None) triggers a TypeError today.

    Expected: a ValueError indicating the type table is incomplete.
    """
    with pytest.raises(ValueError, match="tag"):
        ensure_atom_tags_from_lmp_type_table({"tag": None})
