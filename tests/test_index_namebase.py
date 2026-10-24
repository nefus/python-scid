"""Tests for index and namebase reading/writing."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from scid.index import (
    IndexEntry,
    IndexHeader,
    read_si4,
    read_si5,
    write_si4,
    write_si5,
    encode_si4_entry,
    decode_si4_entry,
    encode_si5_entry,
    decode_si5_entry,
)
from scid.namebase import NameBase


# ---------------------------------------------------------------------------
# SCID4 Index
# ---------------------------------------------------------------------------


class TestSI4Index:
    def test_read_666(self, db_666_path: Path):
        with open(f"{db_666_path}.si4", "rb") as f:
            header, entries = read_si4(f)
        assert len(entries) == 12
        assert isinstance(header, IndexHeader)

    def test_entry_roundtrip(self, db_666_path: Path):
        with open(f"{db_666_path}.si4", "rb") as f:
            header, entries = read_si4(f)

        for i, entry in enumerate(entries):
            raw = encode_si4_entry(entry)
            assert len(raw) == 47, f"Entry {i} encoded to {len(raw)} bytes"
            decoded = decode_si4_entry(raw)
            assert decoded.offset == entry.offset, f"Entry {i} offset"
            assert decoded.game_data_size == entry.game_data_size, f"Entry {i} size"
            assert decoded.white_id == entry.white_id, f"Entry {i} white_id"
            assert decoded.black_id == entry.black_id, f"Entry {i} black_id"
            assert decoded.event_id == entry.event_id, f"Entry {i} event_id"
            assert decoded.result == entry.result, f"Entry {i} result"
            assert decoded.date == entry.date, f"Entry {i} date"

    def test_write_read_roundtrip(self, db_666_path: Path):
        """Write then re-read produces identical entries."""
        with open(f"{db_666_path}.si4", "rb") as f:
            header, entries = read_si4(f)

        buf = io.BytesIO()
        write_si4(buf, header, entries)
        buf.seek(0)
        header2, entries2 = read_si4(buf)

        assert len(entries2) == len(entries)
        for i, (e1, e2) in enumerate(zip(entries, entries2)):
            assert e1.offset == e2.offset, f"Entry {i} offset"
            assert e1.game_data_size == e2.game_data_size, f"Entry {i} size"
            assert e1.white_id == e2.white_id, f"Entry {i} white_id"
            assert e1.black_id == e2.black_id, f"Entry {i} black_id"
            assert e1.result == e2.result, f"Entry {i} result"
            assert e1.date == e2.date, f"Entry {i} date"


# ---------------------------------------------------------------------------
# SCID5 Index
# ---------------------------------------------------------------------------


class TestSI5Index:
    def test_read_kiop23(self, db_kiop23_path: Path):
        with open(f"{db_kiop23_path}.si5", "rb") as f:
            entries = read_si5(f)
        assert len(entries) == 1

    def test_entry_roundtrip(self, db_kiop23_path: Path):
        with open(f"{db_kiop23_path}.si5", "rb") as f:
            entries = read_si5(f)

        for entry in entries:
            raw = encode_si5_entry(entry)
            assert len(raw) == 56
            decoded = decode_si5_entry(raw)
            assert decoded.offset == entry.offset
            assert decoded.game_data_size == entry.game_data_size

    def test_write_roundtrip(self, db_kiop23_path: Path):
        with open(f"{db_kiop23_path}.si5", "rb") as f:
            orig_data = f.read()
            f.seek(0)
            entries = read_si5(f)

        buf = io.BytesIO()
        write_si5(buf, entries)
        assert buf.getvalue() == orig_data


# ---------------------------------------------------------------------------
# SCID4 Namebase
# ---------------------------------------------------------------------------


class TestSN4Namebase:
    def test_read_666(self, db_666_path: Path):
        with open(f"{db_666_path}.sn4", "rb") as f:
            nb = NameBase.read_sn4(f)
        # Should have player names
        assert nb.count(0) > 0  # NAME_PLAYER

    def test_name_lookup(self, db_666_path: Path):
        with open(f"{db_666_path}.sn4", "rb") as f:
            nb = NameBase.read_sn4(f)
        # ID 0 should return a valid name
        name = nb.get_name(0, 0)
        assert isinstance(name, str)

    def test_write_read_roundtrip(self, db_666_path: Path):
        """Write then re-read produces identical names."""
        with open(f"{db_666_path}.sn4", "rb") as f:
            nb = NameBase.read_sn4(f)

        buf = io.BytesIO()
        nb.write_sn4(buf)
        buf.seek(0)
        nb2 = NameBase.read_sn4(buf)

        for name_type in range(4):
            assert nb.count(name_type) == nb2.count(name_type), (
                f"Type {name_type} count mismatch"
            )
            for name_id in range(nb.count(name_type)):
                assert nb.get_name(name_type, name_id) == nb2.get_name(
                    name_type, name_id
                ), f"Type {name_type} id {name_id} mismatch"


# ---------------------------------------------------------------------------
# SCID5 Namebase
# ---------------------------------------------------------------------------


class TestSN5Namebase:
    def test_read_kiop23(self, db_kiop23_path: Path):
        with open(f"{db_kiop23_path}.sn5", "rb") as f:
            nb = NameBase.read_sn5(f)
        assert nb.count(0) > 0  # NAME_PLAYER

    def test_write_read_roundtrip(self, db_kiop23_path: Path):
        """Write then re-read produces identical names."""
        with open(f"{db_kiop23_path}.sn5", "rb") as f:
            nb = NameBase.read_sn5(f)

        buf = io.BytesIO()
        nb.write_sn5(buf)
        buf.seek(0)
        nb2 = NameBase.read_sn5(buf)

        for name_type in range(4):
            assert nb.count(name_type) == nb2.count(name_type), (
                f"Type {name_type} count mismatch"
            )
            for name_id in range(nb.count(name_type)):
                assert nb.get_name(name_type, name_id) == nb2.get_name(
                    name_type, name_id
                ), f"Type {name_type} id {name_id} mismatch"


# ---------------------------------------------------------------------------
# NameBase add_name
# ---------------------------------------------------------------------------


class TestNameBaseAdd:
    def test_add_new_name(self):
        nb = NameBase()
        id1 = nb.add_name(0, "Fischer")
        id2 = nb.add_name(0, "Spassky")
        assert id1 != id2
        assert nb.get_name(0, id1) == "Fischer"
        assert nb.get_name(0, id2) == "Spassky"

    def test_add_duplicate(self):
        nb = NameBase()
        id1 = nb.add_name(0, "Fischer")
        id2 = nb.add_name(0, "Fischer")
        assert id1 == id2

    def test_add_different_types(self):
        nb = NameBase()
        pid = nb.add_name(0, "Test")  # PLAYER
        eid = nb.add_name(1, "Test")  # EVENT
        # Same string but different type -> different ID namespaces
        assert nb.get_name(0, pid) == "Test"
        assert nb.get_name(1, eid) == "Test"
