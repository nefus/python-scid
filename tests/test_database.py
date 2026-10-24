"""Tests for the ScidDatabase high-level API."""

from __future__ import annotations

import os
from pathlib import Path

import chess.pgn
import pytest

from scid.database import ScidDatabase


# ---------------------------------------------------------------------------
# Opening existing databases
# ---------------------------------------------------------------------------


class TestOpenSCID4:
    def test_open_666(self, db_666_path: Path):
        db = ScidDatabase.open(db_666_path)
        assert db.version == 4
        assert len(db) == 12
        db.close()

    def test_context_manager(self, db_666_path: Path):
        with ScidDatabase.open(db_666_path) as db:
            assert len(db) == 12

    def test_getitem(self, db_666_path: Path):
        with ScidDatabase.open(db_666_path) as db:
            game = db[0]
            assert isinstance(game, chess.pgn.Game)
            assert game.headers["White"] == "Aljechin"

    def test_negative_index(self, db_666_path: Path):
        with ScidDatabase.open(db_666_path) as db:
            last = db[-1]
            assert isinstance(last, chess.pgn.Game)

    def test_slice(self, db_666_path: Path):
        with ScidDatabase.open(db_666_path) as db:
            games = db[0:3]
            assert len(games) == 3
            assert all(isinstance(g, chess.pgn.Game) for g in games)

    def test_index_out_of_range(self, db_666_path: Path):
        with ScidDatabase.open(db_666_path) as db:
            with pytest.raises(IndexError):
                db[100]

    def test_headers(self, db_666_path: Path):
        with ScidDatabase.open(db_666_path) as db:
            h = db.headers(0)
            assert "White" in h
            assert "Black" in h
            assert "Result" in h
            assert "Event" in h
            assert "Date" in h

    def test_iterate_all(self, db_666_path: Path):
        with ScidDatabase.open(db_666_path) as db:
            games = list(db)
            assert len(games) == 12


class TestOpenSCID5:
    def test_open_kiop23(self, db_kiop23_path: Path):
        db = ScidDatabase.open(db_kiop23_path)
        assert db.version == 5
        assert len(db) == 1
        db.close()

    def test_getitem(self, db_kiop23_path: Path):
        with ScidDatabase.open(db_kiop23_path) as db:
            game = db[0]
            assert isinstance(game, chess.pgn.Game)
            assert sum(1 for _ in game.mainline_moves()) == 41


class TestOpenErrors:
    def test_nonexistent_path(self):
        with pytest.raises(FileNotFoundError):
            ScidDatabase.open("/nonexistent/path/db")

    def test_readonly_write_fails(self, db_666_path: Path):
        with ScidDatabase.open(db_666_path, mode="r") as db:
            game = chess.pgn.Game()
            with pytest.raises(IOError):
                db.add_game(game)


# ---------------------------------------------------------------------------
# Creating databases
# ---------------------------------------------------------------------------


class TestCreate:
    def test_create_v5(self, tmp_db_path: Path):
        db = ScidDatabase.create(tmp_db_path, version=5)
        assert db.version == 5
        assert len(db) == 0
        db.close()

    def test_create_v4(self, tmp_db_path: Path):
        db = ScidDatabase.create(tmp_db_path, version=4)
        assert db.version == 4
        assert len(db) == 0
        db.close()

    def test_create_invalid_version(self, tmp_db_path: Path):
        with pytest.raises(ValueError):
            ScidDatabase.create(tmp_db_path, version=3)


# ---------------------------------------------------------------------------
# Writing games
# ---------------------------------------------------------------------------


def _make_game(
    white: str = "White",
    black: str = "Black",
    result: str = "1-0",
    moves: list[str] | None = None,
) -> chess.pgn.Game:
    """Helper to create a simple game."""
    game = chess.pgn.Game()
    game.headers["White"] = white
    game.headers["Black"] = black
    game.headers["Result"] = result
    if moves:
        node = game
        for san in moves:
            node = node.add_variation(node.board().parse_san(san))
    return game


class TestAddGame:
    def test_add_and_read_back(self, tmp_db_path: Path):
        game = _make_game(moves=["e4", "e5", "Nf3", "Nc6"])

        with ScidDatabase.create(tmp_db_path, version=5) as db:
            idx = db.add_game(game)
            assert idx == 0
            assert len(db) == 1

            # Read it back immediately
            g = db[0]
            assert g.headers["White"] == "White"
            assert g.headers["Black"] == "Black"
            moves = [str(m) for m in g.mainline_moves()]
            assert len(moves) == 4

    def test_add_multiple(self, tmp_db_path: Path):
        with ScidDatabase.create(tmp_db_path, version=5) as db:
            for i in range(10):
                game = _make_game(
                    white=f"Player{i}",
                    moves=["e4", "e5"],
                )
                db.add_game(game)
            assert len(db) == 10

    def test_save_and_reopen(self, tmp_db_path: Path):
        game = _make_game(
            white="Fischer",
            black="Spassky",
            result="1-0",
            moves=["e4", "c5", "Nf3"],
        )
        game.headers["WhiteElo"] = "2785"
        game.headers["ECO"] = "B20"

        db = ScidDatabase.create(tmp_db_path, version=5)
        db.add_game(game)
        db.save()
        db.close()

        # Reopen
        db2 = ScidDatabase.open(tmp_db_path)
        assert len(db2) == 1
        g = db2[0]
        assert g.headers["White"] == "Fischer"
        assert g.headers["Black"] == "Spassky"
        assert g.headers["Result"] == "1-0"
        assert g.headers["WhiteElo"] == "2785"
        assert g.headers["ECO"] == "B20"
        moves = [str(m) for m in g.mainline_moves()]
        assert len(moves) == 3
        db2.close()


class TestReplaceGame:
    def test_replace(self, tmp_db_path: Path):
        db = ScidDatabase.create(tmp_db_path, version=5)
        db.add_game(_make_game(white="Old", moves=["e4"]))
        db.replace_game(0, _make_game(white="New", moves=["d4", "d5"]))

        g = db[0]
        assert g.headers["White"] == "New"
        assert sum(1 for _ in g.mainline_moves()) == 2
        db.close()


class TestDeleteAndCompact:
    def test_delete_and_compact(self, tmp_db_path: Path):
        db = ScidDatabase.create(tmp_db_path, version=5)
        db.add_game(_make_game(white="A", moves=["e4"]))
        db.add_game(_make_game(white="B", moves=["d4"]))
        db.add_game(_make_game(white="C", moves=["c4"]))
        assert len(db) == 3

        db.delete_game(1)  # Delete "B"
        db.compact()
        assert len(db) == 2

        # Remaining games should be A and C
        assert db[0].headers["White"] == "A"
        assert db[1].headers["White"] == "C"

        # Save and reopen
        db.save()
        db2 = ScidDatabase.open(tmp_db_path)
        assert len(db2) == 2
        assert db2[0].headers["White"] == "A"
        assert db2[1].headers["White"] == "C"
        db2.close()
        db.close()


# ---------------------------------------------------------------------------
# PGN export/import
# ---------------------------------------------------------------------------


class TestPGNExport:
    def test_export_666(self, db_666_path: Path, tmp_path: Path):
        pgn_path = tmp_path / "export.pgn"
        with ScidDatabase.open(db_666_path) as db:
            db.to_pgn(pgn_path)

        content = pgn_path.read_text()
        assert content.count("[Event ") == 12

    def test_export_and_reimport(self, db_666_path: Path, tmp_path: Path):
        pgn_path = tmp_path / "export.pgn"
        db_path = tmp_path / "reimport"

        with ScidDatabase.open(db_666_path) as db:
            db.to_pgn(pgn_path)

        db2 = ScidDatabase.from_pgn(pgn_path, db_path, version=5)
        assert len(db2) == 12

        # Re-export and compare
        pgn_path2 = tmp_path / "re_export.pgn"
        db2.to_pgn(pgn_path2)
        db2.close()

        assert pgn_path.read_text() == pgn_path2.read_text()


class TestPGNImport:
    def test_from_pgn(self, tmp_path: Path):
        pgn_path = tmp_path / "test.pgn"
        pgn_path.write_text(
            '[Event "Test"]\n'
            '[Site "?"]\n'
            '[Date "2026.01.01"]\n'
            '[Round "1"]\n'
            '[White "A"]\n'
            '[Black "B"]\n'
            '[Result "1-0"]\n\n'
            "1. e4 e5 2. Nf3 Nc6 1-0\n\n"
            '[Event "Test"]\n'
            '[Site "?"]\n'
            '[Date "2026.01.01"]\n'
            '[Round "2"]\n'
            '[White "C"]\n'
            '[Black "D"]\n'
            '[Result "0-1"]\n\n'
            "1. d4 d5 0-1\n"
        )

        db_path = tmp_path / "imported"
        db = ScidDatabase.from_pgn(pgn_path, db_path, version=5)
        assert len(db) == 2
        assert db[0].headers["White"] == "A"
        assert db[1].headers["White"] == "C"
        db.close()

        # Verify persistence
        db2 = ScidDatabase.open(db_path)
        assert len(db2) == 2
        db2.close()


# ---------------------------------------------------------------------------
# SCID4 write roundtrip
# ---------------------------------------------------------------------------


class TestSCID4Write:
    def test_create_and_reopen_v4(self, tmp_db_path: Path):
        game = _make_game(
            white="Test",
            black="Player",
            moves=["e4", "e5", "Nf3"],
        )

        db = ScidDatabase.create(tmp_db_path, version=4)
        db.add_game(game)
        db.save()
        db.close()

        db2 = ScidDatabase.open(tmp_db_path)
        assert db2.version == 4
        assert len(db2) == 1
        g = db2[0]
        assert g.headers["White"] == "Test"
        assert sum(1 for _ in g.mainline_moves()) == 3
        db2.close()
