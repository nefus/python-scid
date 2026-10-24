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


# ---------------------------------------------------------------------------
# Index search data written by add_game
# ---------------------------------------------------------------------------


_SEARCH_FLAG_MASK = 0x07  # FLAG_START | FLAG_PROMO | FLAG_UPROMO


class TestAddGameIndexData:
    """add_game() must produce the index data SCID itself writes."""

    @pytest.mark.parametrize("version", [4, 5])
    def test_matches_scid_666(self, db_666_path: Path, tmp_db_path: Path, version):
        self._compare(db_666_path, tmp_db_path, version, 12)

    def test_matches_scid_freechess(self, db_freechess_path: Path, tmp_db_path: Path):
        self._compare(db_freechess_path, tmp_db_path, 4, 300)

    @staticmethod
    def _compare(src_path: Path, tmp_db_path: Path, version: int, limit: int):
        with ScidDatabase.open(src_path) as src:
            dst = ScidDatabase.create(tmp_db_path, version=version)
            for i in range(min(limit, len(src))):
                dst.add_game(src[i])
                expected = src._entries[i]
                actual = dst._entries[i]
                assert actual.flags & _SEARCH_FLAG_MASK == (
                    expected.flags & _SEARCH_FLAG_MASK
                ), i
                assert actual.home_pawn_data == expected.home_pawn_data, i
                assert actual.stored_line_code == expected.stored_line_code, i
                assert actual.final_mat_sig == expected.final_mat_sig, i

    def test_promotion_flags(self, tmp_db_path: Path):
        db = ScidDatabase.create(tmp_db_path, version=5)
        fen = "8/P6k/8/8/8/8/8/K7 w - - 0 1"
        for promo in ("Q", "N"):
            game = chess.pgn.Game()
            game.setup(fen)
            game.add_main_variation(game.board().parse_san(f"a8={promo}"))
            db.add_game(game)
        assert db._entries[0].flags & _SEARCH_FLAG_MASK == 0b011
        assert db._entries[1].flags & _SEARCH_FLAG_MASK == 0b111
        assert db._entries[0].home_pawn_data == b"\x00" * 9
        assert db._entries[0].stored_line_code == 0


class TestDeleteFlag:
    @pytest.mark.parametrize("version", [4, 5])
    def test_delete_sets_scid_delete_flag(self, tmp_db_path: Path, version: int):
        db = ScidDatabase.create(tmp_db_path, version=version)
        db.add_game(_make_game(white="A", moves=["e4"]))
        db.delete_game(0)
        db.save()
        with ScidDatabase.open(tmp_db_path) as reopened:
            flags = reopened._entries[0].flags
            assert flags & (1 << 3)  # FLAG_DELETE
            assert not flags & 0x01  # FLAG_START untouched

    def test_compact_keeps_setup_games(self, tmp_db_path: Path):
        db = ScidDatabase.create(tmp_db_path, version=4)
        setup = chess.pgn.Game()
        setup.setup("8/8/8/4k3/8/8/4K3/8 w - - 0 1")
        setup.headers["White"] = "Setup"
        db.add_game(setup)
        db.add_game(_make_game(white="Deleted", moves=["e4"]))
        db.add_game(_make_game(white="Kept", moves=["d4"]))
        db.delete_game(1)
        db.compact()
        assert [db.headers(i)["White"] for i in range(len(db))] == ["Setup", "Kept"]


# ---------------------------------------------------------------------------
# Index counts and ratings
# ---------------------------------------------------------------------------


class TestAddGameCountsAndElo:
    @pytest.mark.parametrize("version", [4, 5])
    def test_counts_encoded_once(self, tmp_db_path: Path, version: int):
        game = _make_game(moves=["e4", "e5", "Nf3"])
        node = game.variations[0]
        replies = "c5 c6 d5 d6 e6 g6 Nf6 Nc6 a6 b6 h6 a5 b5 f5 g5".split()
        for nag, san in enumerate(replies, start=1):
            node.add_variation(node.board().parse_san(san))
            node.nags.add(nag)
        game.variations[0].comment = "x"
        db = ScidDatabase.create(tmp_db_path, version=version)
        db.add_game(game)
        db.save()
        with ScidDatabase.open(tmp_db_path) as reopened:
            entry = reopened._entries[0]
            assert entry.n_variations == 15
            assert entry.n_nags == 15
            assert entry.n_comments == 1

    @pytest.mark.parametrize("version", [4, 5])
    @pytest.mark.parametrize(
        ("tag", "stored"), [("2850", 2850), ("5000", 4000), ("-5", 0), ("?", 0)]
    )
    def test_elo_clamped(self, tmp_db_path: Path, version: int, tag: str, stored: int):
        game = _make_game(moves=["e4"])
        game.headers["WhiteElo"] = tag
        db = ScidDatabase.create(tmp_db_path, version=version)
        db.add_game(game)
        db.save()
        with ScidDatabase.open(tmp_db_path) as reopened:
            assert reopened._entries[0].white_elo == stored


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


class TestSaveSafety:
    @pytest.mark.parametrize("version", [4, 5])
    def test_unicode_names_save_and_reload(self, tmp_db_path: Path, version: int):
        db = ScidDatabase.create(tmp_db_path, version=version)
        db.add_game(_make_game(white="Дин Лижэнь", black="Müller", moves=["e4"]))
        db.save()
        with ScidDatabase.open(tmp_db_path) as reopened:
            assert reopened.headers(0)["White"] == "Дин Лижэнь"
            assert reopened.search_header(white="müller") == []
            assert reopened.search_header(black="müller") == [0]

    @pytest.mark.parametrize("version", [4, 5])
    def test_failed_save_leaves_files_untouched(
        self, tmp_db_path: Path, version: int, monkeypatch
    ):
        db = ScidDatabase.create(tmp_db_path, version=version)
        db.add_game(_make_game(white="A", moves=["e4"]))
        db.save()
        before = {
            p.name: p.read_bytes() for p in tmp_db_path.parent.iterdir()
        }

        db.add_game(_make_game(white="B", moves=["d4"]))

        def fail(*_args, **_kwargs):
            raise RuntimeError("simulated encoding failure")

        writer = "write_sn4" if version == 4 else "write_sn5"
        monkeypatch.setattr(type(db._namebase), writer, fail)
        with pytest.raises(RuntimeError):
            db.save()

        after = {p.name: p.read_bytes() for p in tmp_db_path.parent.iterdir()}
        assert after == before  # no partial writes, no leftover temp files


class TestDottedBaseName:
    @pytest.mark.parametrize("version", [4, 5])
    def test_open_base_name_with_dot(self, tmp_path: Path, version: int):
        base = tmp_path / "games.2024"
        db = ScidDatabase.create(base, version=version)
        db.add_game(_make_game(white="Dot", moves=["e4"]))
        db.save()
        # Decoy that Path.with_suffix() would pick up with the other version
        (tmp_path / f"games.si{9 - version}").write_bytes(b"decoy")
        with ScidDatabase.open(base) as reopened:
            assert reopened.version == version
            assert reopened.headers(0)["White"] == "Dot"


# ---------------------------------------------------------------------------
# API details
# ---------------------------------------------------------------------------


class TestOpenMode:
    @pytest.mark.parametrize("mode", ["w", "a", "r+", "RW", ""])
    def test_invalid_mode_rejected(self, tmp_db_path: Path, mode: str):
        ScidDatabase.create(tmp_db_path, version=5).save()
        with pytest.raises(ValueError, match="mode"):
            ScidDatabase.open(tmp_db_path, mode=mode)


class TestDescription:
    @pytest.mark.parametrize("version", [4, 5])
    def test_description_roundtrip(self, tmp_db_path: Path, version: int):
        db = ScidDatabase.create(tmp_db_path, version=version)
        db.description = "Meine Partien"
        db.save()
        with ScidDatabase.open(tmp_db_path) as reopened:
            assert reopened.description == "Meine Partien"


class TestContextManagerSave:
    def test_saves_on_normal_exit(self, tmp_db_path: Path):
        ScidDatabase.create(tmp_db_path, version=5).save()
        with ScidDatabase.open(tmp_db_path, mode="rw") as db:
            db.add_game(_make_game(moves=["e4"]))
        with ScidDatabase.open(tmp_db_path) as db:
            assert len(db) == 1

    def test_does_not_save_after_exception(self, tmp_db_path: Path):
        ScidDatabase.create(tmp_db_path, version=5).save()
        with pytest.raises(RuntimeError):
            with ScidDatabase.open(tmp_db_path, mode="rw") as db:
                db.add_game(_make_game(moves=["e4"]))
                raise RuntimeError("boom")
        with ScidDatabase.open(tmp_db_path) as db:
            assert len(db) == 0


class TestPGNEncoding:
    def test_from_pgn_latin1(self, tmp_path: Path):
        pgn = tmp_path / "legacy.pgn"
        pgn.write_bytes(
            '[White "Müller"]\n[Black "Sämisch"]\n[Result "*"]\n\n1. e4 *\n'.encode(
                "latin-1"
            )
        )
        db = ScidDatabase.from_pgn(pgn, tmp_path / "db", encoding="latin-1")
        assert db.headers(0)["White"] == "Müller"

    def test_from_pgn_utf8_with_bom(self, tmp_path: Path):
        pgn = tmp_path / "bom.pgn"
        pgn.write_bytes(
            '﻿[White "Дин"]\n[Black "B"]\n[Result "*"]\n\n1. e4 *\n'.encode("utf-8")
        )
        db = ScidDatabase.from_pgn(pgn, tmp_path / "db")
        assert db.headers(0)["White"] == "Дин"

    def test_to_pgn_utf8(self, tmp_db_path: Path, tmp_path: Path):
        db = ScidDatabase.create(tmp_db_path, version=5)
        db.add_game(_make_game(white="Дин Лижэнь", moves=["e4"]))
        out = tmp_path / "out.pgn"
        db.to_pgn(out)
        assert '[White "Дин Лижэнь"]' in out.read_text(encoding="utf-8")
