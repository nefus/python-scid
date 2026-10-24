"""Tests for game data encoding/decoding."""

from __future__ import annotations

import io
import random
from pathlib import Path

import chess
import chess.pgn
import pytest

from scid.database import ScidDatabase
from scid.dates import encode_date
from scid.gamedata import decode_game_data, encode_game_data


class TestDecodeRealGames:
    """Decode games from real SCID databases and verify no errors."""

    def test_decode_666_all(self, db_666_path: Path):
        """Decode all 12 games from 666."""
        db = ScidDatabase.open(db_666_path)
        for i in range(len(db)):
            game = db[i]
            assert isinstance(game, chess.pgn.Game)
        db.close()

    def test_decode_scandi(self, db_scandi_path: Path):
        db = ScidDatabase.open(db_scandi_path)
        for i in range(len(db)):
            game = db[i]
            assert isinstance(game, chess.pgn.Game)
        db.close()

    def test_decode_repertoire_variations(self, db_repertoire_path: Path):
        """Repertoire has 17 variations in game 0."""
        db = ScidDatabase.open(db_repertoire_path)
        game = db[0]

        def count_vars(node):
            total = 0
            for j, v in enumerate(node.variations):
                if j > 0:
                    total += 1
                total += count_vars(v)
            return total

        assert count_vars(game) == 17
        db.close()

    def test_decode_kiop23(self, db_kiop23_path: Path):
        """SCID5 database with 1 game, 4 variations."""
        db = ScidDatabase.open(db_kiop23_path)
        assert len(db) == 1
        game = db[0]
        assert isinstance(game, chess.pgn.Game)
        assert sum(1 for _ in game.mainline_moves()) == 41
        db.close()


class TestEncodeRoundtrip:
    """Encode a game, then decode it and verify identity."""

    def test_simple_game(self):
        game = chess.pgn.Game()
        game.headers["White"] = "A"
        game.headers["Black"] = "B"
        game.headers["Result"] = "1-0"
        node = game
        for san in ["e4", "e5", "Nf3", "Nc6", "Bb5"]:
            node = node.add_variation(node.board().parse_san(san))

        blob = encode_game_data(game)
        decoded = decode_game_data(blob)

        orig_moves = [str(m) for m in game.mainline_moves()]
        dec_moves = [str(m) for m in decoded.mainline_moves()]
        assert orig_moves == dec_moves

    def test_game_with_comment(self):
        game = chess.pgn.Game()
        node = game
        node = node.add_variation(node.board().parse_san("e4"))
        node.comment = "Best by test"
        node = node.add_variation(node.board().parse_san("e5"))

        blob = encode_game_data(game)
        decoded = decode_game_data(blob)
        assert decoded.variations[0].comment == "Best by test"

    def test_game_with_nag(self):
        game = chess.pgn.Game()
        node = game
        node = node.add_variation(node.board().parse_san("e4"))
        node.nags.add(1)  # !

        blob = encode_game_data(game)
        decoded = decode_game_data(blob)
        assert 1 in decoded.variations[0].nags

    def test_game_with_variation(self):
        game = chess.pgn.Game()
        node = game
        node = node.add_variation(node.board().parse_san("e4"))
        e5 = node.add_variation(node.board().parse_san("e5"))
        # Add variation: 1... c5
        c5 = node.add_variation(node.board().parse_san("c5"))

        blob = encode_game_data(game)
        decoded = decode_game_data(blob)

        # Should have 2 variations after e4
        first_move = decoded.variations[0]
        assert len(first_move.variations) == 2

    def test_fen_game(self):
        fen = "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2"
        game = chess.pgn.Game()
        game.headers["FEN"] = fen
        board = chess.Board(fen)
        node = game
        node = node.add_variation(board.parse_san("d4"))

        blob = encode_game_data(game)
        decoded = decode_game_data(blob)
        assert decoded.headers.get("FEN") == fen

    def test_roundtrip_real_games(self, db_666_path: Path):
        """Encode each 666 game, then decode, and compare moves."""
        db = ScidDatabase.open(db_666_path)
        for i in range(len(db)):
            game = db[i]
            blob = encode_game_data(game)
            decoded = decode_game_data(blob)
            orig = [str(m) for m in game.mainline_moves()]
            dec = [str(m) for m in decoded.mainline_moves()]
            assert orig == dec, f"Game {i} moves don't match after roundtrip"
        db.close()

    def test_roundtrip_exact_blob(self, db_666_path: Path):
        """Re-encode should produce identical blobs for 666 games."""
        with open(f"{db_666_path}.si4", "rb") as f:
            from scid.index import read_si4

            _, entries = read_si4(f)
        with open(f"{db_666_path}.sg4", "rb") as f:
            game_data = f.read()

        for i, entry in enumerate(entries):
            orig_blob = game_data[entry.offset : entry.offset + entry.game_data_size]
            game = decode_game_data(orig_blob)
            re_blob = encode_game_data(game)
            assert re_blob == orig_blob, f"Game {i} blob mismatch on re-encode"


class TestFreechessBatch:
    """Batch decode from freechess to verify no errors on diverse games."""

    def test_decode_first_1000(self, db_freechess_path: Path):
        db = ScidDatabase.open(db_freechess_path)
        errors = 0
        for i in range(min(1000, len(db))):
            try:
                db[i]
            except Exception:
                errors += 1
        db.close()
        assert errors == 0, f"{errors}/1000 games failed to decode"


def _roundtrip(pgn: str) -> tuple[chess.pgn.Game, chess.pgn.Game]:
    game = chess.pgn.read_game(io.StringIO(pgn))
    return game, decode_game_data(encode_game_data(game))


class TestCommentsAndText:
    def test_variation_starting_comments(self):
        game, result = _roundtrip(
            "1. e4 {main} (1. d4 {after d4}) (1. c4) 1... e5 "
            "(1... c5 {sicilian}) 2. Nf3 *"
        )
        game.variations[1].starting_comment = "start d4"
        game.variations[2].starting_comment = "start c4"
        game.variations[0].variations[1].starting_comment = "start c5"
        result = decode_game_data(encode_game_data(game))

        assert result.variations[0].comment == "main"
        assert [(v.starting_comment, v.comment) for v in result.variations] == [
            ("", "main"),
            ("start d4", "after d4"),
            ("start c4", ""),
        ]
        black = result.variations[0].variations
        assert [(v.starting_comment, v.comment) for v in black] == [
            ("", ""),
            ("start c5", "sicilian"),
        ]
        assert str(result.mainline_moves()) == str(game.mainline_moves())

    def test_pregame_comment(self):
        game, result = _roundtrip("{before} 1. e4 {after} *")
        assert result.comment == "before"
        assert result.variations[0].comment == "after"

    def test_unicode_tags_and_comments(self):
        game, result = _roundtrip(
            '[Annotator "Дмитрий"]\n[Opening "Défense française"]\n\n'
            "1. e4 {Großartig ♔} *"
        )
        assert result.headers["Annotator"] == "Дмитрий"
        assert result.headers["Opening"] == "Défense française"
        assert result.variations[0].comment == "Großartig ♔"


class TestCompactEventDateTag:
    """SCID2-era blobs store EventDate as tag 255 + 3 binary bytes."""

    BLOB = (
        bytes([255]) + encode_date(1999, 7, 4).to_bytes(3, "big")  # EventDate
        + bytes([243, 3]) + b"Ann"  # Annotator (common tag)
        + b"\x00"  # end of tags
        + b"\x00"  # standard start
        + bytes([0x0F])  # end of game
    )

    def test_decode(self):
        game = decode_game_data(self.BLOB)
        assert game.headers["EventDate"] == "1999.07.04"
        assert game.headers["Annotator"] == "Ann"


def _random_annotated_game(seed: int, plies: int = 60) -> chess.pgn.Game:
    """Build a game with nested side variations, preferring captures."""
    rng = random.Random(seed)

    def pick(board: chess.Board) -> chess.Move:
        moves = list(board.legal_moves)
        captures = [m for m in moves if board.is_capture(m)]
        return rng.choice(captures if captures and rng.random() < 0.6 else moves)

    def grow(node: chess.pgn.GameNode, depth: int, length: int) -> None:
        for _ in range(length):
            board = node.board()
            if board.is_game_over():
                return
            main = pick(board)
            child = node.add_variation(main)
            if depth < 3 and rng.random() < 0.3:
                for _ in range(rng.randint(1, 3)):
                    alt = pick(board)
                    if alt != main and not node.has_variation(alt):
                        side = node.add_variation(alt)
                        grow(side, depth + 1, rng.randint(1, 8))
            node = child

    game = chess.pgn.Game()
    grow(game, 0, plies)
    return game


class TestVariationPieceLists:
    @pytest.mark.parametrize("seed", range(20))
    def test_random_nested_variations_roundtrip(self, seed: int):
        game = _random_annotated_game(seed)
        result = decode_game_data(encode_game_data(game))
        assert str(result) == str(game)

    def test_fixture_games_reencode(self, db_repertoire_path: Path):
        with ScidDatabase.open(db_repertoire_path) as db:
            for i in range(len(db)):
                game = db[i]
                again = decode_game_data(encode_game_data(game))
                for tag in ("Event", "Site", "Date", "Round", "White", "Black", "Result"):
                    again.headers[tag] = game.headers[tag]
                assert str(again) == str(game)
