"""Tests for low-level modules: dates, eco, matsig, homepawns, constants."""

from __future__ import annotations

import chess
import pytest

from scid.dates import (
    date_from_pgn,
    date_to_pgn,
    decode_date,
    decode_event_date_delta,
    encode_date,
    encode_event_date_delta,
)
from scid.eco import eco_encode, eco_decode
from scid.matsig import encode_matsig, decode_matsig
from scid.homepawns import encode_homepawns, decode_homepawns


# ---------------------------------------------------------------------------
# Date encoding
# ---------------------------------------------------------------------------


class TestDates:
    def test_known_date(self):
        d = encode_date(2026, 3, 6)
        y, m, day = decode_date(d)
        assert (y, m, day) == (2026, 3, 6)

    def test_zero_date(self):
        assert decode_date(0) == (0, 0, 0)

    def test_unknown_date(self):
        assert date_to_pgn(0) == "????.??.??"

    def test_partial_date(self):
        # Year only
        d = encode_date(2000, 0, 0)
        assert date_to_pgn(d) == "2000.??.??"

    def test_pgn_roundtrip(self):
        for pgn in ["2026.03.06", "1990.12.25", "????.??.??"]:
            d = date_from_pgn(pgn)
            assert date_to_pgn(d) == pgn

    def test_pgn_year_only(self):
        d = date_from_pgn("2000.??.??")
        y, m, day = decode_date(d)
        assert y == 2000
        assert m == 0
        assert day == 0


# ---------------------------------------------------------------------------
# ECO codes
# ---------------------------------------------------------------------------


class TestEco:
    def test_basic_eco(self):
        for code in ["A00", "B12", "C60", "D45", "E99"]:
            encoded = eco_encode(code)
            assert encoded > 0
            assert eco_decode(encoded) == code

    def test_extended_eco(self):
        encoded = eco_encode("B12a3")
        assert eco_decode(encoded) == "B12a3"

    def test_empty(self):
        assert eco_encode("") == 0
        assert eco_decode(0) == ""

    def test_roundtrip_all_basic(self):
        for letter in "ABCDE":
            for d1 in range(10):
                for d2 in range(10):
                    code = f"{letter}{d1}{d2}"
                    assert eco_decode(eco_encode(code)) == code


# ---------------------------------------------------------------------------
# Material signatures
# ---------------------------------------------------------------------------


class TestMatSig:
    def test_starting_position(self):
        board = chess.Board()
        sig = encode_matsig(board)
        mat = decode_matsig(sig)
        # Keys are uppercase: WP, BP, WQ, BQ, WR, BR, WN, BN, WB, BB
        assert mat["WP"] == 8
        assert mat["BP"] == 8
        assert mat["WQ"] == 1
        assert mat["BQ"] == 1
        assert mat["WR"] == 2
        assert mat["BR"] == 2
        assert mat["WN"] == 2
        assert mat["BN"] == 2
        assert mat["WB"] == 2
        assert mat["BB"] == 2

    def test_empty_board(self):
        board = chess.Board("8/8/8/8/8/4K3/8/4k3 w - - 0 1")
        sig = encode_matsig(board)
        mat = decode_matsig(sig)
        assert mat["WP"] == 0
        assert mat["BP"] == 0


# ---------------------------------------------------------------------------
# Home pawns
# ---------------------------------------------------------------------------


class TestHomePawns:
    def test_starting_position(self):
        board = chess.Board()
        data = encode_homepawns(board)
        assert len(data) == 9

    def test_roundtrip(self):
        board = chess.Board()
        data = encode_homepawns(board)
        changes = decode_homepawns(data)
        # Starting position: all pawns on home squares, so 0 changes
        assert changes == []

    def test_pawn_moved(self):
        # After 1. e4, the e2 pawn has moved
        board = chess.Board()
        board.push_san("e4")
        data = encode_homepawns(board)
        changes = decode_homepawns(data)
        # e2 pawn -> bit position 11 in SCID's bitmap (a2=15, b2=14, ..., e2=11)
        assert 11 in changes


class TestEventDateDelta:
    GAME = encode_date(2000, 6, 15)

    @pytest.mark.parametrize("years", [-3, -2, -1, 0, 1, 2, 3])
    def test_roundtrip_within_three_years(self, years: int):
        event = encode_date(2000 + years, 5, 1)
        delta = encode_event_date_delta(self.GAME, event)
        assert decode_event_date_delta(self.GAME, delta) == event

    @pytest.mark.parametrize("years", [-5, -4, 4, 5, 11])
    def test_out_of_range_is_dropped(self, years: int):
        event = encode_date(2000 + years, 5, 1)
        assert encode_event_date_delta(self.GAME, event) == 0
