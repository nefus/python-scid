"""Stored opening lines for search optimization.

SCID stores a dictionary of 254 common opening lines.  Each game in the
index has a ``stored_line_code`` (1–254, or 0 for no match) representing
the longest stored line that matches the game's initial moves.

During tree search this allows:
 - Instant matches: if the target position occurs at a known ply inside a
   stored line, all games with that code are matches without decoding.
 - Instant exclusions: if the stored line's final position can never reach
   the target (wrong pawns / material), those games can be skipped.

The 254 lines were extracted from SCID's ``stored.cpp`` and are stored
here as space-separated UCI strings, converted to ``chess.Move`` tuples
at import time.
"""

from __future__ import annotations

from typing import Sequence

import chess

# ---------------------------------------------------------------------------
# Raw data: 254 opening lines as UCI move strings.
# Index 0 is unused (code 0 = "no match").  Index i corresponds to code i.
# ---------------------------------------------------------------------------
_UCI_LINES: tuple[str, ...] = (
    "",  # code 0 – unused
    "b2b3",  # 1: 1.b3
    "c2c4",  # 2: 1.c4
    "c2c4 c7c5",  # 3: 1.c4 c5
    "c2c4 c7c5 g1f3",  # 4: 1.c4 c5 2.Nf3
    "c2c4 e7e5",  # 5: 1.c4 e5
    "c2c4 e7e5 b1c3",  # 6: 1.c4 e5 2.Nc3
    "c2c4 e7e5 b1c3 g8f6",  # 7: 1.c4 e5 2.Nc3 Nf6
    "c2c4 e7e6",  # 8: 1.c4 e6
    "c2c4 e7e6 g1f3",  # 9: 1.c4 e6 2.Nf3
    "c2c4 g7g6",  # 10: 1.c4 g6
    "c2c4 g8f6",  # 11: 1.c4 Nf6
    "c2c4 g8f6 b1c3",  # 12: 1.c4 Nf6 2.Nc3
    "c2c4 g8f6 b1c3 e7e6",  # 13: 1.c4 Nf6 2.Nc3 e6
    "c2c4 g8f6 b1c3 g7g6",  # 14: 1.c4 Nf6 2.Nc3 g6
    "d2d4",  # 15: 1.d4
    "d2d4 d7d5",  # 16: 1.d4 d5
    "d2d4 d7d5 c2c4",  # 17: 1.d4 d5 2.c4
    "d2d4 d7d5 c2c4 c7c6",  # 18: 1.d4 d5 2.c4 c6
    "d2d4 d7d5 c2c4 c7c6 b1c3",  # 19: 1.d4 d5 2.c4 c6 3.Nc3
    "d2d4 d7d5 c2c4 c7c6 b1c3 g8f6",  # 20: 1.d4 d5 2.c4 c6 3.Nc3 Nf6
    "d2d4 d7d5 c2c4 c7c6 b1c3 g8f6 g1f3",  # 21: 1.d4 d5 2.c4 c6 3.Nc3 Nf6 4.Nf3
    "d2d4 d7d5 c2c4 c7c6 g1f3",  # 22: 1.d4 d5 2.c4 c6 3.Nf3
    "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6",  # 23: 1.d4 d5 2.c4 c6 3.Nf3 Nf6
    "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6 b1c3",  # 24: 1.d4 d5 2.c4 c6 3.Nf3 Nf6 4.Nc3
    "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6 b1c3 e7e6",  # 25: 1.d4 d5 2.c4 c6 3.Nf3 Nf6 4.Nc3 e6
    "d2d4 d7d5 c2c4 d5c4",  # 26: 1.d4 d5 2.c4 dxc4
    "d2d4 d7d5 c2c4 d5c4 g1f3",  # 27: 1.d4 d5 2.c4 dxc4 3.Nf3
    "d2d4 d7d5 c2c4 d5c4 g1f3 g8f6",  # 28: 1.d4 d5 2.c4 dxc4 3.Nf3 Nf6
    "d2d4 d7d5 c2c4 e7e6",  # 29: 1.d4 d5 2.c4 e6
    "d2d4 d7d5 c2c4 e7e6 b1c3",  # 30: 1.d4 d5 2.c4 e6 3.Nc3
    "d2d4 d7d5 c2c4 e7e6 b1c3 c7c6",  # 31: 1.d4 d5 2.c4 e6 3.Nc3 c6
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6",  # 32: 1.d4 d5 2.c4 e6 3.Nc3 Nf6
    "d2d4 d7d5 c2c4 e7e6 g1f3",  # 33: 1.d4 d5 2.c4 e6 3.Nf3
    "d2d4 d7d5 g1f3",  # 34: 1.d4 d5 2.Nf3
    "d2d4 d7d5 g1f3 g8f6",  # 35: 1.d4 d5 2.Nf3 Nf6
    "d2d4 d7d5 g1f3 g8f6 c2c4",  # 36: 1.d4 d5 2.Nf3 Nf6 3.c4
    "d2d4 d7d5 g1f3 g8f6 c2c4 c7c6",  # 37: 1.d4 d5 2.Nf3 Nf6 3.c4 c6
    "d2d4 d7d5 g1f3 g8f6 c2c4 e7e6",  # 38: 1.d4 d5 2.Nf3 Nf6 3.c4 e6
    "d2d4 d7d6",  # 39: 1.d4 d6
    "d2d4 d7d6 g1f3",  # 40: 1.d4 d6 2.Nf3
    "d2d4 e7e6",  # 41: 1.d4 e6
    "d2d4 e7e6 c2c4",  # 42: 1.d4 e6 2.c4
    "d2d4 e7e6 c2c4 g8f6",  # 43: 1.d4 e6 2.c4 Nf6
    "d2d4 f7f5",  # 44: 1.d4 f5
    "d2d4 f7f5 g2g3 g8f6 f1g2",  # 45: 1.d4 f5 2.g3 Nf6 3.Bg2
    "d2d4 g7g6",  # 46: 1.d4 g6
    "d2d4 g7g6 c2c4 f8g7",  # 47: 1.d4 g6 2.c4 Bg7
    "d2d4 g8f6",  # 48: 1.d4 Nf6
    "d2d4 g8f6 c1g5",  # 49: 1.d4 Nf6 2.Bg5
    "d2d4 g8f6 c1g5 f6e4",  # 50: 1.d4 Nf6 2.Bg5 Ne4
    "d2d4 g8f6 c2c4",  # 51: 1.d4 Nf6 2.c4
    "d2d4 g8f6 c2c4 c7c5",  # 52: 1.d4 Nf6 2.c4 c5
    "d2d4 g8f6 c2c4 c7c5 d4d5",  # 53: 1.d4 Nf6 2.c4 c5 3.d5
    "d2d4 g8f6 c2c4 c7c5 d4d5 b7b5",  # 54: 1.d4 Nf6 2.c4 c5 3.d5 b5
    "d2d4 g8f6 c2c4 c7c5 d4d5 b7b5 c4b5 a7a6",  # 55: 1.d4 Nf6 2.c4 c5 3.d5 b5 4.cxb5 a6
    "d2d4 g8f6 c2c4 e7e6 g2g3",  # 56: 1.d4 Nf6 2.c4 e6 3.g3
    "d2d4 g8f6 c2c4 e7e6 g2g3 d7d5",  # 57: 1.d4 Nf6 2.c4 e6 3.g3 d5
    "d2d4 g8f6 c2c4 e7e6 b1c3",  # 58: 1.d4 Nf6 2.c4 e6 3.Nc3
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4",  # 59: 1.d4 Nf6 2.c4 e6 3.Nc3 Bb4
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4 e2e3",  # 60: 1.d4 Nf6 2.c4 e6 3.Nc3 Bb4 4.e3
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4 e2e3 e8g8",  # 61: 1.d4 Nf6 2.c4 e6 3.Nc3 Bb4 4.e3 O-O
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4 d1c2",  # 62: 1.d4 Nf6 2.c4 e6 3.Nc3 Bb4 4.Qc2
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4 d1c2 e8g8",  # 63: 1.d4 Nf6 2.c4 e6 3.Nc3 Bb4 4.Qc2 O-O
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4 d1c2 e8g8 a2a3 b4c3 c2c3",  # 64: 1.d4 Nf6 2.c4 e6 3.Nc3 Bb4 4.Qc2 O-O 5.a3 Bxc3+ 6.Qxc3
    "d2d4 g8f6 c2c4 e7e6 b1c3 d7d5",  # 65: 1.d4 Nf6 2.c4 e6 3.Nc3 d5
    "d2d4 g8f6 c2c4 e7e6 g1f3",  # 66: 1.d4 Nf6 2.c4 e6 3.Nf3
    "d2d4 g8f6 c2c4 e7e6 g1f3 b7b6",  # 67: 1.d4 Nf6 2.c4 e6 3.Nf3 b6
    "d2d4 g8f6 c2c4 e7e6 g1f3 b7b6 a2a3",  # 68: 1.d4 Nf6 2.c4 e6 3.Nf3 b6 4.a3
    "d2d4 g8f6 c2c4 e7e6 g1f3 b7b6 g2g3",  # 69: 1.d4 Nf6 2.c4 e6 3.Nf3 b6 4.g3
    "d2d4 g8f6 c2c4 e7e6 g1f3 b7b6 g2g3 c8a6",  # 70: 1.d4 Nf6 2.c4 e6 3.Nf3 b6 4.g3 Ba6
    "d2d4 g8f6 c2c4 e7e6 g1f3 f8b4",  # 71: 1.d4 Nf6 2.c4 e6 3.Nf3 Bb4+
    "d2d4 g8f6 c2c4 e7e6 g1f3 d7d5",  # 72: 1.d4 Nf6 2.c4 e6 3.Nf3 d5
    "d2d4 g8f6 c2c4 e7e6 g1f3 d7d5 b1c3",  # 73: 1.d4 Nf6 2.c4 e6 3.Nf3 d5 4.Nc3
    "d2d4 g8f6 c2c4 g7g6",  # 74: 1.d4 Nf6 2.c4 g6
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7",  # 75: 1.d4 Nf6 2.c4 g6 3.Nc3 Bg7
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4",  # 76: 1.d4 Nf6 2.c4 g6 3.Nc3 Bg7 4.e4
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6",  # 77: 1.d4 Nf6 2.c4 g6 3.Nc3 Bg7 4.e4 d6
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 f1e2 e8g8",  # 78: 1.d4 Nf6 2.c4 g6 3.Nc3 Bg7 4.e4 d6 5.Be2 O-O
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 f1e2 e8g8 g1f3",  # 79: 1.d4 Nf6 2.c4 g6 3.Nc3 Bg7 4.e4 d6 5.Be2 O-O 6.Nf3
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 f2f3",  # 80: 1.d4 Nf6 2.c4 g6 3.Nc3 Bg7 4.e4 d6 5.f3
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 f2f3 e8g8",  # 81: 1.d4 Nf6 2.c4 g6 3.Nc3 Bg7 4.e4 d6 5.f3 O-O
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 f2f3 e8g8 c1e3",  # 82: 1.d4 Nf6 2.c4 g6 3.Nc3 Bg7 4.e4 d6 5.f3 O-O 6.Be3
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 g1f3 e8g8",  # 83: 1.d4 Nf6 2.c4 g6 3.Nc3 Bg7 4.e4 d6 5.Nf3 O-O
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 g1f3 e8g8 f1e2",  # 84: 1.d4 Nf6 2.c4 g6 3.Nc3 Bg7 4.e4 d6 5.Nf3 O-O 6.Be2
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 g1f3 e8g8 f1e2 e7e5",  # 85: 1.d4 Nf6 2.c4 g6 3.Nc3 Bg7 4.e4 d6 5.Nf3 O-O 6.Be2 e5
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 g1f3 e8g8 f1e2 e7e5 e1g1",  # 86: 1.d4 Nf6 2.c4 g6 3.Nc3 Bg7 4.e4 d6 5.Nf3 O-O 6.Be2 e5 7.O-O
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 g1f3 e8g8 f1e2 e7e5 e1g1 b8c6 d4d5 c6e7",  # 87: KID Mar del Plata
    "d2d4 g8f6 c2c4 g7g6 b1c3 d7d5",  # 88: 1.d4 Nf6 2.c4 g6 3.Nc3 d5
    "d2d4 g8f6 c2c4 g7g6 b1c3 d7d5 g1f3",  # 89: 1.d4 Nf6 2.c4 g6 3.Nc3 d5 4.Nf3
    "d2d4 g8f6 c2c4 g7g6 b1c3 d7d5 c4d5 f6d5",  # 90: 1.d4 Nf6 2.c4 g6 3.Nc3 d5 4.cxd5 Nxd5
    "d2d4 g8f6 c2c4 g7g6 b1c3 d7d5 c4d5 f6d5 e2e4 d5c3 b2c3 f8g7",  # 91: Grunfeld Exchange
    "d2d4 g8f6 g1f3",  # 92: 1.d4 Nf6 2.Nf3
    "d2d4 g8f6 g1f3 c7c5",  # 93: 1.d4 Nf6 2.Nf3 c5
    "d2d4 g8f6 g1f3 d7d5",  # 94: 1.d4 Nf6 2.Nf3 d5
    "d2d4 g8f6 g1f3 e7e6",  # 95: 1.d4 Nf6 2.Nf3 e6
    "d2d4 g8f6 g1f3 e7e6 c1g5",  # 96: 1.d4 Nf6 2.Nf3 e6 3.Bg5
    "d2d4 g8f6 g1f3 e7e6 c2c4",  # 97: 1.d4 Nf6 2.Nf3 e6 3.c4
    "d2d4 g8f6 g1f3 g7g6",  # 98: 1.d4 Nf6 2.Nf3 g6
    "d2d4 g8f6 g1f3 g7g6 c1g5",  # 99: 1.d4 Nf6 2.Nf3 g6 3.Bg5
    "d2d4 g8f6 g1f3 g7g6 c2c4",  # 100: 1.d4 Nf6 2.Nf3 g6 3.c4
    "d2d4 g8f6 g1f3 g7g6 c2c4 f8g7",  # 101: 1.d4 Nf6 2.Nf3 g6 3.c4 Bg7
    "d2d4 g8f6 g1f3 g7g6 c2c4 f8g7 b1c3",  # 102: 1.d4 Nf6 2.Nf3 g6 3.c4 Bg7 4.Nc3
    "d2d4 g8f6 g1f3 g7g6 c2c4 f8g7 b1c3 e8g8",  # 103: 1.d4 Nf6 2.Nf3 g6 3.c4 Bg7 4.Nc3 O-O
    "d2d4 g8f6 g1f3 g7g6 g2g3",  # 104: 1.d4 Nf6 2.Nf3 g6 3.g3
    "d2d4 g8f6 g1f3 g7g6 g2g3 f8g7 f1g2",  # 105: 1.d4 Nf6 2.Nf3 g6 3.g3 Bg7 4.Bg2
    "e2e4",  # 106: 1.e4
    "e2e4 c7c5",  # 107: 1.e4 c5
    "e2e4 c7c5 c2c3",  # 108: 1.e4 c5 2.c3
    "e2e4 c7c5 c2c3 d7d5 e4d5 d8d5 d2d4",  # 109: 1.e4 c5 2.c3 d5 3.exd5 Qxd5 4.d4
    "e2e4 c7c5 c2c3 d7d5 e4d5 d8d5 d2d4 g8f6",  # 110: 1.e4 c5 2.c3 d5 3.exd5 Qxd5 4.d4 Nf6
    "e2e4 c7c5 c2c3 g8f6 e4e5 f6d5",  # 111: 1.e4 c5 2.c3 Nf6 3.e5 Nd5
    "e2e4 c7c5 c2c3 g8f6 e4e5 f6d5 d2d4 c5d4",  # 112: 1.e4 c5 2.c3 Nf6 3.e5 Nd5 4.d4 cxd4
    "e2e4 c7c5 d2d4 c5d4",  # 113: 1.e4 c5 2.d4 cxd4
    "e2e4 c7c5 b1c3",  # 114: 1.e4 c5 2.Nc3
    "e2e4 c7c5 b1c3 b8c6",  # 115: 1.e4 c5 2.Nc3 Nc6
    "e2e4 c7c5 b1c3 b8c6 g2g3",  # 116: 1.e4 c5 2.Nc3 Nc6 3.g3
    "e2e4 c7c5 b1c3 b8c6 g2g3 g7g6",  # 117: 1.e4 c5 2.Nc3 Nc6 3.g3 g6
    "e2e4 c7c5 b1c3 b8c6 g2g3 g7g6 f1g2 f8g7",  # 118: 1.e4 c5 2.Nc3 Nc6 3.g3 g6 4.Bg2 Bg7
    "e2e4 c7c5 b1c3 b8c6 g2g3 g7g6 f1g2 f8g7 d2d3",  # 119: 1.e4 c5 2.Nc3 Nc6 3.g3 g6 4.Bg2 Bg7 5.d3
    "e2e4 c7c5 g1f3",  # 120: 1.e4 c5 2.Nf3
    "e2e4 c7c5 g1f3 d7d6",  # 121: 1.e4 c5 2.Nf3 d6
    "e2e4 c7c5 g1f3 d7d6 f1b5",  # 122: 1.e4 c5 2.Nf3 d6 3.Bb5+
    "e2e4 c7c5 g1f3 d7d6 d2d4",  # 123: 1.e4 c5 2.Nf3 d6 3.d4
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4",  # 124: 1.e4 c5 2.Nf3 d6 3.d4 cxd4
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4",  # 125: 1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6",  # 126: Sicilian Najdorf/Scheveningen
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3",  # 127: 1.e4 c5 2.Nf3 d6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6",  # 128: Sicilian Najdorf
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 f1c4",  # 129: Najdorf 6.Bc4
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 f1e2",  # 130: Najdorf 6.Be2
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 c1e3",  # 131: Najdorf 6.Be3
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 c1g5",  # 132: Najdorf 6.Bg5
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6 c1g5 e7e6",  # 133: Najdorf 6.Bg5 e6
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 g7g6",  # 134: Sicilian Dragon
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 g7g6 c1e3 f8g7 f2f3",  # 135: Dragon Yugoslav
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 g7g6 c1e3 f8g7 f2f3 e8g8",  # 136: Dragon Yugoslav O-O
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 b8c6",  # 137: Sicilian Classical
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 b8c6 c1g5",  # 138: Richter-Rauzer
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 b8c6 c1g5 e7e6 d1d2",  # 139: Richter-Rauzer 7.Qd2
    "e2e4 c7c5 g1f3 e7e6",  # 140: 1.e4 c5 2.Nf3 e6
    "e2e4 c7c5 g1f3 e7e6 d2d3",  # 141: 1.e4 c5 2.Nf3 e6 3.d3
    "e2e4 c7c5 g1f3 e7e6 d2d4 c5d4 f3d4",  # 142: 1.e4 c5 2.Nf3 e6 3.d4 cxd4 4.Nxd4
    "e2e4 c7c5 g1f3 e7e6 d2d4 c5d4 f3d4 a7a6",  # 143: Sicilian Kan
    "e2e4 c7c5 g1f3 e7e6 d2d4 c5d4 f3d4 a7a6 f1d3",  # 144: Kan 5.Bd3
    "e2e4 c7c5 g1f3 e7e6 d2d4 c5d4 f3d4 b8c6",  # 145: Sicilian Taimanov
    "e2e4 c7c5 g1f3 e7e6 d2d4 c5d4 f3d4 b8c6 b1c3",  # 146: Taimanov 5.Nc3
    "e2e4 c7c5 g1f3 e7e6 d2d4 c5d4 f3d4 b8c6 b1c3 d8c7",  # 147: Taimanov 5...Qc7
    "e2e4 c7c5 g1f3 e7e6 d2d4 c5d4 f3d4 g8f6",  # 148: Sicilian Scheveningen
    "e2e4 c7c5 g1f3 e7e6 d2d4 c5d4 f3d4 g8f6 b1c3",  # 149: Scheveningen 5.Nc3
    "e2e4 c7c5 g1f3 e7e6 d2d4 c5d4 f3d4 g8f6 b1c3 d7d6",  # 150: Scheveningen 5...d6
    "e2e4 c7c5 g1f3 b8c6",  # 151: 1.e4 c5 2.Nf3 Nc6
    "e2e4 c7c5 g1f3 b8c6 f1b5",  # 152: Rossolimo
    "e2e4 c7c5 g1f3 b8c6 f1b5 g7g6",  # 153: Rossolimo 3...g6
    "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4",  # 154: Open Sicilian Nc6
    "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 e7e5",  # 155: Sveshnikov/Kalashnikov
    "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 g7g6",  # 156: Accelerated Dragon
    "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 g8f6 b1c3",  # 157: Open Sicilian Nc6 Nf6
    "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 g8f6 b1c3 d7d6",  # 158: Classical Sicilian
    "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 g8f6 b1c3 d7d6 c1g5",  # 159: Classical 6.Bg5
    "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 g8f6 b1c3 e7e5",  # 160: Sveshnikov
    "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 g8f6 b1c3 e7e5 d4b5 d7d6",  # 161: Sveshnikov 6.Ndb5 d6
    "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 g8f6 b1c3 e7e5 d4b5 d7d6 c1g5 a7a6",  # 162: Sveshnikov main
    "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4 f3d4 g8f6 b1c3 e7e5 d4b5 d7d6 c1g5 a7a6 b5a3 b7b5",  # 163: Sveshnikov 8...b5
    "e2e4 c7c6",  # 164: 1.e4 c6
    "e2e4 c7c6 d2d4 d7d5",  # 165: Caro-Kann
    "e2e4 c7c6 d2d4 d7d5 e4e5",  # 166: Caro-Kann Advance
    "e2e4 c7c6 d2d4 d7d5 e4e5 c8f5",  # 167: Advance 3...Bf5
    "e2e4 c7c6 d2d4 d7d5 e4d5 c6d5",  # 168: Caro-Kann Exchange
    "e2e4 c7c6 d2d4 d7d5 e4d5 c6d5 c2c4 g8f6 b1c3",  # 169: Panov-Botvinnik
    "e2e4 c7c6 d2d4 d7d5 b1c3",  # 170: Caro-Kann 3.Nc3
    "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4 c3e4",  # 171: Caro-Kann Classical
    "e2e4 c7c6 d2d4 d7d5 b1d2 d5e4 d2e4",  # 172: Caro-Kann 3.Nd2
    "e2e4 d7d5 e4d5 g8f6",  # 173: Scandinavian Nf6
    "e2e4 d7d5 e4d5 d8d5",  # 174: Scandinavian Qxd5
    "e2e4 d7d5 e4d5 d8d5 b1c3",  # 175: Scandinavian 3.Nc3
    "e2e4 d7d5 e4d5 d8d5 b1c3 d5a5",  # 176: Scandinavian 3...Qa5
    "e2e4 d7d6",  # 177: 1.e4 d6
    "e2e4 d7d6 d2d4",  # 178: 1.e4 d6 2.d4
    "e2e4 d7d6 d2d4 g8f6",  # 179: Pirc
    "e2e4 d7d6 d2d4 g8f6 b1c3",  # 180: Pirc 3.Nc3
    "e2e4 d7d6 d2d4 g8f6 b1c3 g7g6",  # 181: Pirc Classical
    "e2e4 d7d6 d2d4 g8f6 b1c3 g7g6 f2f4 f8g7 g1f3",  # 182: Pirc Austrian
    "e2e4 d7d6 d2d4 g8f6 b1c3 g7g6 g1f3 f8g7",  # 183: Pirc 4.Nf3
    "e2e4 e7e5",  # 184: 1.e4 e5
    "e2e4 e7e5 f2f4",  # 185: King's Gambit
    "e2e4 e7e5 b1c3",  # 186: Vienna Game
    "e2e4 e7e5 g1f3",  # 187: 1.e4 e5 2.Nf3
    "e2e4 e7e5 g1f3 b8c6",  # 188: 1.e4 e5 2.Nf3 Nc6
    "e2e4 e7e5 g1f3 b8c6 f1b5",  # 189: Ruy Lopez
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6",  # 190: Ruy Lopez 3...a6
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4",  # 191: Ruy Lopez 4.Ba4
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6",  # 192: Ruy Lopez 4...Nf6
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1",  # 193: Ruy Lopez 5.O-O
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 b7b5 a4b3",  # 194: Ruy Lopez 6.Bb3
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7",  # 195: Closed Ruy Lopez
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 d7d6",  # 196: Closed RL 8.c3 setup
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 d7d6 c2c3 e8g8",  # 197: Closed RL main
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 d7d6 c2c3 e8g8 h2h3",  # 198: Closed RL 9.h3
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 d7d6 c2c3 e8g8 h2h3 c6a5 b3c2 c7c5 d2d4",  # 199: Chigorin RL
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 e8g8",  # 200: Marshall Attack
    "e2e4 e7e5 g1f3 b8c6 f1b5 g8f6",  # 201: Berlin Defense
    "e2e4 e7e5 g1f3 b8c6 f1c4",  # 202: Italian Game
    "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6",  # 203: Two Knights
    "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4",  # 204: Scotch Game
    "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4",  # 205: Scotch 4.Nxd4
    "e2e4 e7e5 g1f3 b8c6 b1c3",  # 206: Four Knights
    "e2e4 e7e5 g1f3 b8c6 b1c3 g8f6",  # 207: Four Knights 3...Nf6
    "e2e4 e7e5 g1f3 g8f6",  # 208: Petroff Defense
    "e2e4 e7e5 g1f3 g8f6 f3e5 d7d6",  # 209: Petroff 3...d6
    "e2e4 e7e6",  # 210: French Defense
    "e2e4 e7e6 d2d3",  # 211: King's Indian Attack
    "e2e4 e7e6 d2d4",  # 212: French 2.d4
    "e2e4 e7e6 d2d4 d7d5",  # 213: French Defense main
    "e2e4 e7e6 d2d4 d7d5 e4e5 c7c5",  # 214: French Advance
    "e2e4 e7e6 d2d4 d7d5 e4e5 c7c5 c2c3",  # 215: French Advance 4.c3
    "e2e4 e7e6 d2d4 d7d5 b1c3",  # 216: French 3.Nc3
    "e2e4 e7e6 d2d4 d7d5 b1c3 f8b4",  # 217: French Winawer
    "e2e4 e7e6 d2d4 d7d5 b1c3 f8b4 e4e5",  # 218: Winawer 4.e5
    "e2e4 e7e6 d2d4 d7d5 b1c3 f8b4 e4e5 c7c5",  # 219: Winawer 4...c5
    "e2e4 e7e6 d2d4 d7d5 b1c3 f8b4 e4e5 c7c5 a2a3 b4c3 b2c3",  # 220: Winawer Poisoned Pawn
    "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6",  # 221: French Classical
    "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6 c1g5",  # 222: French Classical 4.Bg5
    "e2e4 e7e6 d2d4 d7d5 b1d2",  # 223: French Tarrasch
    "e2e4 e7e6 d2d4 d7d5 b1d2 c7c5",  # 224: Tarrasch 3...c5
    "e2e4 e7e6 d2d4 d7d5 b1d2 g8f6",  # 225: Tarrasch 3...Nf6
    "e2e4 e7e6 d2d4 d7d5 b1d2 g8f6 e4e5 f6d7",  # 226: Tarrasch 4.e5 Nfd7
    "e2e4 e7e6 d2d4 d7d5 b1d2 g8f6 e4e5 f6d7 f1d3 c7c5 c2c3 b8c6 g1e2",  # 227: Tarrasch main
    "e2e4 g7g6",  # 228: Modern Defense
    "e2e4 g7g6 d2d4",  # 229: Modern 2.d4
    "e2e4 g7g6 d2d4 f8g7",  # 230: Modern 2...Bg7
    "e2e4 g7g6 d2d4 f8g7 b1c3",  # 231: Modern 3.Nc3
    "e2e4 g7g6 d2d4 f8g7 b1c3 d7d6",  # 232: Modern/Pirc
    "e2e4 g8f6",  # 233: Alekhine Defense
    "e2e4 g8f6 e4e5 f6d5",  # 234: Alekhine 2...Nd5
    "e2e4 g8f6 e4e5 f6d5 d2d4 d7d6",  # 235: Alekhine 3...d6
    "e2e4 g8f6 e4e5 f6d5 d2d4 d7d6 g1f3",  # 236: Alekhine 4.Nf3
    "f2f4",  # 237: Bird's Opening
    "g2g3",  # 238: 1.g3
    "g1f3",  # 239: Reti Opening
    "g1f3 c7c5",  # 240: 1.Nf3 c5
    "g1f3 c7c5 c2c4",  # 241: 1.Nf3 c5 2.c4
    "g1f3 d7d5",  # 242: 1.Nf3 d5
    "g1f3 d7d5 c2c4",  # 243: 1.Nf3 d5 2.c4
    "g1f3 d7d5 d2d4",  # 244: 1.Nf3 d5 2.d4
    "g1f3 d7d5 g2g3",  # 245: 1.Nf3 d5 2.g3
    "g1f3 g7g6",  # 246: 1.Nf3 g6
    "g1f3 g8f6",  # 247: 1.Nf3 Nf6
    "g1f3 g8f6 c2c4",  # 248: 1.Nf3 Nf6 2.c4
    "g1f3 g8f6 c2c4 c7c5",  # 249: 1.Nf3 Nf6 2.c4 c5
    "g1f3 g8f6 c2c4 e7e6",  # 250: 1.Nf3 Nf6 2.c4 e6
    "g1f3 g8f6 c2c4 g7g6",  # 251: 1.Nf3 Nf6 2.c4 g6
    "g1f3 g8f6 c2c4 g7g6 b1c3",  # 252: 1.Nf3 Nf6 2.c4 g6 3.Nc3
    "g1f3 g8f6 g2g3",  # 253: 1.Nf3 Nf6 2.g3
    "g1f3 g8f6 g2g3 g7g6",  # 254: 1.Nf3 Nf6 2.g3 g6
)

# ---------------------------------------------------------------------------
# Convert to tuples of chess.Move at import time.
# STORED_LINES[code] is a tuple of chess.Move for codes 1..254.
# STORED_LINES[0] is an empty tuple (no match).
# ---------------------------------------------------------------------------
STORED_LINES: tuple[tuple[chess.Move, ...], ...] = tuple(
    tuple(chess.Move.from_uci(u) for u in uci.split()) if uci else ()
    for uci in _UCI_LINES
)

NUM_STORED_LINES = 254  # valid codes are 1..254


def classify(moves: Sequence[chess.Move]) -> int:
    """Return the stored line code (1–254) for a game's opening moves.

    Finds the longest stored line that is a prefix of *moves*.
    Returns 0 if no stored line matches.
    """
    best_code = 0
    best_len = 0
    n = len(moves)
    for code in range(1, NUM_STORED_LINES + 1):
        line = STORED_LINES[code]
        line_len = len(line)
        if line_len > n or line_len <= best_len:
            continue
        # Check if line is a prefix of moves
        match = True
        for i, lm in enumerate(line):
            gm = moves[i]
            # Compare from/to squares (castling is already in king-dest form)
            if lm.from_square != gm.from_square or lm.to_square != gm.to_square:
                match = False
                break
        if match:
            best_code = code
            best_len = line_len
    return best_code


def get_line_moves(code: int) -> tuple[chess.Move, ...]:
    """Return the move sequence for a stored line code (1–254)."""
    if 1 <= code <= NUM_STORED_LINES:
        return STORED_LINES[code]
    return ()
