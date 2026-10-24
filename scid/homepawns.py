"""Home pawn data encoding (9 bytes in index entry).

Tracks which pawns have moved from their initial squares (a2-h2 for white,
a7-h7 for black).  Used for fast position search filtering.

The first byte is the count of "change entries" (max 16).
The remaining 8 bytes hold up to 16 half-byte entries, each identifying
a pawn that moved from its home square during the game.

Each nibble stores a bit position (0-15) corresponding to SCID's 16-bit
home pawn signature bitmap:

  bit 15 (0x8000) = a2 (WP)    bit 7 (0x0080) = a7 (BP)
  bit 14 (0x4000) = b2 (WP)    bit 6 (0x0040) = b7 (BP)
  bit 13 (0x2000) = c2 (WP)    bit 5 (0x0020) = c7 (BP)
  bit 12 (0x1000) = d2 (WP)    bit 4 (0x0010) = d7 (BP)
  bit 11 (0x0800) = e2 (WP)    bit 3 (0x0008) = e7 (BP)
  bit 10 (0x0400) = f2 (WP)    bit 2 (0x0004) = f7 (BP)
  bit  9 (0x0200) = g2 (WP)    bit 1 (0x0002) = g7 (BP)
  bit  8 (0x0100) = h2 (WP)    bit 0 (0x0001) = h7 (BP)
"""

from __future__ import annotations

import chess

# Home squares for pawns
_WHITE_HOME = [
    chess.A2,
    chess.B2,
    chess.C2,
    chess.D2,
    chess.E2,
    chess.F2,
    chess.G2,
    chess.H2,
]
_BLACK_HOME = [
    chess.A7,
    chess.B7,
    chess.C7,
    chess.D7,
    chess.E7,
    chess.F7,
    chess.G7,
    chess.H7,
]


def encode_homepawns(board: chess.Board) -> bytes:
    """Compute home pawn data for the given position.

    Returns 9 bytes: 1 count byte + 8 data bytes.
    This encodes which pawns are *missing* from their home squares
    compared to the standard starting position.

    Each nibble is a bit position (0-15) in SCID's home pawn bitmap.
    White pawns: a2=15, b2=14, ... h2=8.
    Black pawns: a7=7, b7=6, ... h7=0.
    """
    changes: list[int] = []

    # Check white home pawns (a2-h2 -> bit positions 15-8)
    for i, sq in enumerate(_WHITE_HOME):
        piece = board.piece_at(sq)
        if (
            piece is None
            or piece.piece_type != chess.PAWN
            or piece.color != chess.WHITE
        ):
            changes.append(15 - i)  # a2=15, b2=14, ..., h2=8

    # Check black home pawns (a7-h7 -> bit positions 7-0)
    for i, sq in enumerate(_BLACK_HOME):
        piece = board.piece_at(sq)
        if (
            piece is None
            or piece.piece_type != chess.PAWN
            or piece.color != chess.BLACK
        ):
            changes.append(7 - i)  # a7=7, b7=6, ..., h7=0

    count = min(len(changes), 16)
    data = bytearray(9)
    data[0] = count

    for idx in range(count):
        byte_pos = 1 + idx // 2
        if idx % 2 == 0:
            data[byte_pos] |= (changes[idx] & 0x0F) << 4
        else:
            data[byte_pos] |= changes[idx] & 0x0F

    return bytes(data)


def decode_homepawns(data: bytes) -> list[int]:
    """Decode home pawn data (9 bytes) to list of bit positions (0-15).

    Each value is a bit position in SCID's 16-bit home pawn bitmap:
    15=a2, 14=b2, ..., 8=h2, 7=a7, 6=b7, ..., 0=h7.
    """
    if len(data) < 9:
        return []
    count = data[0]
    changes: list[int] = []
    for idx in range(min(count, 16)):
        byte_pos = 1 + idx // 2
        if idx % 2 == 0:
            changes.append((data[byte_pos] >> 4) & 0x0F)
        else:
            changes.append(data[byte_pos] & 0x0F)
    return changes
