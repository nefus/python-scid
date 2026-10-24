"""Material signature encoding (24-bit finalMatSig).

Layout (MSB to LSB):
  WQ(2) WR(2) WB(2) WN(2) WP(4) BQ(2) BR(2) BB(2) BN(2) BP(4)
"""

from __future__ import annotations

import chess


def encode_matsig(board: chess.Board) -> int:
    """Compute the 24-bit material signature for a position."""
    sig = 0

    # White pieces
    wq = min(len(board.pieces(chess.QUEEN, chess.WHITE)), 3)
    wr = min(len(board.pieces(chess.ROOK, chess.WHITE)), 3)
    wb = min(len(board.pieces(chess.BISHOP, chess.WHITE)), 3)
    wn = min(len(board.pieces(chess.KNIGHT, chess.WHITE)), 3)
    wp = min(len(board.pieces(chess.PAWN, chess.WHITE)), 8)

    # Black pieces
    bq = min(len(board.pieces(chess.QUEEN, chess.BLACK)), 3)
    br = min(len(board.pieces(chess.ROOK, chess.BLACK)), 3)
    bb = min(len(board.pieces(chess.BISHOP, chess.BLACK)), 3)
    bn = min(len(board.pieces(chess.KNIGHT, chess.BLACK)), 3)
    bp = min(len(board.pieces(chess.PAWN, chess.BLACK)), 8)

    sig = (
        (wq << 22)
        | (wr << 20)
        | (wb << 18)
        | (wn << 16)
        | (wp << 12)
        | (bq << 10)
        | (br << 8)
        | (bb << 6)
        | (bn << 4)
        | bp
    )
    return sig


def decode_matsig(sig: int) -> dict[str, int]:
    """Decode a 24-bit material signature to piece counts."""
    return {
        "WQ": (sig >> 22) & 3,
        "WR": (sig >> 20) & 3,
        "WB": (sig >> 18) & 3,
        "WN": (sig >> 16) & 3,
        "WP": (sig >> 12) & 0xF,
        "BQ": (sig >> 10) & 3,
        "BR": (sig >> 8) & 3,
        "BB": (sig >> 6) & 3,
        "BN": (sig >> 4) & 3,
        "BP": sig & 0xF,
    }


def matsig_matches(sig: int, pattern: dict[str, int]) -> bool:
    """Check if a material signature matches a pattern.

    The pattern dict may contain any subset of piece keys.
    """
    decoded = decode_matsig(sig)
    for key, val in pattern.items():
        if decoded.get(key) != val:
            return False
    return True
