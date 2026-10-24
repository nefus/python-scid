"""Constants for piece types, squares, flags, limits and mappings to python-chess."""

from __future__ import annotations

import chess

# ---------------------------------------------------------------------------
# SCID piece types (3-bit, stored in board array)
# ---------------------------------------------------------------------------
KING = 1
QUEEN = 2
ROOK = 3
BISHOP = 4
KNIGHT = 5
PAWN = 6
EMPTY = 7

# Color flag (bit 3)
WHITE = 0
BLACK = 8

WK, WQ, WR, WB, WN, WP = 1, 2, 3, 4, 5, 6
BK, BQ, BR, BB, BN, BP = 9, 10, 11, 12, 13, 14

# ---------------------------------------------------------------------------
# Square encoding  (A1=0, B1=1, ... H8=63)
# Same as python-chess: chess.A1 == 0, chess.H8 == 63
# ---------------------------------------------------------------------------


def square_rank(sq: int) -> int:
    return sq >> 3


def square_file(sq: int) -> int:
    return sq & 7


def make_square(file: int, rank: int) -> int:
    return (rank << 3) | file


# ---------------------------------------------------------------------------
# Name types
# ---------------------------------------------------------------------------
NAME_PLAYER = 0
NAME_EVENT = 1
NAME_SITE = 2
NAME_ROUND = 3
NAME_INFO = 4  # SCID5 DB metadata

# ---------------------------------------------------------------------------
# Result codes  (2-bit)
# ---------------------------------------------------------------------------
RESULT_NONE = 0  # *
RESULT_WHITE = 1  # 1-0
RESULT_BLACK = 2  # 0-1
RESULT_DRAW = 3  # 1/2-1/2

RESULT_STR = {
    RESULT_NONE: "*",
    RESULT_WHITE: "1-0",
    RESULT_BLACK: "0-1",
    RESULT_DRAW: "1/2-1/2",
}

RESULT_FROM_STR = {v: k for k, v in RESULT_STR.items()}

# ---------------------------------------------------------------------------
# Rating types (3-bit)
# ---------------------------------------------------------------------------
RATING_Elo = 0
RATING_Rating = 1
RATING_Rapid = 2
RATING_ICCF = 3
RATING_USCF = 4
RATING_DWZ = 5
RATING_BCF = 6

# ---------------------------------------------------------------------------
# Index flags (22-bit bitmask)
# ---------------------------------------------------------------------------
FLAG_START = 0  # Non-standard start position
FLAG_PROMO = 1  # Contains promotions
FLAG_UPROMO = 2  # Contains under-promotions
FLAG_DELETE = 3
FLAG_WHITE_OP = 4
FLAG_BLACK_OP = 5
FLAG_MIDDLEGAME = 6
FLAG_ENDGAME = 7
FLAG_NOVELTY = 8
FLAG_PAWN_STRUCT = 9
FLAG_TACTICS = 10
FLAG_KINGSIDE = 11
FLAG_QUEENSIDE = 12
FLAG_BRILLIANCY = 13
FLAG_BLUNDER = 14
FLAG_USER = 15
FLAG_CUSTOM1 = 16
FLAG_CUSTOM2 = 17
FLAG_CUSTOM3 = 18
FLAG_CUSTOM4 = 19
FLAG_CUSTOM5 = 20
FLAG_CUSTOM6 = 21

# ---------------------------------------------------------------------------
# Move encoding markers (King piece index 0, codes 11-15)
# ---------------------------------------------------------------------------
ENCODE_NAG = 11
ENCODE_COMMENT = 12
ENCODE_START_MARKER = 13
ENCODE_END_MARKER = 14
ENCODE_END_GAME = 15

# ---------------------------------------------------------------------------
# Common PGN tags encoded with single-byte shortcuts (241-250)
# ---------------------------------------------------------------------------
COMMON_TAGS = [
    "WhiteCountry",  # 241
    "BlackCountry",  # 242
    "Annotator",  # 243
    "PlyCount",  # 244
    "EventDate",  # 245
    "Opening",  # 246
    "Variation",  # 247
    "Setup",  # 248
    "Source",  # 249
    "SetUp",  # 250
]

COMMON_TAG_BASE = 241

# ---------------------------------------------------------------------------
# Count encoding (4-bit -> actual count, non-linear)
# ---------------------------------------------------------------------------
COUNT_DECODE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50]


def encode_count(x: int) -> int:
    """Encode a count value (0..50+) into 4 bits."""
    if x <= 10:
        return x
    if x <= 12:
        return 10
    if x <= 17:
        return 11
    if x <= 24:
        return 12
    if x <= 34:
        return 13
    if x <= 44:
        return 14
    return 15


def decode_count(raw: int) -> int:
    """Decode a 4-bit raw value to an approximate count."""
    return COUNT_DECODE[raw & 0x0F]


# ---------------------------------------------------------------------------
# Game data block size (128 KB)
# ---------------------------------------------------------------------------
GAME_BLOCK_SIZE = 131072  # 2^17

# ---------------------------------------------------------------------------
# SCID format versions
# ---------------------------------------------------------------------------
SCID4_MAGIC_INDEX = b"Scid.si\x00"  # 8 bytes
SCID4_MAGIC_NAMEBASE = b"Scid.sn\x00"  # 8 bytes
SCID4_VERSION = 400
SCID4_INDEX_HEADER_SIZE = 182
SCID4_INDEX_ENTRY_SIZE = 47

SCID5_INDEX_ENTRY_SIZE = 56

# ---------------------------------------------------------------------------
# SCID piece/square <-> python-chess mappings
# Square encoding is identical: both use A1=0, H8=63
# ---------------------------------------------------------------------------

SCID_PIECE_TO_CHESS = {
    WK: chess.Piece(chess.KING, chess.WHITE),
    WQ: chess.Piece(chess.QUEEN, chess.WHITE),
    WR: chess.Piece(chess.ROOK, chess.WHITE),
    WB: chess.Piece(chess.BISHOP, chess.WHITE),
    WN: chess.Piece(chess.KNIGHT, chess.WHITE),
    WP: chess.Piece(chess.PAWN, chess.WHITE),
    BK: chess.Piece(chess.KING, chess.BLACK),
    BQ: chess.Piece(chess.QUEEN, chess.BLACK),
    BR: chess.Piece(chess.ROOK, chess.BLACK),
    BB: chess.Piece(chess.BISHOP, chess.BLACK),
    BN: chess.Piece(chess.KNIGHT, chess.BLACK),
    BP: chess.Piece(chess.PAWN, chess.BLACK),
}

CHESS_PIECE_TO_SCID_TYPE = {
    chess.KING: KING,
    chess.QUEEN: QUEEN,
    chess.ROOK: ROOK,
    chess.BISHOP: BISHOP,
    chess.KNIGHT: KNIGHT,
    chess.PAWN: PAWN,
}
