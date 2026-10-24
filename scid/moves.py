"""SCID move encoding/decoding.

Converts between SCID's compact 1-byte (or 2-byte for diagonal queen)
move encoding and python-chess moves.

The encoding uses a piece-list indexed by piece number (0-15 per side),
where King is always index 0.  The 1-byte format is:

    byte = (piece_index << 4) | move_code

The move_code interpretation depends on the piece type.
"""

from __future__ import annotations

import chess

# ---------------------------------------------------------------------------
# Lookup tables for move encoding/decoding
# ---------------------------------------------------------------------------

# King direction: delta -> move_code
# Deltas: -9, -8, -7, -1, +1, +7, +8, +9 -> codes 1-8
_KING_DELTA_TO_CODE: dict[int, int] = {
    -9: 1,
    -8: 2,
    -7: 3,
    -1: 4,
    1: 5,
    7: 6,
    8: 7,
    9: 8,
}
_KING_CODE_TO_DELTA: dict[int, int] = {v: k for k, v in _KING_DELTA_TO_CODE.items()}

# Knight direction: delta -> move_code
_KNIGHT_DELTAS = [-17, -15, -10, -6, 6, 10, 15, 17]
_KNIGHT_DELTA_TO_CODE: dict[int, int] = {d: i + 1 for i, d in enumerate(_KNIGHT_DELTAS)}
_KNIGHT_CODE_TO_DELTA: dict[int, int] = {i + 1: d for i, d in enumerate(_KNIGHT_DELTAS)}

# Pawn move code definitions
# Direction codes: 0=capture_left(diff7), 1=forward(diff8), 2=capture_right(diff9)
# Promotion: none(0-2), Q(3-5), R(6-8), B(9-11), N(12-14), double_push(15)
_PAWN_PROMO_PIECES = [
    None,
    None,
    None,
    chess.QUEEN,
    chess.QUEEN,
    chess.QUEEN,
    chess.ROOK,
    chess.ROOK,
    chess.ROOK,
    chess.BISHOP,
    chess.BISHOP,
    chess.BISHOP,
    chess.KNIGHT,
    chess.KNIGHT,
    chess.KNIGHT,
    None,
]
_PAWN_DIRECTIONS = [7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9, 7, 8, 9, 16]


class PieceList:
    """Manages the SCID piece list (0-15 per side) for move encoding/decoding.

    The piece list maps piece indices to board squares.  King is always at
    index 0.  For the standard starting position, SCID uses a hardcoded order:

        0: King (e1/e8)
        1: Rook (a1/a8)
        2: Knight (b1/b8)
        3: Bishop (c1/c8)
        4: Queen (d1/d8)
        5: Bishop (f1/f8)
        6: Knight (g1/g8)
        7: Rook (h1/h8)
        8-15: Pawns a-h (a2-h2 / a7-h7)

    For FEN positions, pieces are added via AddPiece() which scans rank 8→1
    (FEN order).  King goes to index 0 (bumping whatever was there); others
    are appended in scan order.

    Piece count tracks how many active pieces each side has (for swap-and-pop
    capture handling).
    """

    # Standard starting position piece list (same for both colors, mirrored)
    _STD_ORDER_WHITE = [
        chess.E1,  # King
        chess.A1,  # Rook
        chess.B1,  # Knight
        chess.C1,  # Bishop
        chess.D1,  # Queen
        chess.F1,  # Bishop
        chess.G1,  # Knight
        chess.H1,  # Rook
        chess.A2,
        chess.B2,
        chess.C2,
        chess.D2,  # Pawns a-d
        chess.E2,
        chess.F2,
        chess.G2,
        chess.H2,  # Pawns e-h
    ]
    _STD_ORDER_BLACK = [
        chess.E8,  # King
        chess.A8,  # Rook
        chess.B8,  # Knight
        chess.C8,  # Bishop
        chess.D8,  # Queen
        chess.F8,  # Bishop
        chess.G8,  # Knight
        chess.H8,  # Rook
        chess.A7,
        chess.B7,
        chess.C7,
        chess.D7,  # Pawns a-d
        chess.E7,
        chess.F7,
        chess.G7,
        chess.H7,  # Pawns e-h
    ]

    def __init__(self, board: chess.Board) -> None:
        self.board = board
        # Build piece lists for both sides
        self._lists: list[list[int]] = [[], []]  # [WHITE, BLACK]
        self._sq_to_idx: list[dict[int, int]] = [{}, {}]  # square -> piece index
        self._count: list[int] = [0, 0]  # active piece count per side
        self._rebuild()

    def _rebuild(self) -> None:
        """Build piece lists matching SCID's ordering."""
        if self._is_standard_start():
            self._init_standard()
        else:
            self._init_from_fen()

    def _is_standard_start(self) -> bool:
        """Check if the board is at the standard starting position."""
        return self.board.fen() == chess.STARTING_FEN

    def _init_standard(self) -> None:
        """Initialize piece lists for the standard starting position."""
        self._lists[chess.WHITE] = list(self._STD_ORDER_WHITE)
        self._lists[chess.BLACK] = list(self._STD_ORDER_BLACK)
        self._count[chess.WHITE] = 16
        self._count[chess.BLACK] = 16
        for color in (chess.WHITE, chess.BLACK):
            self._sq_to_idx[color] = {sq: i for i, sq in enumerate(self._lists[color])}

    def _init_from_fen(self) -> None:
        """Initialize piece lists for a non-standard position.

        Matches SCID's AddPiece() behavior: FEN is parsed rank 8→1, file a→h.
        King always goes to index 0 (bumping whatever piece was already there).
        Non-king pieces are appended in order.
        """
        for color in (chess.WHITE, chess.BLACK):
            pieces: list[int] = []

            # Scan rank 8→1, file a→h (FEN order), matching SCID's
            # FEN_parsePieces which calls AddPiece in this order
            for rank in range(7, -1, -1):  # 7,6,5,...,0
                for file in range(8):  # 0,1,...,7
                    sq = chess.square(file, rank)
                    p = self.board.piece_at(sq)
                    if p is not None and p.color == color:
                        if p.piece_type == chess.KING:
                            # King goes to index 0; bump existing piece
                            if pieces:
                                pieces.insert(0, sq)
                                # The piece that was at 0 gets moved to end
                                # Actually, SCID does: List[Count] = old_sq_at_0
                                # then List[0] = king_sq
                                # which means: swap index 0 to the current Count position
                                old_sq = pieces[1]  # what was at index 0
                                pieces[0] = sq
                                pieces.pop(1)
                                pieces.append(old_sq)
                            else:
                                pieces.append(sq)
                        else:
                            pieces.append(sq)

            self._lists[color] = pieces
            self._count[color] = len(pieces)
            self._sq_to_idx[color] = {sq: i for i, sq in enumerate(pieces)}

    def get_square(self, color: bool, index: int) -> int:
        """Get the square of piece at given index for given color."""
        return self._lists[color][index]

    def get_index(self, color: bool, sq: int) -> int:
        """Get the piece index for a piece at the given square."""
        return self._sq_to_idx[color][sq]

    def get_count(self, color: bool) -> int:
        """Get the number of active pieces for a color."""
        return self._count[color]

    def apply_move(self, move: chess.Move) -> None:
        """Update the piece list after a move is made on the board.

        Must be called BEFORE board.push(move) — we need the pre-move state
        to identify captured pieces.

        Matches SCID's FastBoard::move() and FastBoard::remove() logic:
        - On capture: decrement count, swap captured piece index with last valid
        - On promotion: change piece type (handled by board, we just update square)
        - On castling: move both king and rook
        """
        board = self.board
        from_sq = move.from_square
        to_sq = move.to_square
        moving_piece = board.piece_at(from_sq)
        if moving_piece is None:
            return

        color = moving_piece.color
        opp = not color

        # Handle castling — move the rook FIRST (before moving the king)
        if board.is_castling(move):
            king_file = chess.square_file(from_sq)
            to_file = chess.square_file(to_sq)
            rank = chess.square_rank(from_sq)
            if to_file > king_file:  # Kingside
                rook_from = chess.square(7, rank)
                rook_to = chess.square(5, rank)
            else:  # Queenside
                rook_from = chess.square(0, rank)
                rook_to = chess.square(3, rank)

            if rook_from in self._sq_to_idx[color]:
                rook_idx = self._sq_to_idx[color][rook_from]
                self._lists[color][rook_idx] = rook_to
                del self._sq_to_idx[color][rook_from]
                self._sq_to_idx[color][rook_to] = rook_idx

        # Move the piece (update from->to)
        if from_sq in self._sq_to_idx[color]:
            piece_idx = self._sq_to_idx[color][from_sq]
            self._lists[color][piece_idx] = to_sq
            del self._sq_to_idx[color][from_sq]
            self._sq_to_idx[color][to_sq] = piece_idx

        # Handle captures (including en passant) — SCID's remove() logic
        captured_sq = to_sq
        if board.is_en_passant(move):
            # En passant: captured pawn is on a different square
            captured_sq = to_sq + (-8 if color == chess.WHITE else 8)

        if captured_sq in self._sq_to_idx[opp]:
            cap_idx = self._sq_to_idx[opp][captured_sq]
            del self._sq_to_idx[opp][captured_sq]

            # Decrement count, then swap with last valid piece
            self._count[opp] -= 1
            last_valid_idx = self._count[opp]

            if cap_idx != last_valid_idx:
                # Swap: move last valid piece into the captured slot
                last_sq = self._lists[opp][last_valid_idx]
                self._lists[opp][cap_idx] = last_sq
                self._sq_to_idx[opp][last_sq] = cap_idx

    def push(self, move: chess.Move) -> None:
        """Update piece list and push move on internal board."""
        self.apply_move(move)
        self.board.push(move)


def encode_move(move: chess.Move, piece_list: PieceList) -> bytes:
    """Encode a chess.Move into SCID's 1-byte (or 2-byte) format.

    Must be called BEFORE piece_list.push(move).
    """
    board = piece_list.board
    to_move = board.turn
    from_sq = move.from_square
    to_sq = move.to_square

    piece = board.piece_at(from_sq)
    if piece is None:
        raise ValueError(f"No piece at {chess.square_name(from_sq)}")

    piece_idx = piece_list.get_index(to_move, from_sq)
    pt = piece.piece_type

    if pt == chess.KING:
        if move == chess.Move.null():
            return bytes([(piece_idx << 4) | 0])
        if board.is_castling(move):
            to_file = chess.square_file(to_sq)
            from_file = chess.square_file(from_sq)
            if to_file > from_file:
                code = 10  # kingside
            else:
                code = 9  # queenside
            return bytes([(piece_idx << 4) | code])
        delta = to_sq - from_sq
        code = _KING_DELTA_TO_CODE.get(delta)
        if code is None:
            raise ValueError(f"Invalid king delta: {delta}")
        return bytes([(piece_idx << 4) | code])

    elif pt == chess.KNIGHT:
        delta = to_sq - from_sq
        code = _KNIGHT_DELTA_TO_CODE.get(delta)
        if code is None:
            raise ValueError(f"Invalid knight delta: {delta}")
        return bytes([(piece_idx << 4) | code])

    elif pt == chess.ROOK:
        if chess.square_rank(from_sq) == chess.square_rank(to_sq):
            code = chess.square_file(to_sq)  # horizontal
        else:
            code = 8 + chess.square_rank(to_sq)  # vertical
        return bytes([(piece_idx << 4) | code])

    elif pt == chess.BISHOP:
        from_file = chess.square_file(from_sq)
        from_rank = chess.square_rank(from_sq)
        to_file = chess.square_file(to_sq)
        to_rank = chess.square_rank(to_sq)
        rank_diff = to_rank - from_rank
        file_diff = to_file - from_file
        if (rank_diff > 0 and file_diff > 0) or (rank_diff < 0 and file_diff < 0):
            # Up-right or down-left diagonal
            code = to_file
        else:
            # Up-left or down-right diagonal
            code = to_file + 8
        return bytes([(piece_idx << 4) | code])

    elif pt == chess.QUEEN:
        from_file = chess.square_file(from_sq)
        from_rank = chess.square_rank(from_sq)
        to_file = chess.square_file(to_sq)
        to_rank = chess.square_rank(to_sq)

        if from_rank == to_rank:
            # Horizontal
            code = to_file
            return bytes([(piece_idx << 4) | code])
        elif from_file == to_file:
            # Vertical
            code = 8 + to_rank
            return bytes([(piece_idx << 4) | code])
        else:
            # Diagonal: 2-byte encoding
            code = from_file  # sentinel
            return bytes([(piece_idx << 4) | code, to_sq + 64])

    elif pt == chess.PAWN:
        # SCID encodes pawns using abs(to - from), color-agnostic.
        # Direction codes: 0=capture-left(abs diff 7), 1=forward(8), 2=capture-right(9)
        # 15=double push(16)
        delta = abs(to_sq - from_sq)
        if delta == 16:
            code = 15  # double push
        else:
            if delta == 7:
                direction = 0  # capture left
            elif delta == 8:
                direction = 1  # forward
            else:  # delta == 9
                direction = 2  # capture right

            promo = move.promotion
            if promo is None:
                code = direction
            elif promo == chess.QUEEN:
                code = 3 + direction
            elif promo == chess.ROOK:
                code = 6 + direction
            elif promo == chess.BISHOP:
                code = 9 + direction
            elif promo == chess.KNIGHT:
                code = 12 + direction
            else:
                code = direction

        return bytes([(piece_idx << 4) | code])

    raise ValueError(f"Cannot encode piece type {pt}")
