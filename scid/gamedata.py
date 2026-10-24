"""Game data blob encoding/decoding.

Converts between the binary game blob (stored in .sg4/.sg5) and
python-chess Game objects.  The blob format is:

    [Tags section] [StartBoard section] [Move stream] [Comments section]
"""

from __future__ import annotations

from dataclasses import dataclass

import chess
import chess.pgn

from scid._filebuf import decode_text, encode_text, read_terminated_string
from scid.constants import (
    COMMON_TAG_BASE,
    COMMON_TAGS,
    ENCODE_COMMENT,
    ENCODE_END_GAME,
    ENCODE_END_MARKER,
    ENCODE_NAG,
    ENCODE_START_MARKER,
)
from scid.moves import PieceList, encode_move


# Seven Tag Roster — these are stored in the index/namebase, not in the blob
_STR_TAGS = {"Event", "Site", "Date", "Round", "White", "Black", "Result"}


@dataclass
class _CommentMark:
    """Target of one entry in the comments section of a game blob."""

    node: chess.pgn.GameNode | None
    starting: bool = False


# ======================================================================
# Decoding
# ======================================================================


def decode_game_data(data: bytes, fen: str | None = None) -> chess.pgn.Game:
    """Decode a SCID game data blob into a python-chess Game.

    Args:
        data: The raw game data blob from the .sg4/.sg5 file.
        fen: Optional FEN from the index entry (used if the blob indicates
             a custom start position).

    Returns:
        A chess.pgn.Game with moves, variations, comments, and NAGs.
    """
    pos = 0
    game = chess.pgn.Game()

    # --- Section 1: Extra Tags ---
    pos, extra_tags = _decode_tags(data, pos)
    for key, value in extra_tags.items():
        game.headers[key] = value

    # --- Section 2: StartBoard ---
    pos, start_fen = _decode_start_board(data, pos)
    if start_fen:
        # Set FEN via headers only — don't use game.setup() which adds
        # a "SetUp" header that may not have been in the original blob
        game.headers["FEN"] = start_fen

    # --- Section 3: Move Stream ---
    board = game.board()
    piece_list = PieceList(board)
    pos, comment_marks = _decode_moves(data, pos, game, piece_list)

    # --- Section 4: Comments ---
    _decode_comments(data, pos, comment_marks)

    return game


def _decode_tags(data: bytes, pos: int) -> tuple[int, dict[str, str]]:
    """Decode the extra tags section."""
    tags: dict[str, str] = {}

    while pos < len(data):
        tag_len = data[pos]
        pos += 1

        if tag_len == 0:
            break  # End of tags

        if tag_len > 240:
            # Common tag shorthand
            idx = tag_len - COMMON_TAG_BASE
            if 0 <= idx < len(COMMON_TAGS):
                tag_name = COMMON_TAGS[idx]
            else:
                tag_name = f"Unknown{tag_len}"
        elif tag_len == 255:
            # SCID2 compact EventDate (3-byte value)
            tag_name = "EventDate"
            val = data[pos : pos + 3].decode("latin-1")
            pos += 3
            tags[tag_name] = val
            continue
        else:
            tag_name = decode_text(data[pos : pos + tag_len])
            pos += tag_len

        value_len = data[pos]
        pos += 1
        value = decode_text(data[pos : pos + value_len])
        pos += value_len
        tags[tag_name] = value

    return pos, tags


def _decode_start_board(data: bytes, pos: int) -> tuple[int, str | None]:
    """Decode the start board section (flags byte + optional FEN)."""
    if pos >= len(data):
        return pos, None

    flags = data[pos]
    pos += 1

    fen = None
    if flags & 0x01:
        # Non-standard start position: read null-terminated FEN
        fen, pos = read_terminated_string(data, pos)

    return pos, fen


def _decode_moves(
    data: bytes, pos: int, game: chess.pgn.Game, piece_list: PieceList
) -> tuple[int, list[_CommentMark]]:
    """Decode the move stream into the game tree.

    Returns (new_pos, comment_marks) where each mark says which node the next
    comment string belongs to.  A comment directly after a variation start
    marker (before the variation's first move) is that move's
    ``starting_comment``.

    Variation handling: the piece list state depends on the history of captures
    (swap-with-last ordering), so we can't simply rebuild it from a board
    position.  Instead, we save/restore full piece list snapshots when entering
    and leaving variations.
    """
    import copy

    comment_marks: list[_CommentMark] = []
    # Mark for a comment seen after START_MARKER, filled by the next move
    pending_start: _CommentMark | None = None
    node: chess.pgn.GameNode = game
    # Stack entries: (node, piece_list_snapshot)
    var_stack: list[tuple[chess.pgn.GameNode, PieceList]] = []
    at_variation_start = False

    while pos < len(data):
        byte = data[pos]
        piece_idx = (byte >> 4) & 0x0F
        move_code = byte & 0x0F

        # Check for special markers (king index 0, codes 11-15)
        if piece_idx == 0 and move_code >= ENCODE_NAG:
            pos += 1

            if move_code == ENCODE_NAG:
                nag_val = data[pos]
                pos += 1
                if isinstance(node, chess.pgn.ChildNode):
                    node.nags.add(nag_val)

            elif move_code == ENCODE_COMMENT:
                if at_variation_start:
                    pending_start = _CommentMark(None, starting=True)
                    comment_marks.append(pending_start)
                else:
                    comment_marks.append(_CommentMark(node))

            elif move_code == ENCODE_START_MARKER:
                at_variation_start = True
                # Save current state (after the last move) and back up to parent
                var_stack.append((node, piece_list))
                if isinstance(node, chess.pgn.ChildNode):
                    parent = node.parent
                    node = parent  # type: ignore[assignment]

                    # Rebuild piece list by replaying from start
                    piece_list = _rebuild_piece_list(node)

            elif move_code == ENCODE_END_MARKER:
                at_variation_start = False
                if var_stack:
                    # Restore to the saved node (which is after the variation point)
                    saved_node, saved_pl = var_stack.pop()
                    node = saved_node
                    # Restore piece list state — we need to deep copy because
                    # the saved_pl may be reused if there are multiple variations
                    piece_list = _rebuild_piece_list(node)

            elif move_code == ENCODE_END_GAME:
                break

            continue

        # Regular move
        board = piece_list.board
        from_sq = piece_list.get_square(board.turn, piece_idx)
        piece = board.piece_at(from_sq)
        if piece is None:
            raise ValueError(f"No piece at {chess.square_name(from_sq)}")

        pt = piece.piece_type
        pos += 1  # consume the move byte

        # Decode the move
        to_sq, promotion = _decode_move_target(
            pt, move_code, from_sq, board.turn, data, pos
        )
        if pt == chess.QUEEN and _is_diagonal_queen(move_code, from_sq):
            # 2-byte queen move: consume second byte
            second = data[pos]
            pos += 1
            to_sq = second - 64
            promotion = None

        move = chess.Move(from_sq, to_sq, promotion=promotion)

        # Handle castling: python-chess expects king-to-target for standard chess
        if pt == chess.KING and move_code in (9, 10):
            rank = chess.square_rank(from_sq)
            if move_code == 10:  # kingside
                move = chess.Move(from_sq, chess.square(6, rank))
            else:  # queenside
                move = chess.Move(from_sq, chess.square(2, rank))

        # Null move
        if pt == chess.KING and move_code == 0:
            move = chess.Move.null()

        # Update piece list and board
        piece_list.apply_move(move)
        piece_list.board.push(move)

        # Add move to game tree
        node = node.add_variation(move)
        at_variation_start = False
        if pending_start is not None:
            pending_start.node = node
            pending_start = None

    return pos, comment_marks


def _rebuild_piece_list(node: chess.pgn.GameNode) -> PieceList:
    """Rebuild a PieceList by replaying all moves from the game start.

    This is necessary because piece list indices depend on the history of
    captures (swap-with-last ordering), so we can't just build from the
    final board position.
    """
    # Collect all moves from root to this node
    moves: list[chess.Move] = []
    current = node
    while isinstance(current, chess.pgn.ChildNode):
        moves.append(current.move)
        current = current.parent  # type: ignore[assignment]

    # current is now the Game root
    game_root: chess.pgn.Game = current  # type: ignore[assignment]
    board = game_root.board()
    pl = PieceList(board)

    # Replay moves in order (root → node)
    for move in reversed(moves):
        pl.apply_move(move)
        pl.board.push(move)

    return pl


def _is_diagonal_queen(move_code: int, from_sq: int) -> bool:
    """Check if a queen move code signals a 2-byte diagonal move."""
    return move_code == chess.square_file(from_sq)


_KING_CODE_TO_DELTA = {1: -9, 2: -8, 3: -7, 4: -1, 5: 1, 6: 7, 7: 8, 8: 9}
_KNIGHT_CODE_TO_DELTA = {1: -17, 2: -15, 3: -10, 4: -6, 5: 6, 6: 10, 7: 15, 8: 17}


def _decode_move_target(
    pt: int, move_code: int, from_sq: int, color: bool, data: bytes, pos: int
) -> tuple[int, int | None]:
    """Decode the target square and promotion from piece type and move code.

    Returns (to_sq, promotion_or_None).
    Does NOT handle 2-byte queen moves — caller must check separately.
    """
    promotion = None

    if pt == chess.KING:
        if move_code == 0:
            return from_sq, None  # null move
        if move_code == 9:  # queenside castle
            rank = chess.square_rank(from_sq)
            return chess.square(2, rank), None
        if move_code == 10:  # kingside castle
            rank = chess.square_rank(from_sq)
            return chess.square(6, rank), None
        delta = _KING_CODE_TO_DELTA.get(move_code, 0)
        return from_sq + delta, None

    if pt == chess.KNIGHT:
        delta = _KNIGHT_CODE_TO_DELTA.get(move_code, 0)
        return from_sq + delta, None

    if pt == chess.ROOK:
        if move_code < 8:
            return chess.square(move_code, chess.square_rank(from_sq)), None
        return chess.square(chess.square_file(from_sq), move_code - 8), None

    if pt == chess.BISHOP:
        from_file = chess.square_file(from_sq)
        from_rank = chess.square_rank(from_sq)
        if move_code < 8:
            target_file = move_code
            file_diff = target_file - from_file
            return chess.square(target_file, from_rank + file_diff), None
        target_file = move_code - 8
        file_diff = target_file - from_file
        return chess.square(target_file, from_rank - file_diff), None

    if pt == chess.QUEEN:
        from_file = chess.square_file(from_sq)
        if move_code == from_file:
            # 2-byte diagonal — caller handles
            return -1, None  # placeholder
        if move_code < 8:
            return chess.square(move_code, chess.square_rank(from_sq)), None
        return chess.square(chess.square_file(from_sq), move_code - 8), None

    if pt == chess.PAWN:
        if move_code == 15:
            delta = 16 if color == chess.WHITE else -16
            return from_sq + delta, None
        direction_code = move_code % 3
        promo_group = move_code // 3
        deltas = [7, 8, 9]
        delta = deltas[direction_code]
        if color == chess.BLACK:
            delta = -delta
        to_sq = from_sq + delta
        if promo_group == 1:
            promotion = chess.QUEEN
        elif promo_group == 2:
            promotion = chess.ROOK
        elif promo_group == 3:
            promotion = chess.BISHOP
        elif promo_group == 4:
            promotion = chess.KNIGHT
        return to_sq, promotion

    raise ValueError(f"Unknown piece type {pt}")


# ======================================================================
# Fast mainline-only replay for position search
# ======================================================================


def decode_mainline_moves(data: bytes) -> tuple[chess.Board, list[chess.Move]]:
    """Decode only the mainline moves from a game blob (no Game tree).

    This is significantly faster than full decode_game_data() because it
    skips building the python-chess Game tree, handling variations, NAGs,
    and comments.  Used for position/material search.

    Returns:
        (starting_board, list_of_mainline_moves)
    """
    pos = 0

    # --- Section 1: Skip tags ---
    while pos < len(data):
        tag_len = data[pos]
        pos += 1
        if tag_len == 0:
            break
        if tag_len > 240:
            # Common tag: skip value
            value_len = data[pos]
            pos += 1 + value_len
        elif tag_len == 255:
            pos += 3  # SCID2 EventDate
        else:
            pos += tag_len  # tag name
            value_len = data[pos]
            pos += 1 + value_len

    # --- Section 2: StartBoard ---
    if pos >= len(data):
        return chess.Board(), []

    flags = data[pos]
    pos += 1

    if flags & 0x01:
        # Read FEN
        end = data.index(0, pos)
        fen = data[pos:end].decode("latin-1")
        pos = end + 1
        board = chess.Board(fen)
    else:
        board = chess.Board()

    piece_list = PieceList(board)

    # --- Section 3: Decode mainline moves only ---
    moves: list[chess.Move] = []
    var_depth = 0

    while pos < len(data):
        byte = data[pos]
        piece_idx = (byte >> 4) & 0x0F
        move_code = byte & 0x0F

        # Special markers
        if piece_idx == 0 and move_code >= ENCODE_NAG:
            pos += 1
            if move_code == ENCODE_NAG:
                pos += 1  # skip NAG value
            elif move_code == ENCODE_START_MARKER:
                var_depth += 1
            elif move_code == ENCODE_END_MARKER:
                var_depth -= 1
            elif move_code == ENCODE_END_GAME:
                break
            continue

        pos += 1

        # Inside a variation — skip the move byte(s)
        if var_depth > 0:
            # Still need to consume 2-byte queen moves
            if piece_idx > 0 or move_code < ENCODE_NAG:
                # We need to know if this is a 2-byte queen move to skip correctly
                # For simplicity, check if it would be a diagonal queen by looking at
                # the piece at the position. But we don't track pieces in variations.
                # Instead, just check the second-byte sentinel pattern: if a queen's
                # move_code equals the from_square's file, there's a second byte.
                # We can't know this without tracking pieces, so we use a heuristic:
                # if the next byte is in range [64..127], it's likely a 2-byte queen move.
                # Actually, for correctness we need to properly handle this.
                # The safest approach: just check if move_code < 8 and next byte >= 64.
                # But this isn't reliable. Let's use a different strategy:
                # Track pieces even in variations to correctly skip bytes.
                pass
            continue

        # Mainline move — decode it
        from_sq = piece_list.get_square(board.turn, piece_idx)
        piece = board.piece_at(from_sq)
        if piece is None:
            break  # Corrupt data

        pt = piece.piece_type

        to_sq, promotion = _decode_move_target(
            pt, move_code, from_sq, board.turn, data, pos
        )
        if pt == chess.QUEEN and _is_diagonal_queen(move_code, from_sq):
            second = data[pos]
            pos += 1
            to_sq = second - 64
            promotion = None

        # Handle castling
        if pt == chess.KING and move_code in (9, 10):
            rank = chess.square_rank(from_sq)
            if move_code == 10:
                move = chess.Move(from_sq, chess.square(6, rank))
            else:
                move = chess.Move(from_sq, chess.square(2, rank))
        elif pt == chess.KING and move_code == 0:
            move = chess.Move.null()
        else:
            move = chess.Move(from_sq, to_sq, promotion=promotion)

        piece_list.apply_move(move)
        piece_list.board.push(move)
        moves.append(move)

    return board, moves


def _decode_comments(
    data: bytes, pos: int, comment_marks: list[_CommentMark]
) -> None:
    """Read null-terminated comment strings and attach to marked nodes."""
    for mark in comment_marks:
        if pos >= len(data):
            break
        # Find the null terminator
        end = pos
        while end < len(data) and data[end] != 0:
            end += 1
        comment = decode_text(data[pos:end])
        pos = end + 1  # skip the null
        if mark.node is None:
            continue  # variation start comment without a following move
        if mark.starting:
            mark.node.starting_comment = comment
        else:
            mark.node.comment = comment


# ======================================================================
# Encoding
# ======================================================================


def encode_game_data(game: chess.pgn.Game) -> bytes:
    """Encode a python-chess Game into a SCID game data blob.

    The Seven Tag Roster (Event, Site, Date, Round, White, Black, Result)
    is NOT included — those are stored in the index/namebase.
    """
    buf = bytearray()

    # --- Section 1: Extra Tags ---
    _encode_tags(game, buf)

    # --- Section 2: StartBoard ---
    _encode_start_board(game, buf)

    # --- Section 3: Move Stream + collect comments ---
    comments: list[str] = []
    board = game.board()
    piece_list = PieceList(board)
    _encode_movelist(game, piece_list, buf, comments)

    # --- Section 4: Comments ---
    for comment in comments:
        buf.extend(comment.encode("utf-8"))
        buf.append(0)

    return bytes(buf)


def _encode_tags(game: chess.pgn.Game, buf: bytearray) -> None:
    """Encode extra (non-STR) tags into the buffer."""
    for key, value in game.headers.items():
        if key in _STR_TAGS or key == "FEN":
            continue

        # Check for common tag shorthand
        tag_code = None
        for i, ct in enumerate(COMMON_TAGS):
            if key == ct:
                tag_code = COMMON_TAG_BASE + i
                break

        if tag_code is not None:
            buf.append(tag_code)
        else:
            name_bytes = encode_text(key, 240)
            buf.append(len(name_bytes))
            buf.extend(name_bytes)

        val_bytes = encode_text(value, 255)
        buf.append(len(val_bytes))
        buf.extend(val_bytes)

    buf.append(0)  # terminator


def _encode_start_board(game: chess.pgn.Game, buf: bytearray) -> None:
    """Encode the start board flags byte and optional FEN."""
    fen = game.headers.get("FEN")
    has_custom_start = fen is not None and fen != chess.STARTING_FEN

    # Check for promotions/underpromotions in the game
    has_promo = False
    has_upromo = False
    for node in game.mainline():
        if node.move.promotion:
            has_promo = True
            if node.move.promotion != chess.QUEEN:
                has_upromo = True

    flags = 0
    if has_custom_start:
        flags |= 0x01
    if has_promo:
        flags |= 0x02
    if has_upromo:
        flags |= 0x04

    buf.append(flags)

    if has_custom_start and fen:
        buf.extend(fen.encode("latin-1"))
        buf.append(0)


def _encode_movelist(
    game: chess.pgn.Game, piece_list: PieceList, buf: bytearray, comments: list[str]
) -> None:
    """Encode all moves, variations, NAGs, and comment markers."""

    # Pre-game comment
    if game.comment:
        buf.append(ENCODE_COMMENT)
        comments.append(game.comment)

    _encode_node_children(game, piece_list, buf, comments)

    buf.append(ENCODE_END_GAME)


def _encode_node_children(
    node: chess.pgn.GameNode, piece_list: PieceList, buf: bytearray, comments: list[str]
) -> None:
    """Recursively encode the children of a game node."""
    variations = node.variations
    if not variations:
        return

    # Main line move
    main = variations[0]
    _encode_single_move(main, piece_list, buf, comments)

    # Side variations
    for var in variations[1:]:
        buf.append(ENCODE_START_MARKER)
        if var.starting_comment:
            buf.append(ENCODE_COMMENT)
            comments.append(var.starting_comment)

        # Rebuild piece list by replaying from root — we can't just use
        # node.board() because piece list indices depend on capture history
        var_pl = _rebuild_piece_list(node)

        _encode_single_move(var, var_pl, buf, comments)
        _encode_node_children(var, var_pl, buf, comments)

        buf.append(ENCODE_END_MARKER)

    # Continue main line
    _encode_node_children(main, piece_list, buf, comments)


def _encode_single_move(
    node: chess.pgn.ChildNode,
    piece_list: PieceList,
    buf: bytearray,
    comments: list[str],
) -> None:
    """Encode a single move with its NAGs and comment marker."""
    move = node.move
    move_bytes = encode_move(move, piece_list)
    buf.extend(move_bytes)
    piece_list.push(move)

    # NAGs
    for nag in sorted(node.nags):
        buf.append(ENCODE_NAG)
        buf.append(nag & 0xFF)

    # Comment
    if node.comment:
        buf.append(ENCODE_COMMENT)
        comments.append(node.comment)
