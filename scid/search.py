"""Search functions for SCID databases.

Implements four search types matching SCID's native search algorithms:

1. **Header search** — Pure index-based filtering on names, dates, ECO,
   elo, results, flags. No game blob decoding required.

2. **Position search** — Find games containing a specific board position.
   Uses three-level pre-filtering (stored lines, home pawns, material
   signature) before full game replay.

3. **Material search** — Find games where material counts fall within
   specified ranges at any point during the game.

4. **Tree search** — Opening tree statistics from a position.  Finds all
   games containing a position, extracts the next move from each, and
   aggregates per-move statistics (frequency, results, Elo, performance).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import chess

from scid.constants import (
    FLAG_PROMO,
    FLAG_START,
    FLAG_UPROMO,
    NAME_EVENT,
    NAME_PLAYER,
    NAME_ROUND,
    NAME_SITE,
    RESULT_BLACK,
    RESULT_DRAW,
    RESULT_FROM_STR,
    RESULT_WHITE,
)
from scid.dates import date_from_pgn
from scid.eco import eco_encode
from scid.index import IndexEntry
from scid.matsig import decode_matsig
from scid.namebase import NameBase
from scid.stored_lines import get_line_moves


# ---------------------------------------------------------------------------
# Home-pawn signature matching
# ---------------------------------------------------------------------------


def _hpsig_make(board: chess.Board) -> tuple[int, int]:
    """Compute home-pawn signature bitmap and count of moved pawns.

    Returns (hp_sig, n_moved) where hp_sig is a 16-bit bitmap with bits set
    for each home pawn that is MISSING from its starting square.

    Bit positions: a2=15, b2=14, ..., h2=8, a7=7, b7=6, ..., h7=0.
    """
    hp_sig = 0
    n_moved = 0

    # White pawns on rank 2: bit positions 15 (a2) down to 8 (h2)
    for i, sq in enumerate(
        (chess.A2, chess.B2, chess.C2, chess.D2, chess.E2, chess.F2, chess.G2, chess.H2)
    ):
        piece = board.piece_at(sq)
        if (
            piece is None
            or piece.piece_type != chess.PAWN
            or piece.color != chess.WHITE
        ):
            hp_sig |= 1 << (15 - i)
            n_moved += 1

    # Black pawns on rank 7: bit positions 7 (a7) down to 0 (h7)
    for i, sq in enumerate(
        (chess.A7, chess.B7, chess.C7, chess.D7, chess.E7, chess.F7, chess.G7, chess.H7)
    ):
        piece = board.piece_at(sq)
        if (
            piece is None
            or piece.piece_type != chess.PAWN
            or piece.color != chess.BLACK
        ):
            hp_sig |= 1 << (7 - i)
            n_moved += 1

    return hp_sig, n_moved


def _hpsig_match(hp_sig: int, n_moved: int, home_pawn_data: bytes) -> bool:
    """Check if a game's home pawn change list is compatible with a target position.

    The change list (from IndexEntry.home_pawn_data) records which pawns left
    their home squares during the game, in order.  For the target position
    to be reachable, the first *n_moved* entries in the change list must
    produce exactly the same bitmap as *hp_sig*.

    Args:
        hp_sig: Target position's home pawn bitmap (16-bit).
        n_moved: Number of moved pawns in target position.
        home_pawn_data: 9-byte home pawn data from IndexEntry.

    Returns:
        True if the game could contain the target position.
    """
    if len(home_pawn_data) < 9:
        return True  # No data available — can't exclude

    count = home_pawn_data[0]

    # Special case: all 16 pawns moved
    if count == 16 and n_moved == 16:
        return True

    # If the game recorded fewer pawn changes than needed, can't match
    if count < n_moved:
        return False

    # No pawns moved from home in target position — matches any game
    if n_moved == 0:
        return hp_sig == 0

    # Reconstruct bitmap from first n_moved entries in the change list.
    # Inline nibble extraction to avoid list allocation.
    sig = 0
    for i in range(n_moved):
        byte_pos = 1 + i // 2
        if i % 2 == 0:
            bit_pos = (home_pawn_data[byte_pos] >> 4) & 0x0F
        else:
            bit_pos = home_pawn_data[byte_pos] & 0x0F
        sig |= 1 << bit_pos

    return sig == hp_sig


# ---------------------------------------------------------------------------
# Material signature pre-filtering
# ---------------------------------------------------------------------------


def _less_mat(
    target_counts: dict[str, int],
    final_matsig: int,
    has_promos: bool,
    has_upromos: bool,
) -> bool:
    """Check if target position requires LESS material than a game's final position.

    Material can only decrease during a game.  If the target position
    has fewer of any piece type than what's left at the end of the game,
    the target position can't have been reached during that game (because
    the count would have to drop below the final value, which didn't happen).

    Returns True if the target is definitely unreachable (game should be
    excluded from search).
    """
    final = decode_matsig(final_matsig)

    # Pawns can never increase — if target has fewer pawns than final,
    # the game never had that few pawns, so exclude.
    wp_diff = target_counts["WP"] - final["WP"]
    bp_diff = target_counts["BP"] - final["BP"]
    if wp_diff < 0 or bp_diff < 0:
        return True

    # Queens: if promotions happened, pawns could have become queens.
    # Extra pawns lost (wp_diff > 0 means target has more pawns than final,
    # so some pawns were captured) could explain extra queens appearing.
    wq_diff = target_counts["WQ"] - final["WQ"]
    bq_diff = target_counts["BQ"] - final["BQ"]
    if has_promos:
        wq_diff += wp_diff  # wp_diff >= 0 here, so this adds pawn surplus
        bq_diff += bp_diff
    if wq_diff < 0 or bq_diff < 0:
        return True

    # With underpromotions, minor pieces could reappear — can't check further
    if has_upromos:
        return False

    # Check non-queen pieces (can never increase without underpromotion)
    if (
        target_counts["WR"] < final["WR"]
        or target_counts["WB"] < final["WB"]
        or target_counts["WN"] < final["WN"]
        or target_counts["BR"] < final["BR"]
        or target_counts["BB"] < final["BB"]
        or target_counts["BN"] < final["BN"]
    ):
        return True

    return False


def _matsig_is_reachable(
    start_sig: int, target_sig: int, has_promos: bool, has_upromos: bool
) -> bool:
    """Check if a material signature transformation is possible.

    Can the game (with final material *start_sig*) have passed through a
    position with material *target_sig*?  Used for material search
    pre-filtering.
    """
    start = decode_matsig(start_sig)
    target = decode_matsig(target_sig)

    # Pawns can never increase
    if start["WP"] < target["WP"] or start["BP"] < target["BP"]:
        return False

    if has_upromos:
        return True

    # Non-queen pieces can't increase without underpromotion
    if (
        start["WR"] < target["WR"]
        or start["WB"] < target["WB"]
        or start["WN"] < target["WN"]
        or start["BR"] < target["BR"]
        or start["BB"] < target["BB"]
        or start["BN"] < target["BN"]
    ):
        return False

    if has_promos:
        return True

    # No promotions: queens can't increase either
    if start["WQ"] < target["WQ"] or start["BQ"] < target["BQ"]:
        return False

    return True


# ---------------------------------------------------------------------------
# Stored line matching
# ---------------------------------------------------------------------------


def _stored_line_match(
    target_board: chess.Board,
    game_stored_code: int,
    _cache: dict[int, list[chess.Board]] = {},
) -> int:
    """Check if a game's stored line code tells us about the target position.

    Uses a cache of board positions at each ply for each stored line code
    to avoid replaying moves on every call.

    Returns:
        >= 0: The ply at which the target position is reached.
        -1: Inconclusive (need full game decode).
        -2: Definitely no match.
    """
    if game_stored_code == 0:
        return -1  # No stored line — inconclusive

    game_line = get_line_moves(game_stored_code)
    if not game_line:
        return -1

    # Build/retrieve cached board positions for this stored line
    if game_stored_code not in _cache:
        boards: list[chess.Board] = []
        board = chess.Board()
        boards.append(board.copy())
        for move in game_line:
            board.push(move)
            boards.append(board.copy())
        _cache[game_stored_code] = boards

    cached_boards = _cache[game_stored_code]

    # Check each ply for a match
    for ply, board in enumerate(cached_boards):
        if board == target_board:
            return ply

    # The stored line didn't contain the target position.
    # But the game could continue beyond the stored line,
    # so we can't exclude it.
    return -1


# ---------------------------------------------------------------------------
# Header Search
# ---------------------------------------------------------------------------


@dataclass
class HeaderSearchCriteria:
    """Criteria for header-based search.

    All fields are optional.  Only non-None fields are checked.
    Multiple criteria are combined with AND logic.
    """

    # Name searches (case-insensitive substring unless quoted for exact match)
    white: str | None = None
    black: str | None = None
    player: str | None = None  # matches either white OR black
    event: str | None = None
    site: str | None = None
    round: str | None = None

    # Date range (inclusive, PGN format "YYYY.MM.DD")
    date_min: str | None = None
    date_max: str | None = None

    # Event date range
    event_date_min: str | None = None
    event_date_max: str | None = None

    # ECO range (inclusive, e.g. "B00" to "B99")
    eco_min: str | None = None
    eco_max: str | None = None

    # Elo range (inclusive)
    white_elo_min: int | None = None
    white_elo_max: int | None = None
    black_elo_min: int | None = None
    black_elo_max: int | None = None
    elo_min: int | None = None  # either player >= this
    elo_max: int | None = None  # either player <= this

    # Half-move count range
    length_min: int | None = None
    length_max: int | None = None

    # Result filter
    results: set[str] | None = None  # e.g. {"1-0", "0-1"}

    # Flag checks (list of flag bit indices that must be set)
    flags_set: list[int] | None = None
    # Flag checks (list of flag bit indices that must NOT be set)
    flags_unset: list[int] | None = None

    # Game number range (0-based, inclusive)
    game_num_min: int | None = None
    game_num_max: int | None = None


def _name_matches(pattern: str, name: str) -> bool:
    """Check if a name matches a search pattern.

    If *pattern* is quoted (starts and ends with '"'), performs exact match.
    Otherwise performs case-insensitive substring match, ignoring spaces.
    """
    if pattern.startswith('"') and pattern.endswith('"'):
        # Exact match (strip quotes)
        return name == pattern[1:-1]

    # Case-insensitive, space-ignoring substring match
    p = pattern.lower().replace(" ", "")
    n = name.lower().replace(" ", "")
    return p in n


def _build_name_mask(
    namebase: NameBase,
    name_type: int,
    pattern: str,
) -> set[int]:
    """Build a set of name IDs matching the pattern."""
    mask: set[int] = set()
    count = namebase.count(name_type)
    for name_id in range(count):
        name = namebase.get_name(name_type, name_id)
        if _name_matches(pattern, name):
            mask.add(name_id)
    return mask


def search_header(
    entries: Sequence[IndexEntry],
    namebase: NameBase,
    criteria: HeaderSearchCriteria,
    game_indices: Sequence[int] | None = None,
) -> list[int]:
    """Search games by header criteria.

    Args:
        entries: List of IndexEntry objects.
        namebase: The database's NameBase.
        criteria: Search criteria.
        game_indices: If given, only search these game indices.
            Otherwise search all games.

    Returns:
        List of matching game indices (0-based).
    """
    c = criteria
    indices: Sequence[int]
    if game_indices is not None:
        indices = game_indices
    else:
        indices = range(len(entries))

    # Pre-build name masks for string searches
    white_mask: set[int] | None = None
    black_mask: set[int] | None = None
    player_mask: set[int] | None = None
    event_mask: set[int] | None = None
    site_mask: set[int] | None = None
    round_mask: set[int] | None = None

    if c.white is not None:
        white_mask = _build_name_mask(namebase, NAME_PLAYER, c.white)
    if c.black is not None:
        black_mask = _build_name_mask(namebase, NAME_PLAYER, c.black)
    if c.player is not None:
        player_mask = _build_name_mask(namebase, NAME_PLAYER, c.player)
    if c.event is not None:
        event_mask = _build_name_mask(namebase, NAME_EVENT, c.event)
    if c.site is not None:
        site_mask = _build_name_mask(namebase, NAME_SITE, c.site)
    if c.round is not None:
        round_mask = _build_name_mask(namebase, NAME_ROUND, c.round)

    # Pre-compute numeric ranges
    date_min_val = date_from_pgn(c.date_min) if c.date_min else None
    date_max_val = date_from_pgn(c.date_max) if c.date_max else None
    edate_min_val = date_from_pgn(c.event_date_min) if c.event_date_min else None
    edate_max_val = date_from_pgn(c.event_date_max) if c.event_date_max else None
    eco_min_val = eco_encode(c.eco_min) if c.eco_min else None
    eco_max_val = eco_encode(c.eco_max) if c.eco_max else None
    result_vals: set[int] | None = None
    if c.results is not None:
        result_vals = {RESULT_FROM_STR.get(r, -1) for r in c.results}

    # Build list of filter functions for efficiency
    filters: list[Callable[[IndexEntry, int], bool]] = []

    if white_mask is not None:
        _wm = white_mask
        filters.append(lambda ie, _i, m=_wm: ie.white_id in m)
    if black_mask is not None:
        _bm = black_mask
        filters.append(lambda ie, _i, m=_bm: ie.black_id in m)
    if player_mask is not None:
        _pm = player_mask
        filters.append(lambda ie, _i, m=_pm: ie.white_id in m or ie.black_id in m)
    if event_mask is not None:
        _em = event_mask
        filters.append(lambda ie, _i, m=_em: ie.event_id in m)
    if site_mask is not None:
        _sm = site_mask
        filters.append(lambda ie, _i, m=_sm: ie.site_id in m)
    if round_mask is not None:
        _rm = round_mask
        filters.append(lambda ie, _i, m=_rm: ie.round_id in m)
    if date_min_val is not None:
        _dmin = date_min_val
        filters.append(lambda ie, _i, v=_dmin: ie.date >= v)
    if date_max_val is not None:
        _dmax = date_max_val
        filters.append(lambda ie, _i, v=_dmax: ie.date <= v)
    if edate_min_val is not None:
        _edmin = edate_min_val
        filters.append(lambda ie, _i, v=_edmin: ie.event_date >= v)
    if edate_max_val is not None:
        _edmax = edate_max_val
        filters.append(lambda ie, _i, v=_edmax: ie.event_date <= v)
    if eco_min_val is not None:
        _ecmin = eco_min_val
        filters.append(lambda ie, _i, v=_ecmin: ie.eco_code >= v)
    if eco_max_val is not None:
        _ecmax = eco_max_val
        filters.append(lambda ie, _i, v=_ecmax: ie.eco_code <= v)
    if c.white_elo_min is not None:
        _wemin = c.white_elo_min
        filters.append(lambda ie, _i, v=_wemin: ie.white_elo >= v)
    if c.white_elo_max is not None:
        _wemax = c.white_elo_max
        filters.append(lambda ie, _i, v=_wemax: ie.white_elo <= v)
    if c.black_elo_min is not None:
        _bemin = c.black_elo_min
        filters.append(lambda ie, _i, v=_bemin: ie.black_elo >= v)
    if c.black_elo_max is not None:
        _bemax = c.black_elo_max
        filters.append(lambda ie, _i, v=_bemax: ie.black_elo <= v)
    if c.elo_min is not None:
        _emin = c.elo_min
        filters.append(lambda ie, _i, v=_emin: ie.white_elo >= v or ie.black_elo >= v)
    if c.elo_max is not None:
        _emax = c.elo_max
        filters.append(lambda ie, _i, v=_emax: ie.white_elo <= v or ie.black_elo <= v)
    if c.length_min is not None:
        _lmin = c.length_min
        filters.append(lambda ie, _i, v=_lmin: ie.num_half_moves >= v)
    if c.length_max is not None:
        _lmax = c.length_max
        filters.append(lambda ie, _i, v=_lmax: ie.num_half_moves <= v)
    if result_vals is not None:
        _rv = result_vals
        filters.append(lambda ie, _i, s=_rv: ie.result in s)
    if c.flags_set is not None:
        for flag_bit in c.flags_set:
            _fb = flag_bit
            filters.append(lambda ie, _i, b=_fb: bool(ie.flags & (1 << b)))
    if c.flags_unset is not None:
        for flag_bit in c.flags_unset:
            _fb = flag_bit
            filters.append(lambda ie, _i, b=_fb: not (ie.flags & (1 << b)))
    if c.game_num_min is not None:
        _gnmin = c.game_num_min
        filters.append(lambda _ie, i, v=_gnmin: i >= v)
    if c.game_num_max is not None:
        _gnmax = c.game_num_max
        filters.append(lambda _ie, i, v=_gnmax: i <= v)

    # Apply all filters
    result: list[int] = []
    for idx in indices:
        ie = entries[idx]
        match = True
        for filt in filters:
            if not filt(ie, idx):
                match = False
                break
        if match:
            result.append(idx)

    return result


# ---------------------------------------------------------------------------
# Position Search
# ---------------------------------------------------------------------------


@dataclass
class PositionSearchResult:
    """Result of a position search for a single game."""

    game_index: int
    ply: int  # 0-based ply where position was found


def _board_material_counts(board: chess.Board) -> dict[str, int]:
    """Count material on the board, returning a dict with matsig keys."""
    return {
        "WQ": len(board.pieces(chess.QUEEN, chess.WHITE)),
        "WR": len(board.pieces(chess.ROOK, chess.WHITE)),
        "WB": len(board.pieces(chess.BISHOP, chess.WHITE)),
        "WN": len(board.pieces(chess.KNIGHT, chess.WHITE)),
        "WP": len(board.pieces(chess.PAWN, chess.WHITE)),
        "BQ": len(board.pieces(chess.QUEEN, chess.BLACK)),
        "BR": len(board.pieces(chess.ROOK, chess.BLACK)),
        "BB": len(board.pieces(chess.BISHOP, chess.BLACK)),
        "BN": len(board.pieces(chess.KNIGHT, chess.BLACK)),
        "BP": len(board.pieces(chess.PAWN, chess.BLACK)),
    }


def _position_index_match(
    ie: IndexEntry,
    target_board: chess.Board,
    target_hp: tuple[int, int],
    target_counts: dict[str, int],
) -> int:
    """Pre-filter a game using index data only.

    Returns:
        >= 0: Matched at this ply (via stored line).
        -1: Inconclusive — need full game decode.
        -2: Definitely no match.
    """
    has_start_flag = bool(ie.flags & (1 << FLAG_START))
    has_promos = bool(ie.flags & (1 << FLAG_PROMO))
    has_upromos = bool(ie.flags & (1 << FLAG_UPROMO))

    if not has_start_flag:
        # Try stored line match
        ply = _stored_line_match(target_board, ie.stored_line_code)
        if ply >= 0:
            return ply  # Exact match at this ply
        # ply == -2 would mean exclude, but our _stored_line_match returns -1

        # Try home pawn signature match
        if not _hpsig_match(target_hp[0], target_hp[1], ie.home_pawn_data):
            return -2

    # Try material signature match
    if _less_mat(target_counts, ie.final_mat_sig, has_promos, has_upromos):
        return -2

    return -1  # Inconclusive — need full decode


def _search_game_for_position(
    game: chess.pgn.Game,
    target_board: chess.Board,
    target_turn: chess.Color,
) -> int:
    """Replay a game's mainline and check for the target position.

    Only checks positions where the side to move matches *target_turn*.

    Returns:
        The ply (0-based) at which the position was found, or -1 if not found.
    """
    board = game.board()  # starting position (may be FEN)

    # Precompute target piece counts for early termination using popcount
    target_white_total = target_board.occupied_co[chess.WHITE].bit_count()
    target_black_total = target_board.occupied_co[chess.BLACK].bit_count()

    ply = 0

    # Check starting position
    if board.turn == target_turn and board == target_board:
        return 0

    for node in game.mainline():
        move = node.move
        is_capture = board.is_capture(move)
        board.push(move)
        ply += 1

        # Early termination: if a capture reduced piece count below target
        if is_capture:
            white_total = board.occupied_co[chess.WHITE].bit_count()
            black_total = board.occupied_co[chess.BLACK].bit_count()
            if white_total < target_white_total or black_total < target_black_total:
                return -1

        # Only check when side to move matches
        if board.turn == target_turn and board == target_board:
            return ply

    return -1


def search_position(
    entries: Sequence[IndexEntry],
    namebase: NameBase,
    game_data: bytes | bytearray,
    target: chess.Board,
    game_indices: Sequence[int] | None = None,
    decode_game: Callable[[int], chess.pgn.Game] | None = None,
) -> list[PositionSearchResult]:
    """Search for games containing a specific board position.

    Uses SCID's three-level pre-filtering to minimize game decoding:
    1. Stored line check (instant match or inconclusive)
    2. Home pawn signature check
    3. Material signature check

    Args:
        entries: List of IndexEntry objects.
        namebase: The database's NameBase (unused but kept for API consistency).
        game_data: Raw game data bytes.
        target: The board position to search for.
        game_indices: If given, only search these game indices.
        decode_game: Callable that takes a game index and returns a
            chess.pgn.Game. If None, games are decoded using gamedata module.

    Returns:
        List of PositionSearchResult for matching games.
    """
    # Precompute target position properties
    target_hp = _hpsig_make(target)
    target_counts = _board_material_counts(target)
    target_turn = target.turn

    indices: Sequence[int]
    if game_indices is not None:
        indices = game_indices
    else:
        indices = range(len(entries))

    results: list[PositionSearchResult] = []

    for idx in indices:
        ie = entries[idx]

        # Stage 1-3: Index-based pre-filtering
        pre = _position_index_match(ie, target, target_hp, target_counts)
        if pre == -2:
            continue  # Definitely no match
        if pre >= 0:
            results.append(PositionSearchResult(game_index=idx, ply=pre))
            continue

        # Stage 4: Full game decode and replay
        if decode_game is not None:
            game = decode_game(idx)
        else:
            from scid.gamedata import decode_game_data

            blob = bytes(game_data[ie.offset : ie.offset + ie.game_data_size])
            game = decode_game_data(blob)

        ply = _search_game_for_position(game, target, target_turn)
        if ply >= 0:
            results.append(PositionSearchResult(game_index=idx, ply=ply))

    return results


# ---------------------------------------------------------------------------
# Material Search
# ---------------------------------------------------------------------------


@dataclass
class MaterialSearchCriteria:
    """Criteria for material-based search.

    Each field is a (min, max) range. If None, that piece type is not
    constrained. Ranges are inclusive.
    """

    wq: tuple[int, int] | None = None  # White queens
    wr: tuple[int, int] | None = None  # White rooks
    wb: tuple[int, int] | None = None  # White bishops
    wn: tuple[int, int] | None = None  # White knights
    wp: tuple[int, int] | None = None  # White pawns
    bq: tuple[int, int] | None = None  # Black queens
    br: tuple[int, int] | None = None  # Black rooks
    bb: tuple[int, int] | None = None  # Black bishops
    bn: tuple[int, int] | None = None  # Black knights
    bp: tuple[int, int] | None = None  # Black pawns

    # If True, both players must match simultaneously (i.e., check each
    # position for ALL constraints at the same ply). Default True.
    simultaneous: bool = True


def _material_matches_criteria(
    board: chess.Board,
    criteria: MaterialSearchCriteria,
) -> bool:
    """Check if a board position's material matches the search criteria."""
    c = criteria
    if c.wq is not None:
        n = len(board.pieces(chess.QUEEN, chess.WHITE))
        if not (c.wq[0] <= n <= c.wq[1]):
            return False
    if c.wr is not None:
        n = len(board.pieces(chess.ROOK, chess.WHITE))
        if not (c.wr[0] <= n <= c.wr[1]):
            return False
    if c.wb is not None:
        n = len(board.pieces(chess.BISHOP, chess.WHITE))
        if not (c.wb[0] <= n <= c.wb[1]):
            return False
    if c.wn is not None:
        n = len(board.pieces(chess.KNIGHT, chess.WHITE))
        if not (c.wn[0] <= n <= c.wn[1]):
            return False
    if c.wp is not None:
        n = len(board.pieces(chess.PAWN, chess.WHITE))
        if not (c.wp[0] <= n <= c.wp[1]):
            return False
    if c.bq is not None:
        n = len(board.pieces(chess.QUEEN, chess.BLACK))
        if not (c.bq[0] <= n <= c.bq[1]):
            return False
    if c.br is not None:
        n = len(board.pieces(chess.ROOK, chess.BLACK))
        if not (c.br[0] <= n <= c.br[1]):
            return False
    if c.bb is not None:
        n = len(board.pieces(chess.BISHOP, chess.BLACK))
        if not (c.bb[0] <= n <= c.bb[1]):
            return False
    if c.bn is not None:
        n = len(board.pieces(chess.KNIGHT, chess.BLACK))
        if not (c.bn[0] <= n <= c.bn[1]):
            return False
    if c.bp is not None:
        n = len(board.pieces(chess.PAWN, chess.BLACK))
        if not (c.bp[0] <= n <= c.bp[1]):
            return False
    return True


def _build_max_matsig(criteria: MaterialSearchCriteria) -> int:
    """Build a material signature from the MAX values of each range.

    Used for matsig_isReachable pre-filtering: if the game's final
    material can't reach a position with these maximum piece counts,
    it definitely can't match.
    """
    wq = min(criteria.wq[1], 3) if criteria.wq else 1
    wr = min(criteria.wr[1], 3) if criteria.wr else 2
    wb = min(criteria.wb[1], 3) if criteria.wb else 2
    wn = min(criteria.wn[1], 3) if criteria.wn else 2
    wp = min(criteria.wp[1], 8) if criteria.wp else 8
    bq = min(criteria.bq[1], 3) if criteria.bq else 1
    br = min(criteria.br[1], 3) if criteria.br else 2
    bb = min(criteria.bb[1], 3) if criteria.bb else 2
    bn = min(criteria.bn[1], 3) if criteria.bn else 2
    bp = min(criteria.bp[1], 8) if criteria.bp else 8

    return (
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


@dataclass
class MaterialSearchResult:
    """Result of a material search for a single game."""

    game_index: int
    ply: int  # 0-based ply where material criteria were first met


def search_material(
    entries: Sequence[IndexEntry],
    game_data: bytes | bytearray,
    criteria: MaterialSearchCriteria,
    game_indices: Sequence[int] | None = None,
    decode_game: Callable[[int], chess.pgn.Game] | None = None,
) -> list[MaterialSearchResult]:
    """Search for games where material counts match the given ranges.

    Pre-filters using matsig_isReachable before full game decode.

    Args:
        entries: List of IndexEntry objects.
        game_data: Raw game data bytes.
        criteria: Material search criteria with min/max ranges.
        game_indices: If given, only search these game indices.
        decode_game: Callable that takes a game index and returns a
            chess.pgn.Game. If None, games are decoded using gamedata module.

    Returns:
        List of MaterialSearchResult for matching games.
    """
    max_sig = _build_max_matsig(criteria)

    indices: Sequence[int]
    if game_indices is not None:
        indices = game_indices
    else:
        indices = range(len(entries))

    results: list[MaterialSearchResult] = []

    for idx in indices:
        ie = entries[idx]
        has_promos = bool(ie.flags & (1 << FLAG_PROMO))
        has_upromos = bool(ie.flags & (1 << FLAG_UPROMO))

        # Pre-filter: can the game's final material reach the target ranges?
        if not _matsig_is_reachable(ie.final_mat_sig, max_sig, has_promos, has_upromos):
            continue

        # Full game decode and ply-by-ply material check
        if decode_game is not None:
            game = decode_game(idx)
        else:
            from scid.gamedata import decode_game_data

            blob = bytes(game_data[ie.offset : ie.offset + ie.game_data_size])
            game = decode_game_data(blob)

        board = game.board()
        ply = 0

        # Check starting position
        if _material_matches_criteria(board, criteria):
            results.append(MaterialSearchResult(game_index=idx, ply=0))
            continue

        found = False
        for node in game.mainline():
            board.push(node.move)
            ply += 1
            if _material_matches_criteria(board, criteria):
                results.append(MaterialSearchResult(game_index=idx, ply=ply))
                found = True
                break

        # (If not found, just continue to next game)

    return results


# ---------------------------------------------------------------------------
# Tree Search (Opening Tree Statistics)
# ---------------------------------------------------------------------------

# FIDE Table 8.1a — maps percentage score (0–100, as int) to expected
# rating difference.  Index i gives the rating diff for i% score.
# Values outside the table (0% and 100%) are clamped to ±800.
_FIDE_DP: tuple[int, ...] = (
    -800,
    -677,
    -589,
    -538,
    -501,
    -470,
    -444,
    -422,
    -401,
    -383,  # 0-9%
    -366,
    -351,
    -336,
    -322,
    -309,
    -296,
    -284,
    -273,
    -262,
    -251,  # 10-19%
    -240,
    -230,
    -220,
    -211,
    -202,
    -193,
    -184,
    -175,
    -166,
    -158,  # 20-29%
    -149,
    -141,
    -133,
    -125,
    -117,
    -110,
    -102,
    -95,
    -87,
    -80,  # 30-39%
    -72,
    -65,
    -57,
    -50,
    -43,
    -36,
    -29,
    -21,
    -14,
    -7,  # 40-49%
    0,
    7,
    14,
    21,
    29,
    36,
    43,
    50,
    57,
    65,  # 50-59%
    72,
    80,
    87,
    95,
    102,
    110,
    117,
    125,
    133,
    141,  # 60-69%
    149,
    158,
    166,
    175,
    184,
    193,
    202,
    211,
    220,
    230,  # 70-79%
    240,
    251,
    262,
    273,
    284,
    296,
    309,
    322,
    336,
    351,  # 80-89%
    366,
    383,
    401,
    422,
    444,
    470,
    501,
    538,
    589,
    677,  # 90-99%
    800,  # 100%
)


@dataclass
class TreeNode:
    """Aggregated statistics for one move in the opening tree.

    Attributes:
        move: The chess.Move played from the search position.
        san: Standard algebraic notation of the move.
        count: Total number of games with this move.
        white_wins: Number of games won by White.
        draws: Number of drawn games.
        black_wins: Number of games won by Black.
        score: Score from White's perspective (0.0 to 1.0).
            Win=1, Draw=0.5, Loss=0.
        avg_elo: Average Elo of the side making the move.
            Only games where both players have Elo are counted.
            0 if no games have Elo.
        performance: FIDE performance rating of the side making the move.
            Opponent's average Elo + FIDE table adjustment.
            0 if fewer than 10 rated games.
        avg_year: Average year of the games. 0 if no dates.
        draw_pct: Percentage of draws (0.0 to 100.0).
    """

    move: chess.Move
    san: str
    count: int
    white_wins: int
    draws: int
    black_wins: int
    score: float
    avg_elo: float
    performance: float
    avg_year: float
    draw_pct: float


@dataclass
class TreeResult:
    """Complete opening tree result from a position.

    Attributes:
        position: The board position searched.
        nodes: Per-move statistics, sorted by frequency descending.
        total: Totals row aggregated across all moves.
        games_searched: Total number of games in the database.
        games_matched: Number of games containing this position.
    """

    position: chess.Board
    nodes: list[TreeNode]
    total: TreeNode | None
    games_searched: int
    games_matched: int


def _compute_score(white_wins: int, draws: int, black_wins: int) -> float:
    """Score from White's perspective: W=1.0, D=0.5, L=0.0."""
    total = white_wins + draws + black_wins
    if total == 0:
        return 0.0
    return (white_wins + 0.5 * draws) / total


def _compute_performance(
    own_elo_sum: int,
    opponent_elo_sum: int,
    elo_count: int,
    score_pct: float,
    is_white_to_move: bool,
) -> float:
    """Compute FIDE performance rating.

    Args:
        own_elo_sum: Sum of Elo ratings of the side making the move.
        opponent_elo_sum: Sum of Elo ratings of the opponent.
        elo_count: Number of games where both players have Elo.
        score_pct: Score percentage (0.0 to 1.0) *from the mover's perspective*.
        is_white_to_move: True if White is the side making the move.

    Returns:
        Performance rating, or 0.0 if fewer than 10 rated games.
    """
    if elo_count < 10:
        return 0.0

    avg_opponent = opponent_elo_sum / elo_count
    # Convert score to percentage index (0–100) for the FIDE table
    pct_index = max(0, min(100, round(score_pct * 100)))
    dp = _FIDE_DP[pct_index]

    return avg_opponent + dp


def search_tree(
    entries: Sequence[IndexEntry],
    namebase: NameBase,
    game_data: bytes | bytearray,
    target: chess.Board,
    game_indices: Sequence[int] | None = None,
    decode_game: Callable[[int], chess.pgn.Game] | None = None,
) -> TreeResult:
    """Compute opening tree statistics from a board position.

    This is a two-phase search matching SCID's tree algorithm:

    **Phase 1**: Position search — find all games containing the target
    position, using the same three-level pre-filtering as
    :func:`search_position` (stored lines, home pawns, material signature).

    **Phase 2**: Statistics — for each matching game, extract the next
    move at the matched ply and aggregate per-move statistics including
    frequency, win/draw/loss counts, average Elo, FIDE performance
    rating, and average year.

    Args:
        entries: List of IndexEntry objects.
        namebase: The database's NameBase.
        game_data: Raw game data bytes.
        target: The board position to compute tree statistics for.
        game_indices: If given, only search these game indices.
        decode_game: Callable that takes a game index and returns a
            chess.pgn.Game. If None, games are decoded using gamedata module.

    Returns:
        A :class:`TreeResult` with per-move nodes and totals.
    """
    # Precompute target position properties for index pre-filtering
    target_hp = _hpsig_make(target)
    target_counts = _board_material_counts(target)
    target_turn = target.turn
    is_white_to_move = target_turn == chess.WHITE

    indices: Sequence[int]
    if game_indices is not None:
        indices = game_indices
    else:
        indices = range(len(entries))

    # Phase 1 + Phase 2 combined: find position, extract next move, aggregate.
    #
    # We inline position search here (rather than calling search_position)
    # because we need to efficiently extract the next move at the matched ply.
    # For stored-line matches, we can get the next move without decoding the
    # game at all.

    # Accumulators keyed by (from_sq, to_sq, promotion) tuple
    move_stats: dict[
        tuple[int, int, int | None],
        list,  # [move, san, count, ww, d, bw, own_elo_sum, opp_elo_sum, elo_count, year_sum, year_count]
    ] = {}

    games_matched = 0

    for idx in indices:
        ie = entries[idx]

        # --- Index pre-filtering (same as search_position) ---
        has_start_flag = bool(ie.flags & (1 << FLAG_START))
        has_promos = bool(ie.flags & (1 << FLAG_PROMO))
        has_upromos = bool(ie.flags & (1 << FLAG_UPROMO))

        ply = -1
        next_move: chess.Move | None = None

        if not has_start_flag:
            # Try stored line match
            sl_ply = _stored_line_match(target, ie.stored_line_code)
            if sl_ply >= 0:
                ply = sl_ply
                # Get next move from stored line (no game decode!)
                line_moves = get_line_moves(ie.stored_line_code)
                if ply < len(line_moves):
                    next_move = line_moves[ply]
                # else: game is at the end of the stored line — need decode

            if ply < 0:
                # Home pawn signature check
                if not _hpsig_match(target_hp[0], target_hp[1], ie.home_pawn_data):
                    continue
            # sl_ply == -2 would skip, but our implementation returns -1

        if ply < 0:
            # Material signature check
            if _less_mat(target_counts, ie.final_mat_sig, has_promos, has_upromos):
                continue

        # If we still need the next move (either no stored line match,
        # or stored line matched but we're past its moves), decode game
        if next_move is None:
            if decode_game is not None:
                game = decode_game(idx)
            else:
                from scid.gamedata import decode_game_data

                blob = bytes(game_data[ie.offset : ie.offset + ie.game_data_size])
                game = decode_game_data(blob)

            # Replay to find position and get next move
            board = game.board()
            found_ply = -1

            if ply >= 0:
                # Stored line told us the ply, just advance
                for i, node in enumerate(game.mainline()):
                    board.push(node.move)
                    if i + 1 == ply:
                        break
                # Now board is at the matched ply, get next move
                node_at_ply = game
                for i, node in enumerate(game.mainline()):
                    if i == ply:
                        next_move = node.move
                        break
                    node_at_ply = node
                found_ply = ply
            else:
                # Full replay needed
                current_ply = 0
                if board.turn == target_turn and board == target:
                    found_ply = 0
                else:
                    for node in game.mainline():
                        move = node.move
                        is_capture = board.is_capture(move)
                        board.push(move)
                        current_ply += 1

                        # Early termination on captures
                        if is_capture:
                            w_total = board.occupied_co[chess.WHITE].bit_count()
                            b_total = board.occupied_co[chess.BLACK].bit_count()
                            t_w = target.occupied_co[chess.WHITE].bit_count()
                            t_b = target.occupied_co[chess.BLACK].bit_count()
                            if w_total < t_w or b_total < t_b:
                                break

                        if board.turn == target_turn and board == target:
                            found_ply = current_ply
                            break

                if found_ply < 0:
                    continue  # Position not found in this game

                # Get next move after found_ply
                ply = found_ply
                move_idx = 0
                for node in game.mainline():
                    if move_idx == ply:
                        next_move = node.move
                        break
                    move_idx += 1

        if next_move is None:
            continue  # No next move (game ended at this position)

        games_matched += 1

        # --- Aggregate statistics ---
        move_key = (next_move.from_square, next_move.to_square, next_move.promotion)

        if move_key not in move_stats:
            # Generate SAN for this move
            san = target.san(next_move)
            move_stats[move_key] = [
                next_move,  # 0: move
                san,  # 1: san
                0,  # 2: count
                0,  # 3: white_wins
                0,  # 4: draws
                0,  # 5: black_wins
                0,  # 6: own_elo_sum (side making the move)
                0,  # 7: opponent_elo_sum
                0,  # 8: elo_count
                0,  # 9: year_sum
                0,  # 10: year_count
            ]

        stats = move_stats[move_key]
        stats[2] += 1  # count

        # Result
        result = ie.result
        if result == RESULT_WHITE:
            stats[3] += 1
        elif result == RESULT_DRAW:
            stats[4] += 1
        elif result == RESULT_BLACK:
            stats[5] += 1

        # Elo — only count if both players have ratings
        w_elo = ie.white_elo
        b_elo = ie.black_elo
        if w_elo > 0 and b_elo > 0:
            if is_white_to_move:
                stats[6] += w_elo  # own = white (the side making the move)
                stats[7] += b_elo  # opponent = black
            else:
                stats[6] += b_elo  # own = black
                stats[7] += w_elo  # opponent = white
            stats[8] += 1

        # Year
        year = (ie.date >> 9) & 0x7FF  # extract year from 20-bit date
        if year > 0:
            stats[9] += year
            stats[10] += 1

    # --- Build TreeNode list ---
    nodes: list[TreeNode] = []
    for stats in move_stats.values():
        move_obj = stats[0]
        san = stats[1]
        count = stats[2]
        ww = stats[3]
        d = stats[4]
        bw = stats[5]
        own_elo_sum = stats[6]
        opp_elo_sum = stats[7]
        elo_count = stats[8]
        year_sum = stats[9]
        year_count = stats[10]

        score = _compute_score(ww, d, bw)
        avg_elo = own_elo_sum / elo_count if elo_count > 0 else 0.0

        # Performance: score from the mover's perspective
        if is_white_to_move:
            mover_score = score  # score is already from White's perspective
        else:
            mover_score = 1.0 - score  # flip for Black

        perf = _compute_performance(
            own_elo_sum, opp_elo_sum, elo_count, mover_score, is_white_to_move
        )

        avg_year = year_sum / year_count if year_count > 0 else 0.0
        draw_pct = 100.0 * d / count if count > 0 else 0.0

        nodes.append(
            TreeNode(
                move=move_obj,
                san=san,
                count=count,
                white_wins=ww,
                draws=d,
                black_wins=bw,
                score=score,
                avg_elo=avg_elo,
                performance=perf,
                avg_year=avg_year,
                draw_pct=draw_pct,
            )
        )

    # Sort by frequency descending
    nodes.sort(key=lambda n: n.count, reverse=True)

    # --- Build totals ---
    total: TreeNode | None = None
    if nodes:
        tot_count = sum(n.count for n in nodes)
        tot_ww = sum(n.white_wins for n in nodes)
        tot_d = sum(n.draws for n in nodes)
        tot_bw = sum(n.black_wins for n in nodes)
        tot_score = _compute_score(tot_ww, tot_d, tot_bw)

        # Aggregate Elo for totals
        tot_own_elo = sum(stats[6] for stats in move_stats.values())
        tot_opp_elo = sum(stats[7] for stats in move_stats.values())
        tot_elo_count = sum(stats[8] for stats in move_stats.values())
        tot_avg_elo = tot_own_elo / tot_elo_count if tot_elo_count > 0 else 0.0

        if is_white_to_move:
            tot_mover_score = tot_score
        else:
            tot_mover_score = 1.0 - tot_score

        tot_perf = _compute_performance(
            tot_own_elo, tot_opp_elo, tot_elo_count, tot_mover_score, is_white_to_move
        )

        tot_year_sum = sum(stats[9] for stats in move_stats.values())
        tot_year_count = sum(stats[10] for stats in move_stats.values())
        tot_avg_year = tot_year_sum / tot_year_count if tot_year_count > 0 else 0.0
        tot_draw_pct = 100.0 * tot_d / tot_count if tot_count > 0 else 0.0

        total = TreeNode(
            move=chess.Move.null(),
            san="Total",
            count=tot_count,
            white_wins=tot_ww,
            draws=tot_d,
            black_wins=tot_bw,
            score=tot_score,
            avg_elo=tot_avg_elo,
            performance=tot_perf,
            avg_year=tot_avg_year,
            draw_pct=tot_draw_pct,
        )

    return TreeResult(
        position=target.copy(),
        nodes=nodes,
        total=total,
        games_searched=len(indices),
        games_matched=games_matched,
    )
