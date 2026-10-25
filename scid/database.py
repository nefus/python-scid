"""High-level SCID database access.

Provides :class:`ScidDatabase`, the main entry point for opening, reading,
writing, and iterating SCID chess databases in both SCID4 (.si4/.sn4/.sg4)
and SCID5 (.si5/.sn5/.sg5) formats.

Usage::

    from scid import ScidDatabase

    db = ScidDatabase.open("~/games/chess/mydb")
    print(len(db))           # number of games
    game = db[0]             # chess.pgn.Game
    for game in db:           # iterate all games
        print(game.headers["White"])
    db.close()

    # Or as context manager:
    with ScidDatabase.open("path/to/db") as db:
        ...
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Iterable, Iterator, overload

import chess.pgn

from scid.constants import (
    FLAG_DELETE,
    FLAG_PROMO,
    FLAG_START,
    FLAG_UPROMO,
    NAME_EVENT,
    NAME_PLAYER,
    NAME_ROUND,
    NAME_SITE,
    RESULT_FROM_STR,
    RESULT_STR,
)
from scid.dates import date_from_pgn, date_to_pgn
from scid.eco import eco_decode, eco_encode
from scid.gamedata import decode_game_data, encode_game_data
from scid.homepawns import encode_homepawns_from_moves
from scid.index import (
    IndexEntry,
    IndexHeader,
    read_si4,
    read_si5,
    write_si4,
    write_si5,
)
from scid.matsig import encode_matsig
from scid.namebase import NameBase
from scid.search import (
    HeaderSearchCriteria,
    MaterialSearchCriteria,
    MaterialSearchResult,
    PositionSearchResult,
    TreeResult,
    iter_search_header,
    iter_search_material,
    iter_search_position,
    search_header,
    search_tree,
)
from scid.stored_lines import classify


# SCID's largest valid rating (MAX_ELO); ratings are stored in 12 bits.
MAX_ELO = 4000


def _parse_elo(value: str) -> int:
    """Parse a PGN Elo tag, clamped to SCID's valid range (0 if invalid)."""
    try:
        elo = int(value)
    except ValueError:
        return 0
    return max(0, min(elo, MAX_ELO))


def _replace_files(contents: dict[str, bytes | bytearray]) -> None:
    """Write each file via a temporary file and atomically rename it into place.

    All contents are encoded before this is called, so encoding errors cannot
    leave a partially written database.  Each temporary file is fsynced before
    any target is replaced, which keeps the window in which the three files
    can disagree as short as possible.
    """
    temporaries: list[tuple[str, str]] = []
    try:
        for target, data in contents.items():
            directory = os.path.dirname(os.path.abspath(target))
            fd, tmp = tempfile.mkstemp(
                prefix=f".{os.path.basename(target)}.", suffix=".tmp", dir=directory
            )
            temporaries.append((tmp, target))
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            if os.path.exists(target):
                os.chmod(tmp, os.stat(target).st_mode & 0o7777)
            else:
                umask = os.umask(0)
                os.umask(umask)
                os.chmod(tmp, 0o666 & ~umask)
        for tmp, target in temporaries:
            os.replace(tmp, target)
    finally:
        for tmp, _target in temporaries:
            if os.path.exists(tmp):
                os.unlink(tmp)


class ScidDatabase:
    """Read/write access to a SCID chess database.

    A SCID database consists of three files sharing a base path:
      - Index (.si4/.si5) — per-game metadata
      - Namebase (.sn4/.sn5) — player/event/site/round strings
      - Games (.sg4/.sg5) — variable-length encoded game data blobs

    The version (4 or 5) is auto-detected from file extensions on open.
    """

    def __init__(self) -> None:
        self._path: str = ""
        self._version: int = 0  # 4 or 5
        self._mode: str = "r"
        self._header: IndexHeader | None = None
        self._entries: list[IndexEntry] = []
        self._namebase: NameBase = NameBase()
        self._game_data: bytes | bytearray = b""
        self._dirty: bool = False

    # ------------------------------------------------------------------
    # Opening / Creating
    # ------------------------------------------------------------------

    @classmethod
    def open(cls, path: str | Path, mode: str = "r") -> ScidDatabase:
        """Open an existing SCID database.

        Args:
            path: Base path without extension (e.g. '/data/mydb').
                  Extensions are added automatically.
            mode: 'r' for read-only, 'rw' for read-write.

        Returns:
            A ScidDatabase instance.

        Raises:
            FileNotFoundError: If the database files don't exist.
            ValueError: If *mode* is not 'r' or 'rw'.
        """
        if mode not in ("r", "rw"):
            raise ValueError(f"mode must be 'r' or 'rw', got {mode!r}")
        db = cls()
        db._path = str(path)
        db._mode = mode
        db._detect_and_load()
        return db

    @classmethod
    def create(cls, path: str | Path, version: int = 5) -> ScidDatabase:
        """Create a new empty SCID database.

        Args:
            path: Base path without extension.
            version: 4 for SCID4, 5 for SCID5. Default is 5.

        Returns:
            A new, empty ScidDatabase instance (writable).
        """
        if version not in (4, 5):
            raise ValueError(f"version must be 4 or 5, got {version}")

        db = cls()
        db._path = str(path)
        db._version = version
        db._mode = "rw"
        db._entries = []
        db._namebase = NameBase()
        db._game_data = bytearray()
        db._dirty = True

        if version == 4:
            db._header = IndexHeader()

        return db

    def _detect_and_load(self) -> None:
        """Detect format version and load all three files."""
        # Append extensions instead of using Path.with_suffix(), which would
        # replace part of a base name such as "games.2024".
        # Try SCID5 first, then SCID4
        if Path(f"{self._path}.si5").exists():
            self._version = 5
            self._load_v5()
        elif Path(f"{self._path}.si4").exists():
            self._version = 4
            self._load_v4()
        else:
            raise FileNotFoundError(
                f"No SCID database found at '{self._path}' "
                f"(tried .si4 and .si5 extensions)"
            )

    def _load_v4(self) -> None:
        path = self._path
        with open(f"{path}.si4", "rb") as f:
            self._header, self._entries = read_si4(f)
        with open(f"{path}.sn4", "rb") as f:
            self._namebase = NameBase.read_sn4(f)
        with open(f"{path}.sg4", "rb") as f:
            self._game_data = f.read()

    def _load_v5(self) -> None:
        path = self._path
        with open(f"{path}.si5", "rb") as f:
            self._entries = read_si5(f)
        with open(f"{path}.sn5", "rb") as f:
            self._namebase = NameBase.read_sn5(f)
        with open(f"{path}.sg5", "rb") as f:
            self._game_data = f.read()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def path(self) -> str:
        """Base path of the database (without extension)."""
        return self._path

    @property
    def version(self) -> int:
        """Database format version (4 or 5)."""
        return self._version

    @property
    def description(self) -> str:
        """Database description.

        Stored in the index header for SCID4 and in the namebase metadata
        for SCID5.
        """
        if self._version == 5:
            return self._namebase.db_info.get("description", "")
        if self._header:
            return self._header.description
        return ""

    @description.setter
    def description(self, value: str) -> None:
        if self._version == 5:
            self._namebase.db_info["description"] = value
        else:
            if self._header is None:
                self._header = IndexHeader()
            self._header.description = value
        self._dirty = True

    # ------------------------------------------------------------------
    # Sequence interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    @overload
    def __getitem__(self, index: int) -> chess.pgn.Game: ...
    @overload
    def __getitem__(self, index: slice) -> list[chess.pgn.Game]: ...

    def __getitem__(self, index: int | slice) -> chess.pgn.Game | list[chess.pgn.Game]:
        if isinstance(index, slice):
            indices = range(*index.indices(len(self._entries)))
            return [self._load_game(i) for i in indices]

        if index < 0:
            index += len(self._entries)
        if not 0 <= index < len(self._entries):
            raise IndexError(f"game index {index} out of range")
        return self._load_game(index)

    def __iter__(self) -> Iterator[chess.pgn.Game]:
        for i in range(len(self._entries)):
            yield self._load_game(i)

    # ------------------------------------------------------------------
    # Searching
    # ------------------------------------------------------------------

    def search_header(
        self,
        criteria: HeaderSearchCriteria | None = None,
        game_indices: list[int] | None = None,
        **kwargs: object,
    ) -> list[int]:
        """Search games by header criteria.

        Criteria can be passed as a :class:`HeaderSearchCriteria` object
        or as keyword arguments (which are forwarded to its constructor).

        Examples::

            # By criteria object
            c = HeaderSearchCriteria(white="Kasparov", date_min="1990.01.01")
            hits = db.search_header(c)

            # By keyword arguments
            hits = db.search_header(player="Carlsen", results={"1-0"})

        Returns:
            List of matching game indices (0-based).
        """
        if criteria is None:
            criteria = HeaderSearchCriteria(**kwargs)  # type: ignore[arg-type]
        return search_header(self._entries, self._namebase, criteria, game_indices)

    def iter_search_header(
        self,
        criteria: HeaderSearchCriteria | None = None,
        game_indices: Iterable[int] | None = None,
        **kwargs: object,
    ) -> Iterator[int]:
        """Yield matching game indices without storing the full result."""
        if criteria is None:
            criteria = HeaderSearchCriteria(**kwargs)  # type: ignore[arg-type]
        return iter_search_header(self._entries, self._namebase, criteria, game_indices)

    def search_position(
        self,
        target: chess.Board,
        game_indices: list[int] | None = None,
    ) -> list[PositionSearchResult]:
        """Search for games containing a specific board position.

        Uses three-level pre-filtering (stored lines, home pawns, material
        signature) before full game decode for maximum performance.

        Args:
            target: The board position to search for.
            game_indices: If given, only search these game indices.

        Returns:
            List of :class:`PositionSearchResult` with game_index and ply.
        """
        return list(self.iter_search_position(target, game_indices))

    def iter_search_position(
        self,
        target: chess.Board,
        game_indices: Iterable[int] | None = None,
    ) -> Iterator[PositionSearchResult]:
        """Yield position matches as the index scan finds them."""
        return iter_search_position(
            self._entries,
            self._namebase,
            self._game_data,
            target,
            game_indices,
            decode_game=self._load_game,
        )

    def search_material(
        self,
        criteria: MaterialSearchCriteria | None = None,
        game_indices: list[int] | None = None,
        **kwargs: object,
    ) -> list[MaterialSearchResult]:
        """Search for games matching material count ranges.

        Criteria can be passed as a :class:`MaterialSearchCriteria` object
        or as keyword arguments.

        Examples::

            # Find games with queen sacrifice (0 white queens at some point)
            hits = db.search_material(wq=(0, 0))

            # Find games reaching a specific endgame
            c = MaterialSearchCriteria(wr=(1, 1), br=(1, 1), wp=(0, 0), bp=(0, 0))
            hits = db.search_material(c)

        Returns:
            List of :class:`MaterialSearchResult` with game_index and ply.
        """
        if criteria is None:
            criteria = MaterialSearchCriteria(**kwargs)  # type: ignore[arg-type]
        return list(self.iter_search_material(criteria, game_indices))

    def iter_search_material(
        self,
        criteria: MaterialSearchCriteria | None = None,
        game_indices: Iterable[int] | None = None,
        **kwargs: object,
    ) -> Iterator[MaterialSearchResult]:
        """Yield material matches as the index scan finds them."""
        if criteria is None:
            criteria = MaterialSearchCriteria(**kwargs)  # type: ignore[arg-type]
        return iter_search_material(
            self._entries,
            self._game_data,
            criteria,
            game_indices,
            decode_game=self._load_game,
        )

    def search_tree(
        self,
        target: chess.Board | None = None,
        game_indices: list[int] | None = None,
    ) -> TreeResult:
        """Compute opening tree statistics from a board position.

        Finds all games containing the target position and aggregates
        per-move statistics: frequency, win/draw/loss counts, average
        Elo, FIDE performance rating, and average year.

        This combines position search with statistics collection,
        matching SCID's native tree algorithm.

        Args:
            target: The board position to compute tree for.  If None,
                uses the standard starting position.
            game_indices: If given, only search these game indices.

        Returns:
            A :class:`TreeResult` with per-move nodes and totals.

        Example::

            import chess
            from scid import ScidDatabase

            with ScidDatabase.open("mydb") as db:
                # Tree from starting position
                tree = db.search_tree()
                for node in tree.nodes[:10]:
                    print(f"{node.san:6s} {node.count:6d}  {node.score:.1%}")

                # Tree from a specific position
                board = chess.Board()
                board.push_san("e4")
                board.push_san("e5")
                tree = db.search_tree(board)
        """
        if target is None:
            target = chess.Board()
        return search_tree(
            self._entries,
            self._namebase,
            self._game_data,
            target,
            game_indices,
            decode_game=self._load_game,
        )

    # ------------------------------------------------------------------
    # Reading games
    # ------------------------------------------------------------------

    def _load_game(self, index: int) -> chess.pgn.Game:
        """Load and return a full chess.pgn.Game for the given index."""
        entry = self._entries[index]
        blob = bytes(
            self._game_data[entry.offset : entry.offset + entry.game_data_size]
        )
        game = decode_game_data(blob)

        # Populate Seven Tag Roster from index/namebase
        nb = self._namebase
        game.headers["Event"] = nb.get_name(NAME_EVENT, entry.event_id)
        game.headers["Site"] = nb.get_name(NAME_SITE, entry.site_id)
        game.headers["Date"] = date_to_pgn(entry.date)
        game.headers["Round"] = nb.get_name(NAME_ROUND, entry.round_id)
        game.headers["White"] = nb.get_name(NAME_PLAYER, entry.white_id)
        game.headers["Black"] = nb.get_name(NAME_PLAYER, entry.black_id)
        game.headers["Result"] = RESULT_STR.get(entry.result, "*")

        # Elo
        if entry.white_elo:
            game.headers["WhiteElo"] = str(entry.white_elo)
        if entry.black_elo:
            game.headers["BlackElo"] = str(entry.black_elo)

        # ECO
        eco = eco_decode(entry.eco_code)
        if eco:
            game.headers["ECO"] = eco

        # Event date
        if entry.event_date:
            game.headers["EventDate"] = date_to_pgn(entry.event_date)

        return game

    def headers(self, index: int) -> dict[str, str]:
        """Return headers for game *index* without decoding moves.

        This is much faster than ``db[index].headers`` for header-only
        queries.
        """
        if index < 0:
            index += len(self._entries)
        if not 0 <= index < len(self._entries):
            raise IndexError(f"game index {index} out of range")

        entry = self._entries[index]
        nb = self._namebase
        h: dict[str, str] = {
            "Event": nb.get_name(NAME_EVENT, entry.event_id),
            "Site": nb.get_name(NAME_SITE, entry.site_id),
            "Date": date_to_pgn(entry.date),
            "Round": nb.get_name(NAME_ROUND, entry.round_id),
            "White": nb.get_name(NAME_PLAYER, entry.white_id),
            "Black": nb.get_name(NAME_PLAYER, entry.black_id),
            "Result": RESULT_STR.get(entry.result, "*"),
        }
        if entry.white_elo:
            h["WhiteElo"] = str(entry.white_elo)
        if entry.black_elo:
            h["BlackElo"] = str(entry.black_elo)
        eco = eco_decode(entry.eco_code)
        if eco:
            h["ECO"] = eco
        if entry.event_date:
            h["EventDate"] = date_to_pgn(entry.event_date)
        return h

    # ------------------------------------------------------------------
    # Writing games
    # ------------------------------------------------------------------

    def _ensure_writable(self) -> None:
        if self._mode == "r":
            raise IOError("Database opened read-only. Use mode='rw'.")

    def add_game(self, game: chess.pgn.Game) -> int:
        """Add a game to the database. Returns the new game index."""
        self._ensure_writable()

        # Encode game blob
        blob = encode_game_data(game)

        # Ensure game_data is mutable
        if not isinstance(self._game_data, bytearray):
            self._game_data = bytearray(self._game_data)

        # Build index entry
        entry = self._build_entry(game, blob)
        entry.offset = len(self._game_data)
        entry.game_data_size = len(blob)

        self._game_data.extend(blob)
        self._entries.append(entry)
        self._dirty = True

        return len(self._entries) - 1

    def replace_game(self, index: int, game: chess.pgn.Game) -> None:
        """Replace the game at *index* with a new game.

        Note: the old blob space is wasted until :meth:`compact` is called.
        """
        self._ensure_writable()

        if index < 0:
            index += len(self._entries)
        if not 0 <= index < len(self._entries):
            raise IndexError(f"game index {index} out of range")

        blob = encode_game_data(game)

        if not isinstance(self._game_data, bytearray):
            self._game_data = bytearray(self._game_data)

        entry = self._build_entry(game, blob)
        entry.offset = len(self._game_data)
        entry.game_data_size = len(blob)

        self._game_data.extend(blob)
        self._entries[index] = entry
        self._dirty = True

    def delete_game(self, index: int) -> None:
        """Mark a game as deleted (sets delete flag).

        The game data is not removed until :meth:`compact` is called.
        """
        self._ensure_writable()

        if index < 0:
            index += len(self._entries)
        if not 0 <= index < len(self._entries):
            raise IndexError(f"game index {index} out of range")

        self._entries[index].flags |= 1 << FLAG_DELETE
        self._dirty = True

    def compact(self) -> None:
        """Remove deleted games and rewrite game data to reclaim space."""
        self._ensure_writable()

        if not isinstance(self._game_data, bytearray):
            self._game_data = bytearray(self._game_data)

        new_data = bytearray()
        new_entries = []

        for entry in self._entries:
            if entry.flags & (1 << FLAG_DELETE):
                continue  # Skip deleted games

            blob = self._game_data[entry.offset : entry.offset + entry.game_data_size]
            entry.offset = len(new_data)
            new_data.extend(blob)
            new_entries.append(entry)

        self._game_data = new_data
        self._entries = new_entries
        self._dirty = True

    def _build_entry(self, game: chess.pgn.Game, blob: bytes) -> IndexEntry:
        """Build an IndexEntry from a python-chess Game."""
        h = game.headers
        nb = self._namebase
        entry = IndexEntry()

        # Names — add to namebase if not present
        entry.white_id = nb.add_name(NAME_PLAYER, h.get("White", "?"))
        entry.black_id = nb.add_name(NAME_PLAYER, h.get("Black", "?"))
        entry.event_id = nb.add_name(NAME_EVENT, h.get("Event", "?"))
        entry.site_id = nb.add_name(NAME_SITE, h.get("Site", "?"))
        entry.round_id = nb.add_name(NAME_ROUND, h.get("Round", "?"))

        # Result
        entry.result = RESULT_FROM_STR.get(h.get("Result", "*"), 0)

        # Date
        entry.date = date_from_pgn(h.get("Date", "????.??.??"))

        # Event date
        event_date_str = h.get("EventDate")
        if event_date_str:
            entry.event_date = date_from_pgn(event_date_str)

        # ECO
        eco_str = h.get("ECO", "")
        if eco_str:
            entry.eco_code = eco_encode(eco_str)

        # Elo
        entry.white_elo = _parse_elo(h.get("WhiteElo", "0"))
        entry.black_elo = _parse_elo(h.get("BlackElo", "0"))

        # Count half-moves
        entry.num_half_moves = min(sum(1 for _ in game.mainline_moves()), 1023)

        # Count variations, comments, NAGs
        var_count = 0
        comment_count = 0
        nag_count = 0

        def _walk(node: chess.pgn.GameNode) -> None:
            nonlocal var_count, comment_count, nag_count
            if node.comment:
                comment_count += 1
            if isinstance(node, chess.pgn.ChildNode):
                nag_count += len(node.nags)
            for i, var in enumerate(node.variations):
                if i > 0:
                    var_count += 1
                _walk(var)

        _walk(game)

        # IndexEntry holds plain counts; the index codecs apply encode_count().
        entry.n_variations = var_count
        entry.n_comments = comment_count
        entry.n_nags = nag_count

        # Index flags and position search data.  SCID leaves the stored line
        # and home pawn data empty for games with a non-standard start.
        start_board = game.board()
        mainline = list(game.mainline_moves())
        promotions = [m.promotion for m in mainline if m.promotion]
        if start_board.fen() != chess.STARTING_FEN:
            entry.flags |= 1 << FLAG_START
        if promotions:
            entry.flags |= 1 << FLAG_PROMO
        if any(p != chess.QUEEN for p in promotions):
            entry.flags |= 1 << FLAG_UPROMO

        if not entry.flags & (1 << FLAG_START):
            entry.stored_line_code = classify(mainline)
            entry.home_pawn_data = encode_homepawns_from_moves(start_board, mainline)

        # Material signature from final position
        board = game.end().board()
        entry.final_mat_sig = encode_matsig(board)

        return entry

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write all changes to disk."""
        self._ensure_writable()
        if not self._dirty:
            return

        path = self._path
        if self._version == 4:
            self._save_v4(path)
        else:
            self._save_v5(path)
        self._dirty = False

    def _save_v4(self, path: str) -> None:
        hdr = self._header or IndexHeader()
        index, names = io.BytesIO(), io.BytesIO()
        write_si4(index, hdr, self._entries)
        self._namebase.write_sn4(names)
        _replace_files(
            {
                f"{path}.sg4": self._game_data,
                f"{path}.sn4": names.getvalue(),
                f"{path}.si4": index.getvalue(),
            }
        )

    def _save_v5(self, path: str) -> None:
        index, names = io.BytesIO(), io.BytesIO()
        write_si5(index, self._entries)
        self._namebase.write_sn5(names)
        _replace_files(
            {
                f"{path}.sg5": self._game_data,
                f"{path}.sn5": names.getvalue(),
                f"{path}.si5": index.getvalue(),
            }
        )

    # ------------------------------------------------------------------
    # PGN export / import
    # ------------------------------------------------------------------

    def to_pgn(self, path: str | Path, encoding: str = "utf-8") -> None:
        """Export the entire database to a PGN file in the given encoding."""
        with open(path, "w", encoding=encoding) as f:
            exporter = chess.pgn.FileExporter(f)
            for i in range(len(self._entries)):
                game = self._load_game(i)
                game.accept(exporter)

    @classmethod
    def from_pgn(
        cls,
        pgn_path: str | Path,
        db_path: str | Path,
        version: int = 5,
        encoding: str = "utf-8-sig",
    ) -> ScidDatabase:
        """Import games from a PGN file into a new database.

        Args:
            pgn_path: Path to the PGN file to import.
            db_path: Base path for the new database.
            version: SCID format version (4 or 5).
            encoding: Text encoding of the PGN file.  The default accepts
                UTF-8 with or without a byte order mark; pass e.g.
                ``"latin-1"`` for legacy files.

        Returns:
            The newly created ScidDatabase.
        """
        db = cls.create(db_path, version=version)

        with open(pgn_path, encoding=encoding) as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                db.add_game(game)

        db.save()
        return db

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Save (if writable and dirty) and release resources."""
        if self._mode != "r" and self._dirty:
            self.save()

    def __enter__(self) -> ScidDatabase:
        return self

    def __exit__(self, exc_type: object, *args: object) -> None:
        # Do not persist a partially modified database after an error.
        if exc_type is None:
            self.close()

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        fmt = f"SCID{self._version}"
        return f"ScidDatabase('{self._path}', {fmt}, {len(self._entries)} games)"
