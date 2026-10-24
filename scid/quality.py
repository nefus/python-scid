"""Stockfish-based quality analysis for a game's main line.

The returned 0–100 index is a provisional measure of move accuracy, not an
estimate of player Elo. It is derived from the expected-score loss of played
moves compared with the engine's best move.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
import hashlib
import json
from math import exp
import os
from pathlib import Path
import tempfile
from threading import Lock, local
from typing import Callable, Sequence, TYPE_CHECKING

import chess
import chess.engine
import chess.pgn

if TYPE_CHECKING:
    from scid.database import ScidDatabase


# Provisional scaling: a mean loss of 0.2 expected points maps to about 37.
# This should be calibrated against a representative collection of games.
_LOSS_SCALE = 5.0
_CHECKPOINT_SCHEMA_VERSION = 1
_CHECKPOINT_RESULT_VERSION = "expected-score-loss-exp-v1"
_CHECKPOINT_MARKER = f"; python-scid-quality-checkpoint-v{_CHECKPOINT_SCHEMA_VERSION} "


@dataclass(frozen=True, slots=True)
class MoveQuality:
    """Engine comparison for one played main-line move.

    ``expected_score_loss`` is measured from the perspective of the player
    who made the move and ranges from 0 (no detected loss) to 1.
    """

    ply: int
    played_move: chess.Move
    played_san: str
    best_move: chess.Move
    best_san: str
    expected_score_loss: float


@dataclass(frozen=True, slots=True)
class GameQuality:
    """Quality summary for a game's main line.

    ``index`` ranges from 0 to 100. It is ``None`` for games without moves,
    because there is no evidence from which to assess move quality.
    """

    index: float | None
    mean_expected_score_loss: float | None
    moves: tuple[MoveQuality, ...]

    @property
    def moves_analyzed(self) -> int:
        """Number of main-line moves analyzed."""

        return len(self.moves)


def _expected_score(
    info: chess.engine.InfoDict, color: chess.Color, ply: int
) -> float:
    """Return an engine score as expected points from ``color``'s perspective."""
    score = info.get("score")
    if score is None:
        raise ValueError("engine analysis did not include a score")
    return score.pov(color).wdl(model="sf", ply=ply).expectation()


def analyze_game_quality(
    game: chess.pgn.Game,
    engine: chess.engine.SimpleEngine,
    limit: chess.engine.Limit | None = None,
) -> GameQuality:
    """Analyze the main line and return a provisional 0–100 quality index.

    For each move, the engine analyzes the position once without restrictions
    and once with the played move forced as the root move. The difference in
    expected score is recorded from the mover's perspective. The game index
    is ``100 * exp(-5 * mean_expected_score_loss)``; the scale is deliberately
    provisional and should be calibrated before comparing games as absolute
    quality levels.

    Args:
        game: A ``python-chess`` game. Only its main line is analyzed.
        engine: A running ``python-chess`` UCI engine, such as Stockfish.
        limit: Search limit applied to each engine analysis. Defaults to
            depth 12. For example, pass ``chess.engine.Limit(time=0.1)``.

    Returns:
        A :class:`GameQuality` containing the index and per-move comparisons.

    Raises:
        ValueError: If an engine response does not contain an evaluation or
            principal variation.
        chess.engine.EngineError: If the engine reports an analysis error.
    """
    if limit is None:
        limit = chess.engine.Limit(depth=12)

    board = game.board()
    move_results: list[MoveQuality] = []

    for ply, node in enumerate(game.mainline(), start=1):
        move = node.move
        mover = board.turn

        best_info = engine.analyse(board, limit)
        best_pv = best_info.get("pv")
        if not best_pv:
            raise ValueError("engine analysis did not include a principal variation")
        best_move = best_pv[0]

        played_info = engine.analyse(board, limit, root_moves=[move])
        best_expectation = _expected_score(best_info, mover, board.ply())
        played_expectation = _expected_score(played_info, mover, board.ply())
        if best_move == move:
            # Independent root-restricted searches can disagree slightly even
            # when the played move is the engine's selected best move.
            loss = 0.0
        else:
            loss = max(0.0, min(1.0, best_expectation - played_expectation))

        move_results.append(
            MoveQuality(
                ply=ply,
                played_move=move,
                played_san=board.san(move),
                best_move=best_move,
                best_san=board.san(best_move),
                expected_score_loss=loss,
            )
        )
        board.push(move)

    if not move_results:
        return GameQuality(index=None, mean_expected_score_loss=None, moves=())

    mean_loss = sum(result.expected_score_loss for result in move_results) / len(
        move_results
    )
    index = 100.0 * exp(-_LOSS_SCALE * mean_loss)
    return GameQuality(
        index=round(index, 1),
        mean_expected_score_loss=mean_loss,
        moves=tuple(move_results),
    )


def _quality_to_json(quality: GameQuality) -> str:
    """Serialize a quality result for a PGN header tag."""
    return json.dumps(
        {
            "index": quality.index,
            "mean_expected_score_loss": quality.mean_expected_score_loss,
            "moves": [
                {
                    "ply": move.ply,
                    "played_move": move.played_move.uci(),
                    "played_san": move.played_san,
                    "best_move": move.best_move.uci(),
                    "best_san": move.best_san,
                    "expected_score_loss": move.expected_score_loss,
                }
                for move in quality.moves
            ],
        },
        separators=(",", ":"),
        allow_nan=False,
    )


def _quality_from_json(payload: str) -> GameQuality:
    """Restore a quality result from a PGN header tag."""
    data = json.loads(payload)
    moves = tuple(
        MoveQuality(
            ply=move["ply"],
            played_move=chess.Move.from_uci(move["played_move"]),
            played_san=move["played_san"],
            best_move=chess.Move.from_uci(move["best_move"]),
            best_san=move["best_san"],
            expected_score_loss=move["expected_score_loss"],
        )
        for move in data["moves"]
    )
    return GameQuality(
        index=data["index"],
        mean_expected_score_loss=data["mean_expected_score_loss"],
        moves=moves,
    )


def _database_fingerprint(database: ScidDatabase) -> dict[str, object]:
    """Hash all three files so a checkpoint cannot silently cross databases."""
    base = Path(database.path).expanduser().resolve()
    extensions = (".si4", ".sn4", ".sg4") if database.version == 4 else (
        ".si5",
        ".sn5",
        ".sg5",
    )
    hashes: dict[str, str] = {}
    for extension in extensions:
        path = Path(f"{base}{extension}")
        digest = hashlib.sha256()
        try:
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise ValueError(
                f"cannot fingerprint database file for checkpoint: {path}"
            ) from exc
        hashes[extension] = digest.hexdigest()
    return {"path": str(base), "version": database.version, "sha256": hashes}


def _limit_fingerprint(limit: chess.engine.Limit) -> dict[str, object]:
    """Return the effective UCI time/depth limit as JSON-safe values."""
    fields = (
        "time",
        "depth",
        "nodes",
        "mate",
        "white_clock",
        "black_clock",
        "white_inc",
        "black_inc",
        "remaining_moves",
        "clock_id",
    )
    return {field: getattr(limit, field, None) for field in fields}


def _engine_identity(
    command: str | list[str], engine_threads: int
) -> dict[str, str]:
    """Probe the UCI engine identity for checkpoint compatibility."""
    engine = chess.engine.SimpleEngine.popen_uci(command)
    try:
        engine.configure({"Threads": engine_threads})
        return dict(engine.id)
    finally:
        try:
            engine.quit()
        except (chess.engine.EngineError, OSError):
            pass


def _checkpoint_record(
    game: chess.pgn.Game, config_id: str
) -> tuple[int, GameQuality] | None:
    """Read and validate a completed checkpoint game from its PGN tags."""
    headers = game.headers
    if "ScidGameIndex" not in headers:
        return None
    saved_config = headers.get("ScidQualityConfig")
    if saved_config is not None and saved_config != config_id:
        raise ValueError(
            "PGN checkpoint settings or database do not match this analysis; "
            "choose another pgn_path or set resume=False with a new file"
        )
    payload = headers.get("ScidQualityData")
    if (
        saved_config is None
        or payload is None
        or headers.get("ScidQualitySchema") != str(_CHECKPOINT_SCHEMA_VERSION)
        or game.errors
    ):
        return None
    try:
        index = int(headers["ScidGameIndex"])
        quality = _quality_from_json(payload)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if sum(1 for _ in game.mainline()) != quality.moves_analyzed:
        return None
    expected_index = f"{quality.index:.1f}" if quality.index is not None else "NA"
    if headers.get("ScidQualityIndex") != expected_index:
        return None
    return index, quality


def _serialize_checkpoint_game(game: chess.pgn.Game) -> str:
    exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
    text = game.accept(exporter)
    return text.rstrip() + "\n\n"


def _repair_checkpoint_tail(path: Path, config_id: str) -> None:
    """Atomically drop an incomplete final PGN game after an interrupted write."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as target:
            target.write(f"{_CHECKPOINT_MARKER}{config_id}\n\n")
            with path.open("r", encoding="utf-8") as source:
                while game := chess.pgn.read_game(source):
                    record = _checkpoint_record(game, config_id)
                    if record is None:
                        break
                    target.write(_serialize_checkpoint_game(game))
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _initialize_pgn_checkpoint(path: Path, config_id: str, resume: bool) -> None:
    """Create a PGN checkpoint marker or verify the existing file belongs here."""
    if path.exists() and not resume:
        raise FileExistsError(
            f"PGN checkpoint already exists; set resume=True to use it: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    marker = f"{_CHECKPOINT_MARKER}{config_id}\n\n"
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", encoding="utf-8", newline="\n") as target:
            target.write(marker)
            target.flush()
            os.fsync(target.fileno())
        return

    with path.open("r", encoding="utf-8") as source:
        first_line = source.readline().rstrip("\r\n")
    if not first_line.startswith(_CHECKPOINT_MARKER):
        raise ValueError(f"not a python-scid quality checkpoint: {path}")
    if first_line != marker.rstrip("\n"):
        raise ValueError(
            "PGN checkpoint settings or database do not match this analysis; "
            "choose another pgn_path or use a new checkpoint file"
        )


def _load_pgn_checkpoint(
    path: Path,
    config_id: str,
    indices: Sequence[int],
    progress_callback: Callable[[int, int], None] | None,
) -> dict[int, GameQuality]:
    """Load completed results, repairing a truncated final PGN record if needed."""
    results: dict[int, GameQuality] = {}
    requested = set(indices)
    incomplete_tail = False
    valid_after_incomplete = False
    with path.open("r", encoding="utf-8") as source:
        while game := chess.pgn.read_game(source):
            record = _checkpoint_record(game, config_id)
            if record is None:
                incomplete_tail = True
                continue
            if incomplete_tail:
                valid_after_incomplete = True
                break
            index, quality = record
            if index in requested:
                previous = results.get(index)
                if previous is not None and previous != quality:
                    raise ValueError(f"conflicting checkpoint results for game {index}")
                if previous is None:
                    results[index] = quality
                    if progress_callback is not None:
                        progress_callback(len(results), len(indices))

    if valid_after_incomplete:
        raise ValueError(
            "checkpoint contains an incomplete PGN game before later records"
        )
    if incomplete_tail:
        _repair_checkpoint_tail(path, config_id)
    return results


def _append_pgn_checkpoint(
    path: Path,
    index: int,
    game: chess.pgn.Game,
    quality: GameQuality,
    config_id: str,
    engine_id: dict[str, str],
    limit: chess.engine.Limit,
) -> None:
    """Append one fully analyzed, tagged game and durably flush it."""
    game.headers["ScidGameIndex"] = str(index)
    game.headers["ScidQualitySchema"] = str(_CHECKPOINT_SCHEMA_VERSION)
    game.headers["ScidQualityIndex"] = (
        f"{quality.index:.1f}" if quality.index is not None else "NA"
    )
    game.headers["ScidQualityMeanLoss"] = (
        f"{quality.mean_expected_score_loss:.8f}"
        if quality.mean_expected_score_loss is not None
        else "NA"
    )
    game.headers["ScidQualityEngine"] = engine_id.get("name", "unknown")
    game.headers["ScidQualityConfig"] = config_id
    game.headers["ScidQualityData"] = _quality_to_json(quality)
    game.headers["ScidQualityLimit"] = json.dumps(
        _limit_fingerprint(limit), sort_keys=True, separators=(",", ":")
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as target:
        target.write(_serialize_checkpoint_game(game))
        target.flush()
        os.fsync(target.fileno())


def analyze_database_quality(
    database: ScidDatabase,
    engine_command: str | Sequence[str] = "stockfish",
    game_indices: Sequence[int] | None = None,
    limit: chess.engine.Limit | None = None,
    workers: int | None = None,
    engine_threads: int = 1,
    progress_callback: Callable[[int, int], None] | None = None,
    pgn_path: str | Path | None = None,
    resume: bool = True,
) -> dict[int, GameQuality]:
    """Analyze selected database games using a pool of UCI engine processes.

    Each worker owns one engine process; an engine instance is never shared
    across threads. Pending work is bounded to twice the worker count, so a
    large database does not require decoding all its games up front. Results
    are returned as a mapping from database game index to quality summary.
    When ``pgn_path`` is set, each completed game is copied to an annotated
    PGN checkpoint with quality tags and flushed to disk. A subsequent call
    with the same path and settings resumes by skipping completed games.

    By default, the worker count is half the logical CPU count (at least one),
    a useful starting point when each engine uses one thread. Set ``workers``
    and ``engine_threads`` explicitly to tune resource use. Keep their product
    at or below the number of CPU cores available to the process.

    Args:
        database: An open :class:`ScidDatabase`. Do not modify it while the
            analysis is running.
        engine_command: Stockfish executable path or argv sequence. The
            default, ``"stockfish"``, is resolved using ``PATH``.
        game_indices: Database game indices to analyze. Defaults to all games.
        limit: Search limit per analysis. Defaults to depth 12.
        workers: Number of concurrent Stockfish processes. Defaults to half
            the logical CPU count, with a minimum of one.
        engine_threads: Number of Stockfish threads per process; defaults to 1.
        progress_callback: Optional callback called as ``callback(done, total)``
            after each completed game, from the calling thread.
        pgn_path: Optional path for an annotated PGN checkpoint. The source
            SCID database is never modified.
        resume: If a checkpoint exists, validate it and reuse its completed
            results. Set to False to refuse an existing file.

    Returns:
        A dictionary keyed by the corresponding database game index.

    Raises:
        ValueError: If worker or engine thread counts are invalid, an engine
            response is missing required analysis data, or the checkpoint is
            incompatible with this database or analysis configuration.
        FileExistsError: If ``pgn_path`` exists and ``resume`` is False.
        chess.engine.EngineError: If a UCI engine reports an analysis error.
    """
    if workers is None:
        workers = max(1, (os.cpu_count() or 1) // 2)
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if engine_threads < 1:
        raise ValueError("engine_threads must be at least 1")

    indices: Sequence[int] = (
        range(len(database)) if game_indices is None else game_indices
    )
    total = len(indices)
    if total == 0:
        return {}
    if len(set(indices)) != total:
        raise ValueError("game_indices must not contain duplicates")

    if limit is None:
        limit = chess.engine.Limit(depth=12)
    command: str | list[str] = (
        engine_command if isinstance(engine_command, str) else list(engine_command)
    )

    checkpoint: Path | None = None
    config_id: str | None = None
    engine_id: dict[str, str] = {}
    if pgn_path is not None:
        checkpoint = Path(pgn_path).expanduser()
        if checkpoint.exists() and not resume:
            raise FileExistsError(
                "PGN checkpoint already exists; set resume=True to use it: "
                f"{checkpoint}"
            )
        base = Path(database.path).expanduser().resolve()
        extensions = (".si4", ".sn4", ".sg4") if database.version == 4 else (
            ".si5",
            ".sn5",
            ".sg5",
        )
        source_files = {Path(f"{base}{extension}") for extension in extensions}
        if checkpoint.resolve() in source_files:
            raise ValueError("pgn_path must not overwrite a SCID database file")

        engine_id = _engine_identity(command, engine_threads)
        manifest = {
            "schema": _CHECKPOINT_SCHEMA_VERSION,
            "result_version": _CHECKPOINT_RESULT_VERSION,
            "database": _database_fingerprint(database),
            "engine": engine_id,
            "engine_threads": engine_threads,
            "limit": _limit_fingerprint(limit),
        }
        config_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        config_id = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
        _initialize_pgn_checkpoint(checkpoint, config_id, resume)
        results = _load_pgn_checkpoint(
            checkpoint, config_id, indices, progress_callback
        )
    else:
        results: dict[int, GameQuality] = {}

    worker_state = local()
    engine_lock = Lock()
    engines: list[chess.engine.SimpleEngine] = []

    def get_engine() -> chess.engine.SimpleEngine:
        engine = getattr(worker_state, "engine", None)
        if engine is None:
            engine = chess.engine.SimpleEngine.popen_uci(command)
            with engine_lock:
                engines.append(engine)
            engine.configure({"Threads": engine_threads})
            worker_state.engine = engine
        return engine

    def analyze_index(
        index: int,
    ) -> tuple[int, GameQuality, chess.pgn.Game]:
        engine = get_engine()
        game = database[index]
        quality = analyze_game_quality(game, engine, limit)
        return index, quality, game

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            indices_iter = iter(indices)
            pending: dict[Future[tuple[int, GameQuality, chess.pgn.Game]], int] = {}
            capacity = workers * 2

            def submit_next() -> bool:
                while True:
                    try:
                        index = next(indices_iter)
                    except StopIteration:
                        return False
                    if index in results:
                        continue
                    pending[executor.submit(analyze_index, index)] = index
                    return True

            for _ in range(min(capacity, total)):
                submit_next()

            completed = len(results)
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future)
                    index, quality, game = future.result()
                    if checkpoint is not None and config_id is not None:
                        _append_pgn_checkpoint(
                            checkpoint,
                            index,
                            game,
                            quality,
                            config_id,
                            engine_id,
                            limit,
                        )
                    results[index] = quality
                    completed += 1
                    if progress_callback is not None:
                        progress_callback(completed, total)
                    submit_next()
    finally:
        for engine in engines:
            try:
                engine.quit()
            except (chess.engine.EngineError, OSError):
                # Preserve the original analysis exception if an engine has
                # already exited or cannot be shut down cleanly.
                pass

    return dict(sorted(results.items()))
