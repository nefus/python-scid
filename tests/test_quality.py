"""Tests for Stockfish-based game quality analysis."""

from __future__ import annotations

from pathlib import Path

import chess
import chess.engine
import chess.pgn
import pytest

from scid import GameQuality, analyze_database_quality, analyze_game_quality


class FakeEngine:
    """Small engine stub that returns predictable root scores."""

    id = {"name": "test engine"}

    def __init__(
        self,
        best_score: int = 100,
        played_score: int = 100,
        best_move: chess.Move | None = None,
    ) -> None:
        self.best_score = best_score
        self.played_score = played_score
        self.best_move = best_move
        self.calls: list[tuple[str, list[chess.Move] | None]] = []
        self.configured_threads: int | None = None
        self.closed = False

    def configure(self, options: dict[str, int]) -> None:
        self.configured_threads = options["Threads"]

    def quit(self) -> None:
        self.closed = True

    def analyse(
        self,
        board: chess.Board,
        limit: chess.engine.Limit,
        *,
        root_moves: list[chess.Move] | None = None,
    ) -> chess.engine.InfoDict:
        del limit
        self.calls.append((board.fen(), root_moves))
        if root_moves:
            move = root_moves[0]
            value = self.played_score
        else:
            move = self.best_move or next(iter(board.legal_moves))
            value = self.best_score
        return {
            "score": chess.engine.PovScore(chess.engine.Cp(value), board.turn),
            "pv": [move],
        }


def _game_with_moves(*uci_moves: str) -> chess.pgn.Game:
    game = chess.pgn.Game()
    node: chess.pgn.GameNode = game
    for uci in uci_moves:
        move = chess.Move.from_uci(uci)
        node = node.add_variation(move)
    return game


def test_perfect_play_has_maximum_index() -> None:
    game = _game_with_moves("e2e4")
    engine = FakeEngine()

    result = analyze_game_quality(game, engine)

    assert isinstance(result, GameQuality)
    assert result.index == 100.0
    assert result.mean_expected_score_loss == 0.0
    assert result.moves_analyzed == 1
    assert result.moves[0].played_san == "e4"
    assert result.moves[0].expected_score_loss == 0.0
    assert len(engine.calls) == 2
    assert engine.calls[1][1] == [chess.Move.from_uci("e2e4")]


def test_selected_best_move_is_not_penalized_for_search_noise() -> None:
    game = _game_with_moves("e2e4")
    engine = FakeEngine(
        best_score=200,
        played_score=-200,
        best_move=chess.Move.from_uci("e2e4"),
    )

    result = analyze_game_quality(game, engine)

    assert result.index == 100.0
    assert result.moves[0].best_move == result.moves[0].played_move
    assert result.moves[0].expected_score_loss == 0.0


def test_suboptimal_moves_lower_index_and_use_mover_perspective() -> None:
    game = _game_with_moves("e2e4", "e7e5")
    engine = FakeEngine(best_score=200, played_score=-200)

    result = analyze_game_quality(game, engine)

    assert result.index is not None and 0 < result.index < 100
    assert result.mean_expected_score_loss is not None
    assert result.mean_expected_score_loss > 0
    assert [move.ply for move in result.moves] == [1, 2]
    assert [move.played_san for move in result.moves] == ["e4", "e5"]


def test_game_without_moves_has_no_index() -> None:
    result = analyze_game_quality(chess.pgn.Game(), FakeEngine())

    assert result.index is None
    assert result.mean_expected_score_loss is None
    assert result.moves == ()
    assert result.moves_analyzed == 0


def test_missing_engine_score_is_reported() -> None:
    class NoScoreEngine(FakeEngine):
        def analyse(self, board, limit, *, root_moves=None):
            return {"pv": [next(iter(board.legal_moves))]}

    with pytest.raises(ValueError, match="did not include a score"):
        analyze_game_quality(_game_with_moves("e2e4"), NoScoreEngine())


def test_batch_analysis_uses_worker_engines_and_returns_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDatabase:
        games = {
            1: _game_with_moves("e2e4"),
            4: _game_with_moves("d2d4"),
            7: _game_with_moves("c2c4"),
        }

        def __len__(self) -> int:
            return len(self.games)

        def __getitem__(self, index: int) -> chess.pgn.Game:
            return self.games[index]

    engines: list[FakeEngine] = []

    def start_engine(command: str | list[str]) -> FakeEngine:
        del command
        engine = FakeEngine()
        engines.append(engine)
        return engine

    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", start_engine)
    progress: list[tuple[int, int]] = []

    result = analyze_database_quality(
        FakeDatabase(),
        game_indices=[7, 1, 4],
        workers=2,
        engine_threads=1,
        progress_callback=lambda done, total: progress.append((done, total)),
    )

    assert list(result) == [1, 4, 7]
    assert all(quality.index == 100.0 for quality in result.values())
    assert progress[-1] == (3, 3)
    assert progress[-1][0] == len(progress)
    assert engines
    assert all(engine.configured_threads == 1 for engine in engines)
    assert all(engine.closed for engine in engines)


def test_empty_batch_does_not_start_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EmptyDatabase:
        def __len__(self) -> int:
            return 0

    def unexpected_engine_start(command: str | list[str]) -> FakeEngine:
        raise AssertionError(f"unexpected engine start: {command}")

    monkeypatch.setattr(
        chess.engine.SimpleEngine, "popen_uci", unexpected_engine_start
    )

    assert analyze_database_quality(EmptyDatabase()) == {}


def test_pgn_checkpoint_resumes_and_recovers_truncated_tail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeDatabase:
        games = {
            0: _game_with_moves("e2e4"),
            1: _game_with_moves("d2d4", "d7d5"),
        }
        path = str(tmp_path / "testdb")
        version = 4

        def __len__(self) -> int:
            return len(self.games)

        def __getitem__(self, index: int) -> chess.pgn.Game:
            return self.games[index]

    for extension in (".si4", ".sn4", ".sg4"):
        Path(f"{FakeDatabase.path}{extension}").write_bytes(extension.encode())

    engines: list[FakeEngine] = []

    def start_engine(command: str | list[str]) -> FakeEngine:
        del command
        engine = FakeEngine()
        engines.append(engine)
        return engine

    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", start_engine)
    checkpoint = tmp_path / "quality.pgn"
    database = FakeDatabase()

    first = analyze_database_quality(
        database,
        game_indices=[0, 1],
        workers=2,
        pgn_path=checkpoint,
    )
    assert set(first) == {0, 1}
    assert all(engine.closed for engine in engines)

    checkpoint.write_text(
        checkpoint.read_text(encoding="utf-8")
        + '\n[Event "interrupted"]\n[ScidGameIndex "9"]\n',
        encoding="utf-8",
    )
    engine_count = len(engines)
    resumed = analyze_database_quality(
        database,
        game_indices=[0, 1],
        workers=2,
        pgn_path=checkpoint,
    )

    assert resumed == first
    assert len(engines) == engine_count + 1  # Identity probe only; no game analysis.
    assert engines[-1].calls == []
    assert engines[-1].closed
    text = checkpoint.read_text(encoding="utf-8")
    assert '[ScidQualityIndex "100.0"]' in text
    assert '[ScidGameIndex "9"]' not in text

    with checkpoint.open(encoding="utf-8") as source:
        saved_games = []
        while game := chess.pgn.read_game(source):
            if "ScidGameIndex" in game.headers:
                saved_games.append(game)
    assert sorted(game.headers["ScidGameIndex"] for game in saved_games) == [
        "0",
        "1",
    ]


def test_pgn_checkpoint_rejects_changed_database(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeDatabase:
        games = {0: _game_with_moves("e2e4")}
        path = str(tmp_path / "testdb")
        version = 4

        def __len__(self) -> int:
            return len(self.games)

        def __getitem__(self, index: int) -> chess.pgn.Game:
            return self.games[index]

    for extension in (".si4", ".sn4", ".sg4"):
        Path(f"{FakeDatabase.path}{extension}").write_bytes(extension.encode())

    monkeypatch.setattr(
        chess.engine.SimpleEngine, "popen_uci", lambda command: FakeEngine()
    )
    checkpoint = tmp_path / "quality.pgn"
    database = FakeDatabase()
    analyze_database_quality(
        database, game_indices=[0], pgn_path=checkpoint, workers=1
    )

    Path(f"{FakeDatabase.path}.sg4").write_bytes(b"changed database")
    with pytest.raises(ValueError, match="do not match"):
        analyze_database_quality(
            database, game_indices=[0], pgn_path=checkpoint, workers=1
        )
