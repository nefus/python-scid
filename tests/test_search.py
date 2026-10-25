"""Tests for search functionality (header, position, material, tree)."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterator

import chess
import chess.pgn
import pytest

from scid import ScidDatabase
from scid.search import (
    HeaderSearchCriteria,
    MaterialSearchCriteria,
    MaterialSearchResult,
    PositionSearchResult,
    TreeNode,
    TreeResult,
    _board_material_counts,
    _compute_performance,
    _compute_score,
    _FIDE_DP,
    _hpsig_make,
    _hpsig_match,
    _less_mat,
    _material_matches_criteria,
    _matsig_is_reachable,
    _stored_line_match,
    search_header,
    search_material,
    search_position,
    search_tree,
)
from scid.homepawns import encode_homepawns
from scid.matsig import decode_matsig, encode_matsig


# =====================================================================
# Unit tests for internal helpers
# =====================================================================


class TestHpSigMake:
    """Test home-pawn signature computation."""

    def test_starting_position(self) -> None:
        board = chess.Board()
        sig, n = _hpsig_make(board)
        assert sig == 0
        assert n == 0

    def test_after_e4(self) -> None:
        board = chess.Board()
        board.push_san("e4")
        sig, n = _hpsig_make(board)
        assert n == 1
        # e2 pawn moved -> change index 4, bit = 15-4 = 11 -> 0x0800
        assert sig == 0x0800

    def test_after_e4_e5(self) -> None:
        board = chess.Board()
        board.push_san("e4")
        board.push_san("e5")
        sig, n = _hpsig_make(board)
        assert n == 2
        # e2 -> change_index 4 -> bit 11 = 0x0800
        # e7 -> change_index 12 -> bit 3 = 0x0008
        assert sig == 0x0808


class TestHpSigMatch:
    """Test home-pawn signature matching."""

    def test_starting_position_matches_everything(self) -> None:
        # Target = starting position (no moved pawns), should match any game
        board = chess.Board()
        hp_data = encode_homepawns(board)
        assert _hpsig_match(0, 0, hp_data)

    def test_after_e4_matches_game_with_e4(self) -> None:
        # Target position has e2 pawn moved
        target = chess.Board()
        target.push_san("e4")
        target_sig, target_n = _hpsig_make(target)

        # Game that played e4 (e2 pawn moved)
        game_board = chess.Board()
        game_board.push_san("e4")
        hp_data = encode_homepawns(game_board)

        assert _hpsig_match(target_sig, target_n, hp_data)

    def test_mismatch_excludes(self) -> None:
        # Target: e4 played (e2 pawn moved)
        target = chess.Board()
        target.push_san("e4")
        target_sig, target_n = _hpsig_make(target)

        # Game that played d4 (d2 pawn moved, NOT e2)
        game_board = chess.Board()
        game_board.push_san("d4")
        hp_data = encode_homepawns(game_board)

        assert not _hpsig_match(target_sig, target_n, hp_data)

    def test_short_changelist_excludes(self) -> None:
        # Target has 3 pawns moved but game only recorded 2
        hp_data = bytes([2, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        assert not _hpsig_match(0x0007, 3, hp_data)


class TestLessMat:
    """Test material pre-filter."""

    def test_equal_material(self) -> None:
        board = chess.Board()
        sig = encode_matsig(board)
        counts = _board_material_counts(board)
        # Starting position should not be excluded
        assert not _less_mat(counts, sig, False, False)

    def test_more_pawns_than_final(self) -> None:
        # Target wants 7 white pawns, but final position has 8
        # (target has FEWER than final -> unreachable, since pawns only decrease)
        counts = {
            "WQ": 1,
            "WR": 2,
            "WB": 2,
            "WN": 2,
            "WP": 7,
            "BQ": 1,
            "BR": 2,
            "BB": 2,
            "BN": 2,
            "BP": 8,
        }
        board = chess.Board()  # starting position: 8 WP, 8 BP
        sig = encode_matsig(board)
        # Target has 7 WP < final's 8 WP -> game never had only 7 WP -> exclude
        assert _less_mat(counts, sig, False, False)

    def test_fewer_pawns_in_final_not_excluded(self) -> None:
        # Target wants 8 white pawns, final has 5 (lost 3 during game)
        # Since material decreases: 8->7->6->5, the 8-pawn position existed
        counts = {
            "WQ": 1,
            "WR": 2,
            "WB": 2,
            "WN": 2,
            "WP": 8,
            "BQ": 1,
            "BR": 2,
            "BB": 2,
            "BN": 2,
            "BP": 8,
        }
        board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PP1PP1PP/RNBQKBNR w KQkq - 0 1")
        sig = encode_matsig(board)  # 6 WP in final
        # Target 8 WP > final 6 WP -> not excluded (position existed earlier)
        assert not _less_mat(counts, sig, False, False)

    def test_promotion_allows_extra_queens(self) -> None:
        # Target wants 0 white queens. Final has 1 queen.
        # Target has fewer queens than final -> exclude (queen count can only decrease)
        counts = {
            "WQ": 0,
            "WR": 2,
            "WB": 2,
            "WN": 2,
            "WP": 8,
            "BQ": 1,
            "BR": 2,
            "BB": 2,
            "BN": 2,
            "BP": 8,
        }
        board = chess.Board()
        sig = encode_matsig(board)
        # Without promo flag: 0 WQ < 1 WQ -> excluded
        assert _less_mat(counts, sig, False, False)
        # With promo flag: wq_diff = 0-1 = -1, wp_diff = 8-8 = 0
        # wq_diff + wp_diff = -1 < 0 -> still excluded (no extra pawns to explain)
        assert _less_mat(counts, sig, True, False)

        # Now: target wants 2 WQ, final has 1 WQ but lost 2 pawns (6 WP final)
        # The 2 lost pawns could have been promoted to queens before capture
        counts2 = {
            "WQ": 2,
            "WR": 2,
            "WB": 2,
            "WN": 2,
            "WP": 7,
            "BQ": 1,
            "BR": 2,
            "BB": 2,
            "BN": 2,
            "BP": 8,
        }
        board2 = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPP1PP/RNBQKBNR w KQkq - 0 1")
        sig2 = encode_matsig(board2)  # 7 WP, 1 WQ
        # Without promo: 2 WQ < 1 WQ?  No, 2 > 1, target has MORE queens than final
        # less_mat checks wq_diff < 0: diff = 2-1 = 1, not < 0 -> not excluded
        # Actually less_mat should NOT exclude here because the extra queen existed earlier
        # But wait, without promos flag, queens can't increase... hmm
        # Actually less_mat is about: can this position be reached?
        # If no promos, queens can only decrease. Target has MORE queens than final -> was possible earlier
        assert not _less_mat(counts2, sig2, False, False)


class TestMatsigIsReachable:
    """Test matsig_isReachable."""

    def test_same_sig(self) -> None:
        board = chess.Board()
        sig = encode_matsig(board)
        assert _matsig_is_reachable(sig, sig, False, False)

    def test_fewer_pawns_unreachable(self) -> None:
        # Game ended with 6 WP, target wants 8 WP
        start = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PP1PPP1P/RNBQKBNR w KQkq - 0 1")
        target = chess.Board()
        ss = encode_matsig(start)
        ts = encode_matsig(target)
        assert not _matsig_is_reachable(ss, ts, False, False)


class TestStoredLineMatch:
    """Test stored line matching."""

    def test_starting_position(self) -> None:
        board = chess.Board()
        # Code 15 is "1.d4" — starting position should be found at ply 0
        ply = _stored_line_match(board, 15)
        assert ply == 0

    def test_after_d4(self) -> None:
        board = chess.Board()
        board.push_san("d4")
        # Code 15 is "d2d4" — position after d4 is at ply 1
        ply = _stored_line_match(board, 15)
        assert ply == 1

    def test_no_stored_line(self) -> None:
        board = chess.Board()
        assert _stored_line_match(board, 0) == -1

    def test_position_not_in_line(self) -> None:
        # A position that doesn't appear in stored line 15
        board = chess.Board()
        board.push_san("e4")
        ply = _stored_line_match(board, 15)
        assert ply == -1  # inconclusive, not in this line


# =====================================================================
# Integration tests: Header Search
# =====================================================================


class TestHeaderSearch:
    """Test header search against the 666 database."""

    def test_search_by_white_name(self, db_666_path: Path) -> None:
        with ScidDatabase.open(db_666_path) as db:
            hits = db.search_header(white="Aljechin")
            assert 0 in hits
            assert len(hits) == 1

    def test_search_by_white_case_insensitive(self, db_666_path: Path) -> None:
        with ScidDatabase.open(db_666_path) as db:
            hits = db.search_header(white="aljechin")
            assert 0 in hits

    def test_search_by_player_matches_either_color(self, db_666_path: Path) -> None:
        with ScidDatabase.open(db_666_path) as db:
            # "Engels" appears as White in game 5 and as Black in game 1
            hits = db.search_header(player="Engels")
            assert 1 in hits
            assert 5 in hits

    def test_search_by_result(self, db_666_path: Path) -> None:
        with ScidDatabase.open(db_666_path) as db:
            hits = db.search_header(results={"1-0"})
            # Games 0, 1, 2, 5, 8 are 1-0
            assert all(i in hits for i in [0, 1, 2, 5, 8])
            # Games 3, 6, 7 are 0-1
            assert all(i not in hits for i in [3, 6, 7])

    def test_search_by_date_range(self, db_666_path: Path) -> None:
        with ScidDatabase.open(db_666_path) as db:
            hits = db.search_header(date_min="1930.01.01", date_max="1939.12.31")
            # Games from the 1930s: 1 (1937), 2 (1938), 4 (1938), 5 (1937)
            assert all(i in hits for i in [1, 2, 4, 5])
            # Game 0 is from 1920, game 8 from 1924
            assert 0 not in hits
            assert 8 not in hits

    def test_search_by_draw_result(self, db_666_path: Path) -> None:
        with ScidDatabase.open(db_666_path) as db:
            hits = db.search_header(results={"1/2-1/2"})
            # Game 4 is the only draw
            assert hits == [4]

    def test_search_combined_criteria(self, db_666_path: Path) -> None:
        with ScidDatabase.open(db_666_path) as db:
            # Search for 1-0 games from the 1930s
            hits = db.search_header(
                results={"1-0"},
                date_min="1930.01.01",
                date_max="1939.12.31",
            )
            # Games 1 (1937, 1-0), 2 (1938, 1-0), 5 (1937, 1-0)
            assert sorted(hits) == [1, 2, 5]

    def test_search_no_results(self, db_666_path: Path) -> None:
        with ScidDatabase.open(db_666_path) as db:
            hits = db.search_header(white="Nonexistent Player")
            assert hits == []

    def test_search_with_game_indices(self, db_666_path: Path) -> None:
        with ScidDatabase.open(db_666_path) as db:
            # Only search in first 3 games
            hits = db.search_header(results={"1-0"}, game_indices=[0, 1, 2])
            assert sorted(hits) == [0, 1, 2]

    def test_search_exact_name(self, db_666_path: Path) -> None:
        with ScidDatabase.open(db_666_path) as db:
            # Exact match with quotes
            hits = db.search_header(white='"Aljechin"')
            assert 0 in hits
            # Partial should NOT match exact
            hits = db.search_header(white='"Alj"')
            assert 0 not in hits

    def test_search_by_length(self, db_666_path: Path) -> None:
        with ScidDatabase.open(db_666_path) as db:
            # Games with >= 30 half-moves
            hits = db.search_header(length_min=30)
            # Games 8 (31), 9 (32), 10 (35), 11 (35) have >= 30 half-moves
            assert all(i in hits for i in [8, 9, 10, 11])

    def test_search_with_criteria_object(self, db_666_path: Path) -> None:
        with ScidDatabase.open(db_666_path) as db:
            c = HeaderSearchCriteria(results={"0-1"})
            hits = db.search_header(c)
            assert all(i in hits for i in [3, 6, 7])

    def test_iter_search_header_matches_list_api(self, db_666_path: Path) -> None:
        with ScidDatabase.open(db_666_path) as db:
            assert list(db.iter_search_header(results={"0-1"})) == db.search_header(results={"0-1"})

    def test_search_header_freechess_by_elo(self, db_freechess_path: Path) -> None:
        with ScidDatabase.open(db_freechess_path) as db:
            hits = db.search_header(
                white_elo_min=2100,
                white_elo_max=2200,
                game_indices=list(range(100)),
            )
            # Verify all results have correct elo
            for idx in hits:
                h = db.headers(idx)
                elo = int(h.get("WhiteElo", "0"))
                assert 2100 <= elo <= 2200

    def test_search_header_freechess_by_eco(self, db_freechess_path: Path) -> None:
        with ScidDatabase.open(db_freechess_path) as db:
            hits = db.search_header(
                eco_min="B01",
                eco_max="B01",
                game_indices=list(range(200)),
            )
            # Verify all results have correct ECO
            for idx in hits:
                h = db.headers(idx)
                eco = h.get("ECO", "")
                assert eco.startswith("B01")


# =====================================================================
# Integration tests: Position Search
# =====================================================================


class TestPositionSearch:
    """Test position search against real databases."""

    def test_search_starting_position(self, db_666_path: Path) -> None:
        with ScidDatabase.open(db_666_path) as db:
            target = chess.Board()
            results = db.search_position(target)
            # All games with standard start should match at ply 0
            # Game 0 starts from FEN, so it should NOT match
            matching_indices = {r.game_index for r in results}
            assert 0 not in matching_indices  # FEN start
            # Most other games should match (standard start)
            # Games 1-11 should all have standard starts
            for r in results:
                if r.game_index != 0:
                    assert r.ply == 0

    def test_search_after_e4(self, db_666_path: Path) -> None:
        with ScidDatabase.open(db_666_path) as db:
            target = chess.Board()
            target.push_san("e4")
            results = db.search_position(target)
            # Games that start with 1.e4 should match at ply 1
            matching_indices = {r.game_index for r in results}
            # Game 8 (Sämisch vs Schlosser) starts 1.e4 e5 2.Nc3 ...
            assert 8 in matching_indices

    def test_search_specific_position(self, db_666_path: Path) -> None:
        with ScidDatabase.open(db_666_path) as db:
            # Search for the position after 1.e4 e5 2.Nc3 Nf6 3.f4
            # (from game 8: Sämisch vs Schlosser)
            target = chess.Board()
            for san in ["e4", "e5", "Nc3", "Nf6", "f4"]:
                target.push_san(san)
            results = db.search_position(target)
            matching_indices = {r.game_index for r in results}
            assert 8 in matching_indices

    def test_search_no_match(self, db_666_path: Path) -> None:
        with ScidDatabase.open(db_666_path) as db:
            # Search for a very specific position unlikely to exist
            target = chess.Board(
                "8/8/8/8/8/8/8/4K2k w - - 0 1"  # K vs K
            )
            results = db.search_position(target)
            assert results == []

    def test_search_with_game_indices(self, db_666_path: Path) -> None:
        with ScidDatabase.open(db_666_path) as db:
            target = chess.Board()
            results = db.search_position(target, game_indices=[8, 9, 10])
            matching_indices = {r.game_index for r in results}
            # Only games 8, 9, 10 should be searched
            assert matching_indices <= {8, 9, 10}

    def test_iter_search_position_matches_list_api(self, db_666_path: Path) -> None:
        with ScidDatabase.open(db_666_path) as db:
            target = chess.Board()
            matches = db.search_position(target)
            assert list(db.iter_search_position(target)) == matches
            first_match = matches[0].game_index

            def first_match_only() -> Iterator[int]:
                yield first_match
                raise AssertionError("the iterator read beyond the first match")

            assert next(db.iter_search_position(target, game_indices=first_match_only())) == matches[0]

    def test_position_search_freechess(self, db_freechess_path: Path) -> None:
        """Search for the Sicilian Defense position (1.e4 c5) in freechess."""
        with ScidDatabase.open(db_freechess_path) as db:
            target = chess.Board()
            target.push_san("e4")
            target.push_san("c5")

            # Only search first 500 games for speed
            results = db.search_position(target, game_indices=list(range(500)))
            assert len(results) > 0

            # Verify by decoding matching games
            for r in results[:3]:  # Check first 3
                game = db[r.game_index]
                board = game.board()
                for i, node in enumerate(game.mainline()):
                    board.push(node.move)
                    if i + 1 == r.ply:
                        assert board == target
                        break


# =====================================================================
# Integration tests: Material Search
# =====================================================================


class TestMaterialSearch:
    """Test material search against real databases."""

    def test_search_starting_material(self, db_666_path: Path) -> None:
        with ScidDatabase.open(db_666_path) as db:
            # Search for starting material (all pieces present)
            c = MaterialSearchCriteria(
                wq=(1, 1),
                wr=(2, 2),
                wb=(2, 2),
                wn=(2, 2),
                wp=(8, 8),
                bq=(1, 1),
                br=(2, 2),
                bb=(2, 2),
                bn=(2, 2),
                bp=(8, 8),
            )
            results = db.search_material(c)
            # All games with standard start should match at ply 0
            assert len(results) > 0
            for r in results:
                assert r.ply == 0

    def test_search_no_queens(self, db_666_path: Path) -> None:
        with ScidDatabase.open(db_666_path) as db:
            # Search for positions where both sides have 0 queens
            c = MaterialSearchCriteria(wq=(0, 0), bq=(0, 0))
            results = db.search_material(c)
            # Verify by checking the position at the match ply
            for r in results[:3]:
                game = db[r.game_index]
                board = game.board()
                for i, node in enumerate(game.mainline()):
                    board.push(node.move)
                    if i + 1 == r.ply:
                        break
                wq = len(board.pieces(chess.QUEEN, chess.WHITE))
                bq = len(board.pieces(chess.QUEEN, chess.BLACK))
                assert wq == 0 and bq == 0

    def test_search_rook_endgame(self, db_666_path: Path) -> None:
        with ScidDatabase.open(db_666_path) as db:
            # Rook endgame: each side has 1 rook, no queens, no bishops, no knights
            c = MaterialSearchCriteria(
                wq=(0, 0),
                bq=(0, 0),
                wb=(0, 0),
                bb=(0, 0),
                wn=(0, 0),
                bn=(0, 0),
                wr=(1, 2),
                br=(1, 2),
            )
            results = db.search_material(c)
            # Verify matches (if any)
            for r in results[:2]:
                game = db[r.game_index]
                board = game.board()
                for i, node in enumerate(game.mainline()):
                    board.push(node.move)
                    if i + 1 == r.ply:
                        break
                assert len(board.pieces(chess.QUEEN, chess.WHITE)) == 0
                assert len(board.pieces(chess.QUEEN, chess.BLACK)) == 0

    def test_material_search_with_criteria_object(self, db_666_path: Path) -> None:
        with ScidDatabase.open(db_666_path) as db:
            c = MaterialSearchCriteria(wp=(0, 4))
            results = db.search_material(c)
            # All results should have <= 4 white pawns at the match ply
            for r in results[:3]:
                game = db[r.game_index]
                board = game.board()
                for i, node in enumerate(game.mainline()):
                    board.push(node.move)
                    if i + 1 == r.ply:
                        break
                wp = len(board.pieces(chess.PAWN, chess.WHITE))
                assert 0 <= wp <= 4

    def test_iter_search_material_matches_list_api(self, db_666_path: Path) -> None:
        with ScidDatabase.open(db_666_path) as db:
            criteria = MaterialSearchCriteria(wq=(0, 0), bq=(0, 0))
            assert list(db.iter_search_material(criteria)) == db.search_material(criteria)

    def test_material_search_freechess(self, db_freechess_path: Path) -> None:
        """Search for queen sacrifice in first 200 freechess games."""
        with ScidDatabase.open(db_freechess_path) as db:
            # White has 0 queens at some point
            c = MaterialSearchCriteria(wq=(0, 0))
            results = db.search_material(c, game_indices=list(range(200)))
            # Some games should involve queen captures
            # Just verify results are returned and valid
            for r in results[:3]:
                game = db[r.game_index]
                board = game.board()
                for i, node in enumerate(game.mainline()):
                    board.push(node.move)
                    if i + 1 == r.ply:
                        break
                assert len(board.pieces(chess.QUEEN, chess.WHITE)) == 0


# =====================================================================
# Tests for ScidDatabase search method API
# =====================================================================


class TestDatabaseSearchAPI:
    """Test the ScidDatabase.search_* methods work correctly."""

    def test_search_header_kwargs(self, db_666_path: Path) -> None:
        with ScidDatabase.open(db_666_path) as db:
            hits = db.search_header(white="Aljechin")
            assert isinstance(hits, list)
            assert all(isinstance(i, int) for i in hits)

    def test_search_header_criteria_obj(self, db_666_path: Path) -> None:
        with ScidDatabase.open(db_666_path) as db:
            c = HeaderSearchCriteria(white="Aljechin")
            hits = db.search_header(c)
            assert 0 in hits

    def test_search_position_returns_results(self, db_666_path: Path) -> None:
        with ScidDatabase.open(db_666_path) as db:
            target = chess.Board()
            results = db.search_position(target)
            assert isinstance(results, list)
            assert all(isinstance(r, PositionSearchResult) for r in results)

    def test_search_material_returns_results(self, db_666_path: Path) -> None:
        with ScidDatabase.open(db_666_path) as db:
            results = db.search_material(wq=(1, 1))
            assert isinstance(results, list)
            assert all(isinstance(r, MaterialSearchResult) for r in results)


# =====================================================================
# Unit tests for tree search helpers
# =====================================================================


class TestComputeScore:
    """Test score computation helper."""

    def test_all_white_wins(self) -> None:
        assert _compute_score(10, 0, 0) == 1.0

    def test_all_black_wins(self) -> None:
        assert _compute_score(0, 0, 10) == 0.0

    def test_all_draws(self) -> None:
        assert _compute_score(0, 10, 0) == 0.5

    def test_mixed(self) -> None:
        # 5 white wins + 2 draws + 3 black wins = (5 + 1) / 10 = 0.6
        assert _compute_score(5, 2, 3) == pytest.approx(0.6)

    def test_zero_games(self) -> None:
        assert _compute_score(0, 0, 0) == 0.0


class TestComputePerformance:
    """Test FIDE performance rating computation."""

    def test_below_threshold(self) -> None:
        """Performance returns 0 if fewer than 10 rated games."""
        assert _compute_performance(10000, 10000, 9, 0.5, True) == 0.0

    def test_50_percent_score(self) -> None:
        """50% score => performance = opponent avg Elo."""
        perf = _compute_performance(20000, 20000, 10, 0.5, True)
        assert perf == pytest.approx(2000.0)

    def test_high_score(self) -> None:
        """High score => performance above opponent avg."""
        perf = _compute_performance(20000, 20000, 10, 0.7, True)
        assert perf > 2000.0

    def test_low_score(self) -> None:
        """Low score => performance below opponent avg."""
        perf = _compute_performance(20000, 20000, 10, 0.3, True)
        assert perf < 2000.0


class TestFideTable:
    """Test FIDE performance rating table."""

    def test_table_length(self) -> None:
        assert len(_FIDE_DP) == 101  # 0% through 100%

    def test_symmetric(self) -> None:
        """Table should be roughly symmetric around 50%."""
        assert _FIDE_DP[50] == 0
        for i in range(1, 50):
            assert _FIDE_DP[50 + i] == -_FIDE_DP[50 - i]

    def test_monotonic(self) -> None:
        """Table values should be monotonically increasing."""
        for i in range(100):
            assert _FIDE_DP[i] < _FIDE_DP[i + 1]


# =====================================================================
# Tree search integration tests — small databases
# =====================================================================


class TestTreeSearchSmall:
    """Tree search tests on small databases."""

    def test_starting_position_666(self, db_666_path: Path) -> None:
        """Tree from start pos in 666 db (12 games, 8 have custom FEN)."""
        with ScidDatabase.open(db_666_path) as db:
            tree = db.search_tree()
            assert isinstance(tree, TreeResult)
            assert tree.games_searched == 12
            # Games 8-11 start from standard position, all play 1.e4
            assert tree.games_matched == 4
            assert len(tree.nodes) == 1
            assert tree.nodes[0].san == "e4"
            assert tree.nodes[0].count == 4

    def test_tree_result_structure(self, db_666_path: Path) -> None:
        """Verify TreeResult and TreeNode dataclass fields."""
        with ScidDatabase.open(db_666_path) as db:
            tree = db.search_tree()
            assert tree.position == chess.Board()
            assert tree.total is not None
            assert tree.total.san == "Total"
            assert tree.total.count == tree.games_matched

            node = tree.nodes[0]
            assert isinstance(node.move, chess.Move)
            assert isinstance(node.san, str)
            assert isinstance(node.count, int)
            assert isinstance(node.white_wins, int)
            assert isinstance(node.draws, int)
            assert isinstance(node.black_wins, int)
            assert isinstance(node.score, float)
            assert isinstance(node.avg_elo, float)
            assert isinstance(node.performance, float)
            assert isinstance(node.avg_year, float)
            assert isinstance(node.draw_pct, float)

    def test_result_counts_consistent(self, db_666_path: Path) -> None:
        """white_wins + draws + black_wins should equal total count
        (excluding unknown result games)."""
        with ScidDatabase.open(db_666_path) as db:
            tree = db.search_tree()
            for n in tree.nodes:
                decided = n.white_wins + n.draws + n.black_wins
                # Some games may have unknown result (*)
                assert decided <= n.count

    def test_scandi_tree(self, db_scandi_path: Path) -> None:
        """Both scandi games should play 1.e4 from start."""
        with ScidDatabase.open(db_scandi_path) as db:
            tree = db.search_tree()
            assert tree.games_matched == 2
            assert len(tree.nodes) == 1
            assert tree.nodes[0].san == "e4"
            assert tree.nodes[0].count == 2

    def test_scandi_after_e4(self, db_scandi_path: Path) -> None:
        """After 1.e4, both scandi games should play 1...d5."""
        with ScidDatabase.open(db_scandi_path) as db:
            board = chess.Board()
            board.push_san("e4")
            tree = db.search_tree(board)
            assert tree.games_matched == 2
            assert tree.nodes[0].san == "d5"

    def test_scandi_after_e4_d5(self, db_scandi_path: Path) -> None:
        """After 1.e4 d5, both should continue 2.exd5."""
        with ScidDatabase.open(db_scandi_path) as db:
            board = chess.Board()
            board.push_san("e4")
            board.push_san("d5")
            tree = db.search_tree(board)
            assert tree.games_matched == 2
            assert tree.nodes[0].san == "exd5"

    def test_score_consistency(self, db_scandi_path: Path) -> None:
        """Score should be computed correctly from results."""
        with ScidDatabase.open(db_scandi_path) as db:
            tree = db.search_tree()
            total = tree.total
            assert total is not None
            ww = total.white_wins
            d = total.draws
            bw = total.black_wins
            n = ww + d + bw
            if n > 0:
                expected_score = (ww + 0.5 * d) / n
                assert total.score == pytest.approx(expected_score)

    def test_empty_tree(self, db_666_path: Path) -> None:
        """Position not in any game should return empty tree."""
        with ScidDatabase.open(db_666_path) as db:
            # A very unusual position
            board = chess.Board("8/8/8/8/4K3/8/8/7k w - - 0 1")
            tree = db.search_tree(board)
            assert tree.games_matched == 0
            assert len(tree.nodes) == 0
            assert tree.total is None

    def test_game_indices_filter(self, db_scandi_path: Path) -> None:
        """game_indices parameter should limit the search."""
        with ScidDatabase.open(db_scandi_path) as db:
            tree_full = db.search_tree()
            tree_partial = db.search_tree(game_indices=[0])
            assert tree_full.games_matched >= tree_partial.games_matched
            assert tree_partial.games_searched == 1

    def test_default_target_is_start_pos(self, db_scandi_path: Path) -> None:
        """Calling search_tree() with no args uses starting position."""
        with ScidDatabase.open(db_scandi_path) as db:
            tree_default = db.search_tree()
            tree_explicit = db.search_tree(chess.Board())
            assert tree_default.games_matched == tree_explicit.games_matched
            assert len(tree_default.nodes) == len(tree_explicit.nodes)
            for a, b in zip(tree_default.nodes, tree_explicit.nodes):
                assert a.san == b.san
                assert a.count == b.count

    def test_nodes_sorted_by_frequency(self, db_666_path: Path) -> None:
        """Tree nodes should be sorted by count descending."""
        with ScidDatabase.open(db_666_path) as db:
            board = chess.Board()
            board.push_san("e4")
            tree = db.search_tree(board)
            counts = [n.count for n in tree.nodes]
            assert counts == sorted(counts, reverse=True)

    def test_draw_percentage(self, db_scandi_path: Path) -> None:
        """Draw percentage should be 100 * draws / count."""
        with ScidDatabase.open(db_scandi_path) as db:
            tree = db.search_tree()
            for n in tree.nodes:
                if n.count > 0:
                    expected = 100.0 * n.draws / n.count
                    assert n.draw_pct == pytest.approx(expected)


# =====================================================================
# Tree search tests — created database
# =====================================================================


class TestTreeSearchCreated:
    """Tree search on a created database with known games."""

    def _make_db(self, tmp_db_path: Path) -> ScidDatabase:
        """Create a small db with 3 games for precise statistics verification."""
        db = ScidDatabase.create(tmp_db_path, version=5)

        # Game 1: 1.e4 e5 2.Nf3 Nc6  White wins
        g1 = chess.pgn.Game()
        g1.headers["White"] = "Player A"
        g1.headers["Black"] = "Player B"
        g1.headers["WhiteElo"] = "2200"
        g1.headers["BlackElo"] = "2000"
        g1.headers["Result"] = "1-0"
        g1.headers["Date"] = "2020.06.15"
        node = g1.add_variation(chess.Move.from_uci("e2e4"))
        node = node.add_variation(chess.Move.from_uci("e7e5"))
        node = node.add_variation(chess.Move.from_uci("g1f3"))
        node = node.add_variation(chess.Move.from_uci("b8c6"))
        db.add_game(g1)

        # Game 2: 1.e4 e5 2.Nf3 Nf6  Draw
        g2 = chess.pgn.Game()
        g2.headers["White"] = "Player C"
        g2.headers["Black"] = "Player D"
        g2.headers["WhiteElo"] = "2100"
        g2.headers["BlackElo"] = "2100"
        g2.headers["Result"] = "1/2-1/2"
        g2.headers["Date"] = "2021.03.10"
        node = g2.add_variation(chess.Move.from_uci("e2e4"))
        node = node.add_variation(chess.Move.from_uci("e7e5"))
        node = node.add_variation(chess.Move.from_uci("g1f3"))
        node = node.add_variation(chess.Move.from_uci("g8f6"))
        db.add_game(g2)

        # Game 3: 1.d4 d5  Black wins
        g3 = chess.pgn.Game()
        g3.headers["White"] = "Player E"
        g3.headers["Black"] = "Player F"
        g3.headers["WhiteElo"] = "1800"
        g3.headers["BlackElo"] = "2300"
        g3.headers["Result"] = "0-1"
        g3.headers["Date"] = "2022.01.01"
        node = g3.add_variation(chess.Move.from_uci("d2d4"))
        node = node.add_variation(chess.Move.from_uci("d7d5"))
        db.add_game(g3)

        db.save()
        return ScidDatabase.open(tmp_db_path)

    def test_starting_tree(self, tmp_db_path: Path) -> None:
        """All 3 games should be in starting position tree."""
        db = self._make_db(tmp_db_path)
        tree = db.search_tree()
        assert tree.games_matched == 3
        assert tree.games_searched == 3

        # Should have 2 moves: e4 (2 games) and d4 (1 game)
        assert len(tree.nodes) == 2
        e4_node = next(n for n in tree.nodes if n.san == "e4")
        d4_node = next(n for n in tree.nodes if n.san == "d4")
        assert e4_node.count == 2
        assert d4_node.count == 1

        # e4 sorted first (more frequent)
        assert tree.nodes[0].san == "e4"
        db.close()

    def test_result_breakdown(self, tmp_db_path: Path) -> None:
        """Verify W/D/B counts per move."""
        db = self._make_db(tmp_db_path)
        tree = db.search_tree()

        e4 = next(n for n in tree.nodes if n.san == "e4")
        assert e4.white_wins == 1
        assert e4.draws == 1
        assert e4.black_wins == 0

        d4 = next(n for n in tree.nodes if n.san == "d4")
        assert d4.white_wins == 0
        assert d4.draws == 0
        assert d4.black_wins == 1
        db.close()

    def test_scores(self, tmp_db_path: Path) -> None:
        """Verify score computation from White's perspective."""
        db = self._make_db(tmp_db_path)
        tree = db.search_tree()

        e4 = next(n for n in tree.nodes if n.san == "e4")
        # 1 win + 0.5 draw = 1.5 / 2 = 0.75
        assert e4.score == pytest.approx(0.75)

        d4 = next(n for n in tree.nodes if n.san == "d4")
        # 0 wins, 0 draws, 1 loss = 0.0
        assert d4.score == pytest.approx(0.0)
        db.close()

    def test_avg_elo_white_to_move(self, tmp_db_path: Path) -> None:
        """Avg Elo should be White's Elo when White is making the move."""
        db = self._make_db(tmp_db_path)
        tree = db.search_tree()  # White to move

        e4 = next(n for n in tree.nodes if n.san == "e4")
        # Game 1: WhiteElo=2200, Game 2: WhiteElo=2100 => avg=2150
        assert e4.avg_elo == pytest.approx(2150.0)

        d4 = next(n for n in tree.nodes if n.san == "d4")
        # Game 3: WhiteElo=1800
        assert d4.avg_elo == pytest.approx(1800.0)
        db.close()

    def test_avg_elo_black_to_move(self, tmp_db_path: Path) -> None:
        """Avg Elo should be Black's Elo when Black is making the move."""
        db = self._make_db(tmp_db_path)
        board = chess.Board()
        board.push_san("e4")
        tree = db.search_tree(board)  # Black to move

        e5 = next(n for n in tree.nodes if n.san == "e5")
        # Game 1: BlackElo=2000, Game 2: BlackElo=2100 => avg=2050
        assert e5.avg_elo == pytest.approx(2050.0)
        db.close()

    def test_avg_year(self, tmp_db_path: Path) -> None:
        """Avg year computed from game dates."""
        db = self._make_db(tmp_db_path)
        tree = db.search_tree()

        e4 = next(n for n in tree.nodes if n.san == "e4")
        # Game 1: 2020, Game 2: 2021 => avg 2020.5
        assert e4.avg_year == pytest.approx(2020.5)

        d4 = next(n for n in tree.nodes if n.san == "d4")
        assert d4.avg_year == pytest.approx(2022.0)
        db.close()

    def test_totals(self, tmp_db_path: Path) -> None:
        """Total row should aggregate across all moves."""
        db = self._make_db(tmp_db_path)
        tree = db.search_tree()

        total = tree.total
        assert total is not None
        assert total.count == 3
        assert total.white_wins == 1
        assert total.draws == 1
        assert total.black_wins == 1
        # Score: (1 + 0.5) / 3 = 0.5
        assert total.score == pytest.approx(0.5)
        assert total.san == "Total"
        db.close()

    def test_deeper_position(self, tmp_db_path: Path) -> None:
        """Tree after 1.e4 e5 should show Nf3 from both games."""
        db = self._make_db(tmp_db_path)
        board = chess.Board()
        board.push_san("e4")
        board.push_san("e5")
        tree = db.search_tree(board)
        assert tree.games_matched == 2
        assert len(tree.nodes) == 1
        assert tree.nodes[0].san == "Nf3"
        assert tree.nodes[0].count == 2
        db.close()

    def test_branching_position(self, tmp_db_path: Path) -> None:
        """Tree after 1.e4 e5 2.Nf3 should show Nc6 and Nf6."""
        db = self._make_db(tmp_db_path)
        board = chess.Board()
        for m in ["e4", "e5", "Nf3"]:
            board.push_san(m)
        tree = db.search_tree(board)
        assert tree.games_matched == 2
        assert len(tree.nodes) == 2
        sans = {n.san for n in tree.nodes}
        assert sans == {"Nc6", "Nf6"}
        db.close()

    def test_no_moves_beyond_game(self, tmp_db_path: Path) -> None:
        """Position at end of a game should not count (no next move)."""
        db = self._make_db(tmp_db_path)
        # Game 3 ends after 1.d4 d5 — search from that position
        board = chess.Board()
        board.push_san("d4")
        board.push_san("d5")
        tree = db.search_tree(board)
        # Game 3 matches but has no next move
        # Games 1&2 don't reach this position
        assert tree.games_matched == 0
        assert len(tree.nodes) == 0
        db.close()


# =====================================================================
# Tree search integration tests — freechess database
# =====================================================================


class TestTreeSearchFreechess:
    """Tree search tests on the freechess database (~193K games)."""

    def test_starting_position_tree(self, db_freechess_path: Path) -> None:
        """Starting position should match almost all games."""
        with ScidDatabase.open(db_freechess_path) as db:
            tree = db.search_tree()
            # Almost all games start from standard position
            assert tree.games_matched > 190000
            # Should have many different first moves
            assert len(tree.nodes) >= 15
            # e4 and d4 should dominate
            top2 = {tree.nodes[0].san, tree.nodes[1].san}
            assert top2 == {"e4", "d4"}
            # Total count should match games_matched
            assert tree.total is not None
            assert tree.total.count == tree.games_matched

    def test_tree_after_e4_e5(self, db_freechess_path: Path) -> None:
        """Tree after 1.e4 e5 should have standard responses."""
        with ScidDatabase.open(db_freechess_path) as db:
            board = chess.Board()
            board.push_san("e4")
            board.push_san("e5")
            tree = db.search_tree(board)
            assert tree.games_matched > 5000
            # Nf3 should be the most popular
            assert tree.nodes[0].san == "Nf3"
            # Score should be reasonable (40-60%)
            assert tree.total is not None
            assert 0.3 <= tree.total.score <= 0.7

    def test_score_range(self, db_freechess_path: Path) -> None:
        """All scores should be between 0 and 1."""
        with ScidDatabase.open(db_freechess_path) as db:
            tree = db.search_tree()
            for n in tree.nodes:
                assert 0.0 <= n.score <= 1.0

    def test_elo_reasonable(self, db_freechess_path: Path) -> None:
        """Average Elo should be in a reasonable range for FICS games."""
        with ScidDatabase.open(db_freechess_path) as db:
            tree = db.search_tree()
            for n in tree.nodes:
                if n.avg_elo > 0:
                    assert 1000 <= n.avg_elo <= 3000

    def test_performance_reasonable(self, db_freechess_path: Path) -> None:
        """Performance rating should be reasonable when present."""
        with ScidDatabase.open(db_freechess_path) as db:
            tree = db.search_tree()
            for n in tree.nodes:
                if n.performance > 0:
                    assert 500 <= n.performance <= 3500

    def test_year_reasonable(self, db_freechess_path: Path) -> None:
        """Average year should be reasonable."""
        with ScidDatabase.open(db_freechess_path) as db:
            tree = db.search_tree()
            for n in tree.nodes:
                if n.avg_year > 0:
                    assert 1990 <= n.avg_year <= 2030


# =====================================================================
# Pre-filtered searches must match a full replay
# =====================================================================


def _brute_force_material(
    db: ScidDatabase, criteria: MaterialSearchCriteria, indices: list[int]
) -> set[int]:
    """Return game indices matching *criteria* by replaying every game."""
    hits: set[int] = set()
    for idx in indices:
        game = db[idx]
        board = game.board()
        if _material_matches_criteria(board, criteria):
            hits.add(idx)
            continue
        for move in game.mainline_moves():
            board.push(move)
            if _material_matches_criteria(board, criteria):
                hits.add(idx)
                break
    return hits


_MATERIAL_CRITERIA = [
    MaterialSearchCriteria(wq=(0, 0), bq=(0, 0)),
    MaterialSearchCriteria(wq=(0, 0)),
    MaterialSearchCriteria(wr=(1, 1), br=(1, 1), wp=(0, 3), bp=(0, 3)),
    MaterialSearchCriteria(wp=(8, 8), bp=(8, 8)),
    MaterialSearchCriteria(wq=(2, 3)),
    MaterialSearchCriteria(wn=(3, 3)),
]


def _promotion_db(path: Path, version: int) -> ScidDatabase:
    """Create a small database with queen and under-promotions."""
    db = ScidDatabase.create(path, version=version)
    pgns = [
        # Queens traded, then promotion to a second white queen
        "1. e4 d5 2. exd5 Qxd5 3. Nc3 Qxd2+ 4. Qxd2 *",
        '[SetUp "1"]\n[FEN "8/P6k/8/8/8/8/8/K7 w - - 0 1"]\n\n1. a8=Q *',
        '[SetUp "1"]\n[FEN "8/P6k/8/8/8/8/8/K6Q w - - 0 1"]\n\n1. a8=Q *',
        '[SetUp "1"]\n[FEN "8/P6k/8/8/8/8/8/KNN5 w - - 0 1"]\n\n1. a8=N *',
        "1. d4 d5 2. c4 e6 *",
    ]
    for pgn in pgns:
        db.add_game(chess.pgn.read_game(io.StringIO(pgn)))
    return db


class TestMaterialPrefilterMatchesBruteForce:
    @pytest.mark.parametrize("criteria", _MATERIAL_CRITERIA)
    @pytest.mark.parametrize("version", [4, 5])
    def test_created_db(
        self, tmp_db_path: Path, version: int, criteria: MaterialSearchCriteria
    ) -> None:
        db = _promotion_db(tmp_db_path, version)
        indices = list(range(len(db)))
        fast = {r.game_index for r in db.search_material(criteria)}
        assert fast == _brute_force_material(db, criteria, indices)

    @pytest.mark.parametrize("criteria", _MATERIAL_CRITERIA)
    def test_freechess(
        self, db_freechess_path: Path, criteria: MaterialSearchCriteria
    ) -> None:
        with ScidDatabase.open(db_freechess_path) as db:
            indices = list(range(300))
            fast = {
                r.game_index
                for r in db.search_material(criteria, game_indices=indices)
            }
            assert fast == _brute_force_material(db, criteria, indices)

    def test_queenless_games_found(self, db_freechess_path: Path) -> None:
        with ScidDatabase.open(db_freechess_path) as db:
            c = MaterialSearchCriteria(wq=(0, 0), bq=(0, 0))
            assert db.search_material(c, game_indices=list(range(300)))


class TestPositionSearchAddedGames:
    """Games added via add_game() must be found by position and tree search."""

    MOVES = "e4 e5 Nc3 Nf6 f4 d6 Nf3 Nc6 Bc4 Be7 O-O O-O".split()

    def _db(self, path: Path, version: int) -> ScidDatabase:
        db = ScidDatabase.create(path, version=version)
        game = chess.pgn.Game()
        node = game
        board = chess.Board()
        for san in self.MOVES:
            node = node.add_variation(board.parse_san(san))
            board.push(node.move)
        db.add_game(game)
        return db

    @pytest.mark.parametrize("version", [4, 5])
    @pytest.mark.parametrize("plies", [0, 1, 4, 5, 8, 12])
    def test_search_position(self, tmp_db_path: Path, version: int, plies: int):
        db = self._db(tmp_db_path, version)
        target = chess.Board()
        for san in self.MOVES[:plies]:
            target.push_san(san)
        assert db.search_position(target) == [
            PositionSearchResult(game_index=0, ply=plies)
        ]
        db.save()
        with ScidDatabase.open(tmp_db_path) as reopened:
            assert reopened.search_position(target) == [
                PositionSearchResult(game_index=0, ply=plies)
            ]

    @pytest.mark.parametrize("version", [4, 5])
    def test_search_tree(self, tmp_db_path: Path, version: int):
        db = self._db(tmp_db_path, version)
        target = chess.Board()
        for san in self.MOVES[:5]:
            target.push_san(san)
        tree = db.search_tree(target)
        assert [n.san for n in tree.nodes] == [self.MOVES[5]]


class TestPositionIgnoresMoveCounters:
    """Positions match regardless of halfmove clock and fullmove number."""

    GAME = "e4 e5 Nc3 Nf6 f4".split()

    def _db(self, path: Path) -> ScidDatabase:
        db = ScidDatabase.create(path, version=5)
        game = chess.pgn.Game()
        node = game
        for san in self.GAME:
            node = node.add_variation(node.board().parse_san(san))
        db.add_game(game)
        return db

    def test_transposition_with_different_halfmove_clock(self, tmp_db_path: Path):
        db = self._db(tmp_db_path)
        target = chess.Board()
        for san in "e4 Nf6 Nc3 e5".split():
            target.push_san(san)
        assert db.search_position(target) == [PositionSearchResult(0, 4)]

    def test_fen_target_with_default_counters(self, tmp_db_path: Path):
        db = self._db(tmp_db_path)
        reached = chess.Board()
        for san in self.GAME:
            reached.push_san(san)
        target = chess.Board(reached.fen().rsplit(" ", 2)[0] + " 0 1")
        assert db.search_position(target) == [PositionSearchResult(0, 5)]
        tree = db.search_tree(target)
        assert tree.games_matched == 0  # game ends here, no next move

        before = chess.Board()
        for san in self.GAME[:4]:
            before.push_san(san)
        target = chess.Board(before.fen().rsplit(" ", 2)[0] + " 0 1")
        assert [n.san for n in db.search_tree(target).nodes] == ["f4"]

    def test_stored_line_ignores_counters(self) -> None:
        board = chess.Board()
        board.push_san("d4")
        target = chess.Board(board.fen().rsplit(" ", 2)[0] + " 7 42")
        assert _stored_line_match(target, 15) == 1

    def test_castling_rights_still_distinguish(self, tmp_db_path: Path):
        db = self._db(tmp_db_path)
        reached = chess.Board()
        for san in self.GAME:
            reached.push_san(san)
        target = reached.copy()
        target.castling_rights = chess.BB_EMPTY
        assert db.search_position(target) == []
