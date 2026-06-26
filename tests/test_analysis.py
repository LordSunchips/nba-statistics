"""Tests for the three high-level analysis functions."""
from __future__ import annotations

import pandas as pd
import pytest

from fantasy_basketball.analysis import (
    generate_vor_rankings,
    random_snake_draft,
    simulate_fantasy_season,
)
from fantasy_basketball.scoring import ScoringRules
from fantasy_basketball.vor import LeagueSettings, VORCalculator, compute_base_value


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_log(pts: float, n_games: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        [{"PTS": pts, "REB": 0, "AST": 0, "STL": 0, "BLK": 0, "TOV": 0,
          "FGM": 0, "FGA": 0, "FTM": 0, "FTA": 0, "3PM": 0}]
        * n_games
    )


def _simple_rules() -> ScoringRules:
    return ScoringRules(
        pts=1.0, reb=0.0, ast=0.0, stl=0.0, blk=0.0, tov=0.0,
        field_goals_made=0.0, field_goals_missed=0.0,
        free_throws_made=0.0, free_throws_missed=0.0,
        three_pointers_made=0.0,
    )


@pytest.fixture
def league():
    return LeagueSettings(
        num_teams=4,
        roster_spots={"G": 2, "F": 1, "C": 1},
        scoring_rules=_simple_rules(),
        season_games=82,
        risk_aversion=0.0,
    )


# 12 guards (4 teams × 2 spots + 4 extras), 8 forwards, 6 centers
@pytest.fixture
def game_logs():
    return {
        **{f"G{i}": _make_log(pts=30 - i) for i in range(1, 13)},
        **{f"F{i}": _make_log(pts=25 - i) for i in range(1, 9)},
        **{f"C{i}": _make_log(pts=20 - i) for i in range(1, 7)},
    }


@pytest.fixture
def player_positions():
    return {
        **{f"G{i}": ["PG"] for i in range(1, 13)},
        **{f"F{i}": ["SF"] for i in range(1, 9)},
        **{f"C{i}": ["C"] for i in range(1, 7)},
    }


# ---------------------------------------------------------------------------
# generate_vor_rankings
# ---------------------------------------------------------------------------

class TestGenerateVorRankings:
    def test_returns_dataframe(self, game_logs, player_positions, league):
        df = generate_vor_rankings(game_logs, player_positions, league)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self, game_logs, player_positions, league):
        df = generate_vor_rankings(game_logs, player_positions, league)
        expected = {"overall_rank", "position_rank", "player", "primary_position",
                    "positions", "base_value", "vor"}
        assert expected.issubset(set(df.columns))

    def test_row_count_equals_player_count(self, game_logs, player_positions, league):
        df = generate_vor_rankings(game_logs, player_positions, league)
        assert len(df) == len(game_logs)

    def test_overall_rank_1_has_highest_vor(self, game_logs, player_positions, league):
        df = generate_vor_rankings(game_logs, player_positions, league)
        assert df.iloc[0]["overall_rank"] == 1
        assert df.iloc[0]["vor"] == df["vor"].max()

    def test_position_rank_1_is_best_in_group(self, game_logs, player_positions, league):
        df = generate_vor_rankings(game_logs, player_positions, league)
        guards = df[df["primary_position"] == "G"]
        top_guard = guards[guards["position_rank"] == 1]
        assert len(top_guard) == 1
        assert top_guard.iloc[0]["vor"] == guards["vor"].max()

    def test_overall_rank_is_sequential(self, game_logs, player_positions, league):
        df = generate_vor_rankings(game_logs, player_positions, league)
        assert list(df["overall_rank"]) == list(range(1, len(df) + 1))

    def test_positions_column_contains_slash_for_multi_position(self, league):
        logs = {"GF": _make_log(pts=20.0)}
        pos = {"GF": ["SG", "PF"]}
        df = generate_vor_rankings(logs, pos, league)
        assert "/" in df.iloc[0]["positions"]

    def test_unknown_positions_marked(self, league):
        logs = {"X": _make_log(pts=10.0)}
        pos = {"X": []}
        df = generate_vor_rankings(logs, pos, league)
        assert df.iloc[0]["primary_position"] == "?"


# ---------------------------------------------------------------------------
# random_snake_draft
# ---------------------------------------------------------------------------

class TestRandomSnakeDraft:
    def test_returns_dataframe(self, game_logs, player_positions, league):
        calc = VORCalculator(league)
        bv = {n: compute_base_value(l, league.scoring_rules) for n, l in game_logs.items()}
        repl = calc.compute_replacement_values(bv, player_positions)
        vor = calc.compute_vor(bv, repl, player_positions)
        df = random_snake_draft(vor, player_positions, league, num_rounds=4)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self, game_logs, player_positions, league):
        calc = VORCalculator(league)
        bv = {n: compute_base_value(l, league.scoring_rules) for n, l in game_logs.items()}
        repl = calc.compute_replacement_values(bv, player_positions)
        vor = calc.compute_vor(bv, repl, player_positions)
        df = random_snake_draft(vor, player_positions, league, num_rounds=4)
        assert {"pick", "round", "team", "player", "primary_position", "vor"}.issubset(df.columns)

    def test_total_picks_equals_teams_times_rounds(self, game_logs, player_positions, league):
        calc = VORCalculator(league)
        bv = {n: compute_base_value(l, league.scoring_rules) for n, l in game_logs.items()}
        repl = calc.compute_replacement_values(bv, player_positions)
        vor = calc.compute_vor(bv, repl, player_positions)
        df = random_snake_draft(vor, player_positions, league, num_rounds=4)
        assert len(df) == league.num_teams * 4

    def test_no_duplicate_players(self, game_logs, player_positions, league):
        calc = VORCalculator(league)
        bv = {n: compute_base_value(l, league.scoring_rules) for n, l in game_logs.items()}
        repl = calc.compute_replacement_values(bv, player_positions)
        vor = calc.compute_vor(bv, repl, player_positions)
        df = random_snake_draft(vor, player_positions, league, num_rounds=4)
        assert df["player"].nunique() == len(df)

    def test_pick_numbers_are_sequential(self, game_logs, player_positions, league):
        calc = VORCalculator(league)
        bv = {n: compute_base_value(l, league.scoring_rules) for n, l in game_logs.items()}
        repl = calc.compute_replacement_values(bv, player_positions)
        vor = calc.compute_vor(bv, repl, player_positions)
        df = random_snake_draft(vor, player_positions, league, num_rounds=4)
        assert list(df["pick"]) == list(range(1, len(df) + 1))

    def test_pick_1_is_highest_vor_player(self, game_logs, player_positions, league):
        calc = VORCalculator(league)
        bv = {n: compute_base_value(l, league.scoring_rules) for n, l in game_logs.items()}
        repl = calc.compute_replacement_values(bv, player_positions)
        vor = calc.compute_vor(bv, repl, player_positions)
        df = random_snake_draft(vor, player_positions, league, num_rounds=4)
        best = max(vor, key=lambda p: (vor[p], p))
        assert df.iloc[0]["player"] == best

    def test_players_drafted_in_vor_order(self, game_logs, player_positions, league):
        calc = VORCalculator(league)
        bv = {n: compute_base_value(l, league.scoring_rules) for n, l in game_logs.items()}
        repl = calc.compute_replacement_values(bv, player_positions)
        vor = calc.compute_vor(bv, repl, player_positions)
        df = random_snake_draft(vor, player_positions, league, num_rounds=4)
        # VOR of each pick should be non-increasing overall (best available always taken)
        assert list(df["vor"]) == sorted(df["vor"], reverse=True)

    def test_raises_on_insufficient_players(self, game_logs, player_positions, league):
        calc = VORCalculator(league)
        bv = {n: compute_base_value(l, league.scoring_rules) for n, l in game_logs.items()}
        repl = calc.compute_replacement_values(bv, player_positions)
        vor = calc.compute_vor(bv, repl, player_positions)
        with pytest.raises(ValueError, match="Not enough players"):
            random_snake_draft(vor, player_positions, league, num_rounds=100)

    def test_snake_order_round1_picks_reverse(self, game_logs, player_positions, league):
        calc = VORCalculator(league)
        bv = {n: compute_base_value(l, league.scoring_rules) for n, l in game_logs.items()}
        repl = calc.compute_replacement_values(bv, player_positions)
        vor = calc.compute_vor(bv, repl, player_positions)
        df = random_snake_draft(vor, player_positions, league, num_rounds=2)
        round1 = df[df["round"] == 1]["team"].tolist()
        round2 = df[df["round"] == 2]["team"].tolist()
        assert round1 == list(range(league.num_teams))
        assert round2 == list(range(league.num_teams - 1, -1, -1))


# ---------------------------------------------------------------------------
# simulate_fantasy_season
# ---------------------------------------------------------------------------

class TestSimulateFantasySeason:
    @pytest.fixture
    def draft_results(self, game_logs, player_positions, league):
        calc = VORCalculator(league)
        bv = {n: compute_base_value(l, league.scoring_rules) for n, l in game_logs.items()}
        repl = calc.compute_replacement_values(bv, player_positions)
        vor = calc.compute_vor(bv, repl, player_positions)
        return calc.simulate_snake_draft(vor, player_positions, num_rounds=4)

    def test_returns_dataframe(self, draft_results, game_logs, league):
        df = simulate_fantasy_season(draft_results, game_logs, league, n_weeks=4, seed=0)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self, draft_results, game_logs, league):
        df = simulate_fantasy_season(draft_results, game_logs, league, n_weeks=4, seed=0)
        expected = {"season_rank", "team", "total_pts", "avg_weekly_pts",
                    "best_week", "worst_week", "wins"}
        assert expected.issubset(set(df.columns))

    def test_row_count_equals_num_teams(self, draft_results, game_logs, league):
        df = simulate_fantasy_season(draft_results, game_logs, league, n_weeks=4, seed=0)
        assert len(df) == league.num_teams

    def test_season_rank_1_has_highest_total_pts(self, draft_results, game_logs, league):
        df = simulate_fantasy_season(draft_results, game_logs, league, n_weeks=4, seed=0)
        assert df.iloc[0]["season_rank"] == 1
        assert df.iloc[0]["total_pts"] == df["total_pts"].max()

    def test_total_pts_positive(self, draft_results, game_logs, league):
        df = simulate_fantasy_season(draft_results, game_logs, league, n_weeks=4, seed=0)
        assert (df["total_pts"] > 0).all()

    def test_best_week_gte_worst_week(self, draft_results, game_logs, league):
        df = simulate_fantasy_season(draft_results, game_logs, league, n_weeks=4, seed=0)
        assert (df["best_week"] >= df["worst_week"]).all()

    def test_wins_are_non_negative(self, draft_results, game_logs, league):
        df = simulate_fantasy_season(draft_results, game_logs, league, n_weeks=4, seed=0)
        assert (df["wins"] >= 0).all()

    def test_seeded_season_is_reproducible(self, draft_results, game_logs, league):
        df1 = simulate_fantasy_season(draft_results, game_logs, league, n_weeks=4, seed=7)
        df2 = simulate_fantasy_season(draft_results, game_logs, league, n_weeks=4, seed=7)
        assert list(df1["total_pts"]) == list(df2["total_pts"])

    def test_missing_player_log_handled_gracefully(self, draft_results, league):
        empty_logs: dict = {}
        df = simulate_fantasy_season(draft_results, empty_logs, league, n_weeks=4, seed=0)
        assert (df["total_pts"] == 0.0).all()
