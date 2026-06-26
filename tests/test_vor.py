"""Tests for the VOR (Value Over Replacement) framework."""
from __future__ import annotations

import pandas as pd
import pytest

from fantasy_basketball.scoring import ScoringRules
from fantasy_basketball.vor import (
    LeagueSettings,
    VORCalculator,
    compute_base_value,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_log(pts: float, n_games: int = 10) -> pd.DataFrame:
    """Minimal game log with only PTS filled, everything else 0."""
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


def _league_8_team() -> LeagueSettings:
    return LeagueSettings(
        num_teams=8,
        roster_spots={"G": 2, "F": 2, "C": 1},
        scoring_rules=_simple_rules(),
        season_games=82,
        risk_aversion=0.0,  # disable risk penalty so base_value == avg_score * availability
    )


# ---------------------------------------------------------------------------
# compute_base_value
# ---------------------------------------------------------------------------

class TestComputeBaseValue:
    def test_empty_log_returns_zero(self):
        log = pd.DataFrame(columns=["PTS"])
        assert compute_base_value(log, _simple_rules()) == 0.0

    def test_full_availability(self):
        log = _make_log(pts=20.0, n_games=82)
        val = compute_base_value(log, _simple_rules(), season_games=82, risk_aversion=0.0)
        assert val == pytest.approx(20.0, abs=1e-3)

    def test_partial_availability(self):
        # 41 games played out of 82 → availability = 0.5
        log = _make_log(pts=30.0, n_games=41)
        val = compute_base_value(log, _simple_rules(), season_games=82, risk_aversion=0.0)
        assert val == pytest.approx(15.0, abs=1e-3)

    def test_risk_aversion_penalizes_std(self):
        # Alternating 10 and 30 → avg=20, std>0
        scores = [10.0, 30.0] * 20  # 40 games
        log = pd.DataFrame(
            [{"PTS": s, "REB": 0, "AST": 0, "STL": 0, "BLK": 0, "TOV": 0,
              "FGM": 0, "FGA": 0, "FTM": 0, "FTA": 0, "3PM": 0}
             for s in scores]
        )
        val_no_risk = compute_base_value(log, _simple_rules(), season_games=82, risk_aversion=0.0)
        val_with_risk = compute_base_value(log, _simple_rules(), season_games=82, risk_aversion=1.0)
        assert val_with_risk < val_no_risk

    def test_single_game_no_std_penalty(self):
        log = _make_log(pts=25.0, n_games=1)
        val = compute_base_value(log, _simple_rules(), season_games=82, risk_aversion=1.0)
        # std for single game = 0, so no penalty beyond availability factor
        assert val == pytest.approx(25.0 / 82, abs=1e-3)


# ---------------------------------------------------------------------------
# VORCalculator._normalize_positions
# ---------------------------------------------------------------------------

class TestNormalizePositions:
    def test_pg_sg_map_to_g(self):
        result = VORCalculator._normalize_positions(["PG", "SG"])
        assert result == ["G"]

    def test_sf_pf_map_to_f(self):
        result = VORCalculator._normalize_positions(["SF", "PF"])
        assert result == ["F"]

    def test_c_maps_to_c(self):
        result = VORCalculator._normalize_positions(["C"])
        assert result == ["C"]

    def test_multi_position_guard_forward(self):
        result = VORCalculator._normalize_positions(["PG", "SF"])
        assert set(result) == {"G", "F"}
        assert len(result) == 2

    def test_unknown_position_dropped(self):
        result = VORCalculator._normalize_positions(["PG", "UTIL"])
        assert result == ["G"]

    def test_empty_positions(self):
        result = VORCalculator._normalize_positions([])
        assert result == []


# ---------------------------------------------------------------------------
# Fixtures for VORCalculator tests
# ---------------------------------------------------------------------------

@pytest.fixture
def calc():
    return VORCalculator(_league_8_team())


@pytest.fixture
def base_values():
    """8-team league, 2G+2F+1C per team → thresholds: G rank 16, F rank 16, C rank 8."""
    return {
        # Guards (16 slots → G16 is replacement at index 16)
        "G1": 40.0, "G2": 38.0, "G3": 36.0, "G4": 34.0,
        "G5": 32.0, "G6": 30.0, "G7": 28.0, "G8": 26.0,
        "G9": 24.0, "G10": 22.0, "G11": 20.0, "G12": 18.0,
        "G13": 16.0, "G14": 14.0, "G15": 12.0, "G16": 10.0,  # last drafted
        "G17": 8.0,  # replacement player
        # Forwards
        "F1": 35.0, "F2": 33.0, "F3": 31.0, "F4": 29.0,
        "F5": 27.0, "F6": 25.0, "F7": 23.0, "F8": 21.0,
        "F9": 19.0, "F10": 17.0, "F11": 15.0, "F12": 13.0,
        "F13": 11.0, "F14": 9.0, "F15": 7.0, "F16": 5.0,   # last drafted
        "F17": 3.0,  # replacement player
        # Centers
        "C1": 45.0, "C2": 42.0, "C3": 39.0, "C4": 36.0,
        "C5": 33.0, "C6": 30.0, "C7": 27.0, "C8": 24.0,    # last drafted
        "C9": 21.0,  # replacement player
        # Cross-listed G+F player
        "GF1": 28.5,
    }


@pytest.fixture
def player_positions():
    return {
        **{f"G{i}": ["PG"] for i in range(1, 18)},
        **{f"F{i}": ["SF"] for i in range(1, 18)},
        **{f"C{i}": ["C"] for i in range(1, 10)},
        "GF1": ["SG", "PF"],  # cross-listed
    }


# ---------------------------------------------------------------------------
# compute_replacement_values
# ---------------------------------------------------------------------------

class TestComputeReplacementValues:
    def test_guard_replacement_correct_rank(self, calc, base_values, player_positions):
        repl = calc.compute_replacement_values(base_values, player_positions)
        # G pool has 17 guards + GF1 (28.5) = 18 players.
        # GF1 inserts between G6(30) and G7(28), shifting G16(10) to index 16.
        # 8 teams * 2 spots = threshold index 16 → G16 value = 10.0
        assert repl["G"] == pytest.approx(10.0)

    def test_forward_replacement_correct_rank(self, calc, base_values, player_positions):
        repl = calc.compute_replacement_values(base_values, player_positions)
        # F pool has 17 forwards + GF1 (28.5) = 18 players.
        # GF1 inserts between F4(29) and F5(27), shifting F16(5) to index 16.
        # 8 teams * 2 spots = threshold index 16 → F16 value = 5.0
        assert repl["F"] == pytest.approx(5.0)

    def test_center_replacement_correct_rank(self, calc, base_values, player_positions):
        repl = calc.compute_replacement_values(base_values, player_positions)
        # 8 teams * 1 spot = index 8 → C9 has value 21.0
        assert repl["C"] == pytest.approx(21.0)

    def test_cross_listed_player_in_both_pools(self, calc, base_values, player_positions):
        # Build pools directly to verify GF1 appears in both G and F pools
        pools = calc._build_position_pools(base_values, player_positions)
        g_names = [name for name, _ in pools["G"]]
        f_names = [name for name, _ in pools["F"]]
        assert "GF1" in g_names
        assert "GF1" in f_names

    def test_empty_pool_returns_zero(self):
        settings = LeagueSettings(
            num_teams=2,
            roster_spots={"G": 1, "C": 1},
            scoring_rules=_simple_rules(),
        )
        calc = VORCalculator(settings)
        # No centers provided
        repl = calc.compute_replacement_values({"P1": 10.0}, {"P1": ["PG"]})
        assert repl["C"] == 0.0

    def test_pool_smaller_than_threshold_uses_last_player(self):
        settings = LeagueSettings(
            num_teams=8,
            roster_spots={"G": 2},
            scoring_rules=_simple_rules(),
        )
        calc = VORCalculator(settings)
        # Only 3 guards, threshold would be index 16 — falls back to last
        bv = {"P1": 30.0, "P2": 20.0, "P3": 10.0}
        pos = {"P1": ["PG"], "P2": ["SG"], "P3": ["PG"]}
        repl = calc.compute_replacement_values(bv, pos)
        assert repl["G"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# compute_vor
# ---------------------------------------------------------------------------

class TestComputeVOR:
    def test_vor_equals_base_minus_replacement(self, calc, base_values, player_positions):
        repl = calc.compute_replacement_values(base_values, player_positions)
        vor = calc.compute_vor(base_values, repl, player_positions)
        assert vor["G1"] == pytest.approx(40.0 - repl["G"], abs=1e-3)
        assert vor["C1"] == pytest.approx(45.0 - repl["C"], abs=1e-3)

    def test_multi_position_uses_min_replacement(self, calc, base_values, player_positions):
        repl = calc.compute_replacement_values(base_values, player_positions)
        vor = calc.compute_vor(base_values, repl, player_positions)
        # GF1 is eligible at G (repl=8.0) and F (repl=3.0) → uses min = 3.0
        assert vor["GF1"] == pytest.approx(28.5 - min(repl["G"], repl["F"]), abs=1e-3)

    def test_player_no_eligible_positions_gets_zero(self):
        settings = LeagueSettings(
            num_teams=2, roster_spots={"G": 1}, scoring_rules=_simple_rules()
        )
        calc = VORCalculator(settings)
        bv = {"P1": 30.0, "P2": 20.0, "unknown": 15.0}
        pos = {"P1": ["PG"], "P2": ["SG"], "unknown": []}
        repl = calc.compute_replacement_values(bv, pos)
        vor = calc.compute_vor(bv, repl, pos)
        assert vor["unknown"] == 0.0

    def test_replacement_player_has_vor_near_zero(self, calc, base_values, player_positions):
        repl = calc.compute_replacement_values(base_values, player_positions)
        vor = calc.compute_vor(base_values, repl, player_positions)
        # G17 is the replacement → VOR should be ≤ 0
        assert vor["G17"] <= 0.0


# ---------------------------------------------------------------------------
# simulate_snake_draft
# ---------------------------------------------------------------------------

class TestSimulateSnakeDraft:
    def test_returns_correct_team_count(self, calc, base_values, player_positions):
        repl = calc.compute_replacement_values(base_values, player_positions)
        vor = calc.compute_vor(base_values, repl, player_positions)
        results = calc.simulate_snake_draft(vor, player_positions, num_rounds=5)
        assert len(results) == 8

    def test_each_team_gets_num_rounds_players(self, calc, base_values, player_positions):
        repl = calc.compute_replacement_values(base_values, player_positions)
        vor = calc.compute_vor(base_values, repl, player_positions)
        results = calc.simulate_snake_draft(vor, player_positions, num_rounds=5)
        assert all(len(team) == 5 for team in results)

    def test_no_duplicate_players(self, calc, base_values, player_positions):
        repl = calc.compute_replacement_values(base_values, player_positions)
        vor = calc.compute_vor(base_values, repl, player_positions)
        results = calc.simulate_snake_draft(vor, player_positions, num_rounds=5)
        all_picks = [p for team in results for p in team]
        assert len(all_picks) == len(set(all_picks))

    def test_snake_order_team0_picks_first_in_round0(self, calc, base_values, player_positions):
        repl = calc.compute_replacement_values(base_values, player_positions)
        vor = calc.compute_vor(base_values, repl, player_positions)
        # Round 0: teams 0..7 in order; highest VOR player goes to team 0
        best_player = max(vor, key=lambda p: vor[p])
        results = calc.simulate_snake_draft(vor, player_positions, num_rounds=1)
        assert best_player in results[0]

    def test_snake_order_round1_reverses(self):
        """Verify snake ordering with a small 2-team, 2-round draft (no ties)."""
        settings = LeagueSettings(
            num_teams=2,
            roster_spots={"G": 1, "F": 1},
            scoring_rules=_simple_rules(),
        )
        calc = VORCalculator(settings)
        # Distinct VOR values to avoid tie-breaking ambiguity
        vor = {"G1": 40.0, "G2": 30.0, "F1": 20.0, "F2": 10.0}
        pos = {"G1": ["PG"], "G2": ["SG"], "F1": ["SF"], "F2": ["PF"]}
        results = calc.simulate_snake_draft(vor, pos, num_rounds=2)
        # Round 0 (forward): team 0 → G1(40), team 1 → F1(20) [next best after G1]
        # Round 1 (reverse): team 1 → G2(30), team 0 → F2(10)
        # team 0 has: G1 + F2; team 1 has: F1 + G2
        assert "G1" in results[0]  # team 0 picks #1 in round 0
        assert "G2" in results[1]  # team 1 picks #1 in round 1 (snake reversal)

    def test_raises_on_insufficient_players(self, calc, base_values, player_positions):
        repl = calc.compute_replacement_values(base_values, player_positions)
        vor = calc.compute_vor(base_values, repl, player_positions)
        with pytest.raises(ValueError, match="Not enough players"):
            calc.simulate_snake_draft(vor, player_positions, num_rounds=100)

    def test_dynamic_mode_removes_drafted_players(self, calc, base_values, player_positions):
        """In dynamic mode the same player cannot be drafted twice."""
        repl = calc.compute_replacement_values(base_values, player_positions)
        vor = calc.compute_vor(base_values, repl, player_positions)
        results = calc.simulate_snake_draft(
            vor, player_positions, num_rounds=5, base_values=base_values
        )
        all_picks = [p for team in results for p in team]
        assert len(all_picks) == len(set(all_picks))


# ---------------------------------------------------------------------------
# compute_team_starting_lineup_vor
# ---------------------------------------------------------------------------

class TestComputeTeamStartingLineupVor:
    def test_returns_entry_per_team(self, calc, base_values, player_positions):
        repl = calc.compute_replacement_values(base_values, player_positions)
        vor = calc.compute_vor(base_values, repl, player_positions)
        results = calc.simulate_snake_draft(vor, player_positions, num_rounds=5)
        team_vors = calc.compute_team_starting_lineup_vor(results, vor, player_positions)
        assert len(team_vors) == 8

    def test_selects_best_starters(self):
        """Manually verify best 2G+2F+1C selection."""
        settings = LeagueSettings(
            num_teams=2,
            roster_spots={"G": 2, "F": 1, "C": 1},
            scoring_rules=_simple_rules(),
        )
        calc = VORCalculator(settings)
        # Build a team roster where we know who the top players are
        draft_results = [
            ["G_high", "G_low", "F_high", "C_high"],
            ["G_mid", "G_bad", "F_low", "C_low"],
        ]
        vor_values = {
            "G_high": 20.0, "G_low": 5.0, "F_high": 15.0, "C_high": 10.0,
            "G_mid": 12.0, "G_bad": 1.0, "F_low": 3.0, "C_low": 2.0,
        }
        pos = {
            "G_high": ["PG"], "G_low": ["SG"], "G_mid": ["PG"], "G_bad": ["SG"],
            "F_high": ["SF"], "F_low": ["PF"], "C_high": ["C"], "C_low": ["C"],
        }
        team_vors = calc.compute_team_starting_lineup_vor(draft_results, vor_values, pos)
        # Team 0: 2G=20+5=25, 1F=15, 1C=10 → 50
        assert team_vors[0] == pytest.approx(50.0)
        # Team 1: 2G=12+1=13, 1F=3, 1C=2 → 18
        assert team_vors[1] == pytest.approx(18.0)
