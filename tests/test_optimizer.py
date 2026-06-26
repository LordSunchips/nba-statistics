"""Tests for the ScoringOptimizer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fantasy_basketball.scoring import CategoryBonus, ScoringRules, StatThresholdBonus
from fantasy_basketball.vor import LeagueSettings, WEIGHT_ORDER
from fantasy_basketball.optimizer import ScoringOptimizer, _DEFAULT_BOUNDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_log(pts: float, reb: float = 5.0, ast: float = 3.0, n_games: int = 41) -> pd.DataFrame:
    return pd.DataFrame(
        [{"PTS": pts, "REB": reb, "AST": ast, "STL": 1.0, "BLK": 0.5,
          "TOV": 1.0, "FGM": 8.0, "FGA": 18.0, "FTM": 3.0, "FTA": 4.0, "3PM": 2.0}]
        * n_games
    )


def _default_rules() -> ScoringRules:
    return ScoringRules.boydfriends()


def _make_league(num_teams: int = 8) -> LeagueSettings:
    return LeagueSettings(
        num_teams=num_teams,
        roster_spots={"G": 2, "F": 2, "C": 1},
        scoring_rules=_default_rules(),
        season_games=82,
        risk_aversion=0.0,
    )


# 20 guards, 20 forwards, 10 centers — enough for 8 teams × 5 picks (40 total)
_GAME_LOGS = {
    **{f"G{i}": _make_log(pts=40 - i, reb=3) for i in range(1, 21)},
    **{f"F{i}": _make_log(pts=35 - i, reb=7) for i in range(1, 21)},
    **{f"C{i}": _make_log(pts=30 - i, reb=10) for i in range(1, 11)},
}
_POSITIONS = {
    **{f"G{i}": ["PG"] for i in range(1, 21)},
    **{f"F{i}": ["SF"] for i in range(1, 21)},
    **{f"C{i}": ["C"] for i in range(1, 11)},
}

_INITIAL_WEIGHTS = np.array([w[0] + (w[1] - w[0]) / 2 for w in _DEFAULT_BOUNDS])


@pytest.fixture
def optimizer():
    return ScoringOptimizer(_GAME_LOGS, _POSITIONS, _make_league())


# ---------------------------------------------------------------------------
# _weights_to_scoring_rules
# ---------------------------------------------------------------------------

class TestWeightsToScoringRules:
    def test_roundtrip_all_fields(self, optimizer):
        weights = np.array([1.0, 1.2, 1.5, 3.0, 3.0, -1.0, 0.5, -0.25, 0.25, -0.25, 0.5])
        rules = optimizer._weights_to_scoring_rules(weights)
        assert rules.pts == pytest.approx(weights[WEIGHT_ORDER.index("pts")])
        assert rules.reb == pytest.approx(weights[WEIGHT_ORDER.index("reb")])
        assert rules.tov == pytest.approx(weights[WEIGHT_ORDER.index("tov")])
        assert rules.three_pointers_made == pytest.approx(weights[WEIGHT_ORDER.index("three_pointers_made")])

    def test_preserves_fixed_bonuses(self):
        dd_bonus = CategoryBonus(threshold=10, min_categories=2, points=1.5)
        td_bonus = CategoryBonus(threshold=10, min_categories=3, points=3.0)
        stat_bonus = StatThresholdBonus(stat="PTS", threshold=40, points=2.0)
        fixed_rules = ScoringRules(
            pts=1.0, reb=1.0, ast=1.0, stl=1.0, blk=1.0, tov=-1.0,
            field_goals_made=0.0, field_goals_missed=0.0,
            free_throws_made=0.0, free_throws_missed=0.0,
            three_pointers_made=0.0,
            category_bonuses=[dd_bonus, td_bonus],
            stat_threshold_bonuses=[stat_bonus],
        )
        optimizer = ScoringOptimizer(
            _GAME_LOGS, _POSITIONS, _make_league(), fixed_scoring_rules=fixed_rules
        )
        rules = optimizer._weights_to_scoring_rules(_INITIAL_WEIGHTS)
        assert len(rules.category_bonuses) == 2
        assert len(rules.stat_threshold_bonuses) == 1

    def test_returns_scoring_rules_instance(self, optimizer):
        rules = optimizer._weights_to_scoring_rules(_INITIAL_WEIGHTS)
        assert isinstance(rules, ScoringRules)


# ---------------------------------------------------------------------------
# _compute_objective
# ---------------------------------------------------------------------------

class TestComputeObjective:
    def test_returns_non_negative_float(self, optimizer):
        obj = optimizer._compute_objective(_INITIAL_WEIGHTS)
        assert isinstance(obj, float)
        assert obj >= 0.0

    def test_zero_weights_still_returns_float(self, optimizer):
        # All-zero weights are unlikely to minimize, but shouldn't crash
        weights = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        obj = optimizer._compute_objective(weights)
        assert isinstance(obj, float)
        assert obj >= 0.0

    def test_objective_changes_with_different_weights(self, optimizer):
        w1 = _INITIAL_WEIGHTS.copy()
        w2 = _INITIAL_WEIGHTS.copy()
        w2[0] = w2[0] + 2.0  # boost pts weight
        assert optimizer._compute_objective(w1) != optimizer._compute_objective(w2)


# ---------------------------------------------------------------------------
# optimize
# ---------------------------------------------------------------------------

class TestOptimize:
    def test_returns_optimize_result(self, optimizer):
        from scipy.optimize import OptimizeResult
        result = optimizer.optimize(_INITIAL_WEIGHTS, n_restarts=1)
        assert isinstance(result, OptimizeResult)

    def test_result_weights_correct_length(self, optimizer):
        result = optimizer.optimize(_INITIAL_WEIGHTS, n_restarts=1)
        assert len(result.x) == 11

    def test_result_does_not_worsen_objective(self, optimizer):
        initial_obj = optimizer._compute_objective(_INITIAL_WEIGHTS)
        result = optimizer.optimize(_INITIAL_WEIGHTS, n_restarts=1)
        # Optimizer should not make the objective worse (with tolerance for fp noise)
        assert result.fun <= initial_obj + 1e-3

    def test_nelder_mead_method(self, optimizer):
        result = optimizer.optimize(_INITIAL_WEIGHTS, method="Nelder-Mead")
        assert len(result.x) == 11

    def test_optimal_weights_stored_after_optimize(self, optimizer):
        optimizer.optimize(_INITIAL_WEIGHTS, n_restarts=1)
        assert optimizer._optimal_weights is not None
        assert len(optimizer._optimal_weights) == 11


# ---------------------------------------------------------------------------
# get_optimal_scoring_rules
# ---------------------------------------------------------------------------

class TestGetOptimalScoringRules:
    def test_raises_before_optimize(self, optimizer):
        with pytest.raises(RuntimeError, match="Call optimize\\(\\) first"):
            optimizer.get_optimal_scoring_rules()

    def test_returns_scoring_rules_after_optimize(self, optimizer):
        optimizer.optimize(_INITIAL_WEIGHTS, n_restarts=1)
        rules = optimizer.get_optimal_scoring_rules()
        assert isinstance(rules, ScoringRules)

    def test_rules_respect_tov_negative_bound(self, optimizer):
        optimizer.optimize(_INITIAL_WEIGHTS, n_restarts=1)
        rules = optimizer.get_optimal_scoring_rules()
        assert rules.tov <= 0.0
