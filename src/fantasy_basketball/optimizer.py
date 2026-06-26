from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize

from fantasy_basketball.scoring import ScoringRules
from fantasy_basketball.vor import (
    WEIGHT_ORDER,
    LeagueSettings,
    VORCalculator,
    compute_base_value,
)

LOGGER = logging.getLogger(__name__)

# Default per-weight bounds, indexed by WEIGHT_ORDER.
# tov and missed-shot weights are constrained non-positive.
_DEFAULT_BOUNDS: List[Tuple[float, float]] = [
    (0.0, 5.0),   # pts
    (0.0, 5.0),   # reb
    (0.0, 5.0),   # ast
    (0.0, 10.0),  # stl
    (0.0, 10.0),  # blk
    (-5.0, 0.0),  # tov
    (0.0, 5.0),   # field_goals_made
    (-5.0, 0.0),  # field_goals_missed
    (0.0, 5.0),   # free_throws_made
    (-5.0, 0.0),  # free_throws_missed
    (0.0, 5.0),   # three_pointers_made
]


class ScoringOptimizer:
    """Calibrates ScoringRules weights to minimize competitive imbalance.

    The objective is the difference in cumulative starting-lineup VOR between
    the best and worst teams after a simulated 8-team snake draft (5 rounds:
    2G + 2F + 1C). Minimizing this spread produces the fairest market allocation.

    Non-linear bonuses (double-doubles, stat thresholds) are excluded from the
    weight vector — they are step functions that gradient-based solvers cannot
    meaningfully optimize. They are preserved unchanged from fixed_scoring_rules
    (or league_settings.scoring_rules if not provided).

    Usage::

        optimizer = ScoringOptimizer(game_logs, positions, league_settings)
        result = optimizer.optimize(initial_weights=np.array([...]))
        optimal_rules = optimizer.get_optimal_scoring_rules()
    """

    def __init__(
        self,
        player_game_logs: Dict[str, pd.DataFrame],
        player_positions: Dict[str, List[str]],
        league_settings: LeagueSettings,
        fixed_scoring_rules: Optional[ScoringRules] = None,
    ) -> None:
        self._game_logs = player_game_logs
        self._positions = player_positions
        self._settings = league_settings
        self._fixed_bonuses_source = fixed_scoring_rules or league_settings.scoring_rules
        self._optimal_weights: Optional[np.ndarray] = None

    def _weights_to_scoring_rules(self, weights: np.ndarray) -> ScoringRules:
        """Unpack a weight vector into a ScoringRules instance.

        Bonuses are taken unchanged from self._fixed_bonuses_source.
        """
        kwargs = dict(zip(WEIGHT_ORDER, weights.tolist()))
        return ScoringRules(
            **kwargs,
            category_bonuses=self._fixed_bonuses_source.category_bonuses,
            stat_threshold_bonuses=self._fixed_bonuses_source.stat_threshold_bonuses,
        )

    def _compute_objective(self, weights: np.ndarray) -> float:
        """Objective function: max(team_vor) - min(team_vor) after a snake draft.

        Steps:
        1. Build ScoringRules from weights.
        2. Compute risk-adjusted base_value for every player.
        3. Compute replacement values and VOR.
        4. Simulate a 5-round snake draft with dynamic VOR recalculation.
        5. Compute each team's starting-lineup VOR (2G + 2F + 1C).
        6. Return the spread between the best and worst team.
        """
        rules = self._weights_to_scoring_rules(weights)

        base_values: Dict[str, float] = {
            name: compute_base_value(
                log,
                rules,
                self._settings.season_games,
                self._settings.risk_aversion,
            )
            for name, log in self._game_logs.items()
        }

        temp_settings = LeagueSettings(
            num_teams=self._settings.num_teams,
            roster_spots=self._settings.roster_spots,
            scoring_rules=rules,
            season_games=self._settings.season_games,
            risk_aversion=self._settings.risk_aversion,
        )
        calc = VORCalculator(temp_settings)

        replacement_values = calc.compute_replacement_values(base_values, self._positions)
        vor_values = calc.compute_vor(base_values, replacement_values, self._positions)

        num_rounds = sum(self._settings.roster_spots.values())
        draft_results = calc.simulate_snake_draft(
            vor_values,
            self._positions,
            num_rounds=num_rounds,
            base_values=base_values,
        )

        team_vors = calc.compute_team_starting_lineup_vor(
            draft_results, vor_values, self._positions
        )

        if not team_vors:
            return 0.0

        return float(max(team_vors.values()) - min(team_vors.values()))

    def _random_weights_in_bounds(
        self,
        bounds: List[Tuple[float, float]],
        rng: np.random.Generator,
    ) -> np.ndarray:
        return np.array([rng.uniform(lo, hi) for lo, hi in bounds])

    def optimize(
        self,
        initial_weights: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]] = None,
        method: str = "L-BFGS-B",
        n_restarts: int = 3,
    ) -> optimize.OptimizeResult:
        """Find scoring weights that minimize competitive imbalance.

        Parameters
        ----------
        initial_weights : Starting weight vector (length 11, WEIGHT_ORDER).
        bounds          : Per-weight (lo, hi) bounds. Defaults to _DEFAULT_BOUNDS.
        method          : "L-BFGS-B" (default, multi-start), "Nelder-Mead"
                          (no bounds), or "differential_evolution" (global).
        n_restarts      : Number of L-BFGS-B restarts (including initial_weights).
        """
        effective_bounds = bounds or _DEFAULT_BOUNDS

        if method == "differential_evolution":
            result = optimize.differential_evolution(
                self._compute_objective,
                bounds=effective_bounds,
                seed=42,
                maxiter=1000,
                tol=1e-6,
            )
        elif method == "Nelder-Mead":
            result = optimize.minimize(
                self._compute_objective,
                x0=initial_weights,
                method="Nelder-Mead",
                options={"maxiter": 10000, "xatol": 1e-5, "fatol": 1e-5},
            )
        else:  # L-BFGS-B with multi-start
            rng = np.random.default_rng(seed=42)
            candidates = [initial_weights] + [
                self._random_weights_in_bounds(effective_bounds, rng)
                for _ in range(n_restarts - 1)
            ]
            best_result: Optional[optimize.OptimizeResult] = None
            for x0 in candidates:
                res = optimize.minimize(
                    self._compute_objective,
                    x0=x0,
                    method="L-BFGS-B",
                    bounds=effective_bounds,
                    options={"maxiter": 500, "ftol": 1e-9, "gtol": 1e-6},
                )
                if best_result is None or res.fun < best_result.fun:
                    best_result = res
            result = best_result  # type: ignore[assignment]

        initial_obj = self._compute_objective(initial_weights)
        if result.fun <= initial_obj:
            self._optimal_weights = result.x
        else:
            LOGGER.warning(
                "Optimizer did not improve on initial weights (%.4f → %.4f). "
                "Retaining initial weights.",
                initial_obj,
                result.fun,
            )
            self._optimal_weights = initial_weights

        return result

    def get_optimal_scoring_rules(self) -> ScoringRules:
        """Return ScoringRules built from the optimized weight vector.

        Raises RuntimeError if optimize() has not been called yet.
        """
        if self._optimal_weights is None:
            raise RuntimeError(
                "No optimized weights available. Call optimize() first."
            )
        return self._weights_to_scoring_rules(self._optimal_weights)
