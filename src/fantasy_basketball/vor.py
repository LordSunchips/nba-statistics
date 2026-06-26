from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from fantasy_basketball.scoring import ScoringRules

LOGGER = logging.getLogger(__name__)

# Maps raw NBA/Sleeper position strings to simplified fantasy positions.
_POSITION_MAP: Dict[str, str] = {
    "PG": "G",
    "SG": "G",
    "SF": "F",
    "PF": "F",
    "C": "C",
}

# Canonical weight vector order — mirrors the linear field declaration order
# in ScoringRules. Used by both VORCalculator and ScoringOptimizer.
WEIGHT_ORDER: List[str] = [
    "pts",
    "reb",
    "ast",
    "stl",
    "blk",
    "tov",
    "field_goals_made",
    "field_goals_missed",
    "free_throws_made",
    "free_throws_missed",
    "three_pointers_made",
]


def compute_base_value(
    game_log: pd.DataFrame,
    rules: ScoringRules,
    season_games: int = 82,
    risk_aversion: float = 0.1,
) -> float:
    """Risk-adjusted player value.

    base_value = avg_score * availability_factor - risk_aversion * std_dev

    availability_factor = games_played / season_games, capped at 1.0.
    Penalizes injury-prone players (low availability) and inconsistent
    players (high standard deviation).
    """
    scores = rules.compute_season_scores(game_log)
    if scores.empty:
        return 0.0
    avg = scores.mean()
    std = float(scores.std(ddof=1)) if len(scores) > 1 else 0.0
    availability = min(len(scores) / season_games, 1.0)
    return round(avg * availability - risk_aversion * std, 4)


@dataclass
class LeagueSettings:
    """Configuration for a fantasy basketball league."""

    num_teams: int
    roster_spots: Dict[str, int]  # e.g. {"G": 2, "F": 2, "C": 1}
    scoring_rules: ScoringRules
    season_games: int = 82
    risk_aversion: float = 0.1


class VORCalculator:
    """Computes Value Over Replacement (VOR) for fantasy basketball players.

    VOR = player's base_value minus the replacement-level player's base_value
    at their position. The replacement player is the first player projected
    to go undrafted at a given position.
    """

    def __init__(self, settings: LeagueSettings) -> None:
        self.settings = settings

    @staticmethod
    def _normalize_positions(positions: List[str]) -> List[str]:
        """Map raw NBA/Sleeper positions to G/F/C. Deduplicate the result."""
        normalized: List[str] = []
        for pos in positions:
            mapped = _POSITION_MAP.get(pos)
            if mapped is None:
                LOGGER.warning("Unknown position '%s' — skipping.", pos)
                continue
            if mapped not in normalized:
                normalized.append(mapped)
        return normalized

    def _build_position_pools(
        self,
        base_values: Dict[str, float],
        player_positions: Dict[str, List[str]],
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Build per-position sorted player pools.

        Cross-listed players appear in every pool they qualify for.
        Returns {position: [(player_name, base_value), ...]} sorted descending.
        """
        pools: Dict[str, List[Tuple[str, float]]] = {
            pos: [] for pos in self.settings.roster_spots
        }
        for name, base_val in base_values.items():
            raw_positions = player_positions.get(name, [])
            if not raw_positions:
                LOGGER.warning("Player '%s' has no positions — excluded from pools.", name)
                continue
            for pos in self._normalize_positions(raw_positions):
                if pos in pools:
                    pools[pos].append((name, base_val))

        for pos in pools:
            pools[pos].sort(key=lambda x: x[1], reverse=True)

        return pools

    def compute_replacement_values(
        self,
        base_values: Dict[str, float],
        player_positions: Dict[str, List[str]],
    ) -> Dict[str, float]:
        """Determine the replacement-level base_value for each position.

        Replacement rank = num_teams * roster_spots[pos] (0-indexed).
        This is the first player projected NOT to be drafted at that position.

        Edge cases:
        - Empty pool: returns 0.0 and logs a warning.
        - Pool smaller than threshold: uses the last player in the pool.
        """
        pools = self._build_position_pools(base_values, player_positions)
        replacement_values: Dict[str, float] = {}
        for pos, spots in self.settings.roster_spots.items():
            pool = pools.get(pos, [])
            if not pool:
                LOGGER.warning(
                    "No players found for position '%s' — replacement value set to 0.0.", pos
                )
                replacement_values[pos] = 0.0
                continue
            threshold_idx = self.settings.num_teams * spots
            if threshold_idx >= len(pool):
                replacement_values[pos] = pool[-1][1]
            else:
                replacement_values[pos] = pool[threshold_idx][1]
        return replacement_values

    def compute_vor(
        self,
        base_values: Dict[str, float],
        replacement_values: Dict[str, float],
        player_positions: Dict[str, List[str]],
    ) -> Dict[str, float]:
        """Compute VOR for each player.

        VOR = base_value - min(replacement_values across eligible positions).
        Using min gives the player the most favorable positional baseline
        (they're valued by the scarcest position they fill).

        Players with no eligible normalized positions receive VOR = 0.0.
        """
        vor: Dict[str, float] = {}
        for name, base_val in base_values.items():
            raw_positions = player_positions.get(name, [])
            norm_positions = self._normalize_positions(raw_positions)
            eligible_replacements = [
                replacement_values[pos]
                for pos in norm_positions
                if pos in replacement_values
            ]
            if not eligible_replacements:
                vor[name] = 0.0
            else:
                vor[name] = round(base_val - min(eligible_replacements), 4)
        return vor

    def simulate_snake_draft(
        self,
        vor_values: Dict[str, float],
        player_positions: Dict[str, List[str]],
        num_rounds: int,
        base_values: Optional[Dict[str, float]] = None,
    ) -> List[List[str]]:
        """Simulate a snake draft with dynamic VOR recalculation.

        Returns a list of length num_teams, where each element is the list
        of players drafted by that team (length num_rounds).

        Snake order: even rounds pick teams 0..N-1, odd rounds pick N-1..0.

        If base_values is provided, replacement values and VOR are recomputed
        after every pick to reflect shifting positional scarcity (dynamic mode).
        Without base_values, the initial VOR values are used throughout (static).

        Raises ValueError if there are not enough players to fill all picks.
        """
        total_picks = num_rounds * self.settings.num_teams
        if total_picks > len(vor_values):
            raise ValueError(
                f"Not enough players ({len(vor_values)}) for "
                f"{self.settings.num_teams} teams × {num_rounds} rounds "
                f"({total_picks} picks)."
            )

        available: set[str] = set(vor_values.keys())
        current_vor = dict(vor_values)
        teams: List[List[str]] = [[] for _ in range(self.settings.num_teams)]

        for round_num in range(num_rounds):
            team_order = (
                range(self.settings.num_teams)
                if round_num % 2 == 0
                else range(self.settings.num_teams - 1, -1, -1)
            )
            for team_idx in team_order:
                best_player = max(available, key=lambda p: (current_vor[p], p))
                teams[team_idx].append(best_player)
                available.remove(best_player)

                # Dynamic recalculation: recompute VOR after each pick
                if base_values is not None:
                    remaining_base = {p: base_values[p] for p in available}
                    remaining_positions = {p: player_positions[p] for p in available}
                    repl = self.compute_replacement_values(remaining_base, remaining_positions)
                    current_vor = self.compute_vor(remaining_base, repl, remaining_positions)

        return teams

    def compute_team_starting_lineup_vor(
        self,
        draft_results: List[List[str]],
        vor_values: Dict[str, float],
        player_positions: Dict[str, List[str]],
    ) -> Dict[int, float]:
        """Compute each team's cumulative starting lineup VOR.

        Starting lineup = best 2G + 2F + 1C by VOR from the team's roster.
        Teams that cannot fill all starting spots receive credit only for
        the spots they can fill.

        Returns {team_index: cumulative_starting_vor}.
        """
        result: Dict[int, float] = {}
        for team_idx, roster in enumerate(draft_results):
            # Group roster by normalized position
            pos_players: Dict[str, List[Tuple[str, float]]] = {
                pos: [] for pos in self.settings.roster_spots
            }
            for player in roster:
                raw_positions = player_positions.get(player, [])
                for pos in self._normalize_positions(raw_positions):
                    if pos in pos_players:
                        pos_players[pos].append((player, vor_values.get(player, 0.0)))

            # Sort each position group descending and pick the top N starters
            total_vor = 0.0
            for pos, spots in self.settings.roster_spots.items():
                starters = sorted(pos_players[pos], key=lambda x: x[1], reverse=True)[:spots]
                total_vor += sum(v for _, v in starters)

            result[team_idx] = round(total_vor, 4)
        return result
