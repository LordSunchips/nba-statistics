"""High-level analysis functions for fantasy basketball.

Three entry points:

    generate_vor_rankings  — rank all players by VOR (overall and by position)
    random_snake_draft     — simulate a random snake draft and report VOR by pick slot
    simulate_fantasy_season — bootstrap-sample player game logs to project season outcomes
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from fantasy_basketball.vor import LeagueSettings, VORCalculator, compute_base_value


def generate_vor_rankings(
    game_logs: Dict[str, pd.DataFrame],
    player_positions: Dict[str, List[str]],
    league_settings: LeagueSettings,
) -> pd.DataFrame:
    """Rank all players by Value Over Replacement.

    Returns a DataFrame sorted by VOR descending with columns:
        overall_rank  — position-agnostic rank across all players
        position_rank — rank within the player's primary position (G / F / C)
        player        — player name
        primary_position — normalized fantasy position (G, F, or C)
        positions     — all normalized positions the player qualifies for (e.g. "G/F")
        base_value    — risk-adjusted average fantasy score
        vor           — Value Over Replacement

    Parameters
    ----------
    game_logs        : {player_name: game_log_DataFrame}
    player_positions : {player_name: [raw_position_strings]}  e.g. ["PG", "SF"]
    league_settings  : LeagueSettings defining num_teams, roster_spots, scoring_rules
    """
    calc = VORCalculator(league_settings)
    rules = league_settings.scoring_rules

    base_values = {
        name: compute_base_value(
            log, rules, league_settings.season_games, league_settings.risk_aversion
        )
        for name, log in game_logs.items()
    }

    replacement_values = calc.compute_replacement_values(base_values, player_positions)
    vor_values = calc.compute_vor(base_values, replacement_values, player_positions)

    rows = []
    for name, vor in vor_values.items():
        raw = player_positions.get(name, [])
        norm = calc._normalize_positions(raw)
        primary = norm[0] if norm else "?"
        rows.append(
            {
                "player": name,
                "primary_position": primary,
                "positions": "/".join(norm) if norm else "?",
                "base_value": base_values[name],
                "vor": vor,
            }
        )

    df = (
        pd.DataFrame(rows)
        .sort_values("vor", ascending=False)
        .reset_index(drop=True)
    )
    df["overall_rank"] = df.index + 1
    df["position_rank"] = (
        df.groupby("primary_position")["vor"]
        .rank(ascending=False, method="first")
        .astype(int)
    )

    return df[
        ["overall_rank", "position_rank", "player", "primary_position", "positions", "base_value", "vor"]
    ]


def random_snake_draft(
    vor_values: Dict[str, float],
    player_positions: Dict[str, List[str]],
    league_settings: LeagueSettings,
    num_rounds: int = 5,
) -> pd.DataFrame:
    """Simulate a snake draft where each pick takes the highest available player by VOR.

    Returns a DataFrame with one row per pick:
        pick             — overall pick number (1-based)
        round            — round number (1-based)
        team             — team index (0-based)
        player           — player name
        primary_position — normalized fantasy position (G / F / C)
        vor              — player's pre-draft VOR

    Parameters
    ----------
    vor_values       : {player_name: vor} — typically from VORCalculator.compute_vor()
    player_positions : {player_name: [raw_position_strings]}
    league_settings  : LeagueSettings (num_teams used for snake order)
    num_rounds       : picks per team (default 5 = 2G + 2F + 1C)
    """
    num_teams = league_settings.num_teams
    total_picks = num_rounds * num_teams

    if total_picks > len(vor_values):
        raise ValueError(
            f"Not enough players ({len(vor_values)}) for "
            f"{num_teams} teams × {num_rounds} rounds ({total_picks} picks)."
        )

    calc = VORCalculator(league_settings)
    available = set(vor_values.keys())

    rows = []
    pick_number = 0
    for round_num in range(num_rounds):
        team_order = (
            range(num_teams)
            if round_num % 2 == 0
            else range(num_teams - 1, -1, -1)
        )
        for team_idx in team_order:
            pick_number += 1
            chosen = max(available, key=lambda p: (vor_values[p], p))
            available.remove(chosen)

            raw = player_positions.get(chosen, [])
            norm = calc._normalize_positions(raw)
            primary = norm[0] if norm else "?"

            rows.append(
                {
                    "pick": pick_number,
                    "round": round_num + 1,
                    "team": team_idx,
                    "player": chosen,
                    "primary_position": primary,
                    "vor": vor_values[chosen],
                }
            )

    return pd.DataFrame(rows)


def simulate_fantasy_season(
    draft_results: List[List[str]],
    game_logs: Dict[str, pd.DataFrame],
    league_settings: LeagueSettings,
    n_weeks: int = 18,
    games_per_week: int = 3,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Project fantasy season standings by bootstrap-sampling player game logs.

    For each week, each player on a team has `games_per_week` games sampled
    with replacement from their historical log. Fantasy points are summed
    per team per week. A team earns a "win" for any week where their score
    exceeds the weekly median across all teams.

    Returns a standings DataFrame sorted by total season fantasy points:
        season_rank    — final ranking (1 = best)
        team           — team index matching draft_results order (0-based)
        total_pts      — cumulative fantasy points over the full season
        avg_weekly_pts — mean points per week
        best_week      — highest single-week score
        worst_week     — lowest single-week score
        wins           — weeks scored above the weekly median

    Parameters
    ----------
    draft_results  : output of VORCalculator.simulate_snake_draft() or random_snake_draft()
                     List[List[str]] where draft_results[i] is team i's roster
    game_logs      : {player_name: game_log_DataFrame}
    league_settings: LeagueSettings (scoring_rules used to compute fantasy points)
    n_weeks        : number of weeks in the simulated season (default 18)
    games_per_week : games sampled per player per week (default 3)
    seed           : random seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    rules = league_settings.scoring_rules
    num_teams = len(draft_results)

    weekly_scores: List[List[float]] = [[] for _ in range(num_teams)]
    weekly_wins: List[int] = [0] * num_teams

    for _week in range(n_weeks):
        week_scores: List[float] = []

        for team_idx, roster in enumerate(draft_results):
            team_pts = 0.0
            for player in roster:
                log = game_logs.get(player)
                if log is None or log.empty:
                    continue
                sample_idx = rng.integers(0, len(log), size=games_per_week)
                sample = log.iloc[sample_idx]
                team_pts += float(rules.compute_season_scores(sample).sum())
            week_scores.append(team_pts)
            weekly_scores[team_idx].append(team_pts)

        median = float(np.median(week_scores))
        for team_idx, score in enumerate(week_scores):
            if score > median:
                weekly_wins[team_idx] += 1

    rows = []
    for team_idx in range(num_teams):
        scores = weekly_scores[team_idx]
        rows.append(
            {
                "team": team_idx,
                "total_pts": round(sum(scores), 2),
                "avg_weekly_pts": round(sum(scores) / len(scores), 2) if scores else 0.0,
                "best_week": round(max(scores), 2) if scores else 0.0,
                "worst_week": round(min(scores), 2) if scores else 0.0,
                "wins": weekly_wins[team_idx],
            }
        )

    df = (
        pd.DataFrame(rows)
        .sort_values("total_pts", ascending=False)
        .reset_index(drop=True)
    )
    df["season_rank"] = df.index + 1
    return df[
        ["season_rank", "team", "total_pts", "avg_weekly_pts", "best_week", "worst_week", "wins"]
    ]
