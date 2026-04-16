from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd


# Stat categories eligible for multi-category bonuses (double-double, triple-double).
BONUS_CATEGORIES = ["PTS", "REB", "AST", "STL", "BLK"]


@dataclass
class CategoryBonus:
    """
    Bonus awarded when a player hits a threshold in N stat categories.

    Example: double-double = threshold=10, min_categories=2, points=1.5
    """
    threshold: int
    min_categories: int
    points: float

    def earned(self, game: pd.Series) -> bool:
        count = sum(
            1 for cat in BONUS_CATEGORIES
            if cat in game and game[cat] >= self.threshold
        )
        return count >= self.min_categories


@dataclass
class StatThresholdBonus:
    """
    Bonus awarded when a single stat exceeds an absolute threshold.

    Example: 40+ PTS game = stat="PTS", threshold=40, points=3.0
    """
    stat: str
    threshold: float
    points: float

    def earned(self, game: pd.Series) -> bool:
        return self.stat in game and game[self.stat] >= self.threshold


@dataclass
class ScoringRules:
    """Encapsulates a fantasy league's scoring configuration.

    Linear multipliers map directly to per-unit stat values (e.g. pts=1.0
    means 1 fantasy point per real point scored). Bonuses are applied on
    top of the linear score and evaluated per game.

    Usage::

        rules = ScoringRules.espn_standard()
        game_scores = rules.compute_season_scores(player_game_log_df)
    """

    # Linear multipliers — one per stat column in the game log
    pts: float
    reb: float
    ast: float
    stl: float
    blk: float
    tov: float
    field_goals_missed: float      # Optional: penalize missed FGs if 'FGA' and 'FGM' are in the game log
    field_goals_made: float        # Optional: reward made FGs if 'FGM' is in the game log
    free_throws_missed: float      # Optional: penalize missed FTs if 'FTA' and 'FTM' are in the game log
    free_throws_made: float        # Optional: reward made FTs if 'FTM' is in the game log
    three_pointers_made: float     # Optional: reward made 3PM if '3PM' is in the game log

    # Non-linear bonuses applied after the linear score is computed
    category_bonuses: List[CategoryBonus] = field(default_factory=list)
    stat_threshold_bonuses: List[StatThresholdBonus] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Core scoring methods
    # ------------------------------------------------------------------ #

    def compute_game_score(self, game: pd.Series) -> float:
        """Compute fantasy points for a single game row."""
        score = (
            game.get("PTS", 0) * self.pts
            + game.get("REB", 0) * self.reb
            + game.get("AST", 0) * self.ast
            + game.get("STL", 0) * self.stl
            + game.get("BLK", 0) * self.blk
            + game.get("TOV", 0) * self.tov
            + game.get("FGA", 0) * self.field_goals_missed
            + game.get("FGM", 0) * self.field_goals_made
            + game.get("FTA", 0) * self.free_throws_missed
            + game.get("FTM", 0) * self.free_throws_made
            + game.get("3PM", 0) * self.three_pointers_made
        )

        for bonus in self.category_bonuses:
            if bonus.earned(game):
                score += bonus.points

        for bonus in self.stat_threshold_bonuses:
            if bonus.earned(game):
                score += bonus.points

        return round(score, 4)

    def compute_season_scores(self, game_log: pd.DataFrame) -> pd.Series:
        """
        Compute per-game fantasy points for a full season game log.

        Returns a Series aligned to the game_log index containing the
        fantasy point total for each game.
        """
        return game_log.apply(self.compute_game_score, axis=1)

    def season_average(self, game_log: pd.DataFrame) -> float:
        """
        Return the mean fantasy points per game over a season.
        """
        scores = self.compute_season_scores(game_log)
        return round(scores.mean(), 4) if not scores.empty else 0.0

    # ------------------------------------------------------------------ #
    # League presets
    # ------------------------------------------------------------------ #

    @classmethod
    def boydfriends(cls) -> ScoringRules:
        """
        Scoring Rules preset for the Boydfriends Fantasy Basketball League
        """
        return cls(
            pts=0.5,
            reb=1.2,
            ast=1.2,
            stl=2.5,
            blk=2.5,
            tov=-1,
            field_goals_made=0.5,
            field_goals_missed=-0.25,
            free_throws_made=0.25,
            free_throws_missed=-0.25,
            three_pointers_made=0.5,
        )

    @classmethod
    def custom(
        cls,
        pts: float = 1.0,
        reb: float = 1.2,
        ast: float = 1.5,
        stl: float = 3.0,
        blk: float = 3.0,
        tov: float = -1.0,
        field_goals_missed: float = 0.0,
        field_goals_made: float = 0.0,
        free_throws_missed: float = 0.0,
        free_throws_made: float = 0.0,
        three_pointers_made: float = 0.0,
        double_double_bonus: float = 0.0,
        triple_double_bonus: float = 0.0,
        stat_bonuses: List[StatThresholdBonus] | None = None,
    ) -> ScoringRules:
        """
        Build a ScoringRules instance for a custom league configuration.
        """
        category_bonuses: List[CategoryBonus] = []
        if double_double_bonus:
            category_bonuses.append(
                CategoryBonus(threshold=10, min_categories=2, points=double_double_bonus)
            )
        if triple_double_bonus:
            category_bonuses.append(
                CategoryBonus(threshold=10, min_categories=3, points=triple_double_bonus)
            )

        return cls(
            pts=pts,
            reb=reb,
            ast=ast,
            stl=stl,
            blk=blk,
            tov=tov,
            field_goals_missed=field_goals_missed,
            field_goals_made=field_goals_made,
            free_throws_missed=free_throws_missed,
            free_throws_made=free_throws_made,
            three_pointers_made=three_pointers_made,
            category_bonuses=category_bonuses,
            stat_threshold_bonuses=stat_bonuses or [],
        )
