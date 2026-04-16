import unittest

import pandas as pd

from fantasy_basketball.scoring import CategoryBonus, ScoringRules, StatThresholdBonus


def _game(**kwargs) -> pd.Series:
    """Build a minimal game Series with zero-defaults for all stat columns."""
    defaults = {"PTS": 0, "REB": 0, "AST": 0, "STL": 0, "BLK": 0, "TOV": 0}
    defaults.update(kwargs)
    return pd.Series(defaults)


def _game_log(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([_game(**r) for r in rows])


def _rules(**overrides) -> ScoringRules:
    """Build a ScoringRules with standard multipliers and zero shooting multipliers."""
    params = dict(
        pts=1.0, reb=1.2, ast=1.5, stl=3.0, blk=3.0, tov=-1.0,
        field_goals_missed=0.0, field_goals_made=0.0,
        free_throws_missed=0.0, free_throws_made=0.0,
        three_pointers_made=0.0,
    )
    params.update(overrides)
    return ScoringRules(**params)


class TestCategoryBonus(unittest.TestCase):

    def setUp(self):
        # Double-double: 10+ in any 2 of PTS/REB/AST/STL/BLK
        self.dd = CategoryBonus(threshold=10, min_categories=2, points=1.5)
        # Triple-double
        self.td = CategoryBonus(threshold=10, min_categories=3, points=4.5)

    def test_earned_exact_threshold(self):
        game = _game(PTS=10, REB=10)
        self.assertTrue(self.dd.earned(game))

    def test_earned_above_threshold(self):
        game = _game(PTS=30, REB=15)
        self.assertTrue(self.dd.earned(game))

    def test_not_earned_one_category(self):
        game = _game(PTS=25, REB=8)
        self.assertFalse(self.dd.earned(game))

    def test_not_earned_all_below(self):
        game = _game(PTS=9, REB=9, AST=9)
        self.assertFalse(self.dd.earned(game))

    def test_triple_double_earned(self):
        game = _game(PTS=20, REB=10, AST=10)
        self.assertTrue(self.td.earned(game))

    def test_triple_double_not_earned_with_only_two(self):
        game = _game(PTS=20, REB=10, AST=9)
        self.assertFalse(self.td.earned(game))

    def test_missing_stat_column_does_not_count(self):
        # Series with only PTS — REB missing entirely
        game = pd.Series({"PTS": 20})
        self.assertFalse(self.dd.earned(game))

    def test_stl_and_blk_count_toward_bonus(self):
        game = _game(STL=10, BLK=10)
        self.assertTrue(self.dd.earned(game))


class TestStatThresholdBonus(unittest.TestCase):

    def setUp(self):
        self.bonus = StatThresholdBonus(stat="PTS", threshold=40, points=3.0)

    def test_earned_exactly_at_threshold(self):
        self.assertTrue(self.bonus.earned(_game(PTS=40)))

    def test_earned_above_threshold(self):
        self.assertTrue(self.bonus.earned(_game(PTS=60)))

    def test_not_earned_below_threshold(self):
        self.assertFalse(self.bonus.earned(_game(PTS=39)))

    def test_missing_stat_returns_false(self):
        self.assertFalse(self.bonus.earned(pd.Series({"REB": 10})))

    def test_non_pts_stat(self):
        reb_bonus = StatThresholdBonus(stat="REB", threshold=20, points=5.0)
        self.assertTrue(reb_bonus.earned(_game(REB=20)))
        self.assertFalse(reb_bonus.earned(_game(REB=19)))


class TestScoringRulesComputeGameScore(unittest.TestCase):

    def setUp(self):
        self.rules = _rules()

    def test_pure_points(self):
        score = self.rules.compute_game_score(_game(PTS=20))
        self.assertAlmostEqual(score, 20.0)

    def test_all_stats_linear(self):
        # PTS=10→10, REB=5→6, AST=4→6, STL=2→6, BLK=1→3, TOV=3→-3  = 28
        game = _game(PTS=10, REB=5, AST=4, STL=2, BLK=1, TOV=3)
        self.assertAlmostEqual(self.rules.compute_game_score(game), 28.0)

    def test_turnovers_are_negative(self):
        score = self.rules.compute_game_score(_game(TOV=5))
        self.assertAlmostEqual(score, -5.0)

    def test_zeros_produce_zero(self):
        self.assertAlmostEqual(self.rules.compute_game_score(_game()), 0.0)

    def test_missing_stat_defaults_to_zero(self):
        game = pd.Series({"PTS": 20})  # no REB/AST/etc.
        self.assertAlmostEqual(self.rules.compute_game_score(game), 20.0)

    def test_result_rounded_to_four_decimal_places(self):
        # 1/3 points per rebound → result should be rounded
        rules = _rules(reb=1 / 3)
        score = rules.compute_game_score(_game(REB=1))
        self.assertEqual(score, round(1 / 3, 4))

    def test_double_double_bonus_applied(self):
        rules = _rules(category_bonuses=[CategoryBonus(threshold=10, min_categories=2, points=1.5)])
        game = _game(PTS=20, REB=10)  # qualifies for DD
        base = 20 * 1.0 + 10 * 1.2
        self.assertAlmostEqual(rules.compute_game_score(game), base + 1.5)

    def test_double_double_bonus_not_applied_when_not_earned(self):
        rules = _rules(category_bonuses=[CategoryBonus(threshold=10, min_categories=2, points=1.5)])
        game = _game(PTS=20, REB=9)  # does NOT qualify
        base = 20 * 1.0 + 9 * 1.2
        self.assertAlmostEqual(rules.compute_game_score(game), base)

    def test_triple_double_bonus_stacks_with_double_double(self):
        """When both DD and TD bonuses exist, both fire on a triple-double."""
        rules = _rules(category_bonuses=[
            CategoryBonus(threshold=10, min_categories=2, points=1.5),
            CategoryBonus(threshold=10, min_categories=3, points=4.5),
        ])
        game = _game(PTS=20, REB=10, AST=10)  # qualifies for both
        base = 20 * 1.0 + 10 * 1.2 + 10 * 1.5
        self.assertAlmostEqual(rules.compute_game_score(game), base + 1.5 + 4.5)

    def test_stat_threshold_bonus_applied(self):
        rules = _rules(stat_threshold_bonuses=[StatThresholdBonus(stat="PTS", threshold=40, points=3.0)])
        game = _game(PTS=45)
        self.assertAlmostEqual(rules.compute_game_score(game), 45.0 + 3.0)

    def test_stat_threshold_bonus_not_applied_below(self):
        rules = _rules(stat_threshold_bonuses=[StatThresholdBonus(stat="PTS", threshold=40, points=3.0)])
        game = _game(PTS=39)
        self.assertAlmostEqual(rules.compute_game_score(game), 39.0)

    def test_multiple_stat_threshold_bonuses(self):
        rules = _rules(stat_threshold_bonuses=[
            StatThresholdBonus(stat="PTS", threshold=40, points=3.0),
            StatThresholdBonus(stat="REB", threshold=20, points=5.0),
        ])
        game = _game(PTS=50, REB=20)
        base = 50 * 1.0 + 20 * 1.2
        self.assertAlmostEqual(rules.compute_game_score(game), base + 3.0 + 5.0)

    def test_shooting_multipliers_applied(self):
        rules = _rules(field_goals_made=0.5, field_goals_missed=-0.25,
                       free_throws_made=0.25, free_throws_missed=-0.25,
                       three_pointers_made=0.5)
        game = pd.Series({"PTS": 0, "REB": 0, "AST": 0, "STL": 0, "BLK": 0, "TOV": 0,
                          "FGM": 8, "FGA": 16, "FTM": 4, "FTA": 6, "3PM": 3})
        expected = (8 * 0.5) + (16 * -0.25) + (4 * 0.25) + (6 * -0.25) + (3 * 0.5)
        self.assertAlmostEqual(rules.compute_game_score(game), expected)


class TestScoringRulesSeasonMethods(unittest.TestCase):

    def setUp(self):
        self.rules = _rules()

    def test_compute_season_scores_returns_series(self):
        log = _game_log([{"PTS": 20}, {"PTS": 30}])
        scores = self.rules.compute_season_scores(log)
        self.assertIsInstance(scores, pd.Series)
        self.assertEqual(len(scores), 2)

    def test_compute_season_scores_values(self):
        log = _game_log([{"PTS": 10, "REB": 5}, {"PTS": 20, "AST": 4}])
        scores = self.rules.compute_season_scores(log)
        self.assertAlmostEqual(scores.iloc[0], 10.0 + 5 * 1.2)
        self.assertAlmostEqual(scores.iloc[1], 20.0 + 4 * 1.5)

    def test_compute_season_scores_aligned_to_index(self):
        log = _game_log([{"PTS": 5}, {"PTS": 15}])
        log.index = [10, 20]  # non-default index
        scores = self.rules.compute_season_scores(log)
        self.assertEqual(list(scores.index), [10, 20])

    def test_season_average_correct(self):
        log = _game_log([{"PTS": 20}, {"PTS": 40}])
        avg = self.rules.season_average(log)
        self.assertAlmostEqual(avg, 30.0)

    def test_season_average_empty_dataframe(self):
        avg = self.rules.season_average(pd.DataFrame(columns=["PTS", "REB"]))
        self.assertEqual(avg, 0.0)

    def test_season_average_rounded_to_four_decimals(self):
        log = _game_log([{"PTS": 10}, {"PTS": 11}, {"PTS": 12}])  # avg = 11.0
        avg = self.rules.season_average(log)
        self.assertEqual(avg, round(avg, 4))


class TestScoringRulesPresets(unittest.TestCase):

    def test_boydfriends_multipliers(self):
        r = ScoringRules.boydfriends()
        self.assertEqual(r.pts, 0.5)
        self.assertEqual(r.reb, 1.2)
        self.assertEqual(r.ast, 1.2)
        self.assertEqual(r.stl, 2.5)
        self.assertEqual(r.blk, 2.5)
        self.assertEqual(r.tov, -1)
        self.assertEqual(r.field_goals_made, 0.5)
        self.assertEqual(r.field_goals_missed, -0.25)
        self.assertEqual(r.free_throws_made, 0.25)
        self.assertEqual(r.free_throws_missed, -0.25)
        self.assertEqual(r.three_pointers_made, 0.5)

    def test_boydfriends_no_bonuses(self):
        r = ScoringRules.boydfriends()
        self.assertEqual(r.category_bonuses, [])
        self.assertEqual(r.stat_threshold_bonuses, [])

    def test_custom_no_bonuses_by_default(self):
        r = ScoringRules.custom(pts=2.0, reb=1.0)
        self.assertEqual(r.pts, 2.0)
        self.assertEqual(r.reb, 1.0)
        self.assertEqual(r.category_bonuses, [])

    def test_custom_with_double_double_bonus(self):
        r = ScoringRules.custom(double_double_bonus=2.0)
        self.assertEqual(len(r.category_bonuses), 1)
        self.assertEqual(r.category_bonuses[0].min_categories, 2)
        self.assertAlmostEqual(r.category_bonuses[0].points, 2.0)

    def test_custom_with_both_bonuses(self):
        r = ScoringRules.custom(double_double_bonus=1.5, triple_double_bonus=3.0)
        self.assertEqual(len(r.category_bonuses), 2)

    def test_custom_with_stat_bonus(self):
        bonus = StatThresholdBonus(stat="PTS", threshold=40, points=5.0)
        r = ScoringRules.custom(stat_bonuses=[bonus])
        self.assertEqual(len(r.stat_threshold_bonuses), 1)
        self.assertEqual(r.stat_threshold_bonuses[0].stat, "PTS")

    def test_custom_zero_bonus_not_added(self):
        """A bonus of 0.0 should not be appended to the bonus list."""
        r = ScoringRules.custom(double_double_bonus=0.0)
        self.assertEqual(r.category_bonuses, [])

    def test_custom_shooting_multipliers(self):
        r = ScoringRules.custom(field_goals_made=0.5, free_throws_missed=-0.25)
        self.assertEqual(r.field_goals_made, 0.5)
        self.assertEqual(r.free_throws_missed, -0.25)
