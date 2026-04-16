import pandas as pd
import unittest
from unittest.mock import patch, MagicMock

from fantasy_basketball.player import Player

_FAKE_PLAYER_NBA = {
    'first_name': 'Fake',
    'last_name': 'Player',
    'full_name': 'Fake Player',
    'id': 9999,
}

_FAKE_GAME_COLS = ['GAME_DATE', 'MATCHUP', 'WL', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']


class TestPlayer(unittest.TestCase):
    def setUp(self):
        Player.get_sleeper_data()

    # ------------------------------------------------------------------
    # __init__ / attribute assignment
    # ------------------------------------------------------------------

    def test_steph_curry_from_file(self):
        steph = Player('Stephen Curry')
        self.assertEqual(steph.full_name, 'Stephen Curry')
        self.assertEqual(steph.first_name, 'Stephen')
        self.assertEqual(steph.last_name, 'Curry')
        self.assertEqual(steph.positions, ['PG'])
        self.assertEqual(steph.team, 'GSW')
        self.assertEqual(steph.sleeper_data['birth_date'], '1988-03-14')

    @patch('requests.get')
    @patch('pathlib.Path.exists', return_value=False)
    @patch('pandas.DataFrame.to_json', return_value=None)
    def test_nikola_jokic_from_sleeper(self, mock_to_json, mock_exists, mock_get):
        mock_sleeper_data = MagicMock()
        mock_sleeper_data.json.return_value = {'123': {
            'full_name': 'Nikola Jokić',
            'first_name': 'Nikola',
            'last_name': 'Jokić',
            'fantasy_positions': ['C'],
            'team': 'DEN',
            'birth_date': '1994-09-20'
        }}
        mock_get.return_value = mock_sleeper_data

        jokic = Player('Nikola Jokić')
        self.assertEqual(jokic.full_name, 'Nikola Jokić')
        self.assertEqual(jokic.first_name, 'Nikola')
        self.assertEqual(jokic.last_name, 'Jokić')
        self.assertEqual(jokic.positions, ['C'])
        self.assertEqual(jokic.team, 'DEN')
        self.assertEqual(jokic.sleeper_data['birth_date'], '1994-09-20')

    def test_nonexistent_player(self):
        with self.assertRaises(ValueError) as context:
            Player('Non Existent Player')
        self.assertIn(
            'Player with full name (Non Existent Player) not found in active players.',
            str(context.exception),
        )

    @patch.dict('fantasy_basketball.player.ACTIVE_PLAYERS', {'Fake Player': _FAKE_PLAYER_NBA})
    @patch.object(Player, 'get_sleeper_data')
    def test_player_not_in_sleeper_data(self, mock_get_sleeper):
        """Player exists in NBA API but is absent from Sleeper data."""
        mock_get_sleeper.return_value = pd.DataFrame(
            columns=['full_name', 'fantasy_positions', 'team', 'player_id']
        )
        with self.assertRaises(ValueError) as ctx:
            Player('Fake Player')
        self.assertIn('Fake Player', str(ctx.exception))
        self.assertIn('sleeper data', str(ctx.exception))

    # ------------------------------------------------------------------
    # Private helpers called after init
    # ------------------------------------------------------------------

    def test_get_positions_missing_key(self):
        steph = Player('Stephen Curry')
        steph.sleeper_data = {}
        self.assertEqual(steph._get_positions(), [])

    def test_get_team_missing_key(self):
        steph = Player('Stephen Curry')
        steph.sleeper_data = {}
        self.assertEqual(steph._get_team(), '')

    def test_get_sleeper_id_missing_key(self):
        steph = Player('Stephen Curry')
        steph.sleeper_data = {}
        self.assertEqual(steph._get_sleeper_id(), '')

    # ------------------------------------------------------------------
    # __repr__
    # ------------------------------------------------------------------

    def test_repr(self):
        steph = Player('Stephen Curry')
        result = repr(steph)
        self.assertIn('Stephen Curry', result)
        self.assertIn('PG', result)
        self.assertIn('GSW', result)

    # ------------------------------------------------------------------
    # get_yearly_game_data
    # ------------------------------------------------------------------

    @patch('pandas.read_csv')
    @patch('pathlib.Path.exists', return_value=True)
    def test_get_yearly_game_data_from_cache(self, mock_exists, mock_read_csv):
        mock_df = pd.DataFrame({'GAME_DATE': ['2024-01-01'], 'PTS': [30]})
        mock_read_csv.return_value = mock_df

        steph = Player('Stephen Curry')
        df, source = steph.get_yearly_game_data('2023-24')

        self.assertEqual(source, 'cache')
        pd.testing.assert_frame_equal(df, mock_df)

    @patch('pandas.DataFrame.to_csv')
    @patch('fantasy_basketball.player.playergamelog.PlayerGameLog')
    @patch('pathlib.Path.exists', return_value=False)
    def test_get_yearly_game_data_from_api(self, mock_exists, mock_gamelog_class, mock_to_csv):
        steph = Player('Stephen Curry')
        raw_df = pd.DataFrame([{col: 'x' for col in _FAKE_GAME_COLS}])
        mock_gamelog = MagicMock()
        mock_gamelog.get_data_frames.return_value = [raw_df]
        mock_gamelog_class.return_value = mock_gamelog

        df, source = steph.get_yearly_game_data('2023-24')

        self.assertEqual(source, 'API')
        self.assertEqual(list(df.columns), _FAKE_GAME_COLS)
        mock_gamelog.get_data_frames.assert_called_once()

    # ------------------------------------------------------------------
    # save_player_game_data
    # ------------------------------------------------------------------

    @patch('fantasy_basketball.player.time.sleep')
    @patch.object(Player, 'get_yearly_game_data', new=MagicMock(return_value=(pd.DataFrame(), 'cache')))
    def test_save_player_game_data_no_sleep_when_cached(self, mock_sleep):
        steph = Player('Stephen Curry')
        steph.save_player_game_data(['2023-24', '2022-23', '2021-22'])
        mock_sleep.assert_not_called()

    @patch('fantasy_basketball.player.random.uniform', new=MagicMock(return_value=1.0))
    @patch('fantasy_basketball.player.time.sleep')
    @patch.object(Player, 'get_yearly_game_data', new=MagicMock(return_value=(pd.DataFrame(), 'API')))
    def test_save_player_game_data_sleeps_on_api_rate_limit(self, mock_sleep):
        steph = Player('Stephen Curry')
        # rate_limit_iterations=1 means sleep fires on every iteration (i % 1 == 0 always)
        steph.save_player_game_data(
            ['2023-24', '2022-23'],
            rate_limit_wait_time=2.0,
            rate_limit_iterations=1,
        )
        self.assertEqual(mock_sleep.call_count, 2)
        mock_sleep.assert_called_with(3.0)  # 2.0 + 1.0 (mocked jitter)

    @patch('fantasy_basketball.player.random.uniform', new=MagicMock(return_value=0.0))
    @patch('fantasy_basketball.player.time.sleep')
    @patch.object(Player, 'get_yearly_game_data', new=MagicMock(return_value=(pd.DataFrame(), 'API')))
    def test_save_player_game_data_skips_sleep_between_iterations(self, mock_sleep):
        """Sleep only triggers at multiples of rate_limit_iterations."""
        steph = Player('Stephen Curry')
        # 3 seasons, rate_limit_iterations=3 → sleep only at i=0
        steph.save_player_game_data(
            ['2023-24', '2022-23', '2021-22'],
            rate_limit_wait_time=1.0,
            rate_limit_iterations=3,
        )
        self.assertEqual(mock_sleep.call_count, 1)
