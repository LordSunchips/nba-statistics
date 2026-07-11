import importlib
import json
import pandas as pd
import unittest
from unittest.mock import patch, mock_open

import fantasy_basketball.data.player as player_module
from fantasy_basketball.data.player import Player

class TestActivePlayers(unittest.TestCase):

    @patch('fantasy_basketball.data.player.open', new_callable=mock_open)
    @patch('nba_api.stats.static.players.get_active_players', return_value=[])
    def test_active_players_loaded_from_cache_when_present(self, mock_get_active_players, mocked_open):
        cached_players = [
            {'full_name': 'LeBron James', 'first_name': 'LeBron', 'last_name': 'James', 'id': 2544}
        ]
        mocked_open.return_value.read.return_value = json.dumps(cached_players)

        importlib.reload(player_module)

        mock_get_active_players.assert_not_called()
        self.assertEqual(player_module.ACTIVE_PLAYERS, cached_players)


    @patch('fantasy_basketball.data.player.open', new_callable=mock_open)
    @patch('nba_api.stats.static.players.get_active_players')
    @patch('fantasy_basketball.data.player.json.dump')
    def test_active_players_fetched_and_cached_when_missing(self, mock_json_dump, mock_get_active_players, mocked_open):
        fetched_players = [
            {'full_name': 'Stephen Curry', 'first_name': 'Stephen', 'last_name': 'Curry', 'id': 201939}
        ]
        mock_get_active_players.return_value = fetched_players
        # First open() (read mode) misses the cache; second open() (write mode) succeeds.
        mocked_open.side_effect = [FileNotFoundError(), mocked_open.return_value]

        importlib.reload(player_module)

        mock_get_active_players.assert_called_once()
        mock_json_dump.assert_called_once()
        self.assertEqual(player_module.ACTIVE_PLAYERS, fetched_players)


    def tearDown(self):
        # Reload with real dependencies so later tests don't inherit mocked ACTIVE_PLAYERS.
        importlib.reload(player_module)


class TestPlayer(unittest.TestCase):

    def setUp(self):
        self.active_players = [
            {'full_name': 'LeBron James', 'first_name': 'LeBron', 'last_name': 'James', 'id': 2544},
            {'full_name': 'Stephen Curry', 'first_name': 'Stephen', 'last_name': 'Curry', 'id': 201939}
        ]

        self.sample_gamelog_df = pd.DataFrame({
            'GAME_DATE': ['2022-10-18', '2022-10-20'],
            'PTS': [30, 25],
            'REB': [10, 8],
            'AST': [5, 7]
        })

        self.sample_sleeper_data = {
            'sleeper_id': '2544',
            'full_name': 'LeBron James',
            'first_name': 'LeBron',
            'last_name': 'James',
            'fantasy_positions': ['G', 'F'],
            'team': 'LAL',
        }

    @patch('fantasy_basketball.data.player.get_sleeper_player_by_name')
    def test_player_initialization(self, mock_get_sleeper_player_by_name):
        mock_get_sleeper_player_by_name.return_value = self.sample_sleeper_data

        player = Player(full_name="LeBron James", first_name="LeBron", last_name="James", nba_id=2544)
        self.assertEqual(player.full_name, "LeBron James")
        self.assertEqual(player.first_name, "LeBron")
        self.assertEqual(player.last_name, "James")
        self.assertEqual(player.nba_id, 2544)
        self.assertEqual(player.fantasy_positions, ['G', 'F'])
        self.assertEqual(player.team, 'LAL')
        self.assertEqual(player.sleeper_id, '2544')

    @patch('fantasy_basketball.data.player.get_sleeper_player_by_name')
    @patch('fantasy_basketball.data.player.get_active_players')
    def test_player_from_nba_id(self, mock_get_active_players, mock_get_sleeper_player_by_name):
        # Mock the active players list
        mock_get_active_players.return_value = self.active_players
        mock_get_sleeper_player_by_name.return_value = self.sample_sleeper_data

        player = Player.from_nba_id(2544)
        self.assertEqual(player.full_name, "LeBron James")
        self.assertEqual(player.first_name, "LeBron")
        self.assertEqual(player.last_name, "James")
        self.assertEqual(player.nba_id, 2544)
        self.assertEqual(player.fantasy_positions, ['G', 'F'])
        self.assertEqual(player.team, 'LAL')
        self.assertEqual(player.sleeper_id, '2544')

    @patch('fantasy_basketball.data.player.get_active_players')
    def test_player_from_nba_id_not_found(self, mock_get_active_players):
        # Mock the active players list to be empty
        mock_get_active_players.return_value = []

        with self.assertRaises(ValueError) as context:
            Player.from_nba_id(9999)

        self.assertEqual(str(context.exception), "No player found with nba_id: 9999")

    @patch('fantasy_basketball.data.player.get_sleeper_player_by_name')
    @patch('fantasy_basketball.data.player.get_active_players')
    def test_player_from_full_name(self, mock_get_active_players, mock_get_sleeper_player_by_name):
        # Mock the active players list
        mock_get_active_players.return_value = self.active_players
        mock_get_sleeper_player_by_name.return_value = self.sample_sleeper_data

        player = Player.from_full_name("LeBron James")
        self.assertEqual(player.full_name, "LeBron James")
        self.assertEqual(player.first_name, "LeBron")
        self.assertEqual(player.last_name, "James")
        self.assertEqual(player.nba_id, 2544)
        self.assertEqual(player.fantasy_positions, ['G', 'F'])
        self.assertEqual(player.team, 'LAL')
        self.assertEqual(player.sleeper_id, '2544')

    @patch('fantasy_basketball.data.player.get_active_players')
    def test_player_from_full_name_not_found(self, mock_get_active_players):
        # Mock the active players list to be empty
        mock_get_active_players.return_value = []

        with self.assertRaises(ValueError) as context:
            Player.from_full_name("Nonexistent Player")

        self.assertEqual(str(context.exception), "No player found with full name: Nonexistent Player")


    @patch('fantasy_basketball.data.player.get_gamelogs')
    @patch('fantasy_basketball.data.player.get_active_players')
    @patch('fantasy_basketball.data.player.get_sleeper_player_by_name')
    def test_get_gamelogs(self, mock_get_sleeper_player_by_name, mock_get_active_players, mock_get_gamelogs):
        # Mock the active players list
        mock_get_active_players.return_value = self.active_players
        mock_get_sleeper_player_by_name.return_value = self.sample_sleeper_data

        mock_get_gamelogs.return_value = [self.sample_gamelog_df]

        player = Player.from_nba_id(2544)
        gamelogs = player.get_gamelogs()
        self.assertIsInstance(gamelogs, list)
        self.assertEqual(len(gamelogs), 1)
        pd.testing.assert_frame_equal(gamelogs[0], self.sample_gamelog_df)

    @patch('fantasy_basketball.data.player.get_gamelogs')
    @patch('fantasy_basketball.data.player.get_active_players')
    @patch('fantasy_basketball.data.player.get_sleeper_player_by_name')
    def test_get_gamelogs_with_specified_seasons(self, mock_get_sleeper_player_by_name, mock_get_active_players, mock_get_gamelogs):
        # Mock the active players list
        mock_get_active_players.return_value = self.active_players
        mock_get_sleeper_player_by_name.return_value = self.sample_sleeper_data

        mock_get_gamelogs.return_value = [self.sample_gamelog_df, self.sample_gamelog_df]

        player = Player.from_nba_id(2544)
        seasons = ['2022-23', '2021-22']
        gamelogs = player.get_gamelogs(seasons=seasons)

        self.assertIsInstance(gamelogs, list)
        self.assertEqual(len(gamelogs), 2)
        pd.testing.assert_frame_equal(gamelogs[0], self.sample_gamelog_df)
        pd.testing.assert_frame_equal(gamelogs[1], self.sample_gamelog_df)


    @patch('fantasy_basketball.data.player.get_sleeper_player_by_name')
    def test_player_repr(self, mock_get_sleeper_player_by_name):
        mock_get_sleeper_player_by_name.return_value = self.sample_sleeper_data

        player = Player(full_name="LeBron James", first_name="LeBron", last_name="James", nba_id=2544)
        expected_repr = 'Player<LeBron James (2544)>'
        self.assertEqual(repr(player), expected_repr)