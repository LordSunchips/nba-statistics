import unittest
from unittest.mock import patch, mock_open

import requests

from fantasy_basketball.data.sleeper import convert_positions_to_fantasy_positions, \
    fetch_sleeper_nba_players_from_api, cache_sleeper_nba_players, \
    get_sleeper_nba_players_from_cache, get_sleeper_nba_players, get_sleeper_player_by_name


class TestConvertPositionsToFantasyPositions(unittest.TestCase):

    def test_guard_positions_map_to_g(self):
        self.assertCountEqual(convert_positions_to_fantasy_positions(['PG', 'SG']), ['G'])

    def test_forward_positions_map_to_f(self):
        self.assertCountEqual(convert_positions_to_fantasy_positions(['SF', 'PF']), ['F'])

    def test_center_maps_to_c(self):
        self.assertCountEqual(convert_positions_to_fantasy_positions(['C']), ['C'])

    def test_multiple_distinct_positions(self):
        self.assertCountEqual(convert_positions_to_fantasy_positions(['PG', 'SF', 'C']), ['G', 'F', 'C'])

    def test_unknown_position_is_dropped(self):
        self.assertEqual(convert_positions_to_fantasy_positions(['XYZ']), [])

    def test_empty_positions_returns_empty_list(self):
        self.assertEqual(convert_positions_to_fantasy_positions([]), [])


class TestSleeperNbaPlayers(unittest.TestCase):

    def setUp(self):
        self.sample_players_data = {
            '1085': {'full_name': 'Stephen Curry', 'first_name': 'Stephen', 'last_name': 'Curry'}
        }

    @patch('fantasy_basketball.data.sleeper.requests.get')
    def test_fetch_sleeper_nba_players_from_api_success(self, mock_get):
        mock_get.return_value.json.return_value = self.sample_players_data

        result = fetch_sleeper_nba_players_from_api()

        mock_get.return_value.raise_for_status.assert_called_once()
        self.assertEqual(result, self.sample_players_data)

    @patch('fantasy_basketball.data.sleeper.requests.get')
    def test_fetch_sleeper_nba_players_from_api_raises_runtime_error_on_failure(self, mock_get):
        mock_get.return_value.raise_for_status.side_effect = requests.exceptions.RequestException("boom")

        with self.assertRaises(RuntimeError):
            fetch_sleeper_nba_players_from_api()

    @patch('fantasy_basketball.data.sleeper.json.dump')
    @patch('fantasy_basketball.data.sleeper.open', new_callable=mock_open)
    def test_cache_sleeper_nba_players_writes_to_cache(self, mocked_open, mock_json_dump):
        cache_sleeper_nba_players(self.sample_players_data)

        mocked_open.assert_called_once()
        args, kwargs = mock_json_dump.call_args
        self.assertEqual(args[0], self.sample_players_data)
        self.assertEqual(kwargs.get('indent'), 4)

    @patch('fantasy_basketball.data.sleeper.Path.exists', return_value=True)
    @patch('fantasy_basketball.data.sleeper.json.load')
    @patch('fantasy_basketball.data.sleeper.open', new_callable=mock_open)
    def test_get_sleeper_nba_players_from_cache_returns_cached_data(self, mocked_open, mock_json_load, mock_path_exists):
        mock_json_load.return_value = self.sample_players_data

        result = get_sleeper_nba_players_from_cache()

        mock_path_exists.assert_called_once()
        self.assertEqual(result, self.sample_players_data)

    @patch('fantasy_basketball.data.sleeper.Path.exists', return_value=False)
    def test_get_sleeper_nba_players_from_cache_raises_when_missing(self, mock_path_exists):
        with self.assertRaises(FileNotFoundError):
            get_sleeper_nba_players_from_cache()

        mock_path_exists.assert_called_once()

    @patch('fantasy_basketball.data.sleeper.fetch_sleeper_nba_players_from_api')
    @patch('fantasy_basketball.data.sleeper.get_sleeper_nba_players_from_cache')
    def test_get_sleeper_nba_players_dispatcher_uses_cache(self, mock_cache, mock_fetch):
        mock_cache.return_value = self.sample_players_data

        result = get_sleeper_nba_players()

        mock_fetch.assert_not_called()
        self.assertEqual(result, self.sample_players_data)

    @patch('fantasy_basketball.data.sleeper.cache_sleeper_nba_players')
    @patch('fantasy_basketball.data.sleeper.fetch_sleeper_nba_players_from_api')
    @patch('fantasy_basketball.data.sleeper.get_sleeper_nba_players_from_cache')
    def test_get_sleeper_nba_players_dispatcher_falls_back_to_api(self, mock_cache, mock_fetch, mock_cache_write):
        mock_cache.side_effect = FileNotFoundError()
        mock_fetch.return_value = self.sample_players_data

        result = get_sleeper_nba_players()

        mock_fetch.assert_called_once()
        mock_cache_write.assert_called_once_with(self.sample_players_data)
        self.assertEqual(result, self.sample_players_data)


class TestGetSleeperPlayerByName(unittest.TestCase):

    @patch('fantasy_basketball.data.sleeper.get_sleeper_nba_players')
    def test_get_sleeper_player_by_name_found(self, mock_get_sleeper_nba_players):
        mock_get_sleeper_nba_players.return_value = {
            '1085': {
                'full_name': 'Stephen Curry',
                'first_name': 'Stephen',
                'last_name': 'Curry',
                'fantasy_positions': ['PG', 'SG'],
                'team': 'GSW',
            }
        }

        result = get_sleeper_player_by_name('Stephen Curry')

        self.assertEqual(result['sleeper_id'], '1085')
        self.assertEqual(result['full_name'], 'Stephen Curry')
        self.assertEqual(result['first_name'], 'Stephen')
        self.assertEqual(result['last_name'], 'Curry')
        self.assertCountEqual(result['fantasy_positions'], ['G'])
        self.assertEqual(result['team'], 'GSW')

    @patch('fantasy_basketball.data.sleeper.get_sleeper_nba_players')
    def test_get_sleeper_player_by_name_falls_back_to_first_last_name(self, mock_get_sleeper_nba_players):
        mock_get_sleeper_nba_players.return_value = {
            '1085': {
                'full_name': None,
                'first_name': 'Stephen',
                'last_name': 'Curry',
                'fantasy_positions': ['PG'],
                'team': 'GSW',
            }
        }

        result = get_sleeper_player_by_name('Stephen Curry')

        self.assertEqual(result['sleeper_id'], '1085')

    @patch('fantasy_basketball.data.sleeper.get_sleeper_nba_players')
    def test_get_sleeper_player_by_name_not_found(self, mock_get_sleeper_nba_players):
        mock_get_sleeper_nba_players.return_value = {}

        with self.assertRaises(ValueError) as context:
            get_sleeper_player_by_name('Nonexistent Player')

        self.assertEqual(str(context.exception), "No player found with full name Nonexistent Player.")


if __name__ == '__main__':
    unittest.main()
