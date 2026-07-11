import pandas as pd
import unittest
from unittest.mock import patch, MagicMock

from fantasy_basketball.data.gamelogs import get_current_season, get_season_gamelog, \
    _get_season_gamelog_cache, _get_season_gamelog_from_nba_api, get_gamelogs

class TestGetCurrentSeason(unittest.TestCase):
    
    def test_get_current_season_format(self):
        season = get_current_season()
        self.assertRegex(season, r'^\d{4}-\d{2}$', "Season format should be 'YYYY-YY'")
        
    @patch('fantasy_basketball.data.gamelogs.datetime')
    def test_get_current_season_before_october(self, mock_datetime):
        mock_datetime.now.return_value = MagicMock(month=9, year=2023)  # September 2023
        season = get_current_season()
        self.assertEqual(season, '2022-23', "Season should be '2022-23' before October 2023")

    @patch('fantasy_basketball.data.gamelogs.datetime')
    def test_get_current_season_after_october(self, mock_datetime):
        mock_datetime.now.return_value = MagicMock(month=11, year=2023)  # November 15, 2023
        season = get_current_season()
        self.assertEqual(season, '2023-24', "Season should be '2023-24' after October 2023")

class TestGamelogs(unittest.TestCase):
    
    def setUp(self):
        self.player_id = 2544
        self.season = '2022-23'
        
        self.sample_df = pd.DataFrame({
            'GAME_ID': [1, 2],
            'PTS': [30, 25],
            'REB': [10, 8],
            'AST': [5, 7]
        })
    
    @patch('fantasy_basketball.data.gamelogs.playergamelog.PlayerGameLog')
    def test_get_season_gamelog_from_nba_api(self, mock_playergamelog):
        # Mock the response from the NBA API
        mock_response = MagicMock()
        mock_response.get_data_frames.return_value = [self.sample_df]
        mock_playergamelog.return_value = mock_response
        
        result = _get_season_gamelog_from_nba_api(self.player_id, self.season)

        # Ensure the PlayerGameLog was called with the correct parameters
        mock_playergamelog.assert_called_once_with(player_id=self.player_id, season=self.season)
        pd.testing.assert_frame_equal(result, self.sample_df)


    @patch('fantasy_basketball.data.gamelogs.Path.exists', return_value=True)
    @patch('fantasy_basketball.data.gamelogs.pd.read_csv')
    def test_get_season_gamelog_cache_returns_cached_df(self, mock_read_csv, mock_path_exists):
        # Mock the cache to return a DataFrame
        mock_read_csv.return_value = self.sample_df

        result = _get_season_gamelog_cache(self.player_id, self.season)

        # Ensure the cache file path was checked and read_csv was called
        mock_path_exists.assert_called_once()
        mock_read_csv.assert_called_once()
        pd.testing.assert_frame_equal(result, self.sample_df)


    @patch('fantasy_basketball.data.gamelogs.Path.exists', return_value=False)
    def test_get_season_gamelog_cache_raises_when_missing(self, mock_path_exists):

        with self.assertRaises(FileNotFoundError):
            _get_season_gamelog_cache(self.player_id, self.season)

        # Ensure the cache file path was checked
        mock_path_exists.assert_called_once()


    @patch('fantasy_basketball.data.gamelogs._get_season_gamelog_cache')
    @patch('fantasy_basketball.data.gamelogs._get_season_gamelog_from_nba_api')
    def test_get_season_gamelog_dispatcher_uses_cache(self, mock_nba_api, mock_cache):
        # Mock the cache to return a DataFrame
        mock_cache.return_value = self.sample_df

        result = get_season_gamelog(self.player_id, self.season)

        # Ensure the cache was called and the NBA API was not called
        mock_cache.assert_called_once_with(self.player_id, self.season)
        mock_nba_api.assert_not_called()
        pd.testing.assert_frame_equal(result, self.sample_df)

    @patch('fantasy_basketball.data.gamelogs.Path.exists', return_value=False)
    @patch('fantasy_basketball.data.gamelogs._get_season_gamelog_from_nba_api')
    def test_get_season_gamelog_dispatcher_falls_back_to_api(self, mock_nba_api, mock_exists):
        # Mock the NBA API to return a DataFrame
        mock_nba_api.return_value = self.sample_df

        get_season_gamelog(self.player_id, self.season)

        # Ensure the cache was checked (raised FileNotFoundError), then the NBA API was called
        mock_nba_api.assert_called_once_with(self.player_id, self.season)
        
    def test_get_gamelogs_returns_list_of_dataframes(self):
        seasons = ['2022-23', '2021-22']
        expected_dfs = [self.sample_df, self.sample_df]

        with patch('fantasy_basketball.data.gamelogs.get_season_gamelog', side_effect=expected_dfs) as mock_get_season_gamelog:
            result = get_gamelogs(self.player_id, seasons)

            # Ensure get_season_gamelog was called for each season
            calls = [((self.player_id, season),) for season in seasons]
            mock_get_season_gamelog.assert_has_calls(calls, any_order=False)
            self.assertEqual(result, expected_dfs)
            
    def test_get_gamelogs_empty_seasons_returns_empty_list(self):
        seasons = []
        result = get_gamelogs(self.player_id, seasons)
        self.assertListEqual(result, [])