import logging
import pandas as pd
import random
import requests
import time
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple

from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog

from fantasy_basketball.utils.constants import Constants

LOGGER = logging.getLogger(__name__)

ACTIVE_PLAYERS = {player[Constants.FULL_NAME]: player for player in players.get_active_players()}

class Player:
    # Fields from NBA API
    full_name: str
    first_name: str
    last_name: str
    nba_id: int
    
    # Fields from Sleeper API
    sleeper_data: Dict
    sleeper_id: str
    positions: List[str]
    team: str
    
    def __init__(self, full_name: str):
        nba_player_info = ACTIVE_PLAYERS.get(full_name)
        if nba_player_info is None:
            raise ValueError(f'Player with full name ({full_name}) not found in active players.')
        
        self.full_name = full_name
        self.first_name = nba_player_info[Constants.FIRST_NAME]
        self.last_name = nba_player_info[Constants.LAST_NAME]
        self.nba_id = nba_player_info[Constants.ID]
        
        self.sleeper_data = self._get_sleeper_data()
        self.positions = self._get_positions()
        self.team = self._get_team()
        self.sleeper_id = self._get_sleeper_id()
        
    @staticmethod
    def get_sleeper_data() -> pd.DataFrame:
        # Check if the data is already cached in ../assets/sleeper_data.json
        assets_dir = Path(__file__).parent / 'assets'
        assets_dir.mkdir(exist_ok=True)
        
        sleeper_data_file = assets_dir / 'sleeper_data.json'
        
        if sleeper_data_file.exists():
            LOGGER.debug(f'Loading sleeper data from cache: {sleeper_data_file}')
            sleeper_data = pd.read_json(sleeper_data_file)
        else:
            # GET https://api.sleeper.app/v1/players/nba and store to ../assets/sleeper_data.json
            LOGGER.debug('Fetching sleeper data from API')
            sleeper_data = pd.DataFrame(requests.get("https://api.sleeper.app/v1/players/nba").json()).T
            sleeper_data = sleeper_data[sleeper_data[Constants.FULL_NAME].notna()]
            
            sleeper_data.to_json(sleeper_data_file)
            
        return sleeper_data
    
    def _get_sleeper_data(self) -> pd.DataFrame:
        sleeper_data = self.get_sleeper_data()
        if self.full_name not in sleeper_data[Constants.FULL_NAME].values:
            raise ValueError(f'Player with full name ({self.full_name}) not found in sleeper data.')
        
        player_data = sleeper_data[sleeper_data[Constants.FULL_NAME] == self.full_name]
        
        return player_data.iloc[0].to_dict()
            
    def _get_positions(self) -> List[str]:
        return self.sleeper_data.get(Constants.FANTASY_POSITIONS, [])
    
    def _get_team(self) -> str:
        return self.sleeper_data.get(Constants.TEAM, "")

    def _get_sleeper_id(self) -> str:
        return self.sleeper_data.get(Constants.PLAYER_ID, "")
    
    def get_yearly_game_data(self, season: str) -> Tuple[pd.DataFrame, str]:
        """
        Obtains a dataframe of the seasonal gamelog for a player.

        Args:
            season (str): The season to obtain the player's gamelog for.

        Returns:
            Tuple[pd.DataFrame, str]: The player gamelog and if the gamelog 
                                      was obtained from cache or API. 
        """

        # Check if the data is already cached
        season_dir = Path(__file__).parent / 'assets' / season
        season_dir.mkdir(parents=True, exist_ok=True)
        
        csv_file = season_dir / f'{self.full_name}.csv'
        
        if csv_file.exists():
            LOGGER.debug(f'Loading game data from cache: {csv_file}')
            df = pd.read_csv(csv_file)
            source = "cache"
        else:
            LOGGER.debug(f'Fetching game data from API for season {season}')
            
            NBA_HEADERS = {
                'Host': 'stats.nba.com',
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                'Accept': 'application/json, text/plain, */*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': 'https://www.nba.com/',
                'Origin': 'https://www.nba.com',
                'Sec-Fetch-Site': 'same-site',
                'Sec-Fetch-Mode': 'cors',
                'Sec-Fetch-Dest': 'empty',
                'x-nba-stats-origin': 'stats',
                'x-nba-stats-token': 'true',
            }
            
            gamelog = playergamelog.PlayerGameLog(
                player_id=self.nba_id, 
                season=season, 
                proxy=None,       # Ensure no proxy is interfering
                headers=NBA_HEADERS, 
                timeout=30,
                get_request=True  # Force GET instead of POST if issues persist
            )
            df = gamelog.get_data_frames()[0]
            
            stats_to_keep = ['GAME_DATE', 'MATCHUP', 'WL', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']
            df = df[stats_to_keep]
            
            df.to_csv(csv_file, index=False)
            source = "API"
        
        return df, source

    def save_player_game_data(self, seasons: List[str], rate_limit_wait_time: float = 1.0, rate_limit_iterations: int = 5):
        for i, season in enumerate(seasons):
            _, source = self.get_yearly_game_data(season)
            if source == "API" and i % rate_limit_iterations == 0:
                LOGGER.debug(f'Rate limit reached. Waiting for {rate_limit_wait_time} seconds before continuing...')
                time.sleep(rate_limit_wait_time + random.uniform(0, 2))  # Add random jitter to avoid hitting rate limits
    
    def __repr__(self) -> str:
        return f'{self.full_name} ({self.positions} - {self.team})'