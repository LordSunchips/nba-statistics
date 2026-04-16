import logging
import pandas as pd
import random
import requests
import time
from pathlib import Path
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
    
    _NAME_SUFFIXES = {'Jr.', 'Sr.', 'II', 'III', 'IV', 'V'}

    def _normalize_name(self, name: str) -> str:
        parts = name.split()
        if parts and parts[-1] in self._NAME_SUFFIXES:
            parts = parts[:-1]
        return ' '.join(parts)

    def _get_sleeper_data(self) -> pd.DataFrame:
        sleeper_data = self.get_sleeper_data()
        names = sleeper_data[Constants.FULL_NAME]

        mask = names == self.full_name
        if not mask.any():
            normalized = self._normalize_name(self.full_name)
            mask = names == normalized

        if not mask.any():
            raise ValueError(f'Player with full name ({self.full_name}) not found in sleeper data.')

        return sleeper_data[mask].iloc[0].to_dict()
            
    def _get_positions(self) -> List[str]:
        return self.sleeper_data.get(Constants.FANTASY_POSITIONS, [])
    
    def _get_team(self) -> str:
        return self.sleeper_data.get(Constants.TEAM, "")

    def _get_sleeper_id(self) -> str:
        return self.sleeper_data.get(Constants.PLAYER_ID, "")
    
    def get_yearly_game_data(self, season: str, max_retries: int = 5) -> Tuple[pd.DataFrame, str]:
        """
        Obtains a dataframe of the seasonal gamelog for a player.

        Args:
            season (str): The season to obtain the player's gamelog for.
            max_retries (int): Maximum number of retry attempts on timeout/error.

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
            return df, "cache"

        LOGGER.debug(f'Fetching game data from API for season {season}')

        for attempt in range(max_retries):
            try:
                gamelog = playergamelog.PlayerGameLog(
                    player_id=self.nba_id,
                    season=season,
                    timeout=5,
                    get_request=True,
                )
                df = gamelog.get_data_frames()[0]
                df.to_csv(csv_file, index=False)
                return df, "API"
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                backoff = (2 ** attempt) + random.uniform(0, 1)
                LOGGER.warning(f'Attempt {attempt + 1}/{max_retries} failed for {self.full_name} ({season}): {e}. Retrying in {backoff:.1f}s...')
                time.sleep(backoff)

    def save_player_game_data(self, seasons: List[str], rate_limit_wait_time: float = 1.0, rate_limit_iterations: int = 1):
        for i, season in enumerate(seasons):
            _, source = self.get_yearly_game_data(season)
            if source == "API" and i % rate_limit_iterations == 0:
                wait = rate_limit_wait_time + random.uniform(0, 1)
                LOGGER.debug(f'Sleeping {wait:.1f}s after API call...')
                time.sleep(wait)
    
    def __repr__(self) -> str:
        return f'{self.full_name} ({self.positions} - {self.team})'