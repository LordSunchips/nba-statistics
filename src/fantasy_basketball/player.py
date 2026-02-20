from urllib import response

import pandas as pd
import requests
from pathlib import Path
from typing import Dict, List

from nba_api.stats.static import players

from fantasy_basketball.utils.constants import Constants

ACTIVE_PLAYERS = {player[Constants.FULL_NAME]: player for player in players.get_active_players()}

class Player():
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
        player = ACTIVE_PLAYERS.get(full_name)
        if player is None:
            raise ValueError(f'Player with full name ({full_name}) not found in active players.')
        
        self.full_name = full_name
        self.first_name = player[Constants.FIRST_NAME]
        self.last_name = player[Constants.LAST_NAME]
        self.nba_id = player[Constants.ID]
        
        self.sleeper_data = self._get_sleeper_data()
        self.positions = self._get_positions()
        self.team = self._get_team()
        self.sleeper_id = self._get_sleeper_id()
        
    def _get_sleeper_data(self) -> pd.DataFrame:
        # Check if the data is already cached in ../assets/sleeper_data.json
        sleeper_data_file = Path(__file__).parent / 'assets' / 'sleeper_data.json'
        
        if sleeper_data_file.exists():
            sleeper_data = pd.read_json(sleeper_data_file)
        else:
            # GET https://api.sleeper.app/v1/players/nba and store to ../assets/sleeper_data.json
            sleeper_data = pd.DataFrame(requests.get("https://api.sleeper.app/v1/players/nba").json()).T
            sleeper_data = sleeper_data[sleeper_data[Constants.FULL_NAME].notna()]
            
            sleeper_data.to_json(sleeper_data_file)
            
        return sleeper_data[sleeper_data[Constants.FULL_NAME] == self.full_name].iloc[0]
    
    def _get_positions(self) -> List[str]:
        return self.sleeper_data.get(Constants.FANTASY_POSITIONS, [])
    
    def _get_team(self) -> str:
        return self.sleeper_data.get(Constants.TEAM, "")

    def _get_sleeper_id(self) -> str:
        return self.sleeper_data.get(Constants.PLAYER_ID, "")
    
    def __repr__(self) -> str:
        return f'{self.full_name} ({self.positions} - {self.team})'