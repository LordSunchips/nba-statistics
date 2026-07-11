import json
import pandas as pd
from pathlib import Path
from time import time
from typing import List, Optional

from nba_api.stats.static.players import get_active_players

from fantasy_basketball.constants import Constants
from fantasy_basketball.data.gamelogs import get_current_season, get_gamelogs
from fantasy_basketball.data.sleeper import get_sleeper_nba_players, \
    get_sleeper_player_by_name

try:
    ACTIVE_PLAYERS = json.load(open(Path(Constants.DATA_CACHE_DIR) / 'active_players.json', 'r'))
except FileNotFoundError:
    ACTIVE_PLAYERS = get_active_players()

    # Save ACTIVE_PLAYERS to a JSON file in the data_cache directory
    cache_file_path = Path(Constants.DATA_CACHE_DIR) / 'active_players.json'
    with open(cache_file_path, 'w') as f:
        json.dump(ACTIVE_PLAYERS, f, indent=4)
        
SLEEPER_NBA_PLAYERS = get_sleeper_nba_players()

class Player:
    """
    Represents an NBA basketball player.
    """
    full_name: str
    first_name: str
    last_name: str
    nba_id: int
    
    fantasy_positions: List[str]
    team: str
    sleeper_id: str
    
    
    def __init__(self, full_name: str, first_name: str, last_name: str, nba_id: int):
        self.full_name = full_name
        self.first_name = first_name
        self.last_name = last_name
        self.nba_id = nba_id
        
        # Get the player's team and fantasy positions from the Sleeper API data
        sleeper_data = get_sleeper_player_by_name(full_name)
        self.fantasy_positions = sleeper_data[Constants.FANTASY_POSITIONS]
        self.team = sleeper_data[Constants.TEAM]
        self.sleeper_id = sleeper_data[Constants.SLEEPER_ID]

    @classmethod
    def from_nba_id(cls, nba_id: int) -> 'Player':
        """
        Creates a `fantasy_basketball.data.player.Player` object from an nba_id.

        :param nba_id: id of the player in the NBA database.
        
        :returns: A `fantasy_basketball.data.player.Player` object.
        :raises ValueError: If no player is found with the given nba_id.
        """
        for player in ACTIVE_PLAYERS:
            if player[Constants.ID] == nba_id:
                return cls(
                    full_name=player[Constants.FULL_NAME],
                    first_name=player[Constants.FIRST_NAME],
                    last_name=player[Constants.LAST_NAME],
                    nba_id=player[Constants.ID]
                )
        raise ValueError(f"No player found with nba_id: {nba_id}")
    
    
    @classmethod
    def from_full_name(cls, full_name: str) -> 'Player':
        """
        Creates a `fantasy_basketball.data.player.Player` object from a full name.

        :param full_name: Full name of the player (e.g., "LeBron James").
        
        :returns: A `fantasy_basketball.data.player.Player` object.
        :raises ValueError: If no player is found with the given full name.
        """
        for player in ACTIVE_PLAYERS:
            if player[Constants.FULL_NAME].lower() == full_name.lower():
                nba_id = player[Constants.ID]
                return cls.from_nba_id(nba_id)
        raise ValueError(f"No player found with full name: {full_name}")
    
    def get_gamelogs(self, seasons: Optional[List[str]] = None) -> List[pd.DataFrame]:
        """
        Retrieves the gamelogs for the player for the specified seasons.

        :param seasons: List of NBA seasons in the format 'YYYY-YY', e.g., ['2022-23', '2021-22'].
                        If None, defaults to the last season.
        
        :returns: A list of `pd.DataFrame` objects containing the gamelogs for each season.
        """
        if seasons is None:
            # Default to the last season
            current_season = get_current_season()
            seasons = [current_season]
        
        return get_gamelogs(self.nba_id, seasons)
    
    def __repr__(self):
        return f'Player<{self.full_name} ({self.nba_id})>'
    
if __name__ == '__main__':
    print(f"Number of active players: {len(ACTIVE_PLAYERS)}")
    print(f'Example player: {Player.from_nba_id(ACTIVE_PLAYERS[0][Constants.ID])}')