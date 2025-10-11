import os
from nba_api.stats.endpoints import playergamelog, commonplayerinfo
from nba_api.stats.static import players as players_static
import pandas as pd
from typing import List, Union

PLAYER_INFO_DIR = 'player_info'
PLAYER_GAMELOGS_DIR = 'player_gamelogs'
PLAYER_TO_ID = {player['full_name']: player for player in players_static.get_active_players()}

def custom_update_position(player_name: str, positions: List[str]) -> List[str]:
    """
    Update the positions of a player manually.

    :param player_name (str): The full name of the player.
    """
    if player_name == 'Giannis Antetokounmpo':
        return['Forward', 'Center']
    elif player_name == 'Alperen Sengun':
        return['Center', 'Forward']
    elif player_name == 'Josh Hart':
        return['Guard', 'Forward']
    elif player_name == 'Onyeka Okongwu':
        return['Center']
    elif player_name == 'Bobby Portis':
        return['Forward', 'Center']
    elif player_name == 'Daniel Gafford':
        return['Center']
    elif player_name == 'Isaiah Hartenstein':
        return['Center']
    elif player_name == 'Nic Claxton':
        return['Center', 'Forward']
    else:
        return positions

def player_as_directory_name(player_name: str) -> str:
    return player_name.replace(' ', '_').replace('.', '').lower()

class Player:
    """
    A class for getting player related information
    """
    def __init__(self, player_name: str, positions: List[str] = []):
        """
        Create an instance of a Player class

        :param player_name (string): player's full name as mentioned in nba_api (Draymond Green)
        :param positions (List[str]): a list of the positions the player plays  (ex. 'PG,SG,SF' or ['PG', 'SG', 'SF])
        """
        self.player_name = player_name
        if self.player_name not in PLAYER_TO_ID:
            raise ValueError(f'Player name {self.player_name} not found in active players!')
        self.positions = positions if positions else self.__get_positions()
        self.positions = custom_update_position(player_name, self.positions) 
        self.player_id = self.__get_player_id()
        self.player_gamelogs = {}
        self.games_played = {}
        self.load_player_gamelogs()
        
    def __get_positions(self) -> list:
        info = players_static.find_players_by_full_name(self.player_name)
        if not info:
            raise ValueError(f"No player found for '{self.player_name}'.")
        player_id = PLAYER_TO_ID[self.player_name]['id']
        
        info_dir = os.path.join(PLAYER_INFO_DIR, player_as_directory_name(self.player_name))
        os.makedirs(info_dir, exist_ok=True)
        info_path = os.path.join(info_dir, "positions.txt")
        
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                positions = f.read().strip()
        else:
            player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
            positions = player_info.loc[0, 'POSITION']
            with open(info_path, "w") as f:
                f.write(positions if positions else "")
        self.positions = positions.split('-') if positions else []
        return self.positions
    
    def __get_player_id(self) -> int:
        """
            Get the Player ID of the player.

        :return int: the player's ID
        """
        return PLAYER_TO_ID[self.player_name]['id']

    def __load_existing_player_gamelogs(self) -> None:
        """
        Loads all the gamelogs saved to file from 'player_gamelogs' to self.player_gamelogs.
        self.player_gamelogs is a Dict[str, pd.DataFrame] where the key is the season_id (e.g. '2024-25')
        and the value is the gamelogs.
        
        :return: None
        """
        gamelog_filepath = os.path.join(PLAYER_GAMELOGS_DIR, player_as_directory_name(self.player_name))
        if not os.path.exists(gamelog_filepath):
            os.makedirs(gamelog_filepath)
        
        for filename in os.listdir(gamelog_filepath):
            if filename.endswith('.csv'):
                season_id = filename.replace('.csv', '')
                full_filepath = os.path.join(gamelog_filepath, filename)
                self.player_gamelogs[season_id] = pd.read_csv(full_filepath)

                        
    def load_player_gamelogs(self, seasons: List[str] = ['2025-26']) -> None:
        """
        Loads all the player gamelogs for each season in `seasons`. 

        Args:
            seasons (List[str], optional): A list of seasons in which to pull gamelogs. 
                                           Defaults to ['2025-26'].
        """
        self.__load_existing_player_gamelogs()
        for season in seasons:
            if season not in self.player_gamelogs:
                try:
                    gamelog = playergamelog.PlayerGameLog(player_id=self.player_id, season=season).get_data_frames()[0]
                except Exception as e:
                    print(f'[ERROR] Could not load gamelog [{self.player_name}, {season}]: {e}')
                    continue                    
                
                self.player_gamelogs[season] = gamelog
                self.games_played[season] = gamelog.shape[0] if not gamelog.empty else 0
                
                # save gamelog to file
                gamelog.to_csv(os.path.join(PLAYER_GAMELOGS_DIR, player_as_directory_name(self.player_name), f'{season}.csv'), index=False)
    
    def games_played_by_season(self, season: str = '2024-25') -> int:
        """
        Get the gamelog for a season and the number of games that this player has played.

        Args:
            season (str, optional): the season_id (typically year formats). Defaults to '2024-25'.

        Returns:
            int: the number of games played for the season.
        """
        filepath = os.path.join(PLAYER_GAMELOGS_DIR, player_as_directory_name(self.player_name), f'{season}.csv')
        if os.path.exists(filepath):
            self.gamelog[season] = pd.read_csv(filepath)
            games_played = self.gamelog[season].shape[0] if not self.gamelog[season].empty else 0
            return int(games_played)
        gamelog = playergamelog.PlayerGameLog(player_id=self.player_id, season=season).get_data_frames()[0]
        self.gamelog[season] = gamelog        
        games_played = gamelog.shape[0] if not gamelog.empty else 0
        return int(games_played)

