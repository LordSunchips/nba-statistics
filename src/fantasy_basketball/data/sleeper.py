import json
import requests
from pathlib import Path
from typing import Dict, List

from fantasy_basketball.constants import Constants

def convert_positions_to_fantasy_positions(positions: List[str]) -> List[str]:
    """
    Converts a list of positions from the Sleeper API to fantasy basketball positions.

    :param positions: A list of positions from the Sleeper API.
    
    :returns: A list of fantasy basketball positions.
    """
    position_mapping = {
        'PG': 'G',
        'SG': 'G',
        'SF': 'F',
        'PF': 'F',
        'C': 'C',
    }
    
    fantasy_positions = []
    for position in positions:
        if position in position_mapping:
            fantasy_position = position_mapping[position]
            fantasy_positions.append(fantasy_position)
    
    return list(set(fantasy_positions))  # Remove duplicates

def fetch_sleeper_nba_players_from_api() -> Dict:
    try:
        response = requests.get(Constants.SLEEPER_NBA_PLAYERS_ENDPOINT)
        response.raise_for_status()  # Raise an error for bad responses
        return response.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to fetch data from Sleeper API ({Constants.SLEEPER_NBA_PLAYERS_ENDPOINT}): {e}")

def cache_sleeper_nba_players(players_data: Dict) -> None:
    cache_file_path = str(Path(Constants.DATA_CACHE_DIR) / 'sleeper_nba_players.json')
    with open(cache_file_path, 'w') as f:
        json.dump(players_data, f, indent=4)

def get_sleeper_nba_players_from_cache() -> Dict:
    cache_file_path = str(Path(Constants.DATA_CACHE_DIR) / 'sleeper_nba_players.json')
    if not Path(cache_file_path).exists():
        raise FileNotFoundError("Cache file for Sleeper NBA players not found.")
    with open(cache_file_path, 'r') as f:
        return json.load(f)
    
def get_sleeper_nba_players() -> Dict:
    """
    Obtains a dictionary containing NBA player data from the Sleeper API, either from the local cache or from the API.

    :returns: A dictionary containing NBA player data.
    """
    try:
        return get_sleeper_nba_players_from_cache()
    except FileNotFoundError:
        players_data = fetch_sleeper_nba_players_from_api()
        cache_sleeper_nba_players(players_data)
        return players_data
    
def get_sleeper_player_by_name(full_name: str) -> Dict:
    """
    Obtains a dictionary containing player data from the Sleeper API for a given NBA player name.

    :param full_name: The full name of the NBA player (see `fantasy_basketball.data.player.Player` for more information).
    
    :returns: A dictionary containing player data from the Sleeper API.
    :raises ValueError: If no player is found with the given full name.
    """
    players_data = get_sleeper_nba_players()
    for sleeper_id, player_info in players_data.items():
        player_full_name = player_info.get(Constants.FULL_NAME)
        if player_full_name is None:
            player_full_name = f"{player_info.get(Constants.FIRST_NAME, '')} {player_info.get(Constants.LAST_NAME, '')}".strip()
        player_full_name = player_full_name
        if player_full_name == full_name:
            return {
                'sleeper_id': sleeper_id,
                'full_name': player_info[Constants.FULL_NAME],
                'first_name': player_info[Constants.FIRST_NAME],
                'last_name': player_info[Constants.LAST_NAME],
                'fantasy_positions': convert_positions_to_fantasy_positions(player_info.get(Constants.FANTASY_POSITIONS, [])),
                'team': player_info.get(Constants.TEAM, None)
            }
    raise ValueError(f"No player found with full name {full_name}.")
    
if __name__ == '__main__':
    players_data = get_sleeper_nba_players()
    print(f"Fetched {len(players_data)} NBA players from Sleeper API.")