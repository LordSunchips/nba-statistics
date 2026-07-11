import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List

from nba_api.stats.endpoints import playergamelog

from fantasy_basketball.constants import Constants

def get_current_season() -> str:
    """
    Returns the current NBA season in the format 'YYYY-YY', e.g., '2022-23'.

    :returns: A string representing the current NBA season.
    """
    now = datetime.now()
    year = now.year
    if now.month < 10:  # Before October, the season is still the previous year
        year -= 1
    return f"{year}-{str(year + 1)[-2:]}"

def _get_season_gamelog_from_nba_api(player_id: int, season: str) -> pd.DataFrame:
    """
    Obtains a `pd.DataFrame` containing the gamelog for a given player in a given season from the NBA API.

    :param player_id: The NBA player ID (see `fantasy_basketball.data.player.Player` for more information).
    :param season: The NBA season in the format 'YYYY-YY', e.g., '2022-23'.
    
    :returns: A `pd.DataFrame` containing the gamelog for the player in the specified season.
    """
    response = playergamelog.PlayerGameLog(player_id=player_id, season=season)
    season_gamelog = response.get_data_frames()[0]
    return season_gamelog


def _get_season_gamelog_cache(player_id: int, season: str) -> pd.DataFrame:
    """
    Obtains a `pd.DataFrame` containing the gamelog for a given player in a given season from the local cache.

    :param player_id: The NBA player ID (see `fantasy_basketball.data.player.Player` for more information).
    :param season: The NBA season in the format 'YYYY-YY', e.g., '2022-23'.
    
    :returns: A `pd.DataFrame` containing the gamelog for the player in the specified season.
    """
    cache_file_path = str(Path(Constants.DATA_CACHE_DIR) / 'gamelogs' / season / f'{player_id}.csv')
    if not Path(cache_file_path).exists():
        raise FileNotFoundError(f"Cache file not found for player_id {player_id} and season {season}.")
    return pd.read_csv(cache_file_path)


def get_season_gamelog(player_id: int, season: str) -> pd.DataFrame:
    """
    Obtains a `pd.DataFrame` containing the gamelog for a given player in a given season, either from the local cache or from the NBA API.

    :param player_id: The NBA player ID (see `fantasy_basketball.data.player.Player` for more information).
    :param season: The NBA season in the format 'YYYY-YY', e.g., '2022-23'.
    
    :returns: A `pd.DataFrame` containing the gamelog for the player in the specified season.
    """
    try:
        return _get_season_gamelog_cache(player_id, season)
    except FileNotFoundError:
        season_gamelog = _get_season_gamelog_from_nba_api(player_id, season)
        
        # Save the gamelog to the local cache
        cache_dir = Path(Constants.DATA_CACHE_DIR) / 'gamelogs' / season
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file_path = cache_dir / f'{player_id}.csv'
        season_gamelog.to_csv(cache_file_path, index=False)

    return season_gamelog

def get_gamelogs(player_id: int, seasons: List[str]) -> List[pd.DataFrame]:
    gamelogs = []
    for season in seasons:
        gamelogs.append(get_season_gamelog(player_id, season))
    return gamelogs

if __name__ == '__main__':
    example_player = 201939
    example_season = '2015-16'
    gamelog_df = get_season_gamelog(player_id=example_player, season=example_season)
    print(f"Gamelog for {example_player} in {example_season}:")
    print(gamelog_df)