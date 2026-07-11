from pathlib import Path

class Constants:
    """
    Constants used by the fantasy basketball module.
    """
    PROJECT_ROOT: str = str(Path(__file__).parent.parent.parent)
    DATA_CACHE_DIR: str = str(Path(PROJECT_ROOT) / 'data_cache')
    
    # Data keys for player information
    FULL_NAME: str = 'full_name'
    FIRST_NAME: str = 'first_name'
    LAST_NAME: str = 'last_name'
    ID: str = 'id'
    
    # Sleeper API and related constants
    SLEEPER_API_BASE_URL = 'https://api.sleeper.app/v1'
    SLEEPER_NBA_PLAYERS_ENDPOINT = f'{SLEEPER_API_BASE_URL}/players/nba'
    FANTASY_POSITIONS: str = 'fantasy_positions'
    TEAM: str = 'team'
    SLEEPER_ID: str = 'sleeper_id'