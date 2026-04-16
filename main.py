from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from fantasy_basketball.player import ACTIVE_PLAYERS, Player
from tqdm import tqdm

SEASONS = ['2021-22', '2022-23', '2023-24', '2024-25', '2025-26']
MAX_WORKERS = 3

ASSETS_DIR = Path(__file__).parent / 'src' / 'fantasy_basketball' / 'assets'


def is_fully_cached(name: str) -> bool:
    return all((ASSETS_DIR / season / f'{name}.csv').exists() for season in SEASONS)


def process_player(name: str):
    if is_fully_cached(name):
        return
    player = Player(name)
    player.save_player_game_data(seasons=SEASONS, rate_limit_wait_time=0.6)


futures = {}
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    for name in ACTIVE_PLAYERS:
        futures[executor.submit(process_player, name)] = name

    with tqdm(total=len(futures)) as pbar:
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f'Error processing player {name}: {e}')
            pbar.update(1)
