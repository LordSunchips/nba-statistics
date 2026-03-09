from fantasy_basketball.player import ACTIVE_PLAYERS, Player
from tqdm import tqdm

for active_player in tqdm(ACTIVE_PLAYERS):
    try:
        player = Player(active_player)

        player.save_player_game_data(
            seasons=['2021-22', '2022-23', '2023-24', '2024-25', '2025-26'],
            rate_limit_wait_time=1.0
        )
    except Exception as e:
        print(f'Error processing player {active_player}: {e}')