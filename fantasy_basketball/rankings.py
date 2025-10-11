import os
import numpy as np
import networkx as nx
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players as players_static
import pandas as pd
from typing import Optional
from time import sleep
import time
from typing import List
from fantasy_basketball.player import Player, player_as_directory_name
from fantasy_basketball.fantasy_league import FantasyLeague
from tqdm import tqdm

PLAYER_GAMELOGS_DIR = 'player_gamelogs'
PLAYER_INFO_DIR = 'player_info'

class Ranking:
    """
    Create a Ranking instance for a Fantasy League
    """
    def __init__(self, fantasy_league: FantasyLeague):
        self.fantasy_league = fantasy_league

    def get_all_active_players(self, season: str = '2025-26') -> List[Player]:
        """
        Get a list of all active players. Initialize their Player object with gamelogs for the specified season and save the gamelogs to file.
        :return List[Player]: list of Player objects 
        """
        all_active_players = players_static.get_active_players()  # returns list of dicts with id, full_name, first_name, last_name, is_active.
        # make the player_gamelogs directory if it doesn't exist
        if not os.path.exists(PLAYER_GAMELOGS_DIR):
            os.makedirs(PLAYER_GAMELOGS_DIR)
        
        last_fetched = False
        for i, player in enumerate(tqdm(all_active_players, desc="Processing Players")):
            # avoid rate limits
            if i % 2 == 0 and last_fetched:
                sleep(1)
            gamelog_filepath = os.path.join(PLAYER_GAMELOGS_DIR, player_as_directory_name(player['full_name']), f'{season}.csv')
            info_filepath = os.path.join(PLAYER_INFO_DIR, player_as_directory_name(player['full_name']), 'positions.txt')
            if not os.path.exists(gamelog_filepath) or not os.path.exists(info_filepath):
                last_fetched = True
                player = Player(player['full_name'])
                player.load_player_gamelogs(seasons=[season])
                player.player_gamelogs[season].to_csv(gamelog_filepath, index=False)
            else:
                last_fetched = False
                player = Player(player['full_name'])
                player.player_gamelogs[season] = pd.read_csv(gamelog_filepath)
            all_active_players[i] = player
        return all_active_players

    def compute_vor(self, df, repl_levels):
        for pos in ['Guard', 'Forward', 'Center']:
            df[f'VOR_{pos}'] = df.apply(
                lambda r: r['BASE_VALUE'] - repl_levels[pos]
                if pos in r['POSITION'] else np.nan, axis=1
            )
        df['VOR'] = df[['VOR_Guard', 'VOR_Forward', 'VOR_Center']].max(axis=1)
        df['BEST_SLOT'] = df[['VOR_Guard', 'VOR_Forward', 'VOR_Center']].idxmax(axis=1).str.replace('VOR_', '')
        
        # move the VOR columns next to BASE_VALUE
        cols = df.columns.tolist()
        base_value_index = cols.index('BASE_VALUE')
        vor_cols = ['VOR', 'VOR_Guard', 'VOR_Forward', 'VOR_Center', 'BEST_SLOT']
        for vor_col in vor_cols:
            cols.remove(vor_col)
        for i, vor_col in enumerate(vor_cols):
            cols.insert(base_value_index + 1 + i, vor_col)
        df = df[cols]
        return df

    def get_replacement_levels_market(self, df, league_teams=10, starters={'Guard': 2, 'Forward': 2, 'Center': 1}):
        repl = {}
        for slot, num_per_team in starters.items():
            elig = df[df["POSITION"].str.contains(slot, case=False, na=False)]
            if elig.empty:
                repl[slot] = 0.0
                continue
            total_starters = league_teams * num_per_team
            total_starters = min(total_starters, len(elig))
            sorted_vals = elig.sort_values("BASE_VALUE", ascending=False).reset_index(drop=True)
            repl_val = sorted_vals.iloc[total_starters - 1]["BASE_VALUE"]
            repl[slot] = repl_val
        # Cast replacement values to float
        repl = {slot: float(val) for slot, val in repl.items()}
        return repl


    
    def rank_players(self, players: List[Player], season: str = '2025-26', month: str = '', stat: str = 'VOR', file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Rank players based on their total fantasy points in a given season.

        Args:
            players (List[Player]): List of Player objects to rank.
            season (str, optional): Season ID. Defaults to '2025-26'.

        Returns:
            pd.DataFrame: DataFrame with players ranked by total fantasy points.
        """
        if os.path.exists(file_path.replace('csv', 'pkl')) and file_path is not None:
            rankings_df = pd.read_pickle(file_path.replace('csv', 'pkl'))
        else:
            rankings = []
            all_active_players = self.get_all_active_players(season=season)
            for i in range(len(all_active_players)):
                player = all_active_players[i]
                summary = self.fantasy_league.get_player_summary(player, season, month=month)
                if summary.get('WEEKS PLAYED', 0) == 0:
                    continue
                rankings.append(summary)
            rankings_df = pd.DataFrame(rankings).fillna(0)
            repl_levels = self.get_replacement_levels_market(rankings_df)
            rankings_df = self.compute_vor(rankings_df, repl_levels=repl_levels)
            rankings_df = rankings_df.sort_values(by=stat, ascending=False).reset_index(drop=True)
        
        if file_path is not None:
            rankings_df.to_pickle(file_path.replace('csv', 'pkl'))
        
        # keep only players in players list
        player_names = [player.player_name for player in players]
        rankings_df = rankings_df[rankings_df['PLAYER NAME'].isin(player_names)].reset_index(drop=True)
        
        # save to csv
        if file_path is not None:
            rankings_df.to_csv(file_path, index=False)
        return rankings_df

    def team_performance_summary(self, team_roster: List[str], season: str = '2025-26', month: str = '', stat: str = 'VOR', pickle_filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Get a summary of team performance based on a list of player names.

        Args:
            team_roster (List[str]): List of player full names.

        Returns:
            pd.DataFrame: DataFrame summarizing the team's performance.
        """
        if pickle_filepath is not None and os.path.exists(pickle_filepath):
            all_players_summary_df = pd.read_pickle(pickle_filepath)
        else:
            all_active_players = self.get_all_active_players(season=season)
            all_players_summary = []
            for player in all_active_players:
                summary = self.fantasy_league.get_player_summary(player, season, month=month)
                if summary.get('WEEKS PLAYED', 0) == 0:
                    continue
                all_players_summary.append(summary)
            all_players_summary_df = pd.DataFrame(all_players_summary).fillna(0)
            if pickle_filepath is not None:
                all_players_summary_df.to_pickle(pickle_filepath)
        
        # Recompute VOR for team players using replacement levels from all players
        repl_levels = self.get_replacement_levels_market(all_players_summary_df)
        print(f"Replacement Levels: {repl_levels}")
        team_summary_df = all_players_summary_df[all_players_summary_df['PLAYER NAME'].isin(team_roster)].reset_index(drop=True)
        team_summary_df = self.compute_vor(team_summary_df, repl_levels=repl_levels).sort_values(by=stat, ascending=False).reset_index(drop=True)
        
        # assign the highest VOR player to each starting position using networkx maximum bipartite matching
        B = nx.Graph()
        positions = ['Guard1', 'Guard2', 'Forward1', 'Forward2', 'Center']
        for pos in positions:
            B.add_node(pos, bipartite=0)
        for i, player in team_summary_df.iterrows():
            player_positions = player['POSITION'].split(',')
            for pos in positions:
                if any(p in pos for p in player_positions):
                    B.add_node(player['PLAYER NAME'], bipartite=1)
                    B.add_edge(pos, player['PLAYER NAME'], weight=player['VOR'])
        matching = nx.algorithms.matching.max_weight_matching(B, maxcardinality=True, weight='weight')
        starting_lineup = []
        for pos, player in matching:
            if pos in positions:
                starting_lineup.append((pos, player))
            else:
                starting_lineup.append((player, pos))
        starting_lineup_df = pd.DataFrame(starting_lineup, columns=['Position', 'Player Name'])
        starting_lineup_df = starting_lineup_df.merge(team_summary_df, left_on='Player Name', right_on='PLAYER NAME', how='left')
        starting_lineup_df = starting_lineup_df[['Position', 'Player Name', 'SEASON', 'WEEKS PLAYED', 'VOR', 'BEST_SLOT', 'POSITION', 'BASE_VALUE', 'average_weekly_max_fpts']]
        
        team_vor = starting_lineup_df['VOR'].sum()
        team_base_value = starting_lineup_df['BASE_VALUE'].sum()
        team_avg_weekly_fpts = starting_lineup_df['average_weekly_max_fpts'].sum()
        
        print(f"Team VOR: {team_vor:.2f}")
        print(f"Team BASE_VALUE: {team_base_value:.2f}")
        print(f"Team Average Weekly Max FPTS: {team_avg_weekly_fpts:.2f}")
        
        print(f"Team Summary for Season {season}:\n{team_summary_df[['PLAYER NAME', 'VOR', 'BEST_SLOT', 'BASE_VALUE', 'average_weekly_max_fpts']]}")
        
        # sort starting lineup by position by Guard, Forward, Center
        position_order = {'Guard': 1, 'Forward': 2, 'Center': 3}
        starting_lineup_df['Position_Order'] = starting_lineup_df['Position'].apply(lambda x: position_order.get(x.replace('1', '').replace('2', ''), 4))
        starting_lineup_df = starting_lineup_df.sort_values(by='Position_Order').drop(columns=['Position_Order']).reset_index(drop=True)
        
        print("\nStarting Lineup:")
        print(starting_lineup_df)
        
        print('Combined VOR: ', starting_lineup_df['VOR'].sum())
        print('Combined BASE_VALUE: ', starting_lineup_df['BASE_VALUE'].sum())
        print('Combined Average Weekly Max FPTS: ', starting_lineup_df['average_weekly_max_fpts'].sum())
        return starting_lineup_df