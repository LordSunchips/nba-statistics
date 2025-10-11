from typing import Dict, List
import pandas as pd
from fantasy_basketball.player import Player
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players as players_static

Z_SCORES = {
    80: 0.8416,
    90: 1.2816,
    95: 1.645,
    99: 2.326
}

def get_all_active_players() -> List[Player]:
    """
    Get a list of all active players
    :return dict with id, full_name, first_name, last_name, is_active.
    """
    players = players_static.get_active_players()
    return [Player(p['full_name']) for p in players]

class FantasyLeague:
    """
    Define an instance of a Fantasy League
    """
    def __init__(self, scoring_rules: Dict[str, float]):
        """
        Create an instance of a FantasyLegaue

        Args:
            scoring_rules (Dict[str, float]): a dictionary with 
                                              keys being the stats being tracked to compute 
                                              fantasy points and values being their multipliers on
                                              raw statistics.
        """
        self.scoring_rules = scoring_rules
    
    def get_player_fantasy_pts_by_game(self, player: Player, season: str = '2024-25') -> pd.DataFrame:
        """
        Get a gamelog for a player's games for a specified season and their fantasy points by game.

        Args:
            player (Player): a Player object
            season (str, optional): season id. Defaults to '2024-25'.

        Returns:
            pd.DataFrame: Columns are what comes with a gamelog and 'Fantasy Points'
        """
        gamelog = player.player_gamelogs[season].copy()
        original_columns = gamelog.columns
        # TODO: implement support for point and stat bonuses
        for key, value in self.scoring_rules.items():
            if key in gamelog.columns:
                gamelog[f"Fantasy {key}"] = gamelog[key] * value
            elif key == 'Field Goals Missed':
                gamelog[f'Fantasy Field Goals Missed'] = (gamelog['FGA'] - gamelog['FGM']) * value
            elif key == 'Free Throws Missed':
                gamelog['Fantasy Free Throws Missed'] = (gamelog['FTA'] - gamelog['FTM']) * value
            elif key == 'Double Double':
                gamelog['Fantasy Double Double'] = gamelog.apply(
                    lambda r: value if sum([
                        r['PTS'] >= 10,
                        r['REB'] >= 10,
                        r['AST'] >= 10,
                        r['STL'] >= 10,
                        r['BLK'] >= 10
                    ]) >= 2 else 0, axis=1
                )
            elif key == 'Triple Double':
                gamelog['Fantasy Triple Double'] = gamelog.apply(
                    lambda r: value if sum([
                        r['PTS'] >= 10,
                        r['REB'] >= 10,
                        r['AST'] >= 10,
                        r['STL'] >= 10,
                        r['BLK'] >= 10
                    ]) >= 3 else 0, axis=1
                )
            elif key == '40+ Points':
                gamelog['Fantasy 40+ Points'] = gamelog.apply(
                    lambda r: value if r['PTS'] >= 40 else 0, axis=1
                )
            elif key == '50+ Points':
                gamelog['Fantasy 50+ Points'] = gamelog.apply(
                    lambda r: value if r['PTS'] >= 50 else 0, axis=1
                )
            elif key == '15+ Assists':
                gamelog['Fantasy 15+ Assists'] = gamelog.apply(
                    lambda r: value if r['AST'] >= 15 else 0, axis=1
                )
            elif key == '20+ Rebounds':
                gamelog['Fantasy 20+ Rebounds'] = gamelog.apply(
                    lambda r: value if r['REB'] >= 20 else 0, axis=1
                )
            else:
                raise KeyError(f'unknown column name passed: {key}')
        scoring_columns = [f'Fantasy {stat}' for stat in self.scoring_rules.keys()]
        gamelog['FPTS'] = gamelog[scoring_columns].sum(axis=1)
        return gamelog[original_columns.tolist() + ['FPTS']]

    def get_max_fantasy_points_by_week(self, player: Player, season: str = '2024-25', month: str = '', week_start_day: str = "MON") -> pd.DataFrame:
        """
        Get the maximum fantasy points in a game per a week for a given Player for a given season.

        Args:
            player (Player): player object
            season (str, optional): season id. Defaults to '2024-25'.
            week_start_day (str, optional): the start day of the week. Defaults to "MON".

        Returns:
            pd.DataFrame: A DataFrame that is grouped by weeks and accumulated using a max function.
        """
        
        # Week normalization to `week_start_day`
        week_anchor_map = dict(MON="W-MON", TUE="W-TUE", WED="W-WED", THU="W-THU",
                               FRI="W-FRI", SAT="W-SAT", SUN="W-SUN")
        anchor = week_anchor_map[week_start_day.upper()]

        player_gamelog = self.get_player_fantasy_pts_by_game(player, season)
        player_gamelog["GAME_DATE"] = pd.to_datetime(player_gamelog["GAME_DATE"])
        week_period = player_gamelog["GAME_DATE"].dt.to_period(anchor)
        player_gamelog["WEEK_START"] = player_gamelog["GAME_DATE"] - pd.to_timedelta(player_gamelog["GAME_DATE"].dt.weekday, unit="D")
        player_gamelog["WEEK_START"] = player_gamelog["WEEK_START"].dt.normalize()
        
        # Group by the WEEK_START column and take a max of the FPTS column for each group
        max_weekly_gamelog = player_gamelog.groupby("WEEK_START", as_index=True)["FPTS"].max().sort_index().reset_index()
        average_weekly_gamelog = player_gamelog.groupby("WEEK_START", as_index=True)["FPTS"].mean().sort_index().reset_index()
        return max_weekly_gamelog, average_weekly_gamelog

    def get_player_summary(self, player: Player, season: str = '2024-25', month: str = '', ignore_dates: tuple = None) -> Dict:
        max_fantasy_points_by_week_df, average_fantasy_points_by_week_df = self.get_max_fantasy_points_by_week(player, season, month)
        if ignore_dates:
            start_date, end_date = ignore_dates
            max_fantasy_points_by_week_df = max_fantasy_points_by_week_df[~((max_fantasy_points_by_week_df['WEEK_START'] >= start_date) & (max_fantasy_points_by_week_df['WEEK_START'] <= end_date))]
        max_fpts = float(max_fantasy_points_by_week_df['FPTS'].max()) if not max_fantasy_points_by_week_df.empty else 0.0
        mean = float(max_fantasy_points_by_week_df['FPTS'].mean()) if max_fantasy_points_by_week_df.size >= 2 else 0.0
        std = float(max_fantasy_points_by_week_df['FPTS'].std(ddof=1)) if max_fantasy_points_by_week_df.size >= 2 else 0.0
        variance = float(max_fantasy_points_by_week_df['FPTS'].var(ddof=1)) if max_fantasy_points_by_week_df.size >= 2 else 0.0
        consistency_index = 1 - std / mean if mean != 0 else 0.0
        ceiling = min(max_fpts, mean + std * Z_SCORES[90]) if max_fpts > 0 else 0.0
        availability_rate = min(player.games_played.get(season, 0) / 23.0, 1.0) # assuming a 23 week season
        average_weekly_average = float(average_fantasy_points_by_week_df['FPTS'].mean()) if not average_fantasy_points_by_week_df.empty else 0.0
        median_weekly_average = float(average_fantasy_points_by_week_df['FPTS'].median()) if not average_fantasy_points_by_week_df.empty else 0.0
                
                
        summary = {
            'average_weekly_max_fpts': round(mean, 3),
            'median_weekly_max_fpts': round(float(max_fantasy_points_by_week_df['FPTS'].median()), 3),
            'max_fpts': round(float(max_fantasy_points_by_week_df['FPTS'].max()), 3),
            'std_weekly_max_fpts': std if max_fantasy_points_by_week_df.size >= 2 else None,
            'consistency_index': 1 - std / mean if mean != 0 else None,
            'quartile_25_weekly_max_fpts': round(float(max_fantasy_points_by_week_df['FPTS'].quantile(0.25)), 3),
            'quartile_75_weekly_max_fpts': round(float(max_fantasy_points_by_week_df['FPTS'].quantile(0.75)), 3),
            'average_weekly_average_fpts': round(average_weekly_average, 3),
            'median_weekly_average_fpts': round(median_weekly_average, 3),
        }
        return {
            'PLAYER ID': player.player_id,
            'PLAYER NAME': player.player_name,
            'SEASON': season,
            'WEEKS PLAYED': max_fantasy_points_by_week_df.shape[0],
            'POSITION': ','.join(player.positions) if player.positions else "Unknown",
            'BASE_VALUE': 0.6 * ceiling + 0.25 * mean + 0.15 * (consistency_index * mean * availability_rate),
            **summary
        }
                