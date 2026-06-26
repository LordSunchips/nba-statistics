import os
import pandas as pd

from pathlib import Path
from fantasy_basketball.player import Player, ACTIVE_PLAYERS
from fantasy_basketball.scoring import ScoringRules
from fantasy_basketball.vor import LeagueSettings, VORCalculator, compute_base_value
from fantasy_basketball.analysis import generate_vor_rankings, random_snake_draft, simulate_fantasy_season

SEASON = '2024-25'
ASSETS = Path('src/fantasy_basketball/assets') / SEASON

gamelogs = {}
for csv in ASSETS.glob('*csv'):
    df = pd.read_csv(csv)
    if not df.empty:
        gamelogs[csv.stem] = df
        
# Build a positions dict from ACTIVE_PLAYERS + Sleeper data for the names we have
# Quick version: instantiate only players we have cached logs for
from fantasy_basketball.player import Player
positions = {}
for name in list(gamelogs.keys()):
    try:
        p = Player(name)
        positions[name] = p.positions
    except ValueError:
        positions[name] = []   # name in cache but not in active roster (retired mid-season)

rules = ScoringRules.boydfriends()

league = LeagueSettings(
    num_teams=8,
    roster_spots={"G": 2, "F": 2, "C": 1},
    scoring_rules=rules,
    season_games=82,
    risk_aversion=0.1,
)

calc = VORCalculator(league)

rankings = generate_vor_rankings(gamelogs, positions, league)

# Full positional-agnostic ranking
print(rankings.head(30).to_string(index=False))

# Just guards, sorted by position rank
print(rankings[rankings["primary_position"] == "G"].head(15).to_string(index=False))

# Dual-position eligible players
print(rankings[rankings["positions"].str.contains("/")].head(10).to_string(index=False))
