"""
Tennis Match Analysis: Adding Elo Ratings
This script simply sorts the matches and calculates Elo ratings for tennis players based on match outcomes.
Elo ratings help measure player skill levels relative to each other.
"""

import pandas as pd
from collections import defaultdict

def get_mens_matches_sorted():
    """
    Load ATP match data and sort chronologically.
    """
    path = "data/matches.csv"

    df = pd.read_csv(path)

    # Filter out Davis Cup matches (D), and Tour Finals (F), as these have different formats.
    # G=Grand Slam, M=Masters, A=ATP
    mens_matches = df[df['tourney_level'].isin(['G', 'M', 'A'])]

    # Convert date format and ensure proper sorting
    mens_matches['tourney_date'] = pd.to_datetime(mens_matches['tourney_date'], format='%Y%m%d')
    mens_matches['match_num'] = pd.to_numeric(mens_matches['match_num'], errors='coerce')

    # Sort by date, tournament, and match number to ensure chronological order
    matches_sorted = mens_matches.sort_values(by=['tourney_date', 'tourney_id', 'match_num']).reset_index(drop=True)
    matches_sorted['match_index'] = matches_sorted.index

    return matches_sorted

df_sorted = get_mens_matches_sorted()

"""
Elo Rating System Explanation:
- Elo ratings measure relative skill between players
- Higher rating = better player
- Ratings change based on expected vs actual match outcomes
- K-factor (32) controls how much ratings change after each match
- Calculate both overall ratings and surface-specific ratings
"""

# Elo calculation helper functions
def expected_score(rating_a, rating_b):
    """
    Calculate the expected probability of player A beating player B.
    Returns a value between 0 and 1.
    """
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(winner_elo, loser_elo, k=32):
    """
    Update Elo ratings after a match.
    Winner gains points, loser loses points based on expected outcome.
    """
    expected_win = expected_score(winner_elo, loser_elo)
    winner_new = winner_elo + k * (1 - expected_win) 
    loser_new = loser_elo + k * (0 - (1 - expected_win)) 
    return winner_new, loser_new

# Initialize Elo rating dictionaries
# All players start with 1500 rating
overall_elo = defaultdict(lambda: 1500) 
surface_elo = {
    'Hard': defaultdict(lambda: 1500),  
    'Clay': defaultdict(lambda: 1500),  
    'Grass': defaultdict(lambda: 1500),  
    'Carpet': defaultdict(lambda: 1500),  # Carpet isn't used on ATP tour anymore, but exists in historical data
}

# Lists to store Elo features for each match, we store ratings BEFORE the match for prediction features
p1_overall_elo_pre = []
p2_overall_elo_pre = []
p1_surface_elo_pre = []
p2_surface_elo_pre = []
elo_diff_surface = []
elo_diff_overall = []

# Process each match to update Elo ratings
K = 32  # K-factor for rating updates

for idx, row in df_sorted.iterrows():
    p1 = row['p1_id']  
    p2 = row['p2_id']  
    result = row['RESULT'] 
    surface = row['surface']  

    # Get current Elo ratings for both players
    p1_o_elo = overall_elo[p1]
    p2_o_elo = overall_elo[p2]

    # Surface-specific ratings
    elo_dict = surface_elo.get(surface, surface_elo['Hard'])  # fallback to hard court if surface not found
    p1_s_elo = elo_dict[p1]
    p2_s_elo = elo_dict[p2]

    # Store Elo features BEFORE the match (these will be used for predictions)
    p1_overall_elo_pre.append(p1_o_elo)
    p2_overall_elo_pre.append(p2_o_elo)
    p1_surface_elo_pre.append(p1_s_elo)
    p2_surface_elo_pre.append(p2_s_elo)
    elo_diff_overall.append(p1_o_elo - p2_o_elo)  # Positive if p1 is rated higher
    elo_diff_surface.append(p1_s_elo - p2_s_elo)  # Surface-specific rating difference

    # Update Elo ratings based on match result
    if result == 1:
        new_p1_o, new_p2_o = update_elo(p1_o_elo, p2_o_elo, K)
        new_p1_s, new_p2_s = update_elo(p1_s_elo, p2_s_elo, K)
    else: 
        new_p2_o, new_p1_o = update_elo(p2_o_elo, p1_o_elo, K)
        new_p2_s, new_p1_s = update_elo(p2_s_elo, p1_s_elo, K)

    # Store the updated ratings
    overall_elo[p1] = new_p1_o
    overall_elo[p2] = new_p2_o
    elo_dict[p1] = new_p1_s
    elo_dict[p2] = new_p2_s

# Add all the calculated Elo features to the DataFrame and save
df_sorted['p1_elo_pre'] = p1_overall_elo_pre
df_sorted['p2_elo_pre'] = p2_overall_elo_pre
df_sorted['elo_diff'] = elo_diff_overall

df_sorted['p1_surface_elo_pre'] = p1_surface_elo_pre
df_sorted['p2_surface_elo_pre'] = p2_surface_elo_pre
df_sorted['elo_diff_surface'] = elo_diff_surface

df_sorted.to_csv("data/mens_matches_sorted_elo.csv", index=False)