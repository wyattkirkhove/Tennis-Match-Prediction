import pandas as pd
import numpy as np

def general_info():
    """
    Extract basic match and player information from the main dataset.
    Returns a clean dataframe with data that will not change.
    Also includes some info that is nice to see in the final output, like player names and IDs.
    """
    df = pd.read_csv('data/mens_matches_sorted_elo.csv')
    output_df = pd.DataFrame()
    output_df['tourney_id'] = df['tourney_id']
    output_df['tourney_name'] = df['tourney_name']
    output_df['surface'] = df['surface']
    output_df['draw_size'] = df['draw_size']
    output_df['best_of'] = df['best_of']
    output_df['tourney_level'] = df['tourney_level']
    output_df['tourney_date'] = df['tourney_date']
    output_df['round'] = df['round']
    output_df['tourney_match_num'] = df['match_num']
    output_df['match_index'] = df['match_index']
    output_df['RESULT'] = df['RESULT']
    output_df['p1_name'] = df['p1_name']
    output_df['p2_name'] = df['p2_name']
    output_df['p1_id'] = df['p1_id']
    output_df['p2_id'] = df['p2_id']
    output_df['p1_hand'] = df['p1_hand']
    output_df['p2_hand'] = df['p2_hand']
    output_df['p1_ht'] = df['p1_ht']
    output_df['p2_ht'] = df['p2_ht']
    output_df['p1_age'] = df['p1_age']
    output_df['p2_age'] = df['p2_age']
    output_df['best_of'] = df['best_of']
    output_df['p1_elo_pre'] = df['p1_elo_pre']
    output_df['p2_elo_pre'] = df['p2_elo_pre']
    output_df['elo_diff'] = df['elo_diff']
    output_df['p1_surface_elo_pre'] = df['p1_surface_elo_pre']
    output_df['p2_surface_elo_pre'] = df['p2_surface_elo_pre']
    output_df['elo_diff_surface'] = df['elo_diff_surface']
    output_df['p1_rank'] = df['p1_rank']
    output_df['p2_rank'] = df['p2_rank']
    output_df['p1_rank_points'] = df['p1_rank_points']
    output_df['p2_rank_points'] = df['p2_rank_points']
    return output_df

def generate_rolling_features(df):
    """
    Generate rolling performance statistics for each player based on their recent matches.
    """

    def build_long_df(df, player_prefix, result_win_flag):
        """
        Convert wide-format match data to long format for easier rolling calculations.
        """
        columns = [
            'match_index', 'tourney_date', f'{player_prefix}_id',
            f'{player_prefix}_ace', f'{player_prefix}_df',
            f'{player_prefix}_svpt', f'{player_prefix}_1stIn', f'{player_prefix}_1stWon',
            f'{player_prefix}_2ndWon', f'{player_prefix}_SvGms',
            f'{player_prefix}_bpSaved', f'{player_prefix}_bpFaced'
        ]
        temp = df[columns + ['RESULT']].copy()
        temp.columns = [col.replace(f'{player_prefix}_', '') for col in columns] + ['RESULT']
        temp['player'] = temp['id']
        temp['is_winner'] = temp['RESULT'] if result_win_flag else 1 - temp['RESULT']
        return temp.drop(columns=['RESULT'])

    # Create long dfs for both players
    p1_long = build_long_df(df, 'p1', result_win_flag=True)
    p2_long = build_long_df(df, 'p2', result_win_flag=False)
    long_df = pd.concat([p1_long, p2_long], ignore_index=True)
    

    long_df = long_df.sort_values(by=['player', 'match_index'])
    long_df['is_first_match'] = long_df.groupby('player').cumcount() == 0

    # Define time windows and statsfor rolling calculations
    rolling_windows = [5, 10, 20, 50, 100]
    raw_stats = ['ace', 'df', 'svpt', '1stIn', '1stWon', '2ndWon', 'SvGms', 'bpSaved', 'bpFaced', 'is_winner']

    # Track how many matches each player has played in each window - this is used to calculate rolling stats. 
    # This is important because it allows us to calculate stats for players who have played less than the window size so we don't have to divide by just the window size.
    for w in rolling_windows:
        long_df[f'match_count_last_{w}'] = long_df.groupby('player').cumcount()
        long_df[f'match_count_last_{w}'] = long_df[f'match_count_last_{w}'].clip(upper=w)

    # Calculate rolling sums for each stat and time window
    for w in rolling_windows:
        for stat in raw_stats:
            colname = f'{stat}_last_{w}'
            long_df[colname] = long_df.groupby('player')[stat].transform(
                lambda x: x.shift(1).rolling(window=w, min_periods=1).sum()
            )

        # A bit of simple math to get the win percentages for first and second serves.
        long_df[f'first_serve_win_pct_last_{w}'] = long_df[f'1stWon_last_{w}'] / long_df[f'1stIn_last_{w}']
        long_df[f'second_serve_win_pct_last_{w}'] = long_df[f'2ndWon_last_{w}'] / (
            long_df[f'svpt_last_{w}'] - long_df[f'1stIn_last_{w}']
        )
        long_df[f'bp_save_pct_last_{w}'] = long_df[f'bpSaved_last_{w}'] / long_df[f'bpFaced_last_{w}']

        # Handle division by zero and infinite values
        for stat in ['first_serve_win_pct', 'second_serve_win_pct', 'bp_save_pct']:
            long_df[f'{stat}_last_{w}'] = long_df[f'{stat}_last_{w}'].replace([np.inf, -np.inf], np.nan)

        # Set features to NaN for players' first matches (no historical data)
        long_df.loc[long_df['is_first_match'], [f'{s}_last_{w}' for s in raw_stats]] = np.nan
        long_df.loc[long_df['is_first_match'], [f'{s}_last_{w}' for s in [
            'first_serve_win_pct', 'second_serve_win_pct', 'bp_save_pct'
        ]]] = np.nan

    #Return performance features
    match_players = pd.DataFrame({
        'match_index': df['match_index'],
        'p1_id': df['p1_id'],
        'p2_id': df['p2_id']
    })

    # Two copies, one for each player's perspective
    p1_perspective = match_players.copy()
    p1_perspective['player'] = p1_perspective['p1_id']
    p1_perspective['opponent'] = p1_perspective['p2_id']
    
    p2_perspective = match_players.copy()
    p2_perspective['player'] = p2_perspective['p2_id']
    p2_perspective['opponent'] = p2_perspective['p1_id']
    
    # Create return stats dataframe
    return_df = pd.DataFrame()
    return_df['match_index'] = df['match_index']
    return_df['tourney_date'] = df['tourney_date']
    
    # Get return stats from opponent's serve data for p1
    p1_return = return_df.copy()
    p1_return['player'] = df['p1_id']
    p1_return['opp_svpt'] = df['p2_svpt']
    p1_return['opp_1stIn'] = df['p2_1stIn']
    p1_return['opp_1stWon'] = df['p2_1stWon']
    p1_return['opp_2ndWon'] = df['p2_2ndWon']
    p1_return['bp_created'] = df['p2_bpFaced']  # p1 created BPs = p2 faced BPs
    p1_return['bp_converted'] = df['p2_bpFaced'] - df['p2_bpSaved']  # p1 converted BPs = p2 faced - p2 saved
    
    # Return stats for p2
    p2_return = return_df.copy()
    p2_return['player'] = df['p2_id']
    p2_return['opp_svpt'] = df['p1_svpt']
    p2_return['opp_1stIn'] = df['p1_1stIn']
    p2_return['opp_1stWon'] = df['p1_1stWon']
    p2_return['opp_2ndWon'] = df['p1_2ndWon']
    p2_return['bp_created'] = df['p1_bpFaced'] 
    p2_return['bp_converted'] = df['p1_bpFaced'] - df['p1_bpSaved']  
    
    # Combine both perspectives
    return_df = pd.concat([p1_return, p2_return], ignore_index=True)
    return_df = return_df.sort_values(by=['player', 'tourney_date', 'match_index'])

    # Merge return stats with the main long dataframe
    long_df = pd.merge(long_df, return_df, on=['match_index', 'player'], how='left')

    # Calculate rolling return performance metrics
    for w in rolling_windows:
        for stat in ['opp_svpt', 'opp_1stIn', 'opp_1stWon', 'opp_2ndWon', 'bp_created', 'bp_converted']:
            colname = f'{stat}_last_{w}'
            long_df[colname] = long_df.groupby('player')[stat].transform(
                lambda x: x.shift(1).rolling(window=w, min_periods=1).sum()
            )

        # Calculate return win percentages and break point conversion rate
        long_df[f'return_1st_win_pct_last_{w}'] = 1 - (
            long_df[f'opp_1stWon_last_{w}'] / long_df[f'opp_1stIn_last_{w}']
        )
        long_df[f'return_2nd_win_pct_last_{w}'] = 1 - (
            long_df[f'opp_2ndWon_last_{w}'] / (long_df[f'opp_svpt_last_{w}'] - long_df[f'opp_1stIn_last_{w}'])
        )
        long_df[f'bp_conversion_pct_last_{w}'] = long_df[f'bp_converted_last_{w}'] / long_df[f'bp_created_last_{w}']

        # Handle division by zero
        for stat in ['return_1st_win_pct', 'return_2nd_win_pct', 'bp_conversion_pct']:
            long_df[f'{stat}_last_{w}'] = long_df[f'{stat}_last_{w}'].replace([np.inf, -np.inf], np.nan)

    # Add surface-specific features
    surface_map = df[['match_index', 'surface']]
    long_df = long_df.merge(surface_map, on='match_index', how='left')

    # Calculate surface-specific win rates
    for w in rolling_windows:
        # Rolling win rate on same surface
        long_df[f'surface_winrate_last_{w}'] = (
            long_df
            .groupby(['player', 'surface'])['is_winner']
            .transform(lambda x: x.shift(1).rolling(window=w, min_periods=1).mean())
        )

    return long_df


start_df = pd.read_csv('data/mens_matches_sorted_elo.csv')
long_stats_df = generate_rolling_features(start_df)

# Separate rolling features for each player with appropriate prefixes
p1_features = long_stats_df.add_prefix('p1_recent_')
p2_features = long_stats_df.add_prefix('p2_recent_')

df = general_info()
# Merge rolling features back to the main dataframe
df = df.merge(
    p1_features[['p1_recent_match_index', 'p1_recent_player'] + 
                [col for col in p1_features.columns if 'last_' in col]],
    left_on=['match_index', 'p1_id'],
    right_on=['p1_recent_match_index', 'p1_recent_player'],
    how='left'
)

df = df.merge(
    p2_features[['p2_recent_match_index', 'p2_recent_player'] + 
                [col for col in p2_features.columns if 'last_' in col]],
    left_on=['match_index', 'p2_id'],
    right_on=['p2_recent_match_index', 'p2_recent_player'],
    how='left'
)

# Drop the temporary columns used for merging
df = df.drop(columns=['p1_recent_match_index', 'p1_recent_player', 
                     'p2_recent_match_index', 'p2_recent_player'])

def compute_h2h(df):
    """
    Calculate head-to-head records between players.
    For each match, computes how many times each player has beaten the other in previous meetings.
    """
    df = df.sort_values('match_index')
    df['h2h_wins_p1'] = 0
    df['h2h_wins_p2'] = 0

    h2h_counts = {}
    h2h_p1 = []
    h2h_p2 = []

    for idx, row in df.iterrows():
        p1, p2 = row['p1_id'], row['p2_id']
        match_key = tuple(sorted((p1, p2)))
        if match_key not in h2h_counts:
            h2h_counts[match_key] = []

        # Count past wins for each player in this matchup
        p1_past_wins = sum(1 for r in h2h_counts[match_key] if r == p1)
        p2_past_wins = sum(1 for r in h2h_counts[match_key] if r == p2)

        h2h_p1.append(p1_past_wins)
        h2h_p2.append(p2_past_wins)

        # Record current winner
        h2h_counts[match_key].append(p1 if row['RESULT'] == 1 else p2)

    df['h2h_wins_p1'] = h2h_p1
    df['h2h_wins_p2'] = h2h_p2
    df['h2h_win_diff'] = df['h2h_wins_p1'] - df['h2h_wins_p2']
    return df

df = compute_h2h(df)


def compute_position_agnostic_elo_growth(df):
    """
    Calculate Elo rating changes and growth trends for each player.
    Creates features that show how players' ratings have evolved over time.
    """
    df = df.sort_values(['tourney_date', 'match_index'])

    # Convert to long format to track each player's Elo history
    p1_elo = df[['match_index', 'tourney_date', 'p1_id', 'p1_elo_pre', 'p1_surface_elo_pre']].copy()
    p1_elo.columns = ['match_index', 'tourney_date', 'player_id', 'elo_pre', 'surface_elo_pre']

    p2_elo = df[['match_index', 'tourney_date', 'p2_id', 'p2_elo_pre', 'p2_surface_elo_pre']].copy()
    p2_elo.columns = ['match_index', 'tourney_date', 'player_id', 'elo_pre', 'surface_elo_pre']

    long_elo = pd.concat([p1_elo, p2_elo], ignore_index=True)
    long_elo = long_elo.sort_values(['player_id', 'tourney_date', 'match_index'])

    # Calculate Elo changes between consecutive matches
    rolling_windows = [5, 10, 20, 50, 100]
    long_elo['elo_diff_since_last'] = long_elo.groupby('player_id')['elo_pre'].diff()
    long_elo['surf_elo_diff_since_last'] = long_elo.groupby('player_id')['surface_elo_pre'].diff()

    # Calculate cumulative Elo growth over different time windows
    for w in rolling_windows:
        long_elo[f'elo_growth_{w}'] = (
            long_elo.groupby('player_id')['elo_pre']
            .transform(lambda x: x.diff().rolling(window=w, min_periods=1).sum())
        )
        long_elo[f'surf_elo_growth_{w}'] = (
            long_elo.groupby('player_id')['surface_elo_pre']
            .transform(lambda x: x.diff().rolling(window=w, min_periods=1).sum())
        )

    # Merge Elo growth features back to the main dataframe
    p1_growth = long_elo.copy()
    p1_growth.columns = ['match_index'] + ['tourney_date'] + ['p1_id'] + [
        f'p1_{col}' for col in long_elo.columns[3:]
    ]

    p2_growth = long_elo.copy()
    p2_growth.columns = ['match_index'] + ['tourney_date'] + ['p2_id'] + [
        f'p2_{col}' for col in long_elo.columns[3:]
    ]

    df = df.merge(p1_growth.drop(columns=['tourney_date']), on=['match_index', 'p1_id'], how='left')
    df = df.merge(p2_growth.drop(columns=['tourney_date']), on=['match_index', 'p2_id'], how='left')

    # Calculate the difference in Elo growth between players
    for w in rolling_windows:
        df[f'elo_growth_diff_{w}'] = df[f'p1_elo_growth_{w}'] - df[f'p2_elo_growth_{w}']
        df[f'surf_elo_growth_diff_{w}'] = df[f'p1_surf_elo_growth_{w}'] - df[f'p2_surf_elo_growth_{w}']

    df['elo_diff_change'] = df['p1_elo_diff_since_last'] - df['p2_elo_diff_since_last']
    df['surf_elo_diff_change'] = df['p1_surf_elo_diff_since_last'] - df['p2_surf_elo_diff_since_last']

    return df



df = compute_position_agnostic_elo_growth(df)

df.to_csv('data/matches_with_rolling_features.csv', index=False)
print("Complete, saved dataset with features.")
