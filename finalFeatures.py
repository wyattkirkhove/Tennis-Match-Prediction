import pandas as pd
"""
Final feature creation function for the model.
"""
def final_features():
    df = pd.read_csv('data/matches_with_rolling_features.csv')
    features = {}

    # Static columns
    features['tourney_id'] = df['tourney_id']
    features['tourney_name'] = df['tourney_name']
    features['surface'] = df['surface']
    features['draw_size'] = df['draw_size']
    features['best_of'] = df['best_of']
    features['tourney_level'] = df['tourney_level']
    features['tourney_date'] = df['tourney_date']
    features['round'] = df['round']
    features['match_index'] = df['match_index']
    features['RESULT'] = df['RESULT']
    features['p1_name'] = df['p1_name']
    features['p2_name'] = df['p2_name']
    features['p1_rank'] = df['p1_rank']
    features['p2_rank'] = df['p2_rank']
    features['rank_diff'] = df['p1_rank'] - df['p2_rank']
    features['p1_rank_points'] = df['p1_rank_points']
    features['p2_rank_points'] = df['p2_rank_points']
    features['p1_rank_points_diff'] = df['p1_rank_points'] - df['p2_rank_points']
    features['p1_ht'] = df['p1_ht']
    features['p2_ht'] = df['p2_ht']
    features['height_diff'] = df['p1_ht'] - df['p2_ht']
    features['p1_age'] = df['p1_age']
    features['p2_age'] = df['p2_age']
    features['age_diff'] = df['p1_age'] - df['p2_age']
    features['p1_elo_pre'] = df['p1_elo_pre_x']
    features['p2_elo_pre'] = df['p2_elo_pre_x']
    features['elo_diff'] = df['elo_diff']
    features['p1_surface_elo_pre'] = df['p1_surface_elo_pre_x']
    features['p2_surface_elo_pre'] = df['p2_surface_elo_pre_x']
    features['elo_diff_surface'] = df['elo_diff_surface']

    rolling_windows = [5, 10, 20, 50, 100]
    for w in rolling_windows:
        features[f'1stIn_pct_diff_last_{w}'] = (df[f'p1_recent_1stIn_last_{w}']/df[f'p1_recent_svpt_last_{w}']) - (df[f'p2_recent_1stIn_last_{w}']/df[f'p2_recent_svpt_last_{w}'])
        features[f'1stWon_pct_diff_last_{w}'] = df[f'p1_recent_first_serve_win_pct_last_{w}'] - df[f'p2_recent_first_serve_win_pct_last_{w}']
        features[f'2ndWon_pct_diff_last_{w}'] = df[f'p1_recent_second_serve_win_pct_last_{w}'] - df[f'p2_recent_second_serve_win_pct_last_{w}']
        features[f'bpSaved_pct_diff_last_{w}'] = (df[f'p1_recent_bpSaved_last_{w}']/df[f'p1_recent_bpFaced_last_{w}']) - (df[f'p2_recent_bpSaved_last_{w}']/df[f'p2_recent_bpFaced_last_{w}'])
        features[f'p1_bpFaced_per_match_last_{w}'] = df[f'p1_recent_bpFaced_last_{w}']/df[f'p1_recent_match_count_last_{w}']
        features[f'p2_bpFaced_per_match_last_{w}'] = df[f'p2_recent_bpFaced_last_{w}']/df[f'p2_recent_match_count_last_{w}']
        features[f'bpConverted_pct_diff_last_{w}'] = df[f'p1_recent_bp_conversion_pct_last_{w}'] - df[f'p2_recent_bp_conversion_pct_last_{w}']
        features[f'p1_bpCreated_per_match_last_{w}'] = df[f'p1_recent_bp_created_last_{w}']/df[f'p1_recent_match_count_last_{w}']
        features[f'p2_bpCreated_per_match_last_{w}'] = df[f'p2_recent_bp_created_last_{w}']/df[f'p2_recent_match_count_last_{w}']
        features[f'ace_per_match_diff_last_{w}'] = (df[f'p1_recent_ace_last_{w}']/df[f'p1_recent_match_count_last_{w}']) - (df[f'p2_recent_ace_last_{w}']/df[f'p2_recent_match_count_last_{w}'])
        features[f'df_per_match_diff_last_{w}'] = (df[f'p1_recent_df_last_{w}']/df[f'p1_recent_match_count_last_{w}']) - (df[f'p2_recent_df_last_{w}']/df[f'p2_recent_match_count_last_{w}'])
        features[f'p1_recent_winrate_last_{w}'] = df[f'p1_recent_is_winner_last_{w}']/df[f'p1_recent_match_count_last_{w}']
        features[f'p2_recent_winrate_last_{w}'] = df[f'p2_recent_is_winner_last_{w}']/df[f'p2_recent_match_count_last_{w}']
        features[f'winrate_diff_last_{w}'] = features[f'p1_recent_winrate_last_{w}'] - features[f'p2_recent_winrate_last_{w}']
        features[f'elo_growth_diff_{w}'] = df[f'elo_growth_diff_{w}']
        features[f'surface_winrate_diff_last_{w}'] = df[f'p1_recent_surface_winrate_last_{w}'] - df[f'p2_recent_surface_winrate_last_{w}']
        features[f'p1_recent_return_points_won_last_{w}'] = df[f'p1_recent_opp_svpt_last_{w}'] - df[f'p1_recent_opp_1stWon_last_{w}'] - df[f'p1_recent_opp_2ndWon_last_{w}']
        features[f'p2_recent_return_points_won_last_{w}'] = df[f'p2_recent_opp_svpt_last_{w}'] - df[f'p2_recent_opp_1stWon_last_{w}'] - df[f'p2_recent_opp_2ndWon_last_{w}']
        features[f'return_points_won_diff_last_{w}'] = features[f'p1_recent_return_points_won_last_{w}'] - features[f'p2_recent_return_points_won_last_{w}']
        features[f'p1_recent_1st_return_points_won_last_{w}'] = df[f'p1_recent_opp_1stIn_last_{w}'] - df[f'p1_recent_opp_1stWon_last_{w}']
        features[f'p2_recent_1st_return_points_won_last_{w}'] = df[f'p2_recent_opp_1stIn_last_{w}'] - df[f'p2_recent_opp_1stWon_last_{w}']
        features[f'1st_return_points_won_diff_last_{w}'] = features[f'p1_recent_1st_return_points_won_last_{w}'] - features[f'p2_recent_1st_return_points_won_last_{w}']
        features[f'p1_recent_return_pts_won_pct_last_{w}'] = df[f'p1_recent_opp_svpt_last_{w}'] - df[f'p1_recent_opp_1stWon_last_{w}'] - df[f'p1_recent_opp_2ndWon_last_{w}']
        features[f'p2_recent_return_pts_won_pct_last_{w}'] = df[f'p2_recent_opp_svpt_last_{w}'] - df[f'p2_recent_opp_1stWon_last_{w}'] - df[f'p2_recent_opp_2ndWon_last_{w}']
        features[f'return_pts_won_pct_diff_last_{w}'] = features[f'p1_recent_return_pts_won_pct_last_{w}'] - features[f'p2_recent_return_pts_won_pct_last_{w}']
        features[f'p1_recent_return_1st_win_pct_last_{w}'] = df[f'p1_recent_opp_1stIn_last_{w}'] - df[f'p1_recent_opp_1stWon_last_{w}']
        features[f'p2_recent_return_1st_win_pct_last_{w}'] = df[f'p2_recent_opp_1stIn_last_{w}'] - df[f'p2_recent_opp_1stWon_last_{w}']
        features[f'return_1st_win_pct_diff_last_{w}'] = features[f'p1_recent_return_1st_win_pct_last_{w}'] - features[f'p2_recent_return_1st_win_pct_last_{w}']
        features[f'p1_recent_return_2nd_win_pct_last_{w}'] = (df[f'p1_recent_opp_svpt_last_{w}'] - df[f'p1_recent_opp_1stIn_last_{w}']) - df[f'p1_recent_opp_2ndWon_last_{w}']
        features[f'p2_recent_return_2nd_win_pct_last_{w}'] = (df[f'p2_recent_opp_svpt_last_{w}'] - df[f'p2_recent_opp_1stIn_last_{w}']) - df[f'p2_recent_opp_2ndWon_last_{w}']
        features[f'return_2nd_win_pct_diff_last_{w}'] = features[f'p1_recent_return_2nd_win_pct_last_{w}'] - features[f'p2_recent_return_2nd_win_pct_last_{w}']

    # H2H features
    features['h2h_wins_p1'] = df['h2h_wins_p1']
    features['h2h_wins_p2'] = df['h2h_wins_p2']
    features['total_h2h_matches'] = features['h2h_wins_p1'] + features['h2h_wins_p2']
    features['has_h2h_history'] = (features['total_h2h_matches'] > 0).astype(int)
    features['h2h_winrate_p1'] = features['h2h_wins_p1'] / features['total_h2h_matches'].replace(0, 1)  # Avoid division by zero
    features['p1_winrate_trend_100_10'] = features['p1_recent_winrate_last_100'] - features['p1_recent_winrate_last_10']
    features['p2_winrate_trend_100_10'] = features['p2_recent_winrate_last_100'] - features['p2_recent_winrate_last_10']
    features['p1_1stWon_trend_100_20'] = df['p1_recent_first_serve_win_pct_last_100'] - df['p1_recent_first_serve_win_pct_last_20']
    features['p2_1stWon_trend_100_20'] = df['p2_recent_first_serve_win_pct_last_100'] - df['p2_recent_first_serve_win_pct_last_20']
    features['p1_2ndWon_trend_100_20'] = df['p1_recent_second_serve_win_pct_last_100'] - df['p1_recent_second_serve_win_pct_last_20']
    features['p2_2ndWon_trend_100_20'] = df['p2_recent_second_serve_win_pct_last_100'] - df['p2_recent_second_serve_win_pct_last_20']
    features['p1_bpFaced_trend_100_20'] = df['p1_recent_bpFaced_last_100'] - df['p1_recent_bpFaced_last_20']
    features['p2_bpFaced_trend_100_20'] = df['p2_recent_bpFaced_last_100'] - df['p2_recent_bpFaced_last_20']
    features['p1_1st_return_points_won_trend_100_20'] = df['p1_recent_return_1st_win_pct_last_100'] - df['p1_recent_return_1st_win_pct_last_20']
    features['p2_1st_return_points_won_trend_100_20'] = df['p2_recent_return_1st_win_pct_last_100'] - df['p2_recent_return_1st_win_pct_last_20']
    features['p1_2nd_return_points_won_trend_100_20'] = df['p1_recent_return_2nd_win_pct_last_100'] - df['p1_recent_return_2nd_win_pct_last_20']
    features['p2_2nd_return_points_won_trend_100_20'] = df['p2_recent_return_2nd_win_pct_last_100'] - df['p2_recent_return_2nd_win_pct_last_20']
    features['winrate_trend_diff_100_10'] = features['p1_winrate_trend_100_10'] - features['p2_winrate_trend_100_10']
    features['1stWon_trend_diff_100_20'] = features['p1_1stWon_trend_100_20'] - features['p2_1stWon_trend_100_20']
    features['2ndWon_trend_diff_100_20'] = features['p1_2ndWon_trend_100_20'] - features['p2_2ndWon_trend_100_20']
    features['bpFaced_trend_diff_100_20'] = features['p1_bpFaced_trend_100_20'] - features['p2_bpFaced_trend_100_20']
    features['1st_return_points_won_trend_diff_100_20'] = features['p1_1st_return_points_won_trend_100_20'] - features['p2_1st_return_points_won_trend_100_20']
    features['winrate_momentum_diff'] = (features['p1_recent_winrate_last_5'] - features['p1_recent_winrate_last_20']) - (features['p2_recent_winrate_last_5'] - features['p2_recent_winrate_last_20'])
    features['surface_momentum_diff'] = (df[f'p1_recent_surface_winrate_last_5'] - df[f'p1_recent_surface_winrate_last_20']) - (df[f'p2_recent_surface_winrate_last_5'] - df[f'p2_recent_surface_winrate_last_20'])
    features['age_x_elo'] = features['age_diff'] * features['elo_diff']
    features['p1_serve_return_ratio'] = (df[f'p1_recent_first_serve_win_pct_last_20'] + df[f'p1_recent_second_serve_win_pct_last_20']) / features[f'p1_recent_return_pts_won_pct_last_20']
    features['p2_serve_return_ratio'] = (df[f'p2_recent_first_serve_win_pct_last_20'] + df[f'p2_recent_second_serve_win_pct_last_20']) / features[f'p2_recent_return_pts_won_pct_last_20']
    features['serve_return_ratio_diff'] = features['p1_serve_return_ratio'] - features['p2_serve_return_ratio']
    features['p1_surface_bias'] = df['p1_recent_surface_winrate_last_10'] - features['p1_recent_winrate_last_50'] 
    features['p2_surface_bias'] = df['p2_recent_surface_winrate_last_10'] - features['p2_recent_winrate_last_50'] 
    features['surface_bias_diff'] = features['p1_surface_bias'] - features['p2_surface_bias']

    output_df = pd.DataFrame(features)
    output_df.to_csv('data/final_features.csv', index=False)

final_features()