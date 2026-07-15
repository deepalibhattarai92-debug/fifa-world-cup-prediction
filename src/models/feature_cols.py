"""Model feature columns — single source of truth for train/eval/simulation."""

FEATURE_COLS_V2 = [
    # Rolling form
    "home_form_win_pct", "home_form_goals_scored", "home_form_goals_conceded",
    "home_form_goal_diff", "home_form_clean_sheet_pct",
    "away_form_win_pct", "away_form_goals_scored", "away_form_goals_conceded",
    "away_form_goal_diff", "away_form_clean_sheet_pct",
    # FIFA rankings
    "home_fifa_rank", "home_fifa_points",
    "away_fifa_rank", "away_fifa_points",
    # Elo
    "home_elo_rating", "away_elo_rating",
    # WC history
    "home_wc_appearances", "home_wc_titles", "home_wc_best_finish",
    "away_wc_appearances", "away_wc_titles", "away_wc_best_finish",
    # Derived
    "rank_diff", "points_diff", "elo_diff", "same_conf",
    # Match context
    "neutral", "match_importance",
    # Head-to-head
    "h2h_win_rate", "h2h_goal_diff",
]

# V3 final-phase: within-tournament path features (FIFA World Cup editions).
TOURNAMENT_PATH_COLS = [
    "home_tournament_matches", "away_tournament_matches",
    "home_tournament_win_pct", "away_tournament_win_pct",
    "home_tournament_goal_diff", "away_tournament_goal_diff",
    "home_days_rest", "away_days_rest",
    "is_knockout",
    "tournament_matches_diff", "days_rest_diff",
]

FEATURE_COLS = FEATURE_COLS_V2 + TOURNAMENT_PATH_COLS
