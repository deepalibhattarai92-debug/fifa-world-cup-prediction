"""
Build Version 2 feature dataset.

Improvements over V1:
  1. Expanded Elo code map (108 teams vs 48) — reduces Elo null from ~70% to ~30%
  2. Match importance score — WC=5, WC qualifier=4, continental=3, other comp=2, friendly=1
  3. Head-to-head features — win rate and goal diff in last 10 meetings
  4. Competitive matches only — friendlies removed from training set

Input files:
    data/processed/clean_results.csv
    data/processed/fifa_rankings.csv
    data/processed/elo_ratings.csv
    data/processed/world_cup_history.csv
    data/processed/former_names.csv
    data/raw/elo_code_map_v2.csv

Output:
    data/processed/features_v2.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_engineering.tournament_path import build_tournament_path_features


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROCESSED_DIR = Path("data/processed")
RAW_DIR       = Path("data/raw")

INPUT_RESULTS  = PROCESSED_DIR / "clean_results.csv"
INPUT_FIFA     = PROCESSED_DIR / "fifa_rankings.csv"
INPUT_ELO      = PROCESSED_DIR / "elo_ratings.csv"
INPUT_WC_HIST  = PROCESSED_DIR / "world_cup_history.csv"
INPUT_FORMER   = PROCESSED_DIR / "former_names.csv"
INPUT_ELO_MAP  = RAW_DIR / "elo_code_map_v2.csv"

OUTPUT_FEATURES = PROCESSED_DIR / "features_v2.csv"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROLLING_WINDOW = 10
H2H_WINDOW     = 10  # last N meetings for head-to-head features

# Tournament importance scores
IMPORTANCE_MAP: dict[str, int] = {
    "FIFA World Cup":                      5,
    "UEFA Euro":                           4,
    "Copa América":                        4,
    "African Cup of Nations":              4,
    "AFC Asian Cup":                       4,
    "Gold Cup":                            4,
    "CONCACAF Nations League":             3,
    "UEFA Nations League":                 3,
    "African Cup of Nations qualification": 3,
    "FIFA World Cup qualification":        3,
    "UEFA Euro qualification":             3,
    "AFC Asian Cup qualification":         3,
    "Gold Cup qualification":              3,
    "CONCACAF":                            3,
    "Copa América qualification":          3,
}
DEFAULT_COMPETITIVE_IMPORTANCE = 2
FRIENDLY_IMPORTANCE            = 1

# WC history name mismatches (fixture name → WC history name)
FIXTURE_TO_WC_NAME: dict[str, str] = {
    "Czechia":         "Czech Republic",
    "Korea Republic":  "South Korea",
    "USA":             "United States",
    "IR Iran":         "Iran",
    "Türkiye":         "Turkey",
    "Côte d'Ivoire":   "Ivory Coast",
    "Congo DR":        "DR Congo",
}

WC_HISTORY_DEFAULTS: dict[str, int] = {
    "appearances": 0,
    "titles":      0,
    "best_finish": 8,
}


# ---------------------------------------------------------------------------
# Step 1 — Load
# ---------------------------------------------------------------------------

def load_data() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    print("Loading source files...")
    results    = pd.read_csv(INPUT_RESULTS, parse_dates=["date"])
    fifa       = pd.read_csv(INPUT_FIFA)
    elo        = pd.read_csv(INPUT_ELO)
    wc_history = pd.read_csv(INPUT_WC_HIST)
    former     = pd.read_csv(INPUT_FORMER)
    elo_map    = pd.read_csv(INPUT_ELO_MAP)

    print(f"  results:    {results.shape}")
    print(f"  fifa:       {fifa.shape}")
    print(f"  elo:        {elo.shape}")
    print(f"  wc_history: {wc_history.shape}")
    print(f"  former:     {former.shape}")
    print(f"  elo_map:    {elo_map.shape}")
    return results, fifa, elo, wc_history, former, elo_map


# ---------------------------------------------------------------------------
# Step 2 — Standardize team names
# ---------------------------------------------------------------------------

def standardize_team_names(results: pd.DataFrame, former: pd.DataFrame) -> pd.DataFrame:
    name_map = dict(zip(former["former"], former["current"]))
    results = results.copy()
    results["home_team"] = results["home_team"].replace(name_map)
    results["away_team"] = results["away_team"].replace(name_map)
    print(f"  Name mappings applied: {len(name_map)}")
    return results


# ---------------------------------------------------------------------------
# Step 3 — Match importance score
# ---------------------------------------------------------------------------

def add_match_importance(results: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a numeric importance score to each match based on tournament type.
    Higher = more competitive / more meaningful for team strength assessment.
    """
    def score(tournament: str) -> int:
        if not isinstance(tournament, str):
            return FRIENDLY_IMPORTANCE
        t_lower = tournament.lower()
        if "friendly" in t_lower:
            return FRIENDLY_IMPORTANCE
        for key, val in IMPORTANCE_MAP.items():
            if key.lower() in t_lower:
                return val
        return DEFAULT_COMPETITIVE_IMPORTANCE

    results = results.copy()
    results["match_importance"] = results["tournament"].apply(score)
    dist = results["match_importance"].value_counts().sort_index()
    print("  Importance distribution:")
    for score_val, count in dist.items():
        label = {5: "WC", 4: "Major tournament", 3: "Qualifier/NL", 2: "Other competitive", 1: "Friendly"}.get(score_val, "?")
        print(f"    {score_val} ({label:20}): {count:,}")
    return results


# ---------------------------------------------------------------------------
# Step 4 — Rolling form features
# ---------------------------------------------------------------------------

def build_form_features(results: pd.DataFrame) -> pd.DataFrame:
    """
    One row per team per match, compute rolling 10-match form with shift(1).
    """
    print("Computing rolling form features...")

    home = results[["date", "home_team", "away_team", "home_score", "away_score", "neutral"]].copy()
    home.columns = ["date", "team", "opponent", "goals_for", "goals_against", "neutral"]
    home["is_home"] = True

    away = results[["date", "away_team", "home_team", "away_score", "home_score", "neutral"]].copy()
    away.columns = ["date", "team", "opponent", "goals_for", "goals_against", "neutral"]
    away["is_home"] = False

    team_matches = pd.concat([home, away], ignore_index=True)
    team_matches = team_matches.sort_values(["team", "date"]).reset_index(drop=True)

    team_matches["win"]         = (team_matches["goals_for"] > team_matches["goals_against"]).astype(int)
    team_matches["clean_sheet"] = (team_matches["goals_against"] == 0).astype(int)

    roll_cols = ["win", "goals_for", "goals_against", "clean_sheet"]
    for col in roll_cols:
        shifted = team_matches.groupby("team", sort=False)[col].shift(1)
        team_matches[f"{col}_rolled"] = (
            shifted.groupby(team_matches["team"])
            .transform(lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean())
        )

    team_matches["form_win_pct"]         = team_matches["win_rolled"]
    team_matches["form_goals_scored"]    = team_matches["goals_for_rolled"]
    team_matches["form_goals_conceded"]  = team_matches["goals_against_rolled"]
    team_matches["form_goal_diff"]       = team_matches["goals_for_rolled"] - team_matches["goals_against_rolled"]
    team_matches["form_clean_sheet_pct"] = team_matches["clean_sheet_rolled"]

    form_cols = [
        "date", "team",
        "form_win_pct", "form_goals_scored", "form_goals_conceded",
        "form_goal_diff", "form_clean_sheet_pct",
    ]
    print(f"  Team-match rows: {len(team_matches):,}")
    return team_matches[form_cols]


# ---------------------------------------------------------------------------
# Step 5 — Head-to-head features
# ---------------------------------------------------------------------------

def build_h2h_features(results: pd.DataFrame) -> pd.DataFrame:
    """
    For each match, compute head-to-head win rate and goal difference
    from the last H2H_WINDOW meetings between the two teams (prior to this match).

    Returns a DataFrame indexed by (date, home_team, away_team).
    """
    print("Computing head-to-head features...")

    df = results[["date", "home_team", "away_team", "home_score", "away_score"]].copy()
    df = df.sort_values("date").reset_index(drop=True)

    h2h_win_rate = []
    h2h_goal_diff = []

    for _, row in df.iterrows():
        t1, t2, dt = row.home_team, row.away_team, row.date

        # All prior meetings between these two teams (in either direction)
        prior = df[
            (df.date < dt) & (
                ((df.home_team == t1) & (df.away_team == t2)) |
                ((df.home_team == t2) & (df.away_team == t1))
            )
        ].tail(H2H_WINDOW)

        if len(prior) == 0:
            h2h_win_rate.append(np.nan)
            h2h_goal_diff.append(np.nan)
            continue

        wins, gd_total = 0, 0
        for _, p in prior.iterrows():
            if p.home_team == t1:
                if p.home_score > p.away_score:
                    wins += 1
                gd_total += p.home_score - p.away_score
            else:  # t1 was away
                if p.away_score > p.home_score:
                    wins += 1
                gd_total += p.away_score - p.home_score

        h2h_win_rate.append(wins / len(prior))
        h2h_goal_diff.append(gd_total / len(prior))

    df["h2h_win_rate"]  = h2h_win_rate
    df["h2h_goal_diff"] = h2h_goal_diff

    print(f"  H2H rows: {len(df):,}  null_rate={df.h2h_win_rate.isna().mean()*100:.1f}%")
    return df[["date", "home_team", "away_team", "h2h_win_rate", "h2h_goal_diff"]]


# ---------------------------------------------------------------------------
# Step 6 — Static lookups
# ---------------------------------------------------------------------------

def build_fifa_lookup(fifa: pd.DataFrame) -> pd.DataFrame:
    return fifa[["team", "fifa_rank", "fifa_points", "confederation"]].copy()


def build_elo_lookup(elo: pd.DataFrame, elo_map: pd.DataFrame) -> pd.DataFrame:
    lookup = elo_map.merge(
        elo[["country_code", "elo_rating"]],
        left_on="elo_code", right_on="country_code", how="left",
    )
    missing = lookup[lookup["elo_rating"].isna()]["team"].tolist()
    if missing:
        raise ValueError(f"Elo rating missing for: {missing}")
    return lookup[["team", "elo_rating"]]


def build_wc_history_lookup(wc_history: pd.DataFrame) -> pd.DataFrame:
    wc = wc_history[["team", "appearances", "titles", "best_finish"]].copy()
    for fixture_name, wc_name in FIXTURE_TO_WC_NAME.items():
        wc.loc[wc["team"] == wc_name, "team"] = fixture_name
    return wc.rename(columns={
        "appearances": "wc_appearances",
        "titles":      "wc_titles",
        "best_finish": "wc_best_finish",
    })


# ---------------------------------------------------------------------------
# Step 7 — Assemble match-level features
# ---------------------------------------------------------------------------

def assemble_features(
    results: pd.DataFrame,
    form_features: pd.DataFrame,
    h2h_features: pd.DataFrame,
    fifa_lookup: pd.DataFrame,
    elo_lookup: pd.DataFrame,
    wc_lookup: pd.DataFrame,
) -> pd.DataFrame:
    print("Assembling match-level feature rows...")

    def _join_side(df, side, form, fifa, elo, wc):
        tc = f"{side}_team"

        form_r = form.rename(columns={
            "team":                  tc,
            "form_win_pct":          f"{side}_form_win_pct",
            "form_goals_scored":     f"{side}_form_goals_scored",
            "form_goals_conceded":   f"{side}_form_goals_conceded",
            "form_goal_diff":        f"{side}_form_goal_diff",
            "form_clean_sheet_pct":  f"{side}_form_clean_sheet_pct",
        })
        df = df.merge(form_r, on=["date", tc], how="left")

        fifa_r = fifa.rename(columns={
            "team": tc, "fifa_rank": f"{side}_fifa_rank",
            "fifa_points": f"{side}_fifa_points", "confederation": f"{side}_confederation",
        })
        df = df.merge(fifa_r, on=tc, how="left")

        elo_r = elo.rename(columns={"team": tc, "elo_rating": f"{side}_elo_rating"})
        df = df.merge(elo_r, on=tc, how="left")

        wc_r = wc.rename(columns={
            "team": tc,
            "wc_appearances": f"{side}_wc_appearances",
            "wc_titles":      f"{side}_wc_titles",
            "wc_best_finish": f"{side}_wc_best_finish",
        })
        df = df.merge(wc_r, on=tc, how="left")

        df[f"{side}_wc_appearances"] = df[f"{side}_wc_appearances"].fillna(WC_HISTORY_DEFAULTS["appearances"])
        df[f"{side}_wc_titles"]      = df[f"{side}_wc_titles"].fillna(WC_HISTORY_DEFAULTS["titles"])
        df[f"{side}_wc_best_finish"] = df[f"{side}_wc_best_finish"].fillna(WC_HISTORY_DEFAULTS["best_finish"])

        return df

    features = results.copy()
    features = _join_side(features, "home", form_features, fifa_lookup, elo_lookup, wc_lookup)
    features = _join_side(features, "away", form_features, fifa_lookup, elo_lookup, wc_lookup)

    # Head-to-head
    features = features.merge(h2h_features, on=["date", "home_team", "away_team"], how="left")

    # Derived
    features["rank_diff"]   = features["home_fifa_rank"]   - features["away_fifa_rank"]
    features["elo_diff"]    = features["home_elo_rating"]  - features["away_elo_rating"]
    features["points_diff"] = features["home_fifa_points"] - features["away_fifa_points"]
    features["same_conf"]   = (features["home_confederation"] == features["away_confederation"]).astype(int)

    # Target
    features["result"] = np.where(
        features["home_score"] > features["away_score"], "home_win",
        np.where(features["home_score"] < features["away_score"], "away_win", "draw")
    )

    print(f"  Feature rows: {len(features):,}  columns: {len(features.columns)}")
    return features


# ---------------------------------------------------------------------------
# Step 8 — Validate and save
# ---------------------------------------------------------------------------

def validate_features(features: pd.DataFrame) -> None:
    checks = {
        "home_form_win_pct":  5.0,
        "away_form_win_pct":  5.0,
        "home_elo_rating":   50.0,   # V2 target: <50% null
        "away_elo_rating":   50.0,
        "result":             1.0,
    }
    for col, threshold in checks.items():
        null_pct = features[col].isna().mean() * 100
        status = "ok" if null_pct <= threshold else "high"
        print(f"  {col}: {null_pct:.1f}% null ({status})")


def save_features(features: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_FEATURES, index=False)
    print(f"\nSaved: {OUTPUT_FEATURES}  ({len(features):,} rows × {len(features.columns)} cols)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_features_v2() -> None:
    results, fifa, elo, wc_history, former, elo_map = load_data()

    print("\nStandardizing team names...")
    results = standardize_team_names(results, former)

    print("\nAdding match importance scores...")
    results = add_match_importance(results)

    print("\nBuilding tournament-path features (FIFA World Cup editions)...")
    results = build_tournament_path_features(results)
    wc_rows = results["tournament"].str.contains("FIFA World Cup", case=False, na=False).sum()
    ko_rows = int(results["is_knockout"].sum())
    print(f"  WC rows: {wc_rows:,}  knockout-flagged: {ko_rows:,}")

    form_features = build_form_features(results)
    h2h_features  = build_h2h_features(results)
    fifa_lookup   = build_fifa_lookup(fifa)
    elo_lookup    = build_elo_lookup(elo, elo_map)
    wc_lookup     = build_wc_history_lookup(wc_history)
    features      = assemble_features(results, form_features, h2h_features,
                                      fifa_lookup, elo_lookup, wc_lookup)

    print("\nValidating null rates:")
    validate_features(features)
    save_features(features)


if __name__ == "__main__":
    build_features_v2()
