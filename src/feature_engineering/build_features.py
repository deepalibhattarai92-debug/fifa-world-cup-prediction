"""
Build the feature dataset for the Version 1 FIFA World Cup prediction model.

Input files (data/processed/):
    clean_results.csv       - Historical international match results
    fifa_rankings.csv       - Current FIFA rankings snapshot
    elo_ratings.csv         - Current World Football Elo ratings
    world_cup_history.csv   - Per-team WC appearances, titles, best finish
    world_cup_fixtures.csv  - 2026 tournament fixture list
    former_names.csv        - Historical team name → current name mapping

Input files (data/raw/):
    elo_code_map.csv        - FIFA 3-letter country code → Elo 2-letter code

Output (data/processed/):
    features.csv            - One row per historical match with home/away features
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")

INPUT_RESULTS  = PROCESSED_DIR / "clean_results.csv"
INPUT_FIFA     = PROCESSED_DIR / "fifa_rankings.csv"
INPUT_ELO      = PROCESSED_DIR / "elo_ratings.csv"
INPUT_WC_HIST  = PROCESSED_DIR / "world_cup_history.csv"
INPUT_FORMER   = PROCESSED_DIR / "former_names.csv"
INPUT_ELO_MAP  = RAW_DIR / "elo_code_map.csv"

OUTPUT_FEATURES = PROCESSED_DIR / "features.csv"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROLLING_WINDOW = 10

# Teams whose FIFA fixture name differs from the world_cup_history.csv name.
# Teams not listed and not in WC history are genuine first-timers — they get
# zeros for all WC history columns.
FIXTURE_TO_WC_NAME: dict[str, str] = {
    "Czechia":         "Czech Republic",
    "Korea Republic":  "South Korea",
    "USA":             "United States",
    "IR Iran":         "Iran",
    "Türkiye":         "Turkey",
    "Côte d'Ivoire":   "Ivory Coast",
    "Congo DR":        "DR Congo",
}

# Default WC history values for first-time qualifiers.
# best_finish=8 means "Group stage" (the worst possible finish).
WC_HISTORY_DEFAULTS: dict[str, int] = {
    "appearances": 0,
    "titles": 0,
    "best_finish": 8,
}


# ---------------------------------------------------------------------------
# Step 1 — Load all source files
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
# Step 2 — Standardize team names in results using former_names.csv
# ---------------------------------------------------------------------------

def standardize_team_names(results: pd.DataFrame, former: pd.DataFrame) -> pd.DataFrame:
    """Apply former→current name replacements to home_team and away_team."""
    print("Standardizing team names...")
    name_map = dict(zip(former["former"], former["current"]))
    results = results.copy()
    results["home_team"] = results["home_team"].replace(name_map)
    results["away_team"] = results["away_team"].replace(name_map)

    replaced = (results["home_team"] != results["home_team"]).sum()
    print(f"  Name mappings applied: {len(name_map)}")
    return results


# ---------------------------------------------------------------------------
# Step 3 — Compute rolling form features
# ---------------------------------------------------------------------------

def build_form_features(results: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape results into one row per team per match, compute rolling form,
    then return a team-match lookup keyed by (date, team).

    Rolling window uses shift(1) to prevent data leakage: the form value for
    match N reflects only matches 0…N-1 for that team.
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
    grp = team_matches.groupby("team", sort=False)

    # Shift within each team group first, then roll on the shifted series.
    for col in roll_cols:
        shifted = grp[col].shift(1)
        team_matches[f"{col}_rolled"] = (
            shifted.groupby(team_matches["team"])
            .transform(lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean())
        )

    team_matches["form_win_pct"]         = team_matches["win_rolled"]
    team_matches["form_goals_scored"]    = team_matches["goals_for_rolled"]
    team_matches["form_goals_conceded"]  = team_matches["goals_against_rolled"]
    team_matches["form_goal_diff"]       = (
        team_matches["goals_for_rolled"] - team_matches["goals_against_rolled"]
    )
    team_matches["form_clean_sheet_pct"] = team_matches["clean_sheet_rolled"]

    form_cols = [
        "date", "team",
        "form_win_pct", "form_goals_scored", "form_goals_conceded",
        "form_goal_diff", "form_clean_sheet_pct",
    ]
    print(f"  Team-match rows: {len(team_matches):,}")
    return team_matches[form_cols]


# ---------------------------------------------------------------------------
# Step 4 — Build static feature lookups (FIFA, Elo, WC history)
# ---------------------------------------------------------------------------

def build_fifa_lookup(fifa: pd.DataFrame) -> pd.DataFrame:
    """Return a team → FIFA rank/points/confederation lookup."""
    return fifa[["team", "fifa_rank", "fifa_points", "confederation"]].copy()


def build_elo_lookup(elo: pd.DataFrame, elo_map: pd.DataFrame) -> pd.DataFrame:
    """
    Join the Elo code map to Elo ratings to produce a team → elo_rating lookup.
    The elo_code_map.csv covers all 48 World Cup teams with manually verified codes.
    """
    lookup = elo_map.merge(
        elo[["country_code", "elo_rating"]],
        left_on="elo_code",
        right_on="country_code",
        how="left",
    )
    missing = lookup[lookup["elo_rating"].isna()]["team"].tolist()
    if missing:
        raise ValueError(f"Elo rating missing for: {missing}")
    return lookup[["team", "elo_rating"]]


def build_wc_history_lookup(wc_history: pd.DataFrame) -> pd.DataFrame:
    """
    Build a team → WC history lookup using the fixture name as key.
    Teams with name mismatches are remapped via FIXTURE_TO_WC_NAME.
    First-time qualifiers not in the source data receive default zero values.
    """
    wc = wc_history[["team", "appearances", "titles", "best_finish"]].copy()

    # Create reverse mapping: wc_name → fixture_name
    for fixture_name, wc_name in FIXTURE_TO_WC_NAME.items():
        mask = wc["team"] == wc_name
        if mask.any():
            wc.loc[mask, "team"] = fixture_name

    wc = wc.rename(columns={
        "appearances": "wc_appearances",
        "titles":      "wc_titles",
        "best_finish": "wc_best_finish",
    })
    return wc


# ---------------------------------------------------------------------------
# Step 5 — Assemble the final match-level feature dataset
# ---------------------------------------------------------------------------

def assemble_features(
    results: pd.DataFrame,
    form_features: pd.DataFrame,
    fifa_lookup: pd.DataFrame,
    elo_lookup: pd.DataFrame,
    wc_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join all features onto match results. Each row is one historical match
    with home_ and away_ prefixed feature columns.
    """
    print("Assembling match-level feature rows...")

    def _join_team_features(
        df: pd.DataFrame,
        side: str,
        form: pd.DataFrame,
        fifa: pd.DataFrame,
        elo: pd.DataFrame,
        wc: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge all features for one side (home or away) onto df."""
        team_col = f"{side}_team"

        # Rolling form (join on date + team — exact match required)
        form_renamed = form.rename(columns={
            "team":                   team_col,
            "form_win_pct":           f"{side}_form_win_pct",
            "form_goals_scored":      f"{side}_form_goals_scored",
            "form_goals_conceded":    f"{side}_form_goals_conceded",
            "form_goal_diff":         f"{side}_form_goal_diff",
            "form_clean_sheet_pct":   f"{side}_form_clean_sheet_pct",
        })
        df = df.merge(form_renamed, on=["date", team_col], how="left")

        # FIFA ranking
        fifa_renamed = fifa.rename(columns={
            "team":          team_col,
            "fifa_rank":     f"{side}_fifa_rank",
            "fifa_points":   f"{side}_fifa_points",
            "confederation": f"{side}_confederation",
        })
        df = df.merge(fifa_renamed, on=team_col, how="left")

        # Elo rating
        elo_renamed = elo.rename(columns={
            "team":       team_col,
            "elo_rating": f"{side}_elo_rating",
        })
        df = df.merge(elo_renamed, on=team_col, how="left")

        # WC history
        wc_renamed = wc.rename(columns={
            "team":            team_col,
            "wc_appearances":  f"{side}_wc_appearances",
            "wc_titles":       f"{side}_wc_titles",
            "wc_best_finish":  f"{side}_wc_best_finish",
        })
        df = df.merge(wc_renamed, on=team_col, how="left")

        # First-time qualifiers get default zero WC history
        df[f"{side}_wc_appearances"] = df[f"{side}_wc_appearances"].fillna(WC_HISTORY_DEFAULTS["appearances"])
        df[f"{side}_wc_titles"]      = df[f"{side}_wc_titles"].fillna(WC_HISTORY_DEFAULTS["titles"])
        df[f"{side}_wc_best_finish"] = df[f"{side}_wc_best_finish"].fillna(WC_HISTORY_DEFAULTS["best_finish"])

        return df

    features = results.copy()
    features = _join_team_features(features, "home", form_features, fifa_lookup, elo_lookup, wc_lookup)
    features = _join_team_features(features, "away", form_features, fifa_lookup, elo_lookup, wc_lookup)

    # Derived match-level features
    features["rank_diff"]     = features["home_fifa_rank"]   - features["away_fifa_rank"]
    features["elo_diff"]      = features["home_elo_rating"]  - features["away_elo_rating"]
    features["points_diff"]   = features["home_fifa_points"] - features["away_fifa_points"]
    features["same_conf"]     = (features["home_confederation"] == features["away_confederation"]).astype(int)

    # Target variable
    features["result"] = np.where(
        features["home_score"] > features["away_score"], "home_win",
        np.where(features["home_score"] < features["away_score"], "away_win", "draw")
    )

    print(f"  Feature rows: {len(features):,}")
    print(f"  Feature columns: {len(features.columns)}")
    return features


# ---------------------------------------------------------------------------
# Step 6 — Validate and save
# ---------------------------------------------------------------------------

def validate_features(features: pd.DataFrame) -> None:
    """Warn or raise depending on null rates in critical columns."""
    # Form and result columns must be almost fully populated
    strict = ["home_form_win_pct", "away_form_win_pct", "result"]
    # Rankings are a current snapshot — historical teams may not appear
    lenient = ["home_fifa_rank", "away_fifa_rank", "home_elo_rating", "away_elo_rating"]

    for col in strict:
        null_pct = features[col].isna().mean() * 100
        if null_pct > 5:
            raise ValueError(f"Column '{col}' is {null_pct:.1f}% null — check join logic.")

    for col in lenient:
        null_pct = features[col].isna().mean() * 100
        status = "ok" if null_pct < 30 else "high — expected for historical/defunct teams"
        print(f"  {col}: {null_pct:.1f}% null ({status})")


def save_features(features: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_FEATURES, index=False)
    print(f"Saved: {OUTPUT_FEATURES}  ({len(features):,} rows × {len(features.columns)} cols)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_features() -> None:
    results, fifa, elo, wc_history, former, elo_map = load_data()

    results        = standardize_team_names(results, former)
    form_features  = build_form_features(results)
    fifa_lookup    = build_fifa_lookup(fifa)
    elo_lookup     = build_elo_lookup(elo, elo_map)
    wc_lookup      = build_wc_history_lookup(wc_history)
    features       = assemble_features(results, form_features, fifa_lookup, elo_lookup, wc_lookup)

    validate_features(features)
    save_features(features)


if __name__ == "__main__":
    build_features()
