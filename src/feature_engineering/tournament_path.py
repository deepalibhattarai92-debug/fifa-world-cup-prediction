"""
Within-tournament path features for FIFA World Cup matches.

Used by build_features_v2 (training) and tournament_state (live simulation)
so both paths share the same feature definitions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.team_names import fixture_to_results

FJELSTUL_APPEARANCES = Path("data/raw/world_cup_team_appearances_20260705.csv")

# Defaults for teams with no prior matches in the current edition.
DEFAULT_TOURNAMENT_MATCHES = 0
DEFAULT_TOURNAMENT_WIN_PCT = 0.5
DEFAULT_TOURNAMENT_GOAL_DIFF = 0.0
DEFAULT_DAYS_REST = 7.0

KNOCKOUT_MARKERS = ("round of", "quarter", "semi", "final", "third")


def is_knockout_stage(stage: str | float | None) -> int:
    if stage is None or (isinstance(stage, float) and np.isnan(stage)):
        return 0
    s = str(stage).strip().lower()
    if not s or "first stage" in s or s.startswith("group"):
        return 0
    return int(any(m in s for m in KNOCKOUT_MARKERS))


def build_fjelstul_knockout_lookup(path: Path = FJELSTUL_APPEARANCES) -> dict[frozenset, int]:
    """
    Map (date, team-pair) → is_knockout from Fjelstul WC appearances.
    One row per team; keep home-perspective rows only.
    """
    if not path.exists():
        return {}

    df = pd.read_csv(path, parse_dates=["match_date"])
    home_rows = df[df["home_team"] == 1].copy()
    lookup: dict[frozenset, int] = {}

    for _, row in home_rows.iterrows():
        d = pd.Timestamp(row["match_date"]).normalize()
        t1 = fixture_to_results(str(row["team_name"]))
        t2 = fixture_to_results(str(row["opponent_name"]))
        key = frozenset((d.strftime("%Y-%m-%d"), t1, t2))
        lookup[key] = int(row.get("knockout_stage", 0) == 1)

    return lookup


def _match_knockout_key(date, home: str, away: str) -> frozenset:
    d = pd.Timestamp(date).strftime("%Y-%m-%d")
    return frozenset((d, home, away))


def enrich_is_knockout(results: pd.DataFrame, fjelstul_lookup: dict[frozenset, int]) -> pd.Series:
    """Resolve is_knockout from stage column or Fjelstul fallback."""
    flags = []
    for _, row in results.iterrows():
        stage = row.get("stage", None)
        if pd.notna(stage) and str(stage).strip():
            flags.append(is_knockout_stage(stage))
            continue
        key = _match_knockout_key(row["date"], row["home_team"], row["away_team"])
        flags.append(fjelstul_lookup.get(key, 0))
    return pd.Series(flags, index=results.index, dtype=int)


def _team_tournament_stats(team_matches: pd.DataFrame) -> pd.DataFrame:
    """Compute pre-match cumulative stats per (tournament_year, team)."""
    tm = team_matches.sort_values(["tournament_year", "team", "date"]).copy()

    tm["win"] = (tm["goals_for"] > tm["goals_against"]).astype(float)
    tm["goal_diff"] = tm["goals_for"] - tm["goals_against"]

    grp = tm.groupby(["tournament_year", "team"], sort=False)

    tm["tournament_matches"] = grp.cumcount()
    tm["tournament_win_pct"] = grp["win"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    tm["tournament_goal_diff"] = grp["goal_diff"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )

    prev_date = grp["date"].shift(1)
    tm["days_rest"] = (tm["date"] - prev_date).dt.days
    tm["days_rest"] = tm["days_rest"].fillna(DEFAULT_DAYS_REST)

    # First match in edition → neutral defaults.
    tm.loc[tm["tournament_matches"] == 0, "tournament_win_pct"] = DEFAULT_TOURNAMENT_WIN_PCT
    tm.loc[tm["tournament_matches"] == 0, "tournament_goal_diff"] = DEFAULT_TOURNAMENT_GOAL_DIFF

    return tm


def build_tournament_path_features(results: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-side tournament-path columns for FIFA World Cup rows.
    Non-WC rows get neutral defaults.
    """
    df = results.copy()
    fjelstul = build_fjelstul_knockout_lookup()
    df["is_knockout"] = enrich_is_knockout(df, fjelstul)

    wc_mask = df["tournament"].str.contains("FIFA World Cup", case=False, na=False)
    wc = df.loc[wc_mask, ["date", "home_team", "away_team", "home_score", "away_score"]].copy()
    wc["tournament_year"] = pd.to_datetime(wc["date"]).dt.year

    home = wc[["date", "tournament_year", "home_team", "away_team", "home_score", "away_score"]].copy()
    home.columns = ["date", "tournament_year", "team", "opponent", "goals_for", "goals_against"]

    away = wc[["date", "tournament_year", "away_team", "home_team", "away_score", "home_score"]].copy()
    away.columns = ["date", "tournament_year", "team", "opponent", "goals_for", "goals_against"]

    team_rows = _team_tournament_stats(pd.concat([home, away], ignore_index=True))

    side_cols = ["date", "team", "tournament_matches", "tournament_win_pct", "tournament_goal_diff", "days_rest"]
    team_feats = team_rows[side_cols]

    home_feats = team_feats.rename(columns={
        "team": "home_team",
        "tournament_matches": "home_tournament_matches",
        "tournament_win_pct": "home_tournament_win_pct",
        "tournament_goal_diff": "home_tournament_goal_diff",
        "days_rest": "home_days_rest",
    })
    away_feats = team_feats.rename(columns={
        "team": "away_team",
        "tournament_matches": "away_tournament_matches",
        "tournament_win_pct": "away_tournament_win_pct",
        "tournament_goal_diff": "away_tournament_goal_diff",
        "days_rest": "away_days_rest",
    })

    out = df.merge(home_feats, on=["date", "home_team"], how="left")
    out = out.merge(away_feats, on=["date", "away_team"], how="left")

    fill_defaults = {
        "home_tournament_matches": DEFAULT_TOURNAMENT_MATCHES,
        "away_tournament_matches": DEFAULT_TOURNAMENT_MATCHES,
        "home_tournament_win_pct": DEFAULT_TOURNAMENT_WIN_PCT,
        "away_tournament_win_pct": DEFAULT_TOURNAMENT_WIN_PCT,
        "home_tournament_goal_diff": DEFAULT_TOURNAMENT_GOAL_DIFF,
        "away_tournament_goal_diff": DEFAULT_TOURNAMENT_GOAL_DIFF,
        "home_days_rest": DEFAULT_DAYS_REST,
        "away_days_rest": DEFAULT_DAYS_REST,
    }
    for col, val in fill_defaults.items():
        out[col] = out[col].fillna(val)

    out["tournament_matches_diff"] = out["home_tournament_matches"] - out["away_tournament_matches"]
    out["days_rest_diff"] = out["home_days_rest"] - out["away_days_rest"]

    return out


def tournament_features_from_state(
    matches: list[dict],
    match_date: pd.Timestamp,
) -> dict[str, float]:
    """
    Build the same tournament-path feature dict used in training,
    from a TeamTournamentState.matches list (live simulation).
    """
    prior = [m for m in matches if m["date"] < match_date]
    n = len(prior)

    if n == 0:
        return {
            "tournament_matches": DEFAULT_TOURNAMENT_MATCHES,
            "tournament_win_pct": DEFAULT_TOURNAMENT_WIN_PCT,
            "tournament_goal_diff": DEFAULT_TOURNAMENT_GOAL_DIFF,
            "days_rest": DEFAULT_DAYS_REST,
        }

    wins = sum(m["won"] for m in prior)
    goal_diff = sum(m["goals_for"] - m["goals_against"] for m in prior)
    last = max(m["date"] for m in prior)
    rest = max(0.0, (match_date - last).days)

    return {
        "tournament_matches": float(n),
        "tournament_win_pct": wins / n,
        "tournament_goal_diff": goal_diff / n,
        "days_rest": rest if rest > 0 else float(DEFAULT_DAYS_REST),
    }
