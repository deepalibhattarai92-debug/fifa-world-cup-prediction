"""
Lightweight knockout match prediction helpers for dashboard + simulation.

Kept separate from simulate_tournament_v2.py so Streamlit can import predictions
without loading the full Monte Carlo module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.feature_cols import FEATURE_COLS
from src.simulation.tournament_state import (
    TournamentStateTracker,
    build_tournament_match_features,
)
from src.utils.team_names import FIXTURE_TO_RESULTS_NAME


FIXTURE_TO_WC_NAME: dict[str, str] = {
    "Czechia": "Czech Republic",
    "Korea Republic": "South Korea",
    "USA": "United States",
    "IR Iran": "Iran",
    "Türkiye": "Turkey",
    "Côte d'Ivoire": "Ivory Coast",
    "Congo DR": "DR Congo",
}

WC_HISTORY_DEFAULTS = {"appearances": 0, "titles": 0, "best_finish": 8}


def build_team_lookup(
    features: pd.DataFrame,
    fifa: pd.DataFrame,
    elo: pd.DataFrame,
    wc_history: pd.DataFrame,
    elo_map: pd.DataFrame,
    teams: list[str],
) -> dict[str, dict]:
    """Build per-team feature dict from latest snapshots."""
    elo_merged = elo_map.merge(
        elo[["country_code", "elo_rating"]],
        left_on="elo_code", right_on="country_code", how="left",
    )
    elo_lookup = dict(zip(elo_merged["team"], elo_merged["elo_rating"]))
    fifa_lookup = fifa.set_index("team")[["fifa_rank", "fifa_points", "confederation"]].to_dict("index")

    wc = wc_history.copy()
    for fixture_name, wc_name in FIXTURE_TO_WC_NAME.items():
        wc.loc[wc["team"] == wc_name, "team"] = fixture_name
    wc_lookup = wc.set_index("team")[["appearances", "titles", "best_finish"]].to_dict("index")

    form_cols = [
        "form_win_pct", "form_goals_scored", "form_goals_conceded",
        "form_goal_diff", "form_clean_sheet_pct",
    ]
    team_lookup: dict[str, dict] = {}

    for team in teams:
        results_name = FIXTURE_TO_RESULTS_NAME.get(team, team)
        home_rows = features[features["home_team"] == results_name].sort_values("date")
        away_rows = features[features["away_team"] == results_name].sort_values("date")

        form: dict = {}
        if len(home_rows):
            last_home = home_rows.iloc[-1]
            if pd.notna(last_home.get("home_form_win_pct")):
                form = {c: last_home[f"home_{c}"] for c in form_cols}
        if not form and len(away_rows):
            last_away = away_rows.iloc[-1]
            if pd.notna(last_away.get("away_form_win_pct")):
                form = {c: last_away[f"away_{c}"] for c in form_cols}
        if not form:
            form = {c: 0.5 for c in form_cols}
            form["form_goals_scored"] = 1.5
            form["form_goals_conceded"] = 1.5
            form["form_goal_diff"] = 0.0
            form["form_clean_sheet_pct"] = 0.3

        fifa_data = fifa_lookup.get(team, {})
        wc_data = wc_lookup.get(team, WC_HISTORY_DEFAULTS)
        team_lookup[team] = {
            "form_win_pct": form.get("form_win_pct", 0.5),
            "form_goals_scored": form.get("form_goals_scored", 1.5),
            "form_goals_conceded": form.get("form_goals_conceded", 1.5),
            "form_goal_diff": form.get("form_goal_diff", 0.0),
            "form_clean_sheet_pct": form.get("form_clean_sheet_pct", 0.3),
            "fifa_rank": fifa_data.get("fifa_rank", 50.0),
            "fifa_points": fifa_data.get("fifa_points", 1200.0),
            "confederation": fifa_data.get("confederation", ""),
            "elo_rating": elo_lookup.get(team, 1700.0),
            "wc_appearances": wc_data.get("appearances", 0),
            "wc_titles": wc_data.get("titles", 0),
            "wc_best_finish": wc_data.get("best_finish", 8),
        }

    return team_lookup


def build_h2h_lookup(
    features: pd.DataFrame,
    teams: list[str],
) -> dict[tuple[str, str], dict]:
    """Latest head-to-head features from features_v2 for fixture team pairs."""
    lookup: dict[tuple[str, str], dict] = {}
    for home in teams:
        hname = FIXTURE_TO_RESULTS_NAME.get(home, home)
        for away in teams:
            if home == away:
                continue
            aname = FIXTURE_TO_RESULTS_NAME.get(away, away)
            rows = features[
                ((features["home_team"] == hname) & (features["away_team"] == aname))
                | ((features["home_team"] == aname) & (features["away_team"] == hname))
            ].sort_values("date")
            if len(rows):
                last = rows.iloc[-1]
                if last["home_team"] == hname:
                    lookup[(home, away)] = {
                        "h2h_win_rate": last["h2h_win_rate"],
                        "h2h_goal_diff": last["h2h_goal_diff"],
                    }
                else:
                    wr = last["h2h_win_rate"]
                    lookup[(home, away)] = {
                        "h2h_win_rate": 1.0 - wr if pd.notna(wr) else wr,
                        "h2h_goal_diff": (
                            -last["h2h_goal_diff"]
                            if pd.notna(last["h2h_goal_diff"])
                            else last["h2h_goal_diff"]
                        ),
                    }
    return lookup


def build_match_row(
    home: str,
    away: str,
    team_lookup: dict[str, dict],
    tournament_feats: dict[str, float] | None = None,
    h2h: dict[str, float] | None = None,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    cols = feature_cols or FEATURE_COLS
    h = team_lookup[home]
    a = team_lookup[away]
    t = tournament_feats or {}

    row = {
        "home_form_win_pct": h["form_win_pct"],
        "home_form_goals_scored": h["form_goals_scored"],
        "home_form_goals_conceded": h["form_goals_conceded"],
        "home_form_goal_diff": h["form_goal_diff"],
        "home_form_clean_sheet_pct": h["form_clean_sheet_pct"],
        "away_form_win_pct": a["form_win_pct"],
        "away_form_goals_scored": a["form_goals_scored"],
        "away_form_goals_conceded": a["form_goals_conceded"],
        "away_form_goal_diff": a["form_goal_diff"],
        "away_form_clean_sheet_pct": a["form_clean_sheet_pct"],
        "home_fifa_rank": h["fifa_rank"],
        "home_fifa_points": h["fifa_points"],
        "away_fifa_rank": a["fifa_rank"],
        "away_fifa_points": a["fifa_points"],
        "home_elo_rating": h["elo_rating"],
        "away_elo_rating": a["elo_rating"],
        "home_wc_appearances": h["wc_appearances"],
        "home_wc_titles": h["wc_titles"],
        "home_wc_best_finish": h["wc_best_finish"],
        "away_wc_appearances": a["wc_appearances"],
        "away_wc_titles": a["wc_titles"],
        "away_wc_best_finish": a["wc_best_finish"],
        "rank_diff": h["fifa_rank"] - a["fifa_rank"],
        "points_diff": h["fifa_points"] - a["fifa_points"],
        "elo_diff": h["elo_rating"] - a["elo_rating"],
        "same_conf": int(h["confederation"] == a["confederation"]),
        "neutral": True,
        "match_importance": 5,
        "h2h_win_rate": (h2h or {}).get("h2h_win_rate", np.nan),
        "h2h_goal_diff": (h2h or {}).get("h2h_goal_diff", np.nan),
        "home_tournament_matches": t.get("home_tournament_matches", 0),
        "away_tournament_matches": t.get("away_tournament_matches", 0),
        "home_tournament_win_pct": t.get("home_tournament_win_pct", 0.5),
        "away_tournament_win_pct": t.get("away_tournament_win_pct", 0.5),
        "home_tournament_goal_diff": t.get("home_tournament_goal_diff", 0.0),
        "away_tournament_goal_diff": t.get("away_tournament_goal_diff", 0.0),
        "home_days_rest": t.get("home_days_rest", 7.0),
        "away_days_rest": t.get("away_days_rest", 7.0),
        "is_knockout": t.get("is_knockout", 1.0),
        "tournament_matches_diff": t.get("tournament_matches_diff", 0.0),
        "days_rest_diff": t.get("days_rest_diff", 0.0),
    }
    return pd.DataFrame([row])[cols]


def predict_home_win_prob(
    home: str,
    away: str,
    team_lookup: dict[str, dict],
    state: TournamentStateTracker,
    pipeline,
    label_encoder,
    match_date: pd.Timestamp,
    is_knockout: int,
    feature_cols: list[str],
    h2h_lookup: dict[tuple[str, str], dict] | None = None,
) -> float:
    """Path-dependent P(home wins) using trained tournament-path features."""
    t_feats = build_tournament_match_features(state, home, away, match_date, is_knockout)
    h2h = (h2h_lookup or {}).get((home, away))
    row = build_match_row(home, away, team_lookup, t_feats, h2h, feature_cols)

    classes = list(label_encoder.classes_)
    p = pipeline.predict_proba(row)[0]
    p_home = p[classes.index("home_win")] + p[classes.index("draw")] * 0.5
    p_away = p[classes.index("away_win")] + p[classes.index("draw")] * 0.5
    total = p_home + p_away
    return p_home / total if total else 0.5
