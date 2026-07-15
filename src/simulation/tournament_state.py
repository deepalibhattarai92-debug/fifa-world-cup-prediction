"""
Sequential tournament-state tracking for live World Cup simulation.

Shares feature definitions with build_features_v2 via tournament_path.py so
training and inference use the same within-tournament columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_engineering.tournament_path import (
    DEFAULT_DAYS_REST,
    DEFAULT_TOURNAMENT_GOAL_DIFF,
    DEFAULT_TOURNAMENT_MATCHES,
    DEFAULT_TOURNAMENT_WIN_PCT,
    is_knockout_stage,
    tournament_features_from_state,
)
from src.utils.team_names import fixture_to_results


# Simulated knockout scoreline when the model does not predict goals.
SIM_WIN_GOALS = 2
SIM_LOSS_GOALS = 1


class TeamTournamentState:
    """Cumulative 2026 World Cup path for one team."""

    def __init__(self) -> None:
        self.matches: list[dict] = []

    def record(
        self,
        match_date: pd.Timestamp,
        goals_for: int,
        goals_against: int,
        won: bool,
        stadium_region: str,
    ) -> None:
        self.matches.append({
            "date": match_date,
            "goals_for": goals_for,
            "goals_against": goals_against,
            "won": won,
            "stadium_region": stadium_region,
        })

    def features_for_match(self, match_date: pd.Timestamp) -> dict[str, float]:
        return tournament_features_from_state(self.matches, match_date)


class TournamentStateTracker:
    """Per-team tournament paths; updated after every real or simulated match."""

    def __init__(self) -> None:
        self._teams: dict[str, TeamTournamentState] = {}

    def _get(self, team: str) -> TeamTournamentState:
        if team not in self._teams:
            self._teams[team] = TeamTournamentState()
        return self._teams[team]

    def copy(self) -> TournamentStateTracker:
        other = TournamentStateTracker()
        for team, ts in self._teams.items():
            other._teams[team] = TeamTournamentState()
            other._teams[team].matches = list(ts.matches)
        return other

    def tournament_snapshot(self, team: str) -> dict[str, float]:
        feats = self._get(team).features_for_match(pd.Timestamp("2099-01-01"))
        return {
            "matches": int(feats["tournament_matches"]),
            "win_pct": feats["tournament_win_pct"],
            "goal_diff": feats["tournament_goal_diff"],
        }

    def days_until_match(self, team: str, match_date: pd.Timestamp) -> float:
        return self._get(team).features_for_match(match_date)["days_rest"]

    def side_features(self, team: str, match_date: pd.Timestamp, prefix: str) -> dict[str, float]:
        """Model-ready columns with home_/away_ prefix."""
        raw = self._get(team).features_for_match(match_date)
        return {
            f"{prefix}tournament_matches": raw["tournament_matches"],
            f"{prefix}tournament_win_pct": raw["tournament_win_pct"],
            f"{prefix}tournament_goal_diff": raw["tournament_goal_diff"],
            f"{prefix}days_rest": raw["days_rest"],
        }

    def record_result(
        self,
        home: str,
        away: str,
        home_goals: int,
        away_goals: int,
        winner: str,
        match_date: pd.Timestamp,
        stadium: str,
    ) -> None:
        region = stadium_region(stadium)
        self._get(home).record(
            match_date, home_goals, away_goals, winner == home, region,
        )
        self._get(away).record(
            match_date, away_goals, home_goals, winner == away, region,
        )


def stadium_region(stadium: str) -> str:
    if not stadium or (isinstance(stadium, float) and np.isnan(stadium)):
        return "unknown"
    name = str(stadium).strip()
    if name.endswith(" Stadium"):
        return name[: -len(" Stadium")]
    return name


def init_state_from_fixtures(
    fixtures: pd.DataFrame,
    teams: set[str],
) -> TournamentStateTracker:
    tracker = TournamentStateTracker()
    played = fixtures[
        fixtures["winner"].notna()
        & (fixtures["winner"].astype(str).str.strip() != "")
        & (fixtures["winner"].astype(str).str.lower() != "nan")
    ].copy()

    played["_date"] = pd.to_datetime(played["match_date"], errors="coerce")
    played = played.sort_values("_date")

    for _, row in played.iterrows():
        home = fixture_to_results(str(row["home_team"]))
        away = fixture_to_results(str(row["away_team"]))
        if home not in teams and away not in teams:
            continue

        hg = int(row["home_score"]) if pd.notna(row["home_score"]) else 0
        ag = int(row["away_score"]) if pd.notna(row["away_score"]) else 0
        tracker.record_result(
            home=home,
            away=away,
            home_goals=hg,
            away_goals=ag,
            winner=fixture_to_results(str(row["winner"])),
            match_date=row["_date"],
            stadium=str(row.get("stadium", "")),
        )

    return tracker


def bracket_teams_from_fixtures(fixtures: pd.DataFrame) -> set[str]:
    teams: set[str] = set()
    for col in ("home_team", "away_team"):
        for name in fixtures[col].dropna().astype(str).unique():
            teams.add(fixture_to_results(name))
    return teams


def match_meta_from_fixtures(
    fixtures: pd.DataFrame,
    home: str,
    away: str,
    fallback_date: str = "2026-07-15",
) -> tuple[pd.Timestamp, str, int]:
    """Return (date, stadium, is_knockout) for a pairing."""
    fx = fixtures.copy()
    fx["_date"] = pd.to_datetime(fx["match_date"], errors="coerce")
    home_f, away_f = fixture_to_results(home), fixture_to_results(away)

    mask = (
        ((fx["home_team"] == home) | (fx["home_team"] == home_f))
        & ((fx["away_team"] == away) | (fx["away_team"] == away_f))
    ) | (
        ((fx["home_team"] == away) | (fx["home_team"] == away_f))
        & ((fx["away_team"] == home) | (fx["away_team"] == home_f))
    )
    hit = fx[mask]
    if len(hit):
        row = hit.iloc[0]
        stage = str(row.get("stage", ""))
        return row["_date"], str(row.get("stadium", "MetLife Stadium")), is_knockout_stage(stage)
    return pd.Timestamp(fallback_date), "MetLife Stadium", 1


def record_simulated_result(
    tracker: TournamentStateTracker,
    home: str,
    away: str,
    winner: str,
    match_date: pd.Timestamp,
    stadium: str,
) -> None:
    if winner == home:
        hg, ag = SIM_WIN_GOALS, SIM_LOSS_GOALS
    else:
        hg, ag = SIM_LOSS_GOALS, SIM_WIN_GOALS
    tracker.record_result(home, away, hg, ag, winner, match_date, stadium)


def build_tournament_match_features(
    state: TournamentStateTracker,
    home: str,
    away: str,
    match_date: pd.Timestamp,
    is_knockout: int,
) -> dict[str, float]:
    """Full tournament-path feature block for one knockout row."""
    h = state.side_features(home, match_date, "home_")
    a = state.side_features(away, match_date, "away_")
    return {
        **h,
        **a,
        "is_knockout": float(is_knockout),
        "tournament_matches_diff": h["home_tournament_matches"] - a["away_tournament_matches"],
        "days_rest_diff": h["home_days_rest"] - a["away_days_rest"],
    }
