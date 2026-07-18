"""
Decide when to refresh predictions around knockout match times.

The FIFA fixtures API only exposes dates (no kickoff times), so important matches
can be overridden in config/match_kickoffs.json.

Refresh windows for each pending knockout match:
  1. At scheduled kickoff
  2. Every poll_interval after expected full-time for poll_duration (or until scored)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

KNOCKOUT_STAGES = {
    "Round of 16",
    "Quarter-final",
    "Semi-final",
    "Final",
    "Match for third place",
    "Play-off for third place",
}

STATE_PATH = PROJECT_ROOT / "data" / "processed" / "match_refresh_state.json"
CONFIG_PATH = PROJECT_ROOT / "config" / "match_kickoffs.json"
FIXTURES_PATH = PROJECT_ROOT / "data" / "processed" / "world_cup_fixtures.csv"


@dataclass(frozen=True)
class RefreshDecision:
    should_refresh: bool
    reason: str
    watch_key: str | None = None


@dataclass(frozen=True)
class MatchWatch:
    key: str
    label: str
    kickoff_utc: datetime
    match_duration_minutes: int
    poll_duration_minutes: int
    poll_interval_minutes: int
    completed: bool


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {"defaults": {}, "matches": {}}
    return json.loads(CONFIG_PATH.read_text())


def _has_winner(value: object) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip() != ""


def _parse_kickoff(
    match_date: str,
    defaults: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> datetime:
    overrides = overrides or {}
    if overrides.get("kickoff_utc"):
        return datetime.fromisoformat(str(overrides["kickoff_utc"]))

    kickoff_time = str(defaults.get("knockout_kickoff_utc", "19:00"))
    hour, minute = map(int, kickoff_time.split(":"))
    day = datetime.fromisoformat(str(match_date)).date()
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=timezone.utc)


def _load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"completed_matches": [], "last_refresh_utc": None, "last_reason": None}
    return json.loads(STATE_PATH.read_text())


def save_state(
    *,
    completed_matches: list[str] | None = None,
    last_reason: str | None = None,
) -> None:
    state = _load_state()
    if completed_matches is not None:
        merged = sorted(set(state.get("completed_matches", [])) | set(completed_matches))
        state["completed_matches"] = merged
    state["last_refresh_utc"] = _utc_now().isoformat()
    if last_reason is not None:
        state["last_reason"] = last_reason
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2) + "\n")


def _bracket_helpers():
    from src.simulation.simulate_tournament_v2 import (
        BRACKET_2026,
        MATCH_ORDER,
        _resolve_source,
        derive_bracket_state,
    )

    return BRACKET_2026, MATCH_ORDER, _resolve_source, derive_bracket_state


def _pending_bracket_slots(fixtures: pd.DataFrame) -> list[tuple[str, str, str]]:
    BRACKET_2026, MATCH_ORDER, _resolve_source, derive_bracket_state = _bracket_helpers()
    real = derive_bracket_state(fixtures)
    pending: list[tuple[str, str, str]] = []
    for match_id in MATCH_ORDER:
        if match_id in real:
            continue
        home = _resolve_source(BRACKET_2026[match_id][0], real)
        away = _resolve_source(BRACKET_2026[match_id][1], real)
        if home and away:
            pending.append((match_id, home, away))
    return pending


def _fixture_row_for_pair(
    fixtures: pd.DataFrame,
    home: str,
    away: str,
) -> pd.Series | None:
    for _, row in fixtures.iterrows():
        pair = {str(row["home_team"]), str(row["away_team"])}
        if pair == {home, away}:
            return row
    return None


def build_watch_list(now: datetime | None = None) -> list[MatchWatch]:
    now = now or _utc_now()
    config = _load_config()
    defaults = config.get("defaults", {})
    overrides = config.get("matches", {})
    state = _load_state()
    completed = set(state.get("completed_matches", []))

    duration = int(defaults.get("match_duration_minutes", 135))
    poll_duration = int(defaults.get("poll_duration_minutes", 60))
    poll_interval = int(defaults.get("poll_interval_minutes", 5))

    watches: list[MatchWatch] = []
    seen_keys: set[str] = set()

    if FIXTURES_PATH.exists():
        fixtures = pd.read_csv(FIXTURES_PATH)
    else:
        fixtures = pd.DataFrame(columns=["stage", "match_date", "home_team", "away_team", "winner", "match_id"])

    # Pending fixtures in the feed without a winner yet.
    if not fixtures.empty:
        pending_fixtures = fixtures[
            fixtures["stage"].isin(KNOCKOUT_STAGES)
            & ~fixtures["winner"].map(_has_winner)
        ]
        for row in pending_fixtures.itertuples(index=False):
            key = str(row.match_id)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            match_overrides = overrides.get(key, {})
            kickoff = _parse_kickoff(str(row.match_date), defaults, match_overrides)
            label = match_overrides.get("label") or f"{row.stage}: {row.home_team} vs {row.away_team}"
            watches.append(
                MatchWatch(
                    key=key,
                    label=str(label),
                    kickoff_utc=kickoff,
                    match_duration_minutes=duration,
                    poll_duration_minutes=poll_duration,
                    poll_interval_minutes=poll_interval,
                    completed=key in completed,
                )
            )

    # Bracket slots with both teams known but no result yet (e.g. Final before feed row).
    for match_id, home, away in _pending_bracket_slots(fixtures):
        if match_id in seen_keys:
            continue
        seen_keys.add(match_id)
        row = _fixture_row_for_pair(fixtures, home, away)
        match_date = (
            str(row["match_date"])
            if row is not None
            else overrides.get(match_id, {}).get("match_date", now.date().isoformat())
        )
        match_overrides = overrides.get(match_id, {})
        kickoff = _parse_kickoff(match_date, defaults, match_overrides)
        label = match_overrides.get("label") or f"{match_id}: {home} vs {away}"
        watches.append(
            MatchWatch(
                key=match_id,
                label=str(label),
                kickoff_utc=kickoff,
                match_duration_minutes=duration,
                poll_duration_minutes=poll_duration,
                poll_interval_minutes=poll_interval,
                completed=match_id in completed,
            )
        )

    return watches


def _in_poll_window(now: datetime, watch: MatchWatch) -> bool:
    expected_end = watch.kickoff_utc + timedelta(minutes=watch.match_duration_minutes)
    poll_end = expected_end + timedelta(minutes=watch.poll_duration_minutes)
    return expected_end <= now <= poll_end


def _at_kickoff(now: datetime, watch: MatchWatch, pre_kickoff_minutes: int) -> bool:
    window_start = watch.kickoff_utc - timedelta(minutes=pre_kickoff_minutes)
    return window_start <= now < watch.kickoff_utc + timedelta(minutes=1)


def decide_refresh(now: datetime | None = None) -> RefreshDecision:
    now = now or _utc_now()
    config = _load_config()
    defaults = config.get("defaults", {})
    pre_kickoff = int(defaults.get("pre_kickoff_minutes", 0))

    watches = build_watch_list(now)
    active = [w for w in watches if not w.completed]

    if not active:
        return RefreshDecision(False, "No pending knockout matches to watch.")

    for watch in active:
        if _at_kickoff(now, watch, pre_kickoff):
            return RefreshDecision(
                True,
                f"Kickoff refresh for {watch.label} ({watch.kickoff_utc.isoformat()}).",
                watch.key,
            )

    for watch in active:
        if _in_poll_window(now, watch):
            return RefreshDecision(
                True,
                (
                    f"Post-match poll for {watch.label} "
                    f"(every {watch.poll_interval_minutes} min for up to "
                    f"{watch.poll_duration_minutes} min after full time)."
                ),
                watch.key,
            )

    next_events = []
    for watch in active:
        kickoff = watch.kickoff_utc.isoformat()
        poll_start = (watch.kickoff_utc + timedelta(minutes=watch.match_duration_minutes)).isoformat()
        next_events.append(f"{watch.label}: kickoff {kickoff}, poll from {poll_start}")
    return RefreshDecision(False, "Outside refresh windows. " + " | ".join(next_events))


def detect_new_results(before: dict[str, str], after: dict[str, str]) -> list[str]:
    changed: list[str] = []
    for match_id, winner in after.items():
        if match_id not in before or before[match_id] != winner:
            changed.append(match_id)
    return sorted(changed)


def snapshot_bracket_winners() -> dict[str, str]:
    if not FIXTURES_PATH.exists():
        return {}
    fixtures = pd.read_csv(FIXTURES_PATH)
    _, _, _, derive_bracket_state = _bracket_helpers()
    return dict(derive_bracket_state(fixtures))


def mark_completed_from_bracket() -> list[str]:
    winners = snapshot_bracket_winners()
    completed = list(winners.keys())
    if completed:
        save_state(completed_matches=completed)
    return completed
