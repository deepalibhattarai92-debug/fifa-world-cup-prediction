"""
Sync completed 2026 World Cup fixtures into the historical results feed.

Closes the gap where knockouts live in world_cup_fixtures.csv but not in
clean_results.csv / features_v2.csv. Appends missing rows to data/raw/results.csv
then rebuilds clean_results.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.team_names import fixture_to_results
RAW_RESULTS = PROJECT_ROOT / "data/raw/results.csv"
PROCESSED_RESULTS = PROJECT_ROOT / "data/processed/clean_results.csv"
FIXTURES_PATH = PROJECT_ROOT / "data/processed/world_cup_fixtures.csv"

KNOCKOUT_STAGE_MARKERS = (
    "round of", "quarter", "semi", "final", "third",
)


def _is_played(row: pd.Series) -> bool:
    w = row.get("winner")
    if pd.isna(w):
        return False
    s = str(w).strip().lower()
    return s not in ("", "nan", "none")


def _match_key(
    date: pd.Timestamp,
    home: str,
    away: str,
) -> tuple[str, str, str]:
    d = pd.Timestamp(date).strftime("%Y-%m-%d")
    pair = tuple(sorted((home, away)))
    return d, pair[0], pair[1]


def _existing_keys(df: pd.DataFrame) -> set[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    for _, row in df.iterrows():
        keys.add(_match_key(row["date"], row["home_team"], row["away_team"]))
    return keys


def fixtures_to_result_rows(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Convert completed fixture rows into results.csv schema."""
    played = fixtures[fixtures.apply(_is_played, axis=1)].copy()
    played["date"] = pd.to_datetime(played["match_date"], errors="coerce")

    rows = []
    for _, row in played.iterrows():
        home = fixture_to_results(str(row["home_team"]))
        away = fixture_to_results(str(row["away_team"]))
        stadium = str(row.get("stadium", "") or "")
        rows.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "home_team": home,
            "away_team": away,
            "home_score": int(row["home_score"]),
            "away_score": int(row["away_score"]),
            "tournament": "FIFA World Cup",
            "city": stadium,
            "country": "USA",  # host region proxy; all neutral-site knockouts
            "neutral": True,
            "stage": str(row.get("stage", "")),
        })

    return pd.DataFrame(rows)


def sync_fixtures_to_results() -> int:
    """
    Append new completed 2026 fixtures to raw results and rebuild clean_results.
    Returns the number of rows appended.
    """
    if not FIXTURES_PATH.exists():
        print(f"  No fixtures file at {FIXTURES_PATH} — skipping sync.")
        return 0

    fixtures = pd.read_csv(FIXTURES_PATH)
    candidates = fixtures_to_result_rows(fixtures)
    if candidates.empty:
        print("  No completed fixtures to sync.")
        return 0

    raw = pd.read_csv(RAW_RESULTS, parse_dates=["date"])
    existing = _existing_keys(raw)

    new_rows = []
    for _, row in candidates.iterrows():
        key = _match_key(row["date"], row["home_team"], row["away_team"])
        if key not in existing:
            new_rows.append(row)
            existing.add(key)

    if not new_rows:
        print("  Fixtures already synced — no new rows.")
        _rebuild_clean_results()
        return 0

    append_df = pd.DataFrame(new_rows)
    for col in append_df.columns:
        if col not in raw.columns:
            raw[col] = pd.NA

    # Keep raw dates as strings (Kaggle format) for consistency.
    raw["date"] = raw["date"].astype(str).str[:10]
    append_df["date"] = append_df["date"].astype(str).str[:10]

    combined = pd.concat([raw, append_df], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["date", "home_team", "away_team"], keep="last",
    )
    combined = combined.sort_values("date").reset_index(drop=True)
    combined.to_csv(RAW_RESULTS, index=False)

    print(f"  Appended {len(append_df)} fixture rows to {RAW_RESULTS.name}")
    for _, r in append_df.iterrows():
        print(f"    {r['date']}  {r['home_team']} {r['home_score']}-{r['away_score']} {r['away_team']}  ({r.get('stage','')})")

    _rebuild_clean_results()
    return len(append_df)


def _rebuild_clean_results() -> None:
    """Regenerate clean_results.csv from raw results."""
    raw = pd.read_csv(RAW_RESULTS)
    raw = raw.dropna(subset=["home_score", "away_score"])
    raw["date"] = pd.to_datetime(raw["date"])
    raw["home_score"] = raw["home_score"].astype(int)
    raw["away_score"] = raw["away_score"].astype(int)
    PROCESSED_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(PROCESSED_RESULTS, index=False)
    print(f"  Rebuilt {PROCESSED_RESULTS.name}  ({len(raw):,} rows)")


if __name__ == "__main__":
    print("Syncing 2026 fixtures → results feed...")
    n = sync_fixtures_to_results()
    print(f"Done. {n} new match(es) added.")
