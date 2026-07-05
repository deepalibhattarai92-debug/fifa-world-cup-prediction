"""
Build World Cup history features from the Fjelstul World Cup Database.

Source: https://github.com/jfjelstul/worldcup (data-csv/)
License: CC-BY-NC-SA 4.0 — attribution required for non-commercial use.

Aggregates men's World Cup records (1930-2022) into team-level experience
features for Version 1: appearances, titles, and best finish.
"""

from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


SOURCE_BASE_URL = (
    "https://raw.githubusercontent.com/jfjelstul/worldcup/master/data-csv"
)
SOURCE_FILES = {
    "team_appearances": "team_appearances.csv",
    "tournament_standings": "tournament_standings.csv",
    "tournaments": "tournaments.csv",
}

STAGE_FINISH_SCORES = {
    "final": 1,
    "third-place match": 3,
    "semi-finals": 4,
    "quarter-finals": 5,
    "round of 16": 6,
    "second group stage": 7,
    "final round": 7,
    "group stage": 8,
}

RAW_DIR = Path("data/raw")
PROCESSED_PATH = Path("data/processed/world_cup_history.csv")

OUTPUT_COLUMNS = [
    "team",
    "country_code",
    "appearances",
    "titles",
    "best_finish",
    "best_finish_label",
    "collected_at",
]


def create_source_session() -> requests.Session:
    """Create a requests session for downloading source CSV files."""
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/137.0.0.0 Safari/537.36"
            ),
            "Accept": "text/csv,text/plain,*/*",
        }
    )
    return session


def download_source_tables(session: requests.Session) -> dict[str, pd.DataFrame]:
    """Download required tables from the Fjelstul World Cup Database."""
    tables = {}

    for table_name, file_name in SOURCE_FILES.items():
        url = f"{SOURCE_BASE_URL}/{file_name}"
        print(f"Downloading {table_name}...")
        response = session.get(url, timeout=30)
        response.raise_for_status()
        tables[table_name] = pd.read_csv(StringIO(response.text))

    return tables


def _mens_tournament_mask(series: pd.Series) -> pd.Series:
    """Return a mask for men's World Cup tournaments only."""
    return (
        series.str.contains("FIFA Men's World Cup", case=False, na=False, regex=False)
        & ~series.str.contains("Women", case=False, na=False, regex=False)
    )


def _finish_label(score: int) -> str:
    """Convert a numeric best-finish score to a readable label."""
    labels = {
        1: "Winner",
        2: "Runner-up",
        3: "Third place",
        4: "Fourth place",
        5: "Quarter-finals",
        6: "Round of 16",
        7: "Second group stage",
        8: "Group stage",
    }
    return labels.get(score, "Unknown")


def _tournament_best_finish(team_matches: pd.DataFrame) -> int:
    """Calculate a team's best finish in a single tournament."""
    standings_finish = team_matches["standing_position"].min()
    if pd.notna(standings_finish):
        return int(standings_finish)

    stage_scores = (
        team_matches["stage_name"]
        .str.lower()
        .map(STAGE_FINISH_SCORES)
        .dropna()
    )

    if stage_scores.empty:
        raise ValueError("Could not determine tournament finish from match data.")

    base_score = int(stage_scores.min())

    final_match = team_matches[
        team_matches["stage_name"].str.lower() == "final"
    ]
    if not final_match.empty:
        if final_match["win"].max() == 1:
            return 1
        return 2

    third_place_match = team_matches[
        team_matches["stage_name"].str.lower() == "third-place match"
    ]
    if not third_place_match.empty:
        if third_place_match["win"].max() == 1:
            return 3
        return 4

    return base_score


def build_world_cup_history(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Aggregate source tables into team-level World Cup history features."""
    appearances = tables["team_appearances"]
    standings = tables["tournament_standings"]
    tournaments = tables["tournaments"]

    mens_appearances = appearances[
        _mens_tournament_mask(appearances["tournament_name"])
    ].copy()

    mens_standings = standings[
        _mens_tournament_mask(standings["tournament_name"])
    ].copy()

    mens_tournaments = tournaments[
        _mens_tournament_mask(tournaments["tournament_name"])
    ].copy()

    if mens_appearances.empty:
        raise ValueError("No men's World Cup appearance records found.")

    standings_lookup = mens_standings[
        ["tournament_id", "team_name", "position"]
    ].rename(columns={"position": "standing_position"})

    appearances_with_standings = mens_appearances.merge(
        standings_lookup,
        on=["tournament_id", "team_name"],
        how="left",
    )

    tournament_finishes = []
    for (team_name, tournament_id), team_matches in appearances_with_standings.groupby(
        ["team_name", "tournament_id"]
    ):
        tournament_finishes.append(
            {
                "team": team_name,
                "country_code": team_matches["team_code"].iloc[0],
                "tournament_id": tournament_id,
                "finish_score": _tournament_best_finish(team_matches),
            }
        )

    finishes_df = pd.DataFrame(tournament_finishes)

    titles = (
        mens_tournaments["winner"]
        .value_counts()
        .rename_axis("team")
        .reset_index(name="titles")
    )

    history_df = (
        finishes_df.groupby(["team", "country_code"], as_index=False)
        .agg(
            appearances=("tournament_id", "nunique"),
            best_finish=("finish_score", "min"),
        )
    )

    history_df = history_df.merge(titles, on="team", how="left")
    history_df["titles"] = history_df["titles"].fillna(0).astype(int)
    history_df["best_finish"] = history_df["best_finish"].astype(int)
    history_df["best_finish_label"] = history_df["best_finish"].map(_finish_label)
    history_df["collected_at"] = datetime.now(timezone.utc).isoformat()

    if history_df["appearances"].isnull().any():
        raise ValueError("Missing appearance counts in World Cup history.")

    return history_df[OUTPUT_COLUMNS]


def build_world_cup_history_dataset() -> None:
    """Download source data and save raw snapshots plus processed history CSV."""
    session = create_source_session()
    tables = download_source_tables(session)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

    snapshot_date = datetime.now(timezone.utc).strftime("%Y%m%d")

    for table_name, table_df in tables.items():
        raw_file = RAW_DIR / f"world_cup_{table_name}_{snapshot_date}.csv"
        table_df.to_csv(raw_file, index=False)
        print(f"Saved raw source table: {raw_file}")

    history_df = build_world_cup_history(tables)
    history_df.to_csv(PROCESSED_PATH, index=False)

    print(f"Saved processed CSV: {PROCESSED_PATH}")
    print(f"Total Teams: {len(history_df)}")


if __name__ == "__main__":
    build_world_cup_history_dataset()
