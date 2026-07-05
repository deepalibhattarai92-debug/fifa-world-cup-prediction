"""
Collect World Football Elo Ratings from eloratings.net.

Source: https://www.eloratings.net/World.tsv

The website renders rankings with JavaScript, but the underlying data is
served as a headerless tab-separated file used by the live rankings table.
"""

from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


ELO_TSV_URL = "https://www.eloratings.net/World.tsv"
MIN_TEAM_COUNT = 200

RAW_DIR = Path("data/raw")
PROCESSED_PATH = Path("data/processed/elo_ratings.csv")

RAW_COLUMNS = [
    "elo_rank",
    "current_rank",
    "country_code",
    "elo_rating",
    "rating_level",
    "peak_elo",
    "peak_rank",
    "average_elo",
    "average_rank",
    "lowest_elo",
    "lowest_rank",
    "three_month_rank_change",
    "three_month_rating_change",
    "six_month_rank_change",
    "six_month_rating_change",
    "one_year_rank_change",
    "one_year_rating_change",
    "two_year_rank_change",
    "two_year_rating_change",
    "five_year_rank_change",
    "five_year_rating_change",
    "ten_year_rank_change",
    "ten_year_rating_change",
    "matches_played",
    "wins",
    "draws",
    "losses",
    "goals_for",
    "goals_against",
    "home_wins",
    "away_wins",
]

OUTPUT_COLUMNS = [
    "elo_rank",
    "country_code",
    "elo_rating",
    "peak_elo",
    "peak_rank",
    "average_elo",
    "lowest_elo",
    "collected_at",
]


def create_elo_session() -> requests.Session:
    """Create a requests session for eloratings.net."""
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/137.0.0.0 Safari/537.36"
            ),
            "Accept": "text/tab-separated-values,text/plain,*/*",
            "Referer": "https://www.eloratings.net/",
        }
    )
    return session


def validate_elo_response(response: requests.Response) -> None:
    """Validate that eloratings.net returned TSV rather than an HTML error page."""
    content_type = response.headers.get("Content-Type", "")
    body_start = response.text[:20].strip().lower()

    if "text/html" in content_type or body_start.startswith("<html"):
        raise RuntimeError(
            "Elo ratings source returned HTML instead of TSV. "
            "The download may have failed."
        )

    response.raise_for_status()


def fetch_elo_ratings(session: requests.Session) -> str:
    """Download the live world Elo ratings TSV."""
    print("Connecting to World Football Elo Ratings...")
    response = session.get(ELO_TSV_URL, timeout=30)

    print(f"HTTP Status: {response.status_code}")
    validate_elo_response(response)

    return response.text


def parse_elo_ratings(tsv_text: str) -> pd.DataFrame:
    """Parse the headerless Elo TSV into a normalized dataframe."""
    ratings_df = pd.read_csv(
        StringIO(tsv_text),
        sep="\t",
        header=None,
        names=RAW_COLUMNS,
        na_filter=False,
    )

    if ratings_df.empty:
        raise ValueError("Elo ratings TSV is empty.")

    if len(ratings_df) < MIN_TEAM_COUNT:
        raise ValueError(
            f"Expected at least {MIN_TEAM_COUNT} teams, received {len(ratings_df)}."
        )

    ratings_df["country_code"] = ratings_df["country_code"].astype(str).str.strip()

    if ratings_df["country_code"].isnull().any():
        raise ValueError("Missing country codes in Elo ratings data.")

    if ratings_df["elo_rating"].isnull().any():
        raise ValueError("Missing Elo ratings values.")

    if ratings_df["country_code"].duplicated().any():
        raise ValueError("Duplicate country codes found in Elo ratings data.")

    ratings_df["collected_at"] = datetime.now(timezone.utc).isoformat()

    return ratings_df[OUTPUT_COLUMNS]


def collect_elo_ratings() -> None:
    """Download current Elo ratings and save raw + processed outputs."""
    session = create_elo_session()
    tsv_text = fetch_elo_ratings(session)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw_file = (
        RAW_DIR /
        f"elo_ratings_{datetime.now(timezone.utc).strftime('%Y%m%d')}.tsv"
    )

    with open(raw_file, "w", encoding="utf-8") as file:
        file.write(tsv_text)

    ratings_df = parse_elo_ratings(tsv_text)
    ratings_df.to_csv(PROCESSED_PATH, index=False)

    print(f"Saved raw TSV: {raw_file}")
    print(f"Saved processed CSV: {PROCESSED_PATH}")
    print(f"Total Teams: {len(ratings_df)}")


if __name__ == "__main__":
    collect_elo_ratings()
