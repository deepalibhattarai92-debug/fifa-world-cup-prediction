"""
Collect FIFA Men's World Rankings from the official FIFA JSON API.

Source: https://api.fifa.com/api/v3/rankings?gender=1
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


API_URL = "https://api.fifa.com/api/v3/rankings"
MENS_GENDER = 1
MIN_TEAM_COUNT = 200

RAW_DIR = Path("data/raw")
PROCESSED_PATH = Path("data/processed/fifa_rankings.csv")

OUTPUT_COLUMNS = [
    "ranking_date",
    "team",
    "country_id",
    "fifa_rank",
    "previous_rank",
    "fifa_points",
    "decimal_points",
    "ranking_movement",
    "confederation",
    "matches_played",
    "collected_at",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/137.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://inside.fifa.com/fifa-world-ranking/men",
    "Origin": "https://inside.fifa.com",
}


def create_fifa_session() -> requests.Session:
    """Create a requests session with Chrome-like headers for Akamai."""
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def validate_api_response(response: requests.Response) -> None:
    """
    Validate the FIFA API response before parsing JSON.

    Akamai may return HTTP 403 with an HTML page when the request
    is classified as a bot (for example, python-requests User-Agent).
    """
    content_type = response.headers.get("Content-Type", "")
    body_start = response.text[:20].strip().lower()

    if response.status_code == 403:
        raise RuntimeError(
            "FIFA API returned HTTP 403. Akamai likely blocked the request. "
            "Ensure a Chrome-like User-Agent is set on the session."
        )

    if "text/html" in content_type or body_start.startswith("<html"):
        raise RuntimeError(
            "FIFA API returned HTML instead of JSON. "
            "The request was likely blocked by Akamai."
        )

    response.raise_for_status()


def fetch_fifa_rankings(
    session: requests.Session,
    gender: int = MENS_GENDER,
    date_id: str | None = None,
) -> dict:
    """Download FIFA rankings from the official JSON API."""
    params = {"gender": gender}

    if date_id:
        params["dateId"] = date_id

    print("Connecting to FIFA API...")
    response = session.get(API_URL, params=params, timeout=30)

    print(f"HTTP Status: {response.status_code}")
    validate_api_response(response)

    return response.json()


def parse_rankings_response(data: dict) -> pd.DataFrame:
    """Normalize FIFA rankings JSON into a flat dataframe."""
    if "Results" not in data:
        raise ValueError("FIFA API response is missing the 'Results' field.")

    results = data["Results"]

    if not isinstance(results, list):
        raise ValueError("FIFA API 'Results' field is not a list.")

    if not results:
        raise ValueError("FIFA API returned an empty 'Results' list.")

    if len(results) < MIN_TEAM_COUNT:
        raise ValueError(
            f"Expected at least {MIN_TEAM_COUNT} teams, received {len(results)}."
        )

    collected_at = datetime.now(timezone.utc).isoformat()
    rankings = []

    for team in results:
        team_name = team.get("TeamName")

        if isinstance(team_name, list) and team_name:
            team_name = team_name[0].get("Description")

        rankings.append({
            "ranking_date": team.get("PubDate"),
            "team": team_name,
            "country_id": team.get("IdCountry"),
            "fifa_rank": team.get("Rank"),
            "previous_rank": team.get("PrevRank"),
            "fifa_points": team.get("TotalPoints"),
            "decimal_points": team.get("DecimalTotalPoints"),
            "ranking_movement": team.get("RankingMovement"),
            "confederation": team.get("ConfederationName"),
            "matches_played": team.get("Matches"),
            "collected_at": collected_at,
        })

    rankings_df = pd.DataFrame(rankings)

    required_columns = ["team", "fifa_rank", "fifa_points"]
    if rankings_df[required_columns].isnull().sum().sum() > 0:
        raise ValueError("Missing required FIFA ranking fields.")

    return rankings_df[OUTPUT_COLUMNS]


def collect_fifa_rankings() -> None:
    """Download the latest FIFA Men's World Rankings and save raw + processed outputs."""
    session = create_fifa_session()
    data = fetch_fifa_rankings(session)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw_file = (
        RAW_DIR /
        f"fifa_rankings_{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"
    )

    with open(raw_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    rankings_df = parse_rankings_response(data)
    rankings_df.to_csv(PROCESSED_PATH, index=False)

    print(f"Saved raw JSON: {raw_file}")
    print(f"Saved processed CSV: {PROCESSED_PATH}")
    print(f"Total Teams: {len(rankings_df)}")


if __name__ == "__main__":
    collect_fifa_rankings()
