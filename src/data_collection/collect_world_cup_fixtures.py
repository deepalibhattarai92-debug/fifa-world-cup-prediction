"""
Collect FIFA World Cup 2026 fixtures from the official FIFA data centre API.

Source: https://inside.fifa.com/api/data-centre/matches
        ?gender=1&competitionClassificationCode=FWC&year=2026
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


API_URL = "https://inside.fifa.com/api/data-centre/matches"
MIN_FIXTURE_COUNT = 48

RAW_DIR = Path("data/raw")
PROCESSED_PATH = Path("data/processed/world_cup_fixtures.csv")

OUTPUT_COLUMNS = [
    "match_id",
    "stage",
    "match_date",
    "home_team",
    "away_team",
    "home_country_code",
    "away_country_code",
    "home_score",
    "away_score",
    "winner",
    "stadium",
    "collected_at",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/137.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://inside.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026",
    "Origin": "https://inside.fifa.com",
}


def create_fifa_session() -> requests.Session:
    """Create a requests session with Chrome-like headers."""
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def validate_response(response: requests.Response) -> None:
    """Detect Akamai blocks or unexpected HTML responses."""
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
            "The request was likely blocked."
        )

    response.raise_for_status()


def fetch_fixtures(session: requests.Session) -> list[dict]:
    """Download 2026 World Cup fixtures from the FIFA data centre API."""
    params = {
        "gender": 1,
        "competitionClassificationCode": "FWC",
        "year": 2026,
        "language": "en",
        "count": 200,
    }

    print("Connecting to FIFA fixtures API...")
    response = session.get(API_URL, params=params, timeout=30)

    print(f"HTTP Status: {response.status_code}")
    validate_response(response)

    data = response.json()

    if not isinstance(data, list):
        raise ValueError(
            f"Expected a list of fixtures, received: {type(data).__name__}"
        )

    return data


def _extract_locale_string(field: object) -> str:
    """Extract the English description from a FIFA locale list field."""
    if isinstance(field, list) and field:
        return field[0].get("description", "")
    return str(field) if field else ""


def _resolve_winner(match: dict) -> str:
    """
    Resolve winner team ID to a team name.

    The FIFA API returns the winner as a team ID string (e.g. "43946").
    Match this against teamAId / teamBId to get the readable name.
    """
    winner_id = match.get("winner")
    if not winner_id:
        return ""

    if winner_id == match.get("teamAId"):
        return _extract_locale_string(match.get("teamAName"))
    if winner_id == match.get("teamBId"):
        return _extract_locale_string(match.get("teamBName"))

    return str(winner_id)


def parse_fixtures(raw_fixtures: list[dict]) -> pd.DataFrame:
    """Normalize raw fixture JSON into a flat dataframe."""
    if not raw_fixtures:
        raise ValueError("No fixture data returned by FIFA API.")

    if len(raw_fixtures) < MIN_FIXTURE_COUNT:
        raise ValueError(
            f"Expected at least {MIN_FIXTURE_COUNT} fixtures, "
            f"received {len(raw_fixtures)}."
        )

    collected_at = datetime.now(timezone.utc).isoformat()
    fixtures = []

    for match in raw_fixtures:
        fixtures.append({
            "match_id": match.get("idMatch"),
            "stage": _extract_locale_string(match.get("stageName")),
            "match_date": match.get("matchDate"),
            "home_team": _extract_locale_string(match.get("teamAName")),
            "away_team": _extract_locale_string(match.get("teamBName")),
            "home_country_code": match.get("teamACountryCode"),
            "away_country_code": match.get("teamBCountryCode"),
            "home_score": match.get("teamAScore"),
            "away_score": match.get("teamBScore"),
            "winner": _resolve_winner(match),
            "stadium": _extract_locale_string(match.get("stadiumName")),
            "collected_at": collected_at,
        })

    fixtures_df = pd.DataFrame(fixtures)

    required_columns = ["match_id", "stage", "match_date"]
    if fixtures_df[required_columns].isnull().sum().sum() > 0:
        raise ValueError("Missing required fixture fields.")

    return fixtures_df[OUTPUT_COLUMNS]


def collect_world_cup_fixtures() -> None:
    """Download 2026 World Cup fixtures and save raw + processed outputs."""
    session = create_fifa_session()
    raw_fixtures = fetch_fixtures(session)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw_file = (
        RAW_DIR /
        f"world_cup_fixtures_{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"
    )

    with open(raw_file, "w", encoding="utf-8") as file:
        json.dump(raw_fixtures, file, indent=4)

    fixtures_df = parse_fixtures(raw_fixtures)
    fixtures_df.to_csv(PROCESSED_PATH, index=False)

    print(f"Saved raw JSON: {raw_file}")
    print(f"Saved processed CSV: {PROCESSED_PATH}")
    print(f"Total Fixtures: {len(fixtures_df)}")


if __name__ == "__main__":
    collect_world_cup_fixtures()
