import pandas as pd
from pathlib import Path


RAW_DATA_PATH = Path("data/raw/shootouts.csv")
PROCESSED_DATA_PATH = Path("data/processed/shootouts.csv")


def clean_shootouts():
    """
    Clean and validate penalty shootout data.

    This dataset is not used in the Version 1 baseline model yet,
    but it is prepared for future simulation and knockout-stage analysis.
    """

    print("Loading shootouts data...")
    shootouts = pd.read_csv(RAW_DATA_PATH)

    print("Cleaning data...")
    shootouts["date"] = pd.to_datetime(shootouts["date"])

    if shootouts[["date", "home_team", "away_team", "winner"]].isnull().sum().sum() > 0:
        raise ValueError("Missing values found in required shootout columns")

    duplicate_count = shootouts.duplicated(
        subset=["date", "home_team", "away_team"]
    ).sum()

    if duplicate_count > 0:
        print(f"Warning: {duplicate_count} duplicate shootout records found")

    print("Saving processed shootouts data...")
    shootouts.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Processed shootouts saved to {PROCESSED_DATA_PATH}")
    print(f"Final dataset shape: {shootouts.shape}")


if __name__ == "__main__":
    clean_shootouts()
