import pandas as pd
from pathlib import Path


RAW_DATA_PATH = Path("data/raw/goalscorers.csv")
PROCESSED_DATA_PATH = Path("data/processed/goalscorers.csv")


def clean_goalscorers():
    """
    Clean and validate international football goalscorer data.

    This dataset is not used in Version 1 but is prepared for future
    player-level feature engineering.
    """

    print("Loading goalscorers data...")
    goalscorers = pd.read_csv(RAW_DATA_PATH)

    print("Cleaning data...")

    goalscorers["date"] = pd.to_datetime(goalscorers["date"])

    if goalscorers[["date", "home_team", "away_team", "team"]].isnull().sum().sum() > 0:
        raise ValueError("Missing values found in required columns")

    duplicate_count = goalscorers.duplicated(
        subset=["date", "home_team", "away_team", "team", "scorer", "minute"]
    ).sum()

    if duplicate_count > 0:
        print(f"Warning: {duplicate_count} duplicate goal records found")

    print("Saving processed goalscorers data...")
    goalscorers.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Processed goalscorers saved to {PROCESSED_DATA_PATH}")
    print(f"Final dataset shape: {goalscorers.shape}")


if __name__ == "__main__":
    clean_goalscorers()