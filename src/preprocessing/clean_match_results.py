import pandas as pd
from pathlib import Path


RAW_DATA_PATH = Path("data/raw/results.csv")
PROCESSED_DATA_PATH = Path("data/processed/clean_results.csv")


def clean_match_results():
    """
    Clean historical international football match results.

    Steps:
    1. Load raw match results.
    2. Remove matches with missing scores.
    3. Convert date column to datetime.
    4. Convert score columns to integers.
    5. Save cleaned data to data/processed.
    """

    print("Loading raw match results...")
    results = pd.read_csv(RAW_DATA_PATH)

    print("Cleaning data...")
    results = results.dropna(subset=["home_score", "away_score"])

    results["date"] = pd.to_datetime(results["date"])

    results["home_score"] = results["home_score"].astype(int)
    results["away_score"] = results["away_score"].astype(int)

    print("Saving cleaned data...")
    results.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Cleaned data saved to {PROCESSED_DATA_PATH}")
    print(f"Final dataset shape: {results.shape}")


if __name__ == "__main__":
    clean_match_results()