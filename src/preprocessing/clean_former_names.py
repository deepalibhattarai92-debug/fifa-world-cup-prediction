import pandas as pd
from pathlib import Path


RAW_DATA_PATH = Path("data/raw/former_names.csv")
PROCESSED_DATA_PATH = Path("data/processed/former_names.csv")


def clean_former_names():
    """
    Validate and prepare former country/team names lookup table.

    This file helps standardize historical team names across datasets.
    """

    print("Loading former names data...")
    former_names = pd.read_csv(RAW_DATA_PATH)

    print("Validating data...")

    if former_names.isnull().sum().sum() > 0:
        raise ValueError("Missing values found in former_names.csv")

    if former_names.duplicated(subset=["former"]).sum() > 0:
        raise ValueError("Duplicate former team names found")

    former_names["start_date"] = pd.to_datetime(former_names["start_date"])
    former_names["end_date"] = pd.to_datetime(former_names["end_date"])

    print("Saving processed former names lookup...")
    former_names.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Processed former names saved to {PROCESSED_DATA_PATH}")
    print(f"Final dataset shape: {former_names.shape}")


if __name__ == "__main__":
    clean_former_names()
