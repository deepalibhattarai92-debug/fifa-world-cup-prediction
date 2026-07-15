"""Shared team-name mappings between FIFA fixtures and Kaggle results."""

# FIFA fixture / dashboard name → historical results name in clean_results.csv
FIXTURE_TO_RESULTS_NAME: dict[str, str] = {
    "USA":            "United States",
    "Korea Republic": "South Korea",
    "IR Iran":        "Iran",
    "Türkiye":        "Turkey",
    "Côte d'Ivoire":  "Ivory Coast",
    "Congo DR":       "DR Congo",
    "Czechia":        "Czech Republic",
    "Cabo Verde":     "Cape Verde",
}

RESULTS_TO_FIXTURE_NAME: dict[str, str] = {
    v: k for k, v in FIXTURE_TO_RESULTS_NAME.items()
}


def fixture_to_results(name: str) -> str:
    return FIXTURE_TO_RESULTS_NAME.get(name, name)


def results_to_fixture(name: str) -> str:
    return RESULTS_TO_FIXTURE_NAME.get(name, name)
