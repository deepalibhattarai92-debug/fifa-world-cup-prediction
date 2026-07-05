"""
Monte Carlo simulation of the 2026 FIFA World Cup from current bracket state.
Version 1 — uses best_model.pkl trained on all matches (V1 feature set).

Current state (as of 2026-07-04):
  - Round of 16: France beat Paraguay, Morocco beat Canada
  - 6 remaining Round of 16 matches to simulate
  - Quarter-finals, Semi-finals, and Final to simulate

Bracket structure (derived from the R32 pairing pattern):
  QF1 slot:  Morocco vs France         (both already through)
  QF2 slot:  Brazil vs England winner  vs  Norway vs Belgium winner
  QF3 slot:  Mexico vs Egypt winner    vs  Portugal vs USA winner
  QF4 slot:  Spain vs Switzerland win  vs  Argentina vs Colombia winner

  SF1: QF1 winner vs QF2 winner
  SF2: QF3 winner vs QF4 winner
  Final: SF1 winner vs SF2 winner

Input:
    data/processed/features.csv
    data/processed/fifa_rankings.csv
    data/processed/elo_ratings.csv
    data/processed/world_cup_history.csv
    data/raw/elo_code_map.csv
    models/best_model.pkl

Output:
    data/processed/simulation_results_v1.csv
"""

import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")

INPUT_FEATURES  = PROCESSED_DIR / "features.csv"
INPUT_FIFA      = PROCESSED_DIR / "fifa_rankings.csv"
INPUT_ELO       = PROCESSED_DIR / "elo_ratings.csv"
INPUT_WC_HIST   = PROCESSED_DIR / "world_cup_history.csv"
INPUT_ELO_MAP   = Path("data/raw/elo_code_map.csv")
INPUT_MODEL     = MODELS_DIR / "best_model.pkl"

OUTPUT_RESULTS  = PROCESSED_DIR / "simulation_results_v1.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SIMULATIONS = 10_000

FIXTURE_TO_RESULTS_NAME: dict[str, str] = {
    "USA": "United States",
}

FIXTURE_TO_WC_NAME: dict[str, str] = {
    "Czechia":         "Czech Republic",
    "Korea Republic":  "South Korea",
    "USA":             "United States",
    "IR Iran":         "Iran",
    "Türkiye":         "Turkey",
    "Côte d'Ivoire":   "Ivory Coast",
    "Congo DR":        "DR Congo",
}

WC_HISTORY_DEFAULTS = {"appearances": 0, "titles": 0, "best_finish": 8}

# ---------------------------------------------------------------------------
# Bracket — current tournament state
# ---------------------------------------------------------------------------

QF_CONFIRMED: list[str] = ["France", "Morocco"]

R16_REMAINING: list[tuple[str, str]] = [
    ("Brazil",    "England"),
    ("Norway",    "Belgium"),
    ("Mexico",    "Egypt"),
    ("Portugal",  "USA"),
    ("Spain",     "Switzerland"),
    ("Argentina", "Colombia"),
]


# ---------------------------------------------------------------------------
# Step 1 — Build team feature lookup
# ---------------------------------------------------------------------------

def build_team_lookup(
    features: pd.DataFrame,
    fifa: pd.DataFrame,
    elo: pd.DataFrame,
    wc_history: pd.DataFrame,
    elo_map: pd.DataFrame,
    teams: list[str],
) -> dict[str, dict]:
    elo_merged = elo_map.merge(elo[["country_code", "elo_rating"]],
                               left_on="elo_code", right_on="country_code", how="left")
    elo_lookup = dict(zip(elo_merged["team"], elo_merged["elo_rating"]))

    fifa_lookup = fifa.set_index("team")[["fifa_rank", "fifa_points", "confederation"]].to_dict("index")

    wc = wc_history.copy()
    for fixture_name, wc_name in FIXTURE_TO_WC_NAME.items():
        wc.loc[wc["team"] == wc_name, "team"] = fixture_name
    wc_lookup = wc.set_index("team")[["appearances", "titles", "best_finish"]].to_dict("index")

    form_cols = [
        "form_win_pct", "form_goals_scored", "form_goals_conceded",
        "form_goal_diff", "form_clean_sheet_pct",
    ]

    team_lookup: dict[str, dict] = {}

    for team in teams:
        results_name = FIXTURE_TO_RESULTS_NAME.get(team, team)

        home_rows = features[features["home_team"] == results_name].sort_values("date")
        away_rows = features[features["away_team"] == results_name].sort_values("date")

        form: dict = {}
        if len(home_rows):
            last_home = home_rows.iloc[-1]
            if pd.notna(last_home.get("home_form_win_pct")):
                form = {c: last_home[f"home_{c}"] for c in form_cols}

        if not form and len(away_rows):
            last_away = away_rows.iloc[-1]
            if pd.notna(last_away.get("away_form_win_pct")):
                form = {c: last_away[f"away_{c}"] for c in form_cols}

        if not form:
            form = {c: 0.5 for c in form_cols}
            form["form_goals_scored"]    = 1.5
            form["form_goals_conceded"]  = 1.5
            form["form_goal_diff"]       = 0.0
            form["form_clean_sheet_pct"] = 0.3

        fifa_data = fifa_lookup.get(team, {})
        wc_data   = wc_lookup.get(team, WC_HISTORY_DEFAULTS)

        team_lookup[team] = {
            "form_win_pct":           form.get("form_win_pct", 0.5),
            "form_goals_scored":      form.get("form_goals_scored", 1.5),
            "form_goals_conceded":    form.get("form_goals_conceded", 1.5),
            "form_goal_diff":         form.get("form_goal_diff", 0.0),
            "form_clean_sheet_pct":   form.get("form_clean_sheet_pct", 0.3),
            "fifa_rank":              fifa_data.get("fifa_rank", 50.0),
            "fifa_points":            fifa_data.get("fifa_points", 1200.0),
            "confederation":          fifa_data.get("confederation", ""),
            "elo_rating":             elo_lookup.get(team, 1700.0),
            "wc_appearances":         wc_data.get("appearances", 0),
            "wc_titles":              wc_data.get("titles", 0),
            "wc_best_finish":         wc_data.get("best_finish", 8),
        }

    return team_lookup


# ---------------------------------------------------------------------------
# Step 2 — Build match feature row (V1 feature set — no match_importance/h2h)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "home_form_win_pct", "home_form_goals_scored", "home_form_goals_conceded",
    "home_form_goal_diff", "home_form_clean_sheet_pct",
    "away_form_win_pct", "away_form_goals_scored", "away_form_goals_conceded",
    "away_form_goal_diff", "away_form_clean_sheet_pct",
    "home_fifa_rank", "home_fifa_points",
    "away_fifa_rank", "away_fifa_points",
    "home_elo_rating", "away_elo_rating",
    "home_wc_appearances", "home_wc_titles", "home_wc_best_finish",
    "away_wc_appearances", "away_wc_titles", "away_wc_best_finish",
    "rank_diff", "points_diff", "elo_diff",
    "same_conf", "neutral",
]


def build_match_row(
    home: str,
    away: str,
    team_lookup: dict[str, dict],
) -> pd.DataFrame:
    h = team_lookup[home]
    a = team_lookup[away]

    row = {
        "home_form_win_pct":          h["form_win_pct"],
        "home_form_goals_scored":     h["form_goals_scored"],
        "home_form_goals_conceded":   h["form_goals_conceded"],
        "home_form_goal_diff":        h["form_goal_diff"],
        "home_form_clean_sheet_pct":  h["form_clean_sheet_pct"],
        "away_form_win_pct":          a["form_win_pct"],
        "away_form_goals_scored":     a["form_goals_scored"],
        "away_form_goals_conceded":   a["form_goals_conceded"],
        "away_form_goal_diff":        a["form_goal_diff"],
        "away_form_clean_sheet_pct":  a["form_clean_sheet_pct"],
        "home_fifa_rank":             h["fifa_rank"],
        "home_fifa_points":           h["fifa_points"],
        "away_fifa_rank":             a["fifa_rank"],
        "away_fifa_points":           a["fifa_points"],
        "home_elo_rating":            h["elo_rating"],
        "away_elo_rating":            a["elo_rating"],
        "home_wc_appearances":        h["wc_appearances"],
        "home_wc_titles":             h["wc_titles"],
        "home_wc_best_finish":        h["wc_best_finish"],
        "away_wc_appearances":        a["wc_appearances"],
        "away_wc_titles":             a["wc_titles"],
        "away_wc_best_finish":        a["wc_best_finish"],
        "rank_diff":                  h["fifa_rank"] - a["fifa_rank"],
        "points_diff":                h["fifa_points"] - a["fifa_points"],
        "elo_diff":                   h["elo_rating"] - a["elo_rating"],
        "same_conf":                  int(h["confederation"] == a["confederation"]),
        "neutral":                    True,
    }
    return pd.DataFrame([row])[FEATURE_COLS]


# ---------------------------------------------------------------------------
# Step 3 — Pre-compute pairwise win probabilities
# ---------------------------------------------------------------------------

def precompute_win_probs(
    teams: list[str],
    team_lookup: dict[str, dict],
    pipeline,
    label_encoder,
) -> dict[tuple[str, str], float]:
    classes  = list(label_encoder.classes_)
    away_idx = classes.index("away_win")
    draw_idx = classes.index("draw")
    home_idx = classes.index("home_win")

    probs: dict[tuple[str, str], float] = {}

    for home in teams:
        for away in teams:
            if home == away:
                continue
            row    = build_match_row(home, away, team_lookup)
            p      = pipeline.predict_proba(row)[0]
            p_home = p[home_idx] + p[draw_idx] * 0.5
            p_away = p[away_idx] + p[draw_idx] * 0.5
            total  = p_home + p_away
            probs[(home, away)] = p_home / total

    return probs


# ---------------------------------------------------------------------------
# Step 4 — Simulate a single match
# ---------------------------------------------------------------------------

def sim_match(
    home: str,
    away: str,
    win_probs: dict[tuple[str, str], float],
    rng: np.random.Generator,
) -> str:
    p_home = win_probs[(home, away)]
    return home if rng.random() < p_home else away


# ---------------------------------------------------------------------------
# Step 5 — Single tournament simulation
# ---------------------------------------------------------------------------

def simulate_once(
    win_probs: dict[tuple[str, str], float],
    rng: np.random.Generator,
) -> dict[str, str]:
    stages: dict[str, str] = {}

    r16_winners: list[str] = list(QF_CONFIRMED)

    for home, away in R16_REMAINING:
        winner = sim_match(home, away, win_probs, rng)
        loser  = away if winner == home else home
        r16_winners.append(winner)
        stages[loser] = "Round of 16"

    qf_winners: list[str] = []
    for i in range(0, 8, 2):
        home   = r16_winners[i]
        away   = r16_winners[i + 1]
        winner = sim_match(home, away, win_probs, rng)
        loser  = away if winner == home else home
        qf_winners.append(winner)
        stages[loser] = "Quarter-finals"

    sf_winners: list[str] = []
    for i in range(0, 4, 2):
        home   = qf_winners[i]
        away   = qf_winners[i + 1]
        winner = sim_match(home, away, win_probs, rng)
        loser  = away if winner == home else home
        sf_winners.append(winner)
        stages[loser] = "Semi-finals"

    finalist_a, finalist_b = sf_winners[0], sf_winners[1]
    champion  = sim_match(finalist_a, finalist_b, win_probs, rng)
    runner_up = finalist_b if champion == finalist_a else finalist_a

    stages[runner_up] = "Runner-up"
    stages[champion]  = "Champion"

    return stages


# ---------------------------------------------------------------------------
# Step 6 — Run Monte Carlo and aggregate
# ---------------------------------------------------------------------------

def run_simulation(
    win_probs: dict[tuple[str, str], float],
    n: int = N_SIMULATIONS,
) -> pd.DataFrame:
    print(f"Running {n:,} simulations...")

    all_teams = list(QF_CONFIRMED) + [t for pair in R16_REMAINING for t in pair]
    stage_counts: dict[str, dict[str, int]] = {
        team: {"Round of 16": 0, "Quarter-finals": 0, "Semi-finals": 0,
               "Runner-up": 0, "Champion": 0}
        for team in all_teams
    }

    rng = np.random.default_rng(seed=42)

    for _ in range(n):
        result = simulate_once(win_probs, rng)
        for team, stage in result.items():
            if team not in stage_counts:
                continue
            if stage in ("Champion", "Runner-up", "Semi-finals", "Quarter-finals", "Round of 16"):
                if team not in QF_CONFIRMED:
                    stage_counts[team]["Round of 16"] += 1
            if stage in ("Champion", "Runner-up", "Semi-finals", "Quarter-finals"):
                stage_counts[team]["Quarter-finals"] += 1
            if stage in ("Champion", "Runner-up", "Semi-finals"):
                stage_counts[team]["Semi-finals"] += 1
            if stage in ("Champion", "Runner-up"):
                stage_counts[team]["Runner-up"] += 1
            if stage == "Champion":
                stage_counts[team]["Champion"] += 1

    for team in QF_CONFIRMED:
        stage_counts[team]["Quarter-finals"] = n
        stage_counts[team]["Round of 16"]    = n

    rows = []
    for team in all_teams:
        counts = stage_counts[team]
        rows.append({
            "team":       team,
            "qf_pct":    round(counts["Quarter-finals"] / n * 100, 1),
            "sf_pct":    round(counts["Semi-finals"]    / n * 100, 1),
            "final_pct": round(counts["Runner-up"]      / n * 100, 1),
            "win_pct":   round(counts["Champion"]       / n * 100, 1),
        })

    df = pd.DataFrame(rows).sort_values("win_pct", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def simulate_tournament() -> None:
    print("Loading data and model (V1)...")
    features   = pd.read_csv(INPUT_FEATURES, parse_dates=["date"])
    fifa       = pd.read_csv(INPUT_FIFA)
    elo        = pd.read_csv(INPUT_ELO)
    wc_history = pd.read_csv(INPUT_WC_HIST)
    elo_map    = pd.read_csv(INPUT_ELO_MAP)

    with open(INPUT_MODEL, "rb") as f:
        model_payload = pickle.load(f)

    pipeline      = model_payload["pipeline"]
    label_encoder = model_payload["label_encoder"]
    print(f"  Model: {model_payload['model_name']}")

    all_teams = list(QF_CONFIRMED) + [t for pair in R16_REMAINING for t in pair]
    print(f"  Teams in simulation: {len(all_teams)}")

    print("Building team feature lookup...")
    team_lookup = build_team_lookup(features, fifa, elo, wc_history, elo_map, all_teams)

    print(f"\nPre-computing pairwise win probabilities ({len(all_teams)} teams)...")
    win_probs = precompute_win_probs(all_teams, team_lookup, pipeline, label_encoder)
    print(f"  {len(win_probs)} ordered pairs computed.")

    results = run_simulation(win_probs)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_RESULTS, index=False)

    print(f"\nSimulation results ({N_SIMULATIONS:,} runs):")
    print(results.to_string(index=False))
    print(f"\nSaved: {OUTPUT_RESULTS}")


if __name__ == "__main__":
    simulate_tournament()
