"""
Monte Carlo simulation of the 2026 FIFA World Cup.

The knockout bracket TREE is defined once in BRACKET_2026 (Round-of-16 matchups
and how winners feed the later rounds). The live STATE — which matches have been
played and who won — is read automatically from the fixtures feed every run via
derive_bracket_state(), so this script never needs manual editing as results come
in. Already-played matches are locked to their real winners; everything still
undecided is simulated.

Input:
    data/processed/features_v2.csv
    data/processed/fifa_rankings.csv
    data/processed/elo_ratings.csv
    data/processed/world_cup_history.csv
    data/raw/elo_code_map_v2.csv
    models/best_model_v2.pkl

Output:
    data/processed/simulation_results.csv
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

INPUT_FEATURES  = PROCESSED_DIR / "features_v2.csv"
INPUT_FIFA      = PROCESSED_DIR / "fifa_rankings.csv"
INPUT_ELO       = PROCESSED_DIR / "elo_ratings.csv"
INPUT_WC_HIST   = PROCESSED_DIR / "world_cup_history.csv"
INPUT_ELO_MAP   = Path("data/raw/elo_code_map_v2.csv")
INPUT_FIXTURES  = PROCESSED_DIR / "world_cup_fixtures.csv"
INPUT_MODEL     = MODELS_DIR / "best_model_v2.pkl"

OUTPUT_RESULTS  = PROCESSED_DIR / "simulation_results.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SIMULATIONS = 10_000

# Maps fixture/FIFA team names to historical results team names.
# Used to look up rolling form from features.csv.
FIXTURE_TO_RESULTS_NAME: dict[str, str] = {
    "USA": "United States",
}

# Maps fixture/FIFA team names to world_cup_history.csv team names.
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
# Bracket — static structure, live state derived from the fixtures feed
# ---------------------------------------------------------------------------
#
# The knockout bracket TREE is fixed for the whole tournament (the Round-of-16
# matchups and how winners feed the quarter-finals, semi-finals, and final).
# The current STATE (who has already won) is read automatically from
# data/processed/world_cup_fixtures.csv every run — so this file never needs to
# be hand-edited as results come in.
#
# Each match maps to a pair of "sources". A source is either:
#   - a literal team name (str)                → a Round-of-16 participant
#   - ("W", "<match_id>")                      → the winner of an earlier match
#
# Official 2026 bracket (Round of 16 → Final):
BRACKET_2026: dict[str, tuple] = {
    "R16_1": ("France",      "Paraguay"),
    "R16_2": ("Morocco",     "Canada"),
    "R16_3": ("Norway",      "Brazil"),
    "R16_4": ("England",     "Mexico"),
    "R16_5": ("Spain",       "Portugal"),
    "R16_6": ("USA",         "Belgium"),
    "R16_7": ("Argentina",   "Egypt"),
    "R16_8": ("Switzerland", "Colombia"),
    "QF_1":  (("W", "R16_1"), ("W", "R16_2")),
    "QF_2":  (("W", "R16_5"), ("W", "R16_6")),
    "QF_3":  (("W", "R16_3"), ("W", "R16_4")),
    "QF_4":  (("W", "R16_7"), ("W", "R16_8")),
    "SF_1":  (("W", "QF_1"),  ("W", "QF_2")),
    "SF_2":  (("W", "QF_3"),  ("W", "QF_4")),
    "FINAL": (("W", "SF_1"),  ("W", "SF_2")),
}

# Topological order (feeders before dependents) for resolving/simulating.
MATCH_ORDER: list[str] = [
    "R16_1", "R16_2", "R16_3", "R16_4", "R16_5", "R16_6", "R16_7", "R16_8",
    "QF_1", "QF_2", "QF_3", "QF_4", "SF_1", "SF_2", "FINAL",
]

# The elimination stage assigned to the LOSER of each match.
STAGE_OF_LOSS: dict[str, str] = {
    **{f"R16_{i}": "Round of 16"    for i in range(1, 9)},
    **{f"QF_{i}":  "Quarter-finals" for i in range(1, 5)},
    "SF_1": "Semi-finals", "SF_2": "Semi-finals",
    "FINAL": "Runner-up",
}

# Progress ranking used to compute cumulative reach percentages.
STAGE_RANK: dict[str, int] = {
    "Round of 16": 1,
    "Quarter-finals": 2,
    "Semi-finals": 3,
    "Runner-up": 4,
    "Champion": 5,
}

# The 16 Round-of-16 participants (bracket literals).
BRACKET_TEAMS: list[str] = [
    team for mid in MATCH_ORDER if mid.startswith("R16_")
    for team in BRACKET_2026[mid]
]


def _resolve_source(source, winners: dict[str, str]) -> str | None:
    """Resolve a bracket source to a concrete team name, or None if undecided."""
    if isinstance(source, str):
        return source
    _, match_id = source  # ("W", match_id)
    return winners.get(match_id)


def derive_bracket_state(fixtures: pd.DataFrame) -> dict[str, str]:
    """
    Read completed knockout results from the fixtures feed and map them onto the
    bracket tree. Returns {match_id: winning_team} for every match already played.

    A match is considered played when the feed contains a completed row whose two
    teams match the (resolved) participants of that bracket slot. Higher rounds
    resolve automatically once their feeder matches have real winners, so quarter-
    finals, semi-finals, and the final get picked up with no code changes.
    """
    played = fixtures[
        fixtures["winner"].notna()
        & (fixtures["winner"].astype(str).str.strip() != "")
        & (fixtures["winner"].astype(str).str.lower() != "nan")
    ]

    result_by_pair: dict[frozenset, str] = {}
    for _, row in played.iterrows():
        home, away = str(row["home_team"]), str(row["away_team"])
        result_by_pair[frozenset((home, away))] = str(row["winner"])

    real_winners: dict[str, str] = {}
    for match_id in MATCH_ORDER:
        src_a, src_b = BRACKET_2026[match_id]
        team_a = _resolve_source(src_a, real_winners)
        team_b = _resolve_source(src_b, real_winners)
        if team_a is None or team_b is None:
            continue  # participants not decided yet → not played
        winner = result_by_pair.get(frozenset((team_a, team_b)))
        if winner is not None:
            real_winners[match_id] = winner

    return real_winners


def alive_teams_from_state(real_winners: dict[str, str]) -> list[str]:
    """Return bracket teams that have not lost a completed knockout match."""
    eliminated: set[str] = set()
    for match_id, winner in real_winners.items():
        src_a, src_b = BRACKET_2026[match_id]
        team_a = _resolve_source(src_a, real_winners)
        team_b = _resolve_source(src_b, real_winners)
        loser = team_b if winner == team_a else team_a
        eliminated.add(loser)
    return [t for t in BRACKET_TEAMS if t not in eliminated]


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
    """
    Build a feature dictionary for each team, using their most recent match
    state from features.csv for form, and current snapshots for ranking/rating.
    """
    # Elo lookup: team → elo_rating
    elo_merged = elo_map.merge(elo[["country_code", "elo_rating"]],
                               left_on="elo_code", right_on="country_code", how="left")
    elo_lookup = dict(zip(elo_merged["team"], elo_merged["elo_rating"]))

    # FIFA lookup: team → {fifa_rank, fifa_points, confederation}
    fifa_lookup = fifa.set_index("team")[["fifa_rank", "fifa_points", "confederation"]].to_dict("index")

    # WC history lookup: team → {appearances, titles, best_finish}
    wc = wc_history.copy()
    for fixture_name, wc_name in FIXTURE_TO_WC_NAME.items():
        wc.loc[wc["team"] == wc_name, "team"] = fixture_name
    wc_lookup = wc.set_index("team")[["appearances", "titles", "best_finish"]].to_dict("index")

    # Form lookup: get most recent form values from features.csv per team
    # Features.csv uses historical result names; map fixture names where needed
    form_cols = [
        "form_win_pct", "form_goals_scored", "form_goals_conceded",
        "form_goal_diff", "form_clean_sheet_pct",
    ]

    team_lookup: dict[str, dict] = {}

    for team in teams:
        results_name = FIXTURE_TO_RESULTS_NAME.get(team, team)

        # Try home then away for form
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

        # Fill missing form with 0.5 neutral defaults
        if not form:
            form = {c: 0.5 for c in form_cols}
            form["form_goals_scored"]   = 1.5
            form["form_goals_conceded"] = 1.5
            form["form_goal_diff"]      = 0.0
            form["form_clean_sheet_pct"] = 0.3

        # Get static features
        fifa_data = fifa_lookup.get(team, {})
        wc_data   = wc_lookup.get(team, WC_HISTORY_DEFAULTS)

        team_lookup[team] = {
            **{f"form_{k}": v for k, v in form.items() if not k.startswith("form_")},
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
# Step 2 — Build match feature row
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
    "match_importance",   # V2 new
    "h2h_win_rate",       # V2 new
    "h2h_goal_diff",      # V2 new
]


def build_match_row(
    home: str,
    away: str,
    team_lookup: dict[str, dict],
) -> pd.DataFrame:
    """Build a single feature row for a match between home and away teams."""
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
        "neutral":                    True,   # all knockout matches at neutral venues
        "match_importance":           5,      # World Cup knockout = max importance
        "h2h_win_rate":               np.nan, # no prior context for this specific matchup
        "h2h_goal_diff":              np.nan,
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
    """
    Pre-compute P(home wins) for every ordered team pair.
    Draws are split equally between home and away.
    Called once; simulation loop then uses the cached values.
    """
    classes = list(label_encoder.classes_)
    away_idx = classes.index("away_win")
    draw_idx = classes.index("draw")
    home_idx = classes.index("home_win")

    probs: dict[tuple[str, str], float] = {}

    for i, home in enumerate(teams):
        for j, away in enumerate(teams):
            if home == away:
                continue
            row  = build_match_row(home, away, team_lookup)
            p    = pipeline.predict_proba(row)[0]
            p_home = p[home_idx] + p[draw_idx] * 0.5
            p_away = p[away_idx] + p[draw_idx] * 0.5
            total  = p_home + p_away
            probs[(home, away)] = p_home / total

    return probs


# ---------------------------------------------------------------------------
# Step 4 — Simulate a single match (using cached probs)
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
    real_winners: dict[str, str],
) -> dict[str, str]:
    """
    Run one complete simulation of the bracket, honouring matches that have
    already been played (real_winners) and simulating the rest.
    Returns a dict: team → furthest stage reached.
    """
    winners: dict[str, str] = dict(real_winners)  # seed with actual results
    stages: dict[str, str] = {}

    for match_id in MATCH_ORDER:
        src_a, src_b = BRACKET_2026[match_id]
        team_a = _resolve_source(src_a, winners)
        team_b = _resolve_source(src_b, winners)

        if match_id in winners:
            winner = winners[match_id]           # real, already-played result
        else:
            winner = sim_match(team_a, team_b, win_probs, rng)
            winners[match_id] = winner

        loser = team_b if winner == team_a else team_a
        stages[loser] = STAGE_OF_LOSS[match_id]

    stages[winners["FINAL"]] = "Champion"
    return stages


# ---------------------------------------------------------------------------
# Step 5 — Run Monte Carlo and aggregate
# ---------------------------------------------------------------------------

def run_simulation(
    win_probs: dict[tuple[str, str], float],
    real_winners: dict[str, str],
    alive_teams: list[str],
    n: int = N_SIMULATIONS,
) -> pd.DataFrame:
    """Run N simulations and return a dataframe of reach probabilities per team."""
    print(f"Running {n:,} simulations...")

    counts = {team: {"qf": 0, "sf": 0, "final": 0, "win": 0} for team in alive_teams}
    rng = np.random.default_rng(seed=42)

    for _ in range(n):
        stages = simulate_once(win_probs, rng, real_winners)
        for team in alive_teams:
            rank = STAGE_RANK[stages[team]]
            if rank >= 2:  counts[team]["qf"]    += 1  # lost in QF or later ⇒ reached QF
            if rank >= 3:  counts[team]["sf"]    += 1
            if rank >= 4:  counts[team]["final"] += 1
            if rank == 5:  counts[team]["win"]   += 1

    rows = [
        {
            "team":      team,
            "qf_pct":    round(counts[team]["qf"]    / n * 100, 1),
            "sf_pct":    round(counts[team]["sf"]    / n * 100, 1),
            "final_pct": round(counts[team]["final"] / n * 100, 1),
            "win_pct":   round(counts[team]["win"]   / n * 100, 1),
        }
        for team in alive_teams
    ]

    df = pd.DataFrame(rows).sort_values("win_pct", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def simulate_tournament() -> None:
    print("Loading data and model...")
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

    # --- Derive live bracket state from the fixtures feed ---
    fixtures     = pd.read_csv(INPUT_FIXTURES)
    real_winners = derive_bracket_state(fixtures)
    alive_teams  = alive_teams_from_state(real_winners)

    print("\nBracket state (auto-derived from fixtures feed):")
    for match_id in MATCH_ORDER:
        src_a, src_b = BRACKET_2026[match_id]
        team_a = _resolve_source(src_a, real_winners) or "TBD"
        team_b = _resolve_source(src_b, real_winners) or "TBD"
        if match_id in real_winners:
            status = f"played → {real_winners[match_id]}"
        elif team_a != "TBD" and team_b != "TBD":
            status = "to simulate"
        else:
            status = "awaiting feeders"
        print(f"  {match_id:6} {team_a:14} vs {team_b:14} | {status}")
    print(f"  Teams still alive: {len(alive_teams)}")

    print("\nBuilding team feature lookup...")
    team_lookup = build_team_lookup(features, fifa, elo, wc_history, elo_map, alive_teams)

    # Print team strengths for verification
    print("\nTeam feature snapshot:")
    print(f"  {'Team':25} {'FIFA Rank':>10} {'Elo':>8} {'WC Titles':>10} {'Form Win%':>10}")
    for team in sorted(alive_teams):
        t = team_lookup[team]
        print(f"  {team:25} {t['fifa_rank']:>10.0f} {t['elo_rating']:>8.0f} {t['wc_titles']:>10.0f} {t['form_win_pct']:>10.2f}")

    print(f"\nPre-computing pairwise win probabilities ({len(alive_teams)} teams)...")
    win_probs = precompute_win_probs(alive_teams, team_lookup, pipeline, label_encoder)
    print(f"  {len(win_probs)} ordered pairs computed.")

    results = run_simulation(win_probs, real_winners, alive_teams)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_RESULTS, index=False)

    print(f"\nSimulation results ({N_SIMULATIONS:,} runs):")
    print(results.to_string(index=False))
    print(f"\nSaved: {OUTPUT_RESULTS}")


if __name__ == "__main__":
    simulate_tournament()
