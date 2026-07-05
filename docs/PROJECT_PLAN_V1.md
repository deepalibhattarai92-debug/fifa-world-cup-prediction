# FIFA World Cup Prediction

## Version 1 Project Plan

---

# Project Objective

The objective of Version 1 is to build an end-to-end machine learning pipeline that predicts FIFA World Cup match outcomes and estimates each team's probability of winning the tournament.

The project will:

- Collect football data from free public data sources.
- Build a historical training dataset.
- Clean, preprocess, and integrate multiple datasets.
- Engineer football performance features.
- Train and compare baseline machine learning models.
- Predict match outcomes.
- Simulate the FIFA World Cup using Monte Carlo Simulation.
- Build an interactive Streamlit dashboard.
- Document the complete development process on GitHub and Substack.

---

# Scope

Version 1 focuses on building a complete baseline prediction system using publicly available football data.

## Data Sources

- Historical International Match Results
- FIFA World Rankings
- World Football Elo Ratings
- FIFA World Cup Historical Records
- FIFA World Cup Fixtures

## Machine Learning Models

- Logistic Regression
- Random Forest
- XGBoost

The best-performing model will be selected based on evaluation metrics.

## Dashboard Features

The dashboard will display:

- Team Strength Rankings
- Match Predictions
- Tournament Winner Probabilities
- Monte Carlo Simulation Results
- Model Performance Metrics
- Feature Importance

## Deliverables

The final Version 1 deliverables include:

- Automated data collection pipeline
- Feature engineering pipeline
- Machine learning prediction model
- Tournament simulation engine
- Interactive Streamlit dashboard
- GitHub repository
- Project documentation

## System Architecture

The FIFA World Cup Prediction system will follow the pipeline below:

Data Sources
        │
        ▼
Data Collection
        │
        ▼
Data Cleaning & Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
Machine Learning Model
        │
        ▼
Match Prediction
        │
        ▼
Monte Carlo Tournament Simulation
        │
        ▼
Interactive Streamlit Dashboard

## Data Sources

Version 1 will use five primary datasets.

| Dataset | Purpose | Collection Method | Refresh Frequency |
|----------|---------|-------------------|-------------------|
| Historical International Match Results | Train the machine learning model and calculate team performance metrics | Initial download + future automated updates | Periodically |
| FIFA World Rankings | Measure official team strength | Automated Python download | Monthly |
| World Football Elo Ratings | Measure dynamic team strength | Automated Python download | Daily |
| FIFA World Cup Historical Records | Calculate tournament experience features | One-time download | Rarely |
| FIFA World Cup Fixtures | Simulate the current tournament | Automated Python download | During the tournament |

## Feature Mapping

The following table documents every feature used in the machine learning model, its source, how it is calculated, and whether it is included in Version 1.

| Feature | Category | Source Dataset | Raw Variables Required | Engineered | Version 1 | Notes |
|---------|----------|----------------|------------------------|------------|------------|------|
| FIFA Rank | Team Strength | FIFA World Rankings | Rank | No | ✅ | Lower rank indicates stronger team |
| FIFA Points | Team Strength | FIFA World Rankings | Points | No | ✅ | Official FIFA rating |
| Elo Rating | Team Strength | World Football Elo Ratings | Elo Rating | No | ✅ | Dynamic team strength |
| Win Percentage | Recent Form | Historical Match Results | Match Result | Yes | ✅ | Last 10 matches |
| Goals Scored per Match | Attack | Historical Match Results | Goals Scored | Yes | ✅ | Last 10 matches |
| Goals Conceded per Match | Defense | Historical Match Results | Goals Conceded | Yes | ✅ | Last 10 matches |
| Goal Difference | Attack/Defense | Historical Match Results | Goals Scored, Goals Conceded | Yes | ✅ | Last 10 matches |
| Clean Sheet Percentage | Defense | Historical Match Results | Goals Conceded | Yes | ✅ | Last 10 matches |
| Average Opponent Elo | Strength of Schedule | Historical Matches + Elo | Opponent, Elo | Yes | ✅ | Average opponent quality |
| Average Opponent FIFA Rank | Strength of Schedule | Historical Matches + FIFA | Opponent Rank | Yes | ✅ | Average opponent ranking |
| World Cup Appearances | Experience | World Cup History | Appearances | No | ✅ | Tournament experience |
| World Cup Championships | Experience | World Cup History | Titles | No | ✅ | Winning pedigree |
| Best World Cup Finish | Experience | World Cup History | Best Finish | Yes | ✅ | Converted to numeric score |
| Host Nation | Match Context | World Cup Fixtures | Host Country | Yes | ✅ | Binary feature |
| Neutral Venue | Match Context | Historical Match Results | Neutral Venue | Yes | ✅ | Binary feature |
| Confederation | Geography | FIFA World Rankings | Confederation | No | ✅ | UEFA, CONMEBOL, etc. |
| Recent Form Score | Engineered | Calculated | Win %, Goal Difference | Yes | ✅ | Composite score |
| Attack Score | Engineered | Calculated | Goals/Game, Goal Difference | Yes | ✅ | Composite score |
| Defense Score | Engineered | Calculated | Goals Conceded, Clean Sheets | Yes | ✅ | Composite score |
| Experience Score | Engineered | Calculated | Appearances, Titles, Best Finish | Yes | ✅ | Composite score |
| Team Strength Score | Dashboard | Calculated | Elo, FIFA, Form, Experience | Yes | Dashboard Only | Used for visualization, not training |

## Feature Engineering

Feature engineering transforms raw football data into meaningful variables that improve the predictive performance of the machine learning models.

The following engineered features will be created during Version 1.

| Feature | Formula / Method | Reason |
|----------|------------------|--------|
| Win Percentage | Wins ÷ Last 10 Matches | Measures recent performance |
| Goals Scored per Match | Total Goals Scored ÷ Matches Played | Measures attacking ability |
| Goals Conceded per Match | Total Goals Conceded ÷ Matches Played | Measures defensive ability |
| Goal Difference | Goals Scored − Goals Conceded | Overall team quality |
| Clean Sheet Percentage | Clean Sheets ÷ Matches Played | Defensive consistency |
| Average Opponent Elo | Mean Elo Rating of Opponents | Measures strength of schedule |
| Average Opponent FIFA Rank | Mean FIFA Rank of Opponents | Additional schedule strength |
| Recent Form Score | Weighted combination of Win %, Goal Difference and Goals/Game | Measures current momentum |
| Attack Score | Weighted attacking metrics | Overall offensive capability |
| Defense Score | Weighted defensive metrics | Overall defensive capability |
| Experience Score | Weighted combination of World Cup appearances, titles and best finish | Tournament experience |
| Team Strength Score | Weighted combination of Elo, FIFA Rank, Form and Experience | Dashboard visualization only |

## Machine Learning Pipeline

Version 1 will compare multiple supervised machine learning models to determine which model best predicts international football match outcomes.

### Workflow

Historical Data

↓

Data Cleaning

↓

Feature Engineering

↓

Train/Test Split

↓

Model Training

↓

Model Evaluation

↓

Best Model Selection

↓

Match Prediction

↓

Tournament Simulation

### Models

- Logistic Regression
- Random Forest
- XGBoost

The same training dataset and engineered features will be used for each model to ensure a fair comparison.

## Model Evaluation

The following evaluation metrics will be used to compare machine learning models.

| Metric | Purpose |
|---------|---------|
| Accuracy | Overall prediction accuracy |
| Precision | Correct positive predictions |
| Recall | Ability to identify true positives |
| F1 Score | Balance between Precision and Recall |
| ROC-AUC | Measures model discrimination |
| Log Loss | Evaluates probability predictions |

The best-performing model will be selected based on overall performance across these evaluation metrics rather than a single metric.

## Tournament Simulation

After predicting match probabilities, the selected machine learning model will be used to simulate the FIFA World Cup tournament.

Simulation Process

1. Predict the probability of every match.
2. Simulate each match outcome using predicted probabilities.
3. Advance the winning team through the tournament bracket.
4. Repeat the tournament thousands of times using Monte Carlo Simulation.
5. Calculate the probability of each team reaching every tournament stage.

### Simulation Output

- Group Stage Qualification Probability
- Round of 16 Probability
- Quarter-final Probability
- Semi-final Probability
- Final Probability
- Championship Probability

## Dashboard Design

The Streamlit dashboard will present predictions and simulation results through an interactive web application.

### Dashboard Pages

### 1. Tournament Overview

Displays:

- Team Strength Rankings
- Championship Probabilities
- Top Tournament Contenders

### 2. Match Predictor

Users can:

- Select Team A
- Select Team B

Outputs:

- Win Probability
- Draw Probability
- Loss Probability

### 3. Tournament Simulation

Displays:

- Tournament Bracket
- Stage Advancement Probabilities
- Championship Probability

### 4. Model Performance

Displays:

- Model Comparison
- Feature Importance
- Confusion Matrix
- Evaluation Metrics

## Repository Roadmap

| Stage | Status |
|---------|---------|
| Project Setup | ✅ Complete |
| Data Collection | ✅ Complete |
| Data Cleaning | ⬜ Planned |
| Feature Engineering | ⬜ Planned |
| Model Training | ⬜ Planned |
| Model Evaluation | ⬜ Planned |
| Tournament Simulation | ⬜ Planned |
| Dashboard Development | ⬜ Planned |
| Deployment | ⬜ Planned |

## Future Improvements

Version 2

- Additional engineered features
- Rolling performance metrics
- Hyperparameter tuning
- Automated feature selection

Version 3

- Squad market values
- Player ratings
- Expected Goals (xG)
- Injury information

Version 4

- Automated data refresh pipeline
- CI/CD deployment
- Docker containerization
- Cloud deployment
- Real-time prediction updates

# Design Iterations

This section documents design decisions, architectural changes, and implementation improvements made during the development of Version 1. The original project scope remains unchanged unless explicitly stated.

---

## Iteration 1

**Date:** 2026-07-04

### Stage

Data Exploration & Preprocessing

### Original Plan

The original Version 1 plan assumed preprocessing would primarily focus on the historical match results dataset before moving on to collecting external datasets (FIFA Rankings, Elo Ratings, and World Cup History).

### Observation

During exploration of the downloaded historical football dataset, four local datasets were identified:

- results.csv
- former_names.csv
- shootouts.csv
- goalscorers.csv

Rather than treating `results.csv` as the only local dataset, the remaining files were evaluated to determine their role within the overall machine learning pipeline.

### Decision

The preprocessing phase was expanded to include every local dataset before beginning external data collection.

The following preprocessing scripts were added:

- clean_match_results.py
- clean_former_names.py
- clean_shootouts.py
- clean_goalscorers.py

### Rationale

This change establishes a consistent preprocessing pipeline for every locally available dataset.

Benefits include:

- Standardized preprocessing architecture.
- Early validation of all available data.
- Improved scalability for future model versions.
- Reduced technical debt when introducing new features.

### Version 1 Impact

The Version 1 machine learning model remains unchanged.

Only `results.csv` and `former_names.csv` are expected to contribute directly to the baseline model.

`shootouts.csv` and `goalscorers.csv` have been prepared for future versions without affecting the Version 1 feature set.

### Status

✅ Completed

---

## Iteration 2

**Date:** 2026-07-05

### Stage

Data Collection

### Original Plan

The original plan listed five data sources to be collected using "automated Python download" as the collection method for all external datasets. No specific source URLs, API endpoints, or technical constraints were documented.

### Observations

Several design decisions and technical issues were encountered during implementation.

**FIFA Rankings — Akamai bot protection**

The initial attempt used `pandas.read_html()` against the FIFA website. This failed for two reasons:

1. `lxml` was not installed, causing an `ImportError`.
2. Even after fixing the import, the FIFA rankings page is a Next.js client-rendered app. The `<tbody>` is empty in server-returned HTML. No ranking rows exist in the static page.

Investigation of the page's network requests revealed that the browser calls a JSON API directly:

`https://api.fifa.com/api/v3/rankings?gender=1`

This endpoint returns structured JSON with all 211 men's teams. However, requests using the default `python-requests` User-Agent (`python-requests/2.34.2`) receive HTTP 403.

Root cause: Akamai Bot Manager blocks requests whose TLS fingerprint matches the known `python-requests` signature. The fix is a session-level Chrome-like `User-Agent` — not cookies, not `Referer`, not `Origin` headers.

**World Football Elo Ratings — direct TSV**

The eloratings.net website is also JavaScript-rendered but serves its data as a plain tab-separated file:

`https://www.eloratings.net/World.tsv`

No authentication or bot protection was encountered. A minor parsing issue was found: `pandas.read_csv()` treated `NA` (Namibia's country code) as a null value. Fixed with `na_filter=False`.

**World Cup History — women's data contamination**

The Fjelstul World Cup Database was selected as the source for World Cup history features. The database contains both men's and women's tournaments under the same `WC-YYYY` ID scheme.

Initial filtering used:

```python
series.str.contains("Men", case=False, na=False)
```

This accidentally matched `"Women's World Cup"` because `pandas` treats the pattern as regex by default, where `.` matches any character including the apostrophe in `"Women's"`.

Fixed with:

```python
series.str.contains("FIFA Men's World Cup", case=False, na=False, regex=False)
& ~series.str.contains("Women", case=False, na=False, regex=False)
```

After the fix, Brazil's appearances dropped from 30 to 22, which matches the correct historical record.

**World Cup Fixtures — FIFA data centre API**

The original plan did not specify a source for 2026 fixtures. Investigated the FIFA website and identified an undocumented internal API:

`https://inside.fifa.com/api/data-centre/matches?gender=1&competitionClassificationCode=FWC&year=2026`

This returns 90 matches with stage, date, teams, scores, and stadium. The `winner` field is a team ID string, not a team name — resolved by matching against `teamAId` / `teamBId` in the same record. The `stadiumName` field is a locale list — extracted using a shared helper function.

### Decisions

1. **FIFA Rankings** — collect from `api.fifa.com/api/v3/rankings?gender=1` using a `requests.Session()` with Chrome-like headers. Validate that the response is JSON and not an Akamai HTML block page before parsing.

2. **Elo Ratings** — download `eloratings.net/World.tsv` directly. Use `na_filter=False` to prevent `NA` (Namibia) being treated as null.

3. **World Cup History** — build from the Fjelstul World Cup Database using `team_appearances`, `tournament_standings`, and `tournaments` tables. Filter to men's tournaments with `regex=False`.

4. **World Cup Fixtures** — collect from `inside.fifa.com/api/data-centre/matches`. Resolve team ID fields to readable names using the same match record.

5. **Scope confirmed: Version 1 is men's only.** All datasets and filters were audited to confirm no women's data enters the processed pipeline.

### Version 1 Impact

Data collection is complete. All five datasets planned for Version 1 have been collected and saved to `data/processed/`.

| Dataset | Processed File |
|---------|---------------|
| Match Results | `clean_results.csv` |
| FIFA Rankings | `fifa_rankings.csv` |
| Elo Ratings | `elo_ratings.csv` |
| World Cup History | `world_cup_history.csv` |
| World Cup Fixtures | `world_cup_fixtures.csv` |

### Status

✅ Completed

---

## Iteration 3

**Date:** 2026-07-04

### Stage

Feature Engineering

### Original Plan

The original plan described feature engineering as a single step: "build rolling form, join rankings and history." No detail was given on how the five datasets would be joined, what the country code schemes were, or how defunct historical teams would be handled.

### Observations

**Elo country code system mismatch**

The `elo_ratings.csv` file uses its own 2-letter country codes (e.g. `PT`, `CH`, `DE`) which are neither ISO 3166-1 alpha-2 nor FIFA's 3-letter codes. A naive approach of truncating the FIFA 3-letter code to 2 characters partially worked (181 of 211 teams matched) but produced four silent collisions:

| Truncated code | Team A | Team B |
|---|---|---|
| `AU` | Australia | Austria |
| `IR` | IR Iran | Iraq |
| `PA` | Panama | Paraguay |
| `CO` | Colombia | Congo DR |

In each collision, the truncation maps two different teams to the same 2-letter code. One team gets the correct Elo rating; the other silently gets the wrong team's rating. This is worse than a null — it introduces incorrect data without any error.

Eight additional teams have codes where truncation simply produces the wrong result (e.g. FIFA `POR` → truncated `PO`, but Elo uses `PT`; FIFA `SUI` → `SU`, but Elo uses `CH`).

**Decision:** Build a curated `data/raw/elo_code_map.csv` — a one-time 48-row file covering all 2026 World Cup fixture teams with manually verified FIFA-to-Elo code mappings. All 48 codes were verified against the live Elo ratings file before saving.

**Elo null rate in training data (70%) is expected**

The `elo_code_map.csv` covers the 48 teams in the 2026 tournament. The training dataset spans matches from 1872 onwards and includes ~200+ distinct nations, many of which are now defunct (West Germany, Soviet Union, Czechoslovakia, Yugoslavia, Saarland, etc.). These teams have no entry in the current Elo ratings file and therefore get null Elo features in `features.csv`.

This null rate (70%) is not a bug. It reflects a structural limitation: the current Elo ratings are a live snapshot, not a historical time series. For Version 1, this is acceptable — the model will learn from rows where Elo is available, and null rows can be imputed or filtered. A future version could use historical Elo snapshots to fill these gaps.

**pandas 3.0 breaking change in `groupby().apply()`**

The rolling form logic was initially written using `groupby("team").apply(func)`, which is the standard pattern in pandas 2.x. In pandas 3.0, the group-key column (`team`) is excluded from the DataFrame passed to the applied function. The `include_groups=True` argument that allowed restoring the old behaviour was fully removed in pandas 3.0 (not deprecated — hard removed).

The fix was to rewrite the rolling form logic without `apply()` entirely:

```python
# Shift within group
shifted = team_matches.groupby("team")[col].shift(1)

# Roll on the shifted series
team_matches[f"{col}_rolled"] = (
    shifted.groupby(team_matches["team"])
    .transform(lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean())
)
```

This approach is more explicit and forward-compatible with pandas 3+.

**World Cup history — 7 name mismatches and 4 genuine first-timers**

The fixture names used by the FIFA API do not always match the names used in the Fjelstul World Cup Database. Seven teams required a manual name map:

| Fixture name | WC history name |
|---|---|
| Czechia | Czech Republic |
| Korea Republic | South Korea |
| USA | United States |
| IR Iran | Iran |
| Türkiye | Turkey |
| Côte d'Ivoire | Ivory Coast |
| Congo DR | DR Congo |

Four teams (Cabo Verde, Curaçao, Uzbekistan, Jordan) have no WC history at all — they are genuine first-time qualifiers. These received default values: `appearances=0`, `titles=0`, `best_finish=8` (group stage).

### Decisions

1. **Elo code mapping** — build `data/raw/elo_code_map.csv` with manually verified FIFA-to-Elo code mappings for all 48 fixture teams. Do not use truncation, which causes silent data corruption via collisions.

2. **70% Elo null rate in training data is accepted** — the Elo lookup only covers 48 current teams. Historical matches involving defunct nations are expected to have null Elo features. Version 1 model will impute or filter these at training time.

3. **Rolling form rewritten for pandas 3.0** — use `groupby().shift()` and `groupby().transform()` instead of `groupby().apply()`. The old `apply()` pattern is incompatible with pandas 3.0.

4. **WC history name map baked into `build_features.py`** — the `FIXTURE_TO_WC_NAME` dictionary maps FIFA fixture names to WC history names. First-time qualifiers are filled with zeros, not dropped.

### Version 1 Impact

Feature engineering is complete. `data/processed/features.csv` is ready for model training.

| Output | Details |
|--------|---------|
| `features.csv` | 49,832 rows × 38 columns |
| Features per team | 5 rolling form, 3 FIFA, 1 Elo, 3 WC history |
| Derived features | `rank_diff`, `elo_diff`, `points_diff`, `same_conf` |
| Target variable | `result` — `home_win` / `away_win` / `draw` |

### Status

✅ Completed