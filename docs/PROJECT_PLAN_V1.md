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

---

## Iteration 4

**Date:** 2026-07-05

### Stage

Model Training

### Original Plan

The original plan listed three models to compare (Logistic Regression, Random Forest, XGBoost) and six evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC, Log Loss). No implementation details were specified.

### Observations

**Train/test split strategy — temporal, not random**

A random train/test split would leak future information into training: a model could learn from a 2022 match while being asked to predict a 2010 match in the test set. Football has temporal structure — team strength, styles, and rosters change over time. A temporal split was used instead: all matches before 2018-01-01 form the training set, all matches from 2018 onward form the test set.

This gives 41,627 training rows (1872–2017) and 8,205 test rows (2018–2026), which includes the 2018, 2022, and 2026 World Cups in the evaluation window.

**scikit-learn 1.7 removed `multi_class` argument from LogisticRegression**

The `multi_class="multinomial"` argument was removed in scikit-learn 1.7 — multinomial output is now the default for `lbfgs`. The script raised a `TypeError` on first run. Fixed by removing the argument.

**XGBoost 3.x removed `use_label_encoder`**

The `use_label_encoder=False` argument was removed in XGBoost 3.x and the script produced a `UserWarning` on first run. Fixed by removing the argument.

**Primary selection metric: log loss**

Log loss measures how well-calibrated the predicted probabilities are, not just whether the top class is correct. This is the right metric for tournament simulation, where each match outcome probability feeds directly into the bracket progression logic. A model that correctly outputs 60% home win / 25% draw / 15% away win is more useful than one that is slightly more "accurate" in binary classification but poorly calibrated.

### Results

All three models were evaluated on the held-out test set (2018–2026).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Log Loss |
|---|---|---|---|---|---|---|
| XGBoost | 0.5918 | 0.5332 | 0.5918 | 0.5185 | 0.7467 | **0.8863** |
| Logistic Regression | 0.5905 | 0.5320 | 0.5905 | 0.5091 | 0.7419 | 0.8935 |
| Random Forest | 0.5909 | 0.4587 | 0.5909 | 0.5046 | 0.7474 | 0.8959 |

XGBoost achieved the best log loss (0.8863), accuracy (0.5918), and F1 (0.5185). All three models performed within a narrow band — football match outcomes are inherently uncertain and ~59% accuracy on a 3-class problem is a solid baseline.

### Decisions

1. **Temporal split at 2018-01-01** — train on 1872–2017, test on 2018–2026. Prevents future-data leakage and tests on the most recent tournament era.

2. **Primary selection metric: log loss** — probability calibration matters more than accuracy for simulation.

3. **XGBoost selected as best model** — lowest log loss, highest accuracy and F1. Saved to `models/best_model.pkl` for use in the simulation stage.

4. **All models use median imputation** — handles null FIFA rank (13%) and Elo rating (70%) consistently across all three pipelines. XGBoost can handle nulls natively but was wrapped in the same imputer for consistency.

### Version 1 Impact

Model training is complete. All three models trained and compared. Best model saved to disk.

| Output | Details |
|--------|---------|
| `data/processed/model_comparison.csv` | Metrics table for all three models |
| `models/best_model.pkl` | XGBoost pipeline + feature list + label encoder |
| `models/label_encoder.pkl` | LabelEncoder fitted on `away_win / draw / home_win` |

### Status

✅ Completed

---

## Iteration 5

**Date:** 2026-07-05

### Stage

Tournament Simulation

### Original Plan

The original plan described a Monte Carlo simulation: predict probabilities for every match, simulate thousands of times, and report the probability of each team reaching each stage. No implementation details were given.

### Observations

**Fixture file only covers matches up to the point of collection**

The `world_cup_fixtures.csv` file (collected during data collection) had 90 matches — all group stage, all Round of 32, and 2 Round of 16 matches. The remaining 6 Round of 16 matches, plus Quarter-finals, Semi-finals, and Final, were not yet scheduled in the FIFA API at collection time.

This meant the simulation could not be built by simply iterating the fixture file. The bracket had to be reconstructed from first principles.

**Bracket reconstruction from R32 pairing pattern**

The Round of 32 matchups follow a fixed bracket drawn before the tournament. By looking at which R32 winners played each other in the known Round of 16 results (Canada vs Morocco, Paraguay vs France), the bracket pairing pattern was identified:

- R32 matches are grouped in pairs: (1,6), (2,4), (3,8), (5,7), (9,14), (10,12), (11,15), (13,16)
- The two confirmed R16 results matched slots (1,6) and (2,4) exactly

From this, the remaining 6 Round of 16 matchups were derived:

| Match | Home | Away |
|---|---|---|
| R16-3 | Brazil | England |
| R16-4 | Norway | Belgium |
| R16-5 | Mexico | Egypt |
| R16-6 | Portugal | USA |
| R16-7 | Spain | Switzerland |
| R16-8 | Argentina | Colombia |

**Single-row model prediction is slow — fixed by pre-computing pairwise probabilities**

The first implementation called `pipeline.predict_proba()` inside the simulation loop: once per match per simulation. With 7 matches per simulation and 10,000 runs, that is 70,000 model calls at 1.3ms each — approximately 90 seconds total. The script ran silently for 2.5 minutes before being killed.

Fix: pre-compute P(home wins) for all 182 ordered team pairs (14 × 13) before the simulation loop. The loop then just does `random.random() < p_home`. Total model calls reduced from 70,000 to 182. Runtime dropped from ~90 seconds to 4 seconds.

**Draw resolution in knockout rounds**

The model outputs three-class probabilities: home_win, draw, away_win. In knockout rounds there are no draws. The draw probability was split equally between home win and away win, and the match was decided by sampling from the adjusted two-class distribution.

**USA name mismatch between historical results and FIFA names**

In `features.csv`, the United States appears as "United States" (from the historical results dataset). The FIFA rankings, Elo code map, and fixture file all use "USA". A `FIXTURE_TO_RESULTS_NAME` mapping was added to the simulation to correctly look up USA's rolling form from the historical data.

### Results

10,000 Monte Carlo simulations from current bracket state (Round of 16, 2026-07-05):

| Team | QF% | SF% | Final% | Win% |
|---|---|---|---|---|
| France | 100.0 | 63.9 | 36.0 | **21.8** |
| Brazil | 58.9 | 40.3 | 20.8 | 13.5 |
| Spain | 71.7 | 41.8 | 25.5 | 13.4 |
| Argentina | 68.1 | 36.3 | 21.9 | 11.3 |
| Morocco | 100.0 | 36.1 | 15.6 | 7.6 |
| England | 41.1 | 24.3 | 12.6 | 7.1 |
| Belgium | 64.3 | 26.7 | 12.3 | 7.1 |
| Portugal | 65.8 | 36.9 | 19.0 | 7.0 |
| Mexico | 63.6 | 33.1 | 15.1 | 4.8 |
| Colombia | 31.9 | 12.8 | 6.2 | 2.2 |
| USA | 34.2 | 15.6 | 4.6 | 1.3 |
| Switzerland | 28.3 | 9.0 | 4.0 | 1.2 |
| Norway | 35.7 | 8.7 | 2.7 | 0.9 |
| Egypt | 36.4 | 14.4 | 3.5 | 0.9 |

France is the model's pick to win the 2026 World Cup at 21.8%, followed by Brazil (13.5%) and Spain (13.4%).

### Decisions

1. **Bracket hard-coded from R32 pairing pattern** — the remaining R16 pairings were derived from the confirmed matches, not fetched from an API. All 8 R16 matchups are documented in the script.

2. **Pre-compute all pairwise win probabilities** — 182 model calls before the simulation, zero model calls inside the loop. 22× speedup (90s → 4s).

3. **Draw probability split 50/50** — draw probability is redistributed equally between home and away win for knockout rounds. Simple and unbiased.

4. **FIXTURE_TO_RESULTS_NAME map** — handles USA/United States name discrepancy for rolling form lookup.

### Version 1 Impact

Tournament simulation is complete. Championship probabilities produced for all 14 remaining teams.

| Output | Details |
|--------|---------|
| `data/processed/simulation_results.csv` | Per-team QF/SF/Final/Win probabilities from 10,000 simulations |

### Status

✅ Completed