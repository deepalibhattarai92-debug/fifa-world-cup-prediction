# FIFA World Cup Prediction

## Version 2 Project Plan (Final Version)

---

# Project Objective

Version 2 is the final version of the FIFA World Cup Prediction project. It builds on the complete Version 1 baseline pipeline and addresses every known limitation identified during V1 development.

The goal is to produce the most accurate possible match outcome predictions and World Cup championship probabilities using publicly available football data, without requiring paid data sources.

---

# What Changed from Version 1

| Area | Version 1 | Version 2 |
|---|---|---|
| Elo coverage | 48 teams (70% null) | 108 teams (36% null) |
| Training data | All matches including friendlies | Competitive matches only |
| Match importance | Not included | Score 1–5 by tournament type |
| Head-to-head | Not included | Win rate + goal diff (last 10) |
| Model tuning | Default hyperparameters | RandomizedSearchCV (40 configs × 5 folds) |
| Model accuracy | 59.2% | **62.1%** |
| Log Loss | 0.886 | **0.851** |

---

# Scope

## Data Sources

All data sources from Version 1 are retained. No new external sources are introduced.

| Dataset | Source | Notes |
|---|---|---|
| Historical Match Results | Kaggle (1872–2026) | Used for form + H2H features |
| FIFA World Rankings | `api.fifa.com/api/v3/rankings` | Current snapshot |
| World Football Elo Ratings | `eloratings.net/World.tsv` | Current snapshot |
| World Cup History | Fjelstul World Cup Database | 1930–2022 |
| 2026 WC Fixtures | `inside.fifa.com/api/data-centre/matches` | Collected pre-tournament |

## New Reference Files

| File | Purpose |
|---|---|
| `data/raw/elo_code_map_v2.csv` | Expanded FIFA→Elo code mapping (108 teams) |

## Machine Learning Models

All three V1 models retrained on the V2 feature set. Hyperparameter tuning applied to XGBoost.

- Logistic Regression
- Random Forest
- XGBoost (base)
- **XGBoost (tuned)** ← selected as best model

---

# Feature Set

49,848 match rows × 41 columns. 30 model features.

| Feature Group | Features | Notes vs V1 |
|---|---|---|
| Rolling form (home + away) | 10 features | Unchanged |
| FIFA ranking | 4 features | Unchanged |
| Elo rating | 2 features | Null rate halved (70% → 36%) |
| World Cup history | 6 features | Unchanged |
| Derived match features | rank_diff, elo_diff, points_diff, same_conf | Unchanged |
| Match context | neutral | Unchanged |
| Match importance ⭐ | 1 feature (score 1–5) | **New in V2** |
| Head-to-head ⭐ | h2h_win_rate, h2h_goal_diff | **New in V2** |

---

# Machine Learning Pipeline

```
Competitive matches only (21,325 rows)
        │
        ▼
Train / Test split (temporal: pre-2018 / 2018+)
        │
        ▼
Median imputation for nulls
        │
        ▼
Base model comparison (LR, RF, XGBoost)
        │
        ▼
XGBoost hyperparameter tuning
(RandomizedSearchCV, TimeSeriesSplit, 40 configs × 5 folds)
        │
        ▼
Best model selection on log loss
        │
        ▼
Monte Carlo simulation (10,000 runs)
```

## Hyperparameter Search Space

| Parameter | Values Searched |
|---|---|
| n_estimators | 200, 300, 400, 500 |
| max_depth | 3, 4, 5, 6, 7 |
| learning_rate | 0.01, 0.03, 0.05, 0.1 |
| subsample | 0.6, 0.7, 0.8, 0.9, 1.0 |
| colsample_bytree | 0.6, 0.7, 0.8, 0.9, 1.0 |
| min_child_weight | 1, 3, 5, 10 |
| gamma | 0, 0.1, 0.2, 0.5 |
| reg_alpha | 0, 0.01, 0.1, 1.0 |
| reg_lambda | 0.5, 1.0, 2.0, 5.0 |

**Best configuration found:**
`n_estimators=500, max_depth=5, learning_rate=0.01, subsample=0.6, colsample_bytree=0.7, min_child_weight=3, gamma=0.2, reg_alpha=0, reg_lambda=2.0`

---

# Model Evaluation

Same six metrics as Version 1.

| Metric | Purpose |
|---|---|
| Accuracy | Overall prediction accuracy |
| Precision | Correct positive predictions |
| Recall | Ability to identify true positives |
| F1 Score | Balance between Precision and Recall |
| ROC-AUC | Measures model discrimination |
| Log Loss | Evaluates probability calibration (primary metric) |

Log Loss is the primary selection metric because well-calibrated probabilities matter more than accuracy for tournament simulation.

---

# Tournament Simulation

Same Monte Carlo approach as Version 1, updated to use the V2 model.

- 10,000 simulations from current bracket state (Round of 16)
- Pairwise win probabilities pre-computed for all 182 ordered team pairs
- Draws redistributed 50/50 between home and away win (knockout rounds)
- Output: QF / SF / Final / Championship probability for each of 14 remaining teams

---

# Repository Roadmap

| Stage | Status |
|---|---|
| Data Collection | ✅ Complete |
| Preprocessing | ✅ Complete |
| Feature Engineering V1 | ✅ Complete |
| Model Training V1 | ✅ Complete |
| Tournament Simulation V1 | ✅ Complete |
| Feature Engineering V2 | ✅ Complete |
| Model Training V2 | ✅ Complete |
| Tournament Simulation V2 | ✅ Complete |
| Dashboard Development | ⬜ In Progress |

---

# Design Iterations

## Iteration 1 — V2 Feature Engineering and Model Improvement

**Date:** 2026-07-05

### Observations

**Elo null rate cause identified and fixed**

The 70% Elo null rate in V1 was caused by `elo_code_map.csv` only covering 48 teams. Most of the unmatched teams were either historical name aliases (United States → US, Ivory Coast → CI, Turkey → TR, Czech Republic → CZ, South Korea → KR, Iran → IR, DR Congo → CD) or active nations not included in the 48-team WC-specific map.

Solution: built `elo_code_map_v2.csv` with 108 entries. Null rate dropped from 70% to 36%.

**Competitive matches only — the most impactful single change**

Filtering out 18,388 friendlies from training improved accuracy by 2.9 percentage points (59.2% → 62.1%). Friendly matches are weak signal: teams rotate squads, experiment tactically, and are not strongly motivated to win. Training exclusively on competitive matches (FIFA WC, qualifiers, continental tournaments, Nations League) gives the model cleaner examples of genuine team strength.

**joblib multiprocessing blocked in sandbox**

`RandomizedSearchCV` with `n_jobs=-1` failed with `PermissionError` from `os.sysconf("SC_SEM_NSEMS_MAX")`. Fixed by setting `n_jobs=1`. Training ran in ~3 minutes single-threaded.

### Results

| Model | Accuracy | F1 | ROC-AUC | Log Loss |
|---|---|---|---|---|
| **XGBoost (tuned)** | **62.08%** | **0.5427** | 0.7752 | **0.8510** |
| Random Forest | 61.94% | 0.5383 | 0.7764 | 0.8548 |
| Logistic Regression | 61.50% | 0.5376 | 0.7703 | 0.8561 |
| XGBoost (base) | 61.86% | 0.5600 | 0.7689 | 0.8574 |

### Status

✅ Completed
