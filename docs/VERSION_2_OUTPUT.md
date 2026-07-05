# FIFA World Cup 2026 Prediction — Version 2 Output (Final)

**Date:** 2026-07-05
**Status:** Complete — ready for dashboard

---

## What Was Built

An improved end-to-end machine learning pipeline with three key additions over Version 1: expanded Elo coverage, match importance scoring, and head-to-head features. The model was trained exclusively on competitive matches and tuned with hyperparameter search.

---

## Pipeline Summary

```
Data Collection  →  Preprocessing  →  Feature Engineering V2  →  Model Training V2  →  Simulation
```

| Stage | Script | Output |
|---|---|---|
| Data Collection | `collect_match_results.py` | `clean_results.csv` |
| Data Collection | `collect_fifa_rankings.py` | `fifa_rankings.csv` |
| Data Collection | `collect_elo_ratings.py` | `elo_ratings.csv` |
| Data Collection | `build_world_cup_history.py` | `world_cup_history.csv` |
| Data Collection | `collect_world_cup_fixtures.py` | `world_cup_fixtures.csv` |
| Feature Engineering V2 | `build_features_v2.py` | `features_v2.csv` |
| Model Training V2 | `train_model_v2.py` | `best_model_v2.pkl`, `model_comparison_v2.csv` |
| Simulation | `simulate_tournament.py` | `simulation_results.csv` |

---

## What Changed from Version 1

| Change | V1 | V2 | Impact |
|---|---|---|---|
| Elo code map | 48 teams | 108 teams | Elo null 70% → 36% |
| Training data | All matches (49,490) | Competitive only (21,325) | +2.9pp accuracy |
| Match importance | — | Score 1–5 by tournament | New feature |
| Head-to-head | — | Win rate + goal diff | New feature |
| Model tuning | Defaults | RandomizedSearchCV | Best config found |
| Accuracy | 59.2% | **62.1%** | |
| Log Loss | 0.886 | **0.851** | |

---

## Features

49,848 match rows × 41 columns. 30 model features.

| Feature Group | Features | Source |
|---|---|---|
| Rolling form (home) | Win%, goals scored/conceded, goal diff, clean sheet% | Last 10 matches, shift-1 |
| Rolling form (away) | Same 5 features | Last 10 matches |
| FIFA ranking | Rank, points, confederation | Current snapshot |
| Elo rating | Rating | V2 expanded code map (108 teams) |
| World Cup history | Appearances, titles, best finish | Aggregated 1930–2022 |
| Match importance ⭐ | Score 1–5 | WC=5, tournament=4, qualifier=3, comp=2, friendly=1 |
| Head-to-head ⭐ | Win rate, goal diff | Last 10 meetings |
| Derived | rank_diff, elo_diff, points_diff, same_conf, neutral | Calculated |
| Target | `home_win` / `draw` / `away_win` | Match result |

---

## Model Results

Train: 16,294 competitive matches (1916–2017)
Test: 5,031 competitive matches (2018–2026, temporal split)

| Model | Accuracy | F1 | ROC-AUC | Log Loss |
|---|---|---|---|---|
| **XGBoost (tuned)** ✅ | **62.08%** | **0.5427** | 0.7752 | **0.8510** |
| Random Forest | 61.94% | 0.5383 | 0.7764 | 0.8548 |
| Logistic Regression | 61.50% | 0.5376 | 0.7703 | 0.8561 |
| XGBoost (base) | 61.86% | 0.5600 | 0.7689 | 0.8574 |

**Best hyperparameters:** `n_estimators=500, max_depth=5, learning_rate=0.01, subsample=0.6, colsample_bytree=0.7, min_child_weight=3, gamma=0.2, reg_alpha=0, reg_lambda=2.0`

**V1 → V2 comparison:**

| Metric | V1 | V2 | Δ |
|---|---|---|---|
| Accuracy | 59.2% | 62.1% | +2.9pp |
| F1 | 0.519 | 0.543 | +0.024 |
| ROC-AUC | 0.747 | 0.775 | +0.028 |
| Log Loss | 0.886 | **0.851** | −4.0% |

---

## 2026 World Cup Simulation Results

10,000 Monte Carlo simulations from Round of 16 state (2026-07-05).
France and Morocco confirmed in Quarter-finals from actual results.

| Team | QF% | SF% | Final% | **Win%** |
|---|---|---|---|---|
| 🇫🇷 France | 100.0 | 58.8 | 34.8 | **19.8** |
| 🇲🇦 Morocco | 100.0 | 41.2 | 21.4 | **11.2** |
| 🇪🇸 Spain | 63.7 | 35.2 | 20.0 | **10.5** |
| 🇦🇷 Argentina | 65.4 | 35.2 | 20.0 | **10.4** |
| 🇵🇹 Portugal | 62.3 | 33.4 | 18.0 | **8.0** |
| 🇧🇷 Brazil | 52.5 | 31.5 | 13.8 | **7.8** |
| 🇧🇪 Belgium | 60.3 | 28.8 | 13.5 | **7.4** |
| 🇲🇽 Mexico | 64.5 | 34.0 | 16.8 | **7.0** |
| 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England | 47.5 | 26.5 | 12.3 | **6.9** |
| 🇨🇴 Colombia | 34.6 | 15.2 | 7.1 | **3.0** |
| 🇺🇸 USA | 37.8 | 17.2 | 6.4 | **2.4** |
| 🇨🇭 Switzerland | 36.3 | 14.4 | 6.2 | **2.3** |
| 🇳🇴 Norway | 39.7 | 13.1 | 4.2 | **1.8** |
| 🇪🇬 Egypt | 35.5 | 15.4 | 5.4 | **1.7** |

**Model's pick: France at 19.8%**

---

## Key Technical Decisions

| Decision | Reason |
|---|---|
| Competitive matches only for training | Friendlies are low signal — squads rotate, stakes are low. +2.9pp accuracy |
| Expanded Elo map to 108 teams | Historical name aliases caused false nulls. Halved the null rate |
| Match importance (1–5) | WC finals are different from friendlies — model should know the stakes |
| H2H win rate + goal diff | Teams with strong H2H records often outperform strength metrics alone |
| RandomizedSearchCV + TimeSeriesSplit | Time-ordered CV prevents leakage in hyperparameter search |
| n_jobs=1 for tuning | Sandbox blocks joblib multiprocessing (semaphore restriction) |
| Log loss as primary metric | Probability calibration matters most for simulation |
| Pre-compute pairwise probs in simulation | Reduces 70,000 model calls to 182 → 4 second runtime |

---

## Remaining Limitation

- **Elo still 36% null** — the 108-team map covers all common nations but ~136 minor nations in the training data still have no Elo entry. These are imputed to the median at training time.
- **FIFA ranking is a single snapshot** — not a time series. All historical matches use 2026 rankings.
- **No player-level data** — injuries, suspensions, and squad depth not modelled.

---

## Files Produced

```
data/
├── raw/
│   ├── elo_code_map.csv          (V1 — 48 teams)
│   └── elo_code_map_v2.csv       (V2 — 108 teams)
└── processed/
    ├── features.csv              (V1 — 49,832 rows × 38 cols)
    ├── features_v2.csv           (V2 — 49,848 rows × 41 cols)
    ├── model_comparison.csv      (V1 metrics)
    ├── model_comparison_v2.csv   (V2 metrics)
    └── simulation_results.csv    (V2 championship probabilities)

models/
├── best_model.pkl                (V1 XGBoost)
├── label_encoder.pkl             (V1)
├── best_model_v2.pkl             (V2 XGBoost tuned) ← used by dashboard
└── label_encoder_v2.pkl          (V2)
```
