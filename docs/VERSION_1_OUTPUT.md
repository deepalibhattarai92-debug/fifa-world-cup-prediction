# FIFA World Cup 2026 Prediction — Version 1 Output

**Date:** 2026-07-05
**Status:** Complete (simulation stage reached; dashboard pending)

---

## What Was Built

An end-to-end machine learning pipeline that collects football data, engineers features, trains a prediction model, and simulates the 2026 FIFA World Cup tournament using Monte Carlo simulation.

---

## Pipeline Summary

```
Data Collection  →  Preprocessing  →  Feature Engineering  →  Model Training  →  Simulation
```

| Stage | Script | Output |
|---|---|---|
| Data Collection | `collect_match_results.py` | `clean_results.csv` |
| Data Collection | `collect_fifa_rankings.py` | `fifa_rankings.csv` |
| Data Collection | `collect_elo_ratings.py` | `elo_ratings.csv` |
| Data Collection | `build_world_cup_history.py` | `world_cup_history.csv` |
| Data Collection | `collect_world_cup_fixtures.py` | `world_cup_fixtures.csv` |
| Feature Engineering | `build_features.py` | `features.csv` |
| Model Training | `train_model.py` | `best_model.pkl`, `model_comparison.csv` |
| Simulation | `simulate_tournament.py` | `simulation_results.csv` |

---

## Datasets

| Dataset | Source | Rows |
|---|---|---|
| Historical match results | Kaggle (1872–2026) | 49,490 |
| FIFA World Rankings | `api.fifa.com/api/v3/rankings` | 211 teams |
| World Football Elo Ratings | `eloratings.net/World.tsv` | 244 teams |
| World Cup History | Fjelstul World Cup Database (1930–2022) | 85 teams |
| 2026 World Cup Fixtures | `inside.fifa.com/api/data-centre/matches` | 90 matches |

All data is men's international football only.

---

## Features

49,832 match rows × 38 columns. One row per historical match.

| Feature Group | Features | Source |
|---|---|---|
| Rolling form (home) | Win%, goals scored, goals conceded, goal diff, clean sheet% | Last 10 matches, shift-1 leakage protection |
| Rolling form (away) | Same 5 features | Last 10 matches |
| FIFA ranking | Rank, points, confederation | Current snapshot |
| Elo rating | Rating | Current snapshot via curated code map |
| World Cup history | Appearances, titles, best finish | Aggregated 1930–2022 |
| Match context | Rank diff, Elo diff, points diff, same confederation, neutral venue | Derived |
| Target | `home_win` / `draw` / `away_win` | Match result |

---

## Model Results

Train: 41,627 matches (1872–2017)
Test: 8,205 matches (2018–2026, temporal split)

| Model | Accuracy | F1 | ROC-AUC | Log Loss |
|---|---|---|---|---|
| **XGBoost** ✅ | **59.18%** | **0.5185** | 0.7467 | **0.8863** |
| Logistic Regression | 59.05% | 0.5091 | 0.7419 | 0.8935 |
| Random Forest | 59.09% | 0.5046 | 0.7474 | 0.8959 |

XGBoost selected as best model on log loss (best for probability-calibrated simulation).

---

## 2026 World Cup Simulation Results

10,000 Monte Carlo simulations from Round of 16 state (2026-07-05).
France and Morocco confirmed in Quarter-finals from actual results.

| Team | QF% | SF% | Final% | **Win%** |
|---|---|---|---|---|
| 🇫🇷 France | 100.0 | 63.9 | 36.0 | **21.8** |
| 🇧🇷 Brazil | 58.9 | 40.3 | 20.8 | **13.5** |
| 🇪🇸 Spain | 71.7 | 41.8 | 25.5 | **13.4** |
| 🇦🇷 Argentina | 68.1 | 36.3 | 21.9 | **11.3** |
| 🇲🇦 Morocco | 100.0 | 36.1 | 15.6 | **7.6** |
| 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England | 41.1 | 24.3 | 12.6 | **7.1** |
| 🇧🇪 Belgium | 64.3 | 26.7 | 12.3 | **7.1** |
| 🇵🇹 Portugal | 65.8 | 36.9 | 19.0 | **7.0** |
| 🇲🇽 Mexico | 63.6 | 33.1 | 15.1 | **4.8** |
| 🇨🇴 Colombia | 31.9 | 12.8 | 6.2 | **2.2** |
| 🇺🇸 USA | 34.2 | 15.6 | 4.6 | **1.3** |
| 🇨🇭 Switzerland | 28.3 | 9.0 | 4.0 | **1.2** |
| 🇳🇴 Norway | 35.7 | 8.7 | 2.7 | **0.9** |
| 🇪🇬 Egypt | 36.4 | 14.4 | 3.5 | **0.9** |

**Model's pick: France at 21.8%**

---

## Key Technical Decisions

| Decision | Reason |
|---|---|
| Temporal train/test split at 2018 | Prevents future-data leakage; tests on the modern tournament era |
| Log loss as primary selection metric | Probability calibration matters more than accuracy for simulation |
| Median imputation for null features | Handles 13% null FIFA rank and 70% null Elo (historical/defunct teams) |
| Pre-compute pairwise win probabilities | Reduced simulation runtime from ~90s to 4s (70,000 → 182 model calls) |
| Elo code map (curated, 48 teams) | Avoids 4 silent data collisions from naive 2-char truncation |
| `regex=False` for men's WC filter | Prevents women's tournament contamination via regex `.` wildcard |
| `na_filter=False` for Elo TSV | Prevents Namibia (`NA`) being parsed as null |

---

## Limitations

- **Elo features sparse in training data (70% null)** — the Elo lookup covers only the 48 current WC teams. Historical teams (West Germany, Soviet Union, etc.) have no Elo rating.
- **FIFA rankings are a single current snapshot** — not a time series. Historical matches use today's ranking, not the ranking at the time of the match.
- **No player-level features** — goalscorers and shootout datasets collected but not used in Version 1.
- **No injury/suspension data** — team strength assumed static per snapshot.
- **Bracket structure hard-coded** — the remaining Round of 16 pairings were derived from the bracket pattern, not fetched from an official API.

---

## Version 2 Ideas

- Historical FIFA ranking snapshots instead of single current snapshot
- Historical Elo time series to fill the 70% null gap
- Player availability and injury data
- Head-to-head record features
- Tournament-specific form (WC qualifying form vs friendlies)
- Hyperparameter tuning (cross-validated grid search)
- Neural network model comparison

---

## Files Produced

```
data/
├── raw/
│   ├── results.csv
│   ├── former_names.csv
│   ├── shootouts.csv
│   ├── goalscorers.csv
│   ├── elo_code_map.csv
│   └── fifa_rankings_YYYYMMDD.json
└── processed/
    ├── clean_results.csv
    ├── former_names.csv
    ├── fifa_rankings.csv
    ├── elo_ratings.csv
    ├── world_cup_history.csv
    ├── world_cup_fixtures.csv
    ├── features.csv
    ├── model_comparison.csv
    └── simulation_results.csv

models/
├── best_model.pkl
└── label_encoder.pkl
```
