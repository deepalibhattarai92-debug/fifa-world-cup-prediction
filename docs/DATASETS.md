# Dataset Inventory

| Dataset | File / Source | Collection Method | Status | Used in Version 1 | Purpose |
|---|---|---|---|---|---|
| Historical Match Results | `data/raw/results.csv` | Kaggle download | ✅ Collected + preprocessed | ✅ Yes | Training data and recent performance features |
| Former Team Names | `data/raw/former_names.csv` | Kaggle download | ✅ Collected + preprocessed | ✅ Yes | Standardize historical team names |
| Shootouts | `data/raw/shootouts.csv` | Kaggle download | ✅ Collected + preprocessed | Future | Penalty shootout analysis |
| Goalscorers | `data/raw/goalscorers.csv` | Kaggle download | ✅ Collected + preprocessed | Future | Player-level features |
| FIFA World Rankings | `api.fifa.com/api/v3/rankings` via `collect_fifa_rankings.py` | Official JSON API | ✅ Collected | ✅ Yes | FIFA rank and FIFA points |
| World Football Elo Ratings | `eloratings.net/World.tsv` via `collect_elo_ratings.py` | Official TSV download | ✅ Collected | ✅ Yes | Dynamic team strength |
| World Cup History | Fjelstul World Cup Database via `build_world_cup_history.py` | Curated aggregation (1930-2022) | ✅ Collected | ✅ Yes | Appearances, titles, best finish |
| World Cup Fixtures | `inside.fifa.com/api/data-centre/matches` via `collect_world_cup_fixtures.py` | Official FIFA API | ✅ Collected | ✅ Yes | Tournament simulation |