# Update Log

A dated record of every refresh of the live 2026 World Cup prediction.

Each run archives the exact simulation script and its output under
`archive/<date>/`, so every published prediction is fully reproducible and the
original code is never lost.

**How to read this:** the canonical files the dashboard uses always live at their
normal paths (`src/simulation/simulate_tournament_v2.py`,
`data/processed/simulation_results.csv`). Dated copies in `archive/` are frozen
snapshots for documentation.

---

## 2026-07-09 — automated refresh

*Generated 2026-07-09 15:20 UTC by `run_daily_update.py`.*

**Archive:** [`archive/2026-07-09/`](../archive/2026-07-09/)

**Knockout results detected (8 teams still alive):**
  - R16_1: France vs Paraguay → **France**
  - R16_2: Morocco vs Canada → **Morocco**
  - R16_3: Norway vs Brazil → **Norway**
  - R16_4: England vs Mexico → **England**
  - R16_5: Spain vs Portugal → **Spain**
  - R16_6: USA vs Belgium → **Belgium**
  - R16_7: Argentina vs Egypt → **Argentina**
  - R16_8: Switzerland vs Colombia → **Switzerland**

**Model:** XGBoost (tuned) — accuracy 0.6215, log loss 0.8511, ROC-AUC 0.7751

**Champion odds (top 5, 10,000 simulations):**

| Rank | Team | Win % |
|------|------|-------|
| 1 | Argentina | 19.2 |
| 2 | France | 18.2 |
| 3 | England | 15.7 |
| 4 | Spain | 15.6 |
| 5 | Belgium | 11.1 |

---


## 2026-07-08 — automated refresh

*Generated 2026-07-08 14:25 UTC by `run_daily_update.py`.*

**Archive:** [`archive/2026-07-08/`](../archive/2026-07-08/)

**Knockout results detected (8 teams still alive):**
  - R16_1: France vs Paraguay → **France**
  - R16_2: Morocco vs Canada → **Morocco**
  - R16_3: Norway vs Brazil → **Norway**
  - R16_4: England vs Mexico → **England**
  - R16_5: Spain vs Portugal → **Spain**
  - R16_6: USA vs Belgium → **Belgium**
  - R16_7: Argentina vs Egypt → **Argentina**
  - R16_8: Switzerland vs Colombia → **Switzerland**

**Model:** XGBoost (tuned) — accuracy 0.6215, log loss 0.8511, ROC-AUC 0.7751

**Champion odds (top 5, 10,000 simulations):**

| Rank | Team | Win % |
|------|------|-------|
| 1 | Argentina | 19.2 |
| 2 | France | 18.2 |
| 3 | England | 15.7 |
| 4 | Spain | 15.6 |
| 5 | Belgium | 11.1 |

---


## 2026-07-07 — automated refresh

*Generated 2026-07-08 02:34 UTC by `run_daily_update.py`.*

**Archive:** [`archive/2026-07-07/`](../archive/2026-07-07/)

**Knockout results detected (8 teams still alive):**
  - R16_1: France vs Paraguay → **France**
  - R16_2: Morocco vs Canada → **Morocco**
  - R16_3: Norway vs Brazil → **Norway**
  - R16_4: England vs Mexico → **England**
  - R16_5: Spain vs Portugal → **Spain**
  - R16_6: USA vs Belgium → **Belgium**
  - R16_7: Argentina vs Egypt → **Argentina**
  - R16_8: Switzerland vs Colombia → **Switzerland**

**Model:** XGBoost (tuned) — accuracy 0.6215, log loss 0.8511, ROC-AUC 0.7751

**Champion odds (top 5, 10,000 simulations):**

| Rank | Team | Win % |
|------|------|-------|
| 1 | Argentina | 19.4 |
| 2 | France | 18.2 |
| 3 | England | 15.7 |
| 4 | Spain | 15.6 |
| 5 | Belgium | 11.1 |

---



## 2026-07-06 — Round of 16 in progress (first real bracket + retrain)

**Archive:** [`archive/2026-07-06/`](../archive/2026-07-06/)

**Milestones this run:**
- Switched the simulation to **auto-derive the bracket from the FIFA fixtures feed**
  (`derive_bracket_state`) — no more manual bracket edits.
- Introduced the one-command daily runner (`src/pipeline/run_daily_update.py`).
- Retrained the model to check for drift (see table) — confirmed, not changed.

**Knockout results detected (11 teams still alive):**
  - R16_1: France vs Paraguay → **France**
  - R16_2: Morocco vs Canada → **Morocco**
  - R16_3: Norway vs Brazil → **Norway**
  - R16_4: England vs Mexico → **England**
  - R16_5: Spain vs Portugal → **Spain**
  - Still to play: USA vs Belgium, Argentina vs Egypt, Switzerland vs Colombia

**Retrain outcome (essentially unchanged):**

| Metric | 2026-07-04 model | 2026-07-06 model |
|--------|------------------|------------------|
| Accuracy | 0.6208 | 0.6215 |
| Log loss | 0.851 | 0.8511 |
| ROC-AUC | 0.7752 | 0.7751 |

**Champion odds (top 5, 10,000 simulations):**

| Rank | Team | Win % |
|------|------|-------|
| 1 | France | 18.3 |
| 2 | Spain | 17.1 |
| 3 | England | 16.4 |
| 4 | Argentina | 14.7 |
| 5 | Morocco | 11.0 |

---



## 2026-07-04 — Original knockout prediction (Version 2 launch)

**Archive:** [`archive/2026-07-04/`](../archive/2026-07-04/)

**Bracket state (as modeled at launch):**
- France and Morocco confirmed into the quarter-finals.
- Six placeholder Round-of-16 matches simulated.

**Champion odds (10,000 simulations):**

| Rank | Team | Win % |
|------|------|-------|
| 1 | France | 19.8 |
| 2 | Morocco | 11.2 |
| 3 | Spain | 10.5 |
| 4 | Argentina | 10.4 |
| 5 | Portugal | 8.0 |

*Note:* this run used a placeholder bracket created before the real Round-of-16
pairings were known. The 2026-07-06 run replaced it with the actual bracket.
