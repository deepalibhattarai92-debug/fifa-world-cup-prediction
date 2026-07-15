# Model Improvement and Lessons Learned

**Project:** FIFA World Cup 2026 Prediction
**Versions Compared:** V1 (baseline) vs V2 (final)
**Date:** 2026-07-05

---

## How Much Better Is V2?

| Metric | V1 | V2 | Change | What It Means |
|---|---|---|---|---|
| Accuracy | 59.18% | 62.08% | **+2.9pp** | Predicts the correct match outcome nearly 3 in every 100 additional matches |
| F1 Score | 0.519 | 0.543 | **+0.024** | Better balance between precision and recall across all three outcomes |
| ROC-AUC | 0.747 | 0.775 | **+0.028** | Model is better at separating winners from losers |
| Log Loss | 0.886 | 0.851 | **−4.0%** | Probabilities are more calibrated — the model is more confident when it should be, and less confident when it shouldn't be |

These numbers come from the same test period (2018–2026 competitive matches) so the comparison is fair.

---

## Why Is V2 Better?

There were four changes between V1 and V2. Here is what each one actually contributed and why it worked.

---

### 1. Training on Competitive Matches Only — The Single Biggest Win

**What changed:** V1 trained on all 49,490 matches (1872–2026), including friendlies. V2 filtered training to 21,325 competitive matches only.

**Why it worked:**

Friendly matches are structurally different from competitive ones:

- Squads are rotated. The starting 11 in a friendly may share almost nobody with the World Cup squad.
- Coaches experiment with formations and roles, not trying to win.
- Motivation is low. Top players are often rested to avoid injury.
- The result is often random relative to true team strength.

Including friendlies added 28,165 rows of mostly-noise data. The model learned patterns from games where outcome was not closely tied to team quality. Removing them meant every training example was a genuine signal about how teams perform when it matters.

This is the most important lesson of the whole project: **more data is not always better**. Cleaner, more relevant data beats volume.

**Impact:** The largest single driver of the +2.9pp accuracy improvement.

---

### 2. Expanding the Elo Code Map (48 → 108 Teams)

**What changed:** V1 used a hand-curated mapping of FIFA 3-letter codes to Elo 2-letter codes for just 48 teams (the 2026 World Cup participants). V2 expanded this to 108 teams, adding historical name aliases and frequently appearing nations.

**Why it worked:**

In V1, 70% of Elo rating values were null in the training data — not because those teams had no Elo rating, but because the mapping was incomplete. Teams like:

- United States → the code was `US`, not what the dataset used
- Ivory Coast → `CI` vs the historical name used in match results
- Czech Republic → `CZ` vs the older name used in historical data
- South Korea, Iran, Turkey, DR Congo — all had silent mismatches

Every null Elo was imputed to the median at training time. The model was receiving the same imputed value for Germany, Costa Rica, and an obscure team with no Elo data — which was wrong.

With 108 teams in the map, the null rate dropped from 70% to 36%. The model now has real Elo ratings for all the historically strong and frequently-appearing nations that appear throughout the training data.

**Impact:** The model now has better data quality for the teams that matter most in training. Elo rating is one of the strongest signal features (it measures true long-run team strength), so getting it right directly improves predictions.

---

### 3. Match Importance Feature

**What changed:** Added a numeric feature (`match_importance`, scored 1–5) that tells the model what type of match it is:
- 5 = FIFA World Cup
- 4 = Continental tournament (Euros, Copa America, AFCON, etc.)
- 3 = World Cup qualifier
- 2 = Other competitive (Nations League, Confederations Cup)
- 1 = Friendly

**Why it worked:**

Before this feature existed, the model had no way to distinguish a World Cup final from a mid-week qualifier. The match context is important because:

- Teams perform differently under different levels of pressure
- The features themselves (rolling form, Elo, rankings) do not capture the stakes of the match
- A team's 10-match rolling form might be built mostly on friendlies, which tells you less about how they will perform in a major tournament

The match importance score gives the model a direct signal about how seriously to weight other features in context.

**Impact:** Improved calibration (lower log loss) more than raw accuracy. The model became better at outputting meaningful probabilities, especially at the high end of team quality differences.

---

### 4. Head-to-Head Features

**What changed:** Added `h2h_win_rate` (win rate in last 10 meetings) and `h2h_goal_diff` (average goal difference in last 10 meetings) for each team against each opponent.

**Why it worked:**

Some matchups have persistent patterns that general strength metrics miss. For example:

- A team might consistently underperform against teams that play a specific defensive style
- Psychological factors — certain national teams have a documented history of performing or struggling against specific opponents
- These patterns exist even when overall strength ratings say the match should be close

Head-to-head data captures relationships between specific pairs of teams, which no individual strength metric can do on its own.

**Impact:** Most useful for frequently-played matchups (regional rivals, recurring tournament fixtures). Sparse for rare pairings, where it is left as null and imputed to the median.

---

### 5. Hyperparameter Tuning

**What changed:** V1 used default XGBoost parameters. V2 searched 40 configurations using RandomizedSearchCV with TimeSeriesSplit (5 folds).

**Best config found:** `n_estimators=500, max_depth=5, learning_rate=0.01, subsample=0.6, colsample_bytree=0.7, min_child_weight=3, gamma=0.2, reg_alpha=0, reg_lambda=2.0`

**Why it worked:**

The key parameters that differed from defaults:
- Lower `learning_rate` (0.01 vs default 0.1) with more trees (500 vs 100) — learns more gradually, avoids overfitting
- Lower `subsample` (0.6) — each tree only sees 60% of the training data, preventing overfitting to specific matches
- Higher regularisation (`reg_lambda=2.0`) — penalises complexity in the tree weights
- `min_child_weight=3` — prevents trees from making splits on very small groups of matches

The TimeSeriesSplit was critical: it ensured that validation folds always used future data to evaluate past-trained models. Using a random split for time-series data would leak future information and produce falsely optimistic scores.

**Impact:** Contributed a smaller but real improvement on top of the data quality changes. Tuning matters less if the underlying data is noisy — the data changes were prerequisite for tuning to help.

---

## What Changed in the Simulation

Both V1 and V2 named France as the most likely winner. However, V2 made several notable shifts:

| Team | V1 Win% | V2 Win% | Change |
|---|---|---|---|
| France | 21.8% | 19.8% | −2.0pp |
| Brazil | 13.5% | 7.8% | −5.7pp |
| Morocco | 7.6% | 11.2% | +4.6pp |
| Portugal | 7.0% | 8.0% | +1.0pp |
| Belgium | 7.1% | 7.4% | +0.3pp |

**Interpretation:**

- Morocco rising is the most significant shift. The H2H features and match importance score, combined with better Elo coverage, gave Morocco credit for its recent competitive record in major tournaments (2022 WC semi-finalist) that the V1 model underweighted.
- Brazil falling is likely explained by the competitive-only filter. Brazil plays many high-profile friendlies that inflated their rolling form features in V1. Stripping friendlies from training reduced the weight given to that data.
- France remaining top is consistent — their recent competitive record, high Elo, and strong FIFA ranking all align.

---

## Lessons Learned

### Data Quality Beats Data Volume

The single biggest improvement came from removing data (friendlies), not adding it. When building a training dataset, asking "is this data representative of what I want to predict?" matters more than maximising row count.

### Silent Joins Are Dangerous

The Elo code mismatch caused 70% of Elo values to silently become null. There was no error — the merge just returned NaN, which got imputed to the median. The model trained and ran without complaint but was using wrong data for a significant feature. The fix was not in the code logic but in the reference table. Always validate join quality explicitly — check null rates before and after every merge.

### Temporal Splits Are Non-Negotiable for Sports Data

Using a random train/test split in sports prediction would leak future data into training. A team's "form" features and rankings are derived from future matches. Temporal splits ensure the model genuinely learns from the past to predict the future, not the reverse.

### Log Loss Is the Right Metric for Simulation

A model can have good accuracy and poor log loss. Accuracy measures whether you picked the right outcome. Log loss measures whether your probabilities were right. For Monte Carlo simulation, probabilities matter — a model that says "70% home win" when it should say "52%" will produce a simulation that is wrong even if individual match predictions look correct.

### Imputation Hides Data Gaps

Median imputation for null features is necessary to keep pipelines running, but it hides the truth: many training examples have no Elo data. The model cannot distinguish between "this team has median Elo strength" and "we have no data for this team". Ideally, a separate binary indicator feature (`has_elo_rating`) should accompany each imputed feature to let the model learn the difference. This is left for a future version.

### FIFA Rankings Are a Snapshot Problem

Both V1 and V2 use current (2026) FIFA rankings for all historical matches. A match from 2010 uses 2026 rankings. This is incorrect — the ranking of teams changes substantially over time. Historical ranking snapshots would dramatically improve the ranking features. This was left out because obtaining them requires either a paid data source or scraping years of archived pages.

### Hyperparameter Tuning Is Not Free

RandomizedSearchCV with `n_jobs=-1` failed in the sandbox environment because multiprocessing was restricted. This was unexpected — most local development environments allow it freely. The lesson is to test computational requirements early, not when you are deep in a tuning run. The single-threaded fallback (`n_jobs=1`) worked but took longer.

### Precomputing Probabilities Transformed Simulation Speed

The first simulation implementation made one `predict_proba()` call per match per simulation. With 10,000 simulations and up to 7 rounds each, this was 70,000+ model calls and took ~90 seconds. Precomputing all 182 pairwise win probabilities once and sampling from them during simulation reduced this to 4 seconds. For anything with repeated inference, batch once and index many.

### Sequential Tournament State (Live Knockout Phase)

As of the semi-final stage, the simulation no longer uses a static pairwise cache for remaining knockouts. `TournamentStateTracker` (`src/simulation/tournament_state.py`) rebuilds each team's 2026 path from completed fixtures and updates it along every Monte Carlo branch. Tournament win rate, goals, rest days, and stadium-region changes feed into the existing `form_*` and `elo_rating` columns at inference time — no retrain required. Trade-off: ~2–3 model calls per simulation when only a few rounds remain (~2.5 min for 10k runs); early tournament still benefits from static precompute when many teams are alive.

---

## What Would Make V3 Better

The remaining gaps after V2, in order of likely impact:

| Improvement | Why It Would Help | Complexity |
|---|---|---|
| Historical FIFA ranking time series | Match rankings would reflect the actual ranking at the time of the match | High |
| Historical Elo time series | Fill the remaining 36% null and make historical training examples accurate | Medium |
| Squad availability / injury data | Team strength at match time, not just long-run average | High |
| Expected Goals (xG) features | Better signal on attacking and defensive quality than raw goal counts | Medium |
| Player rating aggregation (FIFA/Sofascore) | Captures squad depth and individual quality | High |
| Neutral venue correction | Matches at neutral venues behave differently from home/away | Low |
| Separate binary imputation flag | Model learns the difference between "median team" and "unknown team" | Low |
