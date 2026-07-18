# Automating the Daily Refresh

The prediction can be refreshed with a single command. The simulation reads the
live bracket straight from the FIFA fixtures feed, so **nothing needs to be
hand-edited** as results come in — new match results are detected automatically.

## The one command

```bash
python src/pipeline/run_daily_update.py
```

This runs the full light-refresh pipeline:

1. Pull fresh data — World Cup fixtures, FIFA rankings, Elo ratings
2. Rebuild the V2 feature dataset
3. Re-run the 10,000-simulation tournament forecast (bracket auto-derived)
4. Freeze a dated snapshot into `archive/<date>/`
5. Prepend a dated entry to `docs/UPDATE_LOG.md`

Useful flags:

| Flag | Effect |
|------|--------|
| *(none)* | Light refresh — recommended day to day |
| `--retrain` | Also retrain the model (rarely changes anything mid-tournament) |
| `--no-pull` | Skip live data pulls (offline re-run / testing) |

To also commit and push in one shot (updates the live dashboard):

```bash
AUTO_COMMIT=1 ./scripts/daily_update.sh
```

## Option A — Local schedule with cron (most reliable)

Runs on your Mac whenever it's awake. Because the pull happens from your own IP,
the FIFA API works reliably.

```bash
crontab -e
```

Add a line to run every day at 9:00 am (adjust the path):

```cron
0 9 * * * cd /Users/deepalibhattarai/Documents/GitHub/fifa-world-cup-prediction && AUTO_COMMIT=1 ./scripts/daily_update.sh >> /tmp/wc_update.log 2>&1
```

Check `/tmp/wc_update.log` for output. (If your Mac is asleep at 9am, the job is
skipped that day — run the command manually to catch up.)

## Option B — GitHub Actions (fully hands-off, with a caveat)

`.github/workflows/daily-update.yml` runs the refresh in the cloud daily and
pushes the result, so the dashboard updates with zero involvement from you.

**Caveat:** the FIFA fixtures/rankings APIs use Akamai bot protection and may
return HTTP 403 from GitHub's shared runner IPs. The runner is resilient (it keeps
the last committed data and still re-simulates), but a blocked pull means no new
fixture results that day. Elo ratings and the simulation always work. You can also
trigger it manually from the Actions tab ("Run workflow").

**Recommendation:** use local cron (Option A) for dependable daily fixture updates;
keep the Actions workflow as an on-demand / backup trigger.

## Option C — Match-time auto refresh (knockout stage)

During knockouts the dashboard can refresh automatically around each match:

1. **At kickoff** — pull fixtures and re-simulate
2. **After full time** — poll every **5 minutes for 1 hour** until FIFA publishes the score

### GitHub Actions (hands-off)

`.github/workflows/match-refresh.yml` runs every 5 minutes. Most runs no-op instantly;
during an active match window it runs the full pipeline and pushes updates.

Manual force refresh from the Actions tab: **Match-time prediction refresh → Run workflow → force: true**.

### Local cron (best FIFA API reliability)

Run the same scheduler every 5 minutes on your Mac:

```bash
crontab -e
```

```cron
*/5 * * * * cd /Users/deepalibhattarai/Documents/GitHub/fifa-world-cup-prediction && AUTO_COMMIT=1 ./scripts/match_refresh.sh >> /tmp/wc_match_refresh.log 2>&1
```

Useful commands:

```bash
./scripts/match_refresh.sh --status     # show upcoming watch windows
./scripts/match_refresh.sh --dry-run    # print decision without running
./scripts/match_refresh.sh --force      # refresh now, ignore schedule
```

### Kickoff times

FIFA's API only provides **dates**, not kickoff times. Defaults and overrides live in
`config/match_kickoffs.json`. The World Cup Final is pre-configured for **Jul 19,
2026 at 19:00 UTC** (3:00 pm ET).

Adjust `match_duration_minutes` (default 135) and `poll_duration_minutes` (default 60)
there if needed.

## What still needs a human

- **Match results:** handled automatically via the fixtures feed. ✅
- **Historical training data** (`data/raw/results.csv`): only refreshed by a manual
  Kaggle download. Not needed mid-tournament; do it once after the final for a
  clean post-event retrain.
