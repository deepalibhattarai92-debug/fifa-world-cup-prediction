"""
One-command daily refresh of the 2026 World Cup prediction.

Runs the light-refresh pipeline end to end:
  1. Pull fresh data   — World Cup fixtures, FIFA rankings, Elo ratings
  2. Sync fixtures     — merge completed 2026 knockouts into results feed
  3. Rebuild features  — build_features_v2 (incl. tournament-path features)
  4. (optional) retrain the model  — train_model_v2  [--retrain]
  5. Simulate          — simulate_tournament_v2 (trained tournament-path features)
  6. Archive           — freeze dated snapshots under archive/<date>/
  7. Document          — prepend a dated entry to docs/UPDATE_LOG.md

The simulation reads the live bracket state straight from the fixtures feed, so no
file ever needs hand-editing as results come in.

Usage (from project root):
    python src/pipeline/run_daily_update.py            # light refresh (no retrain)
    python src/pipeline/run_daily_update.py --retrain  # also retrain the model
    python src/pipeline/run_daily_update.py --no-pull  # skip live data pulls (offline test)
"""

import argparse
import shutil
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.simulate_tournament_v2 import (  # noqa: E402
    BRACKET_2026,
    MATCH_ORDER,
    _resolve_source,
    alive_teams_from_state,
    derive_bracket_state,
)

PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS    = PROJECT_ROOT / "models"
ARCHIVE   = PROJECT_ROOT / "archive"
LOG_PATH  = PROJECT_ROOT / "docs" / "UPDATE_LOG.md"

PY = sys.executable


def run_step(label: str, script: str, required: bool) -> bool:
    """Run a pipeline script as a subprocess. Returns True on success."""
    print(f"\n{'=' * 70}\n▶ {label}\n{'=' * 70}")
    script_path = PROJECT_ROOT / script
    # Module path for scripts under src/ that import the package.
    if script_path.parts[-3:-1] == ("src",) or "src/" in script:
        module = script.replace("/", ".").replace(".py", "")
        cmd = [PY, "-m", module]
    else:
        cmd = [PY, str(script_path)]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    ok = result.returncode == 0
    if not ok:
        msg = f"  ✗ {label} failed (exit {result.returncode})."
        if required:
            print(msg + " Aborting.")
            sys.exit(result.returncode)
        print(msg + " Continuing with existing data.")
    return ok


def archive_run(run_date: str) -> Path:
    """Freeze the current canonical outputs into archive/<run_date>/."""
    dest = ARCHIVE / run_date
    (dest / "eval").mkdir(parents=True, exist_ok=True)

    files = [
        (PROJECT_ROOT / "src/simulation/simulate_tournament_v2.py", dest / "simulate_tournament_v2.py"),
        (PROCESSED / "simulation_results.csv",  dest / "simulation_results.csv"),
        (PROCESSED / "model_comparison_v2.csv", dest / "model_comparison_v2.csv"),
        (MODELS / "best_model_v2.pkl",          dest / "best_model_v2.pkl"),
        (MODELS / "label_encoder_v2.pkl",       dest / "label_encoder_v2.pkl"),
    ]
    for src, dst in files:
        if src.exists():
            shutil.copy2(src, dst)

    eval_dir = PROCESSED / "eval"
    if eval_dir.exists():
        for csv in eval_dir.glob("*.csv"):
            shutil.copy2(csv, dest / "eval" / csv.name)

    print(f"\n  Archived snapshot → {dest.relative_to(PROJECT_ROOT)}")
    return dest


def build_log_entry(run_date: str) -> str:
    """Generate a Markdown log entry from the current outputs."""
    fixtures     = pd.read_csv(PROCESSED / "world_cup_fixtures.csv")
    real_winners = derive_bracket_state(fixtures)
    alive        = alive_teams_from_state(real_winners)

    played_lines = []
    for match_id in MATCH_ORDER:
        if match_id in real_winners:
            a = _resolve_source(BRACKET_2026[match_id][0], real_winners)
            b = _resolve_source(BRACKET_2026[match_id][1], real_winners)
            played_lines.append(f"  - {match_id}: {a} vs {b} → **{real_winners[match_id]}**")
    played_block = "\n".join(played_lines) if played_lines else "  - (no knockout matches played yet)"

    sim = pd.read_csv(PROCESSED / "simulation_results.csv").head(5)
    odds_rows = "\n".join(
        f"| {i + 1} | {r.team} | {r.win_pct} |"
        for i, r in enumerate(sim.itertuples(index=False))
    )

    model_line = ""
    comp_path = PROCESSED / "model_comparison_v2.csv"
    if comp_path.exists():
        comp = pd.read_csv(comp_path).sort_values("log_loss").iloc[0]
        model_line = (
            f"**Model:** {comp['model']} — accuracy "
            f"{comp['accuracy']:.4f}, log loss {comp['log_loss']:.4f}, "
            f"ROC-AUC {comp['roc_auc']:.4f}\n\n"
        )

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (
        f"## {run_date} — automated refresh\n\n"
        f"*Generated {ts} by `run_daily_update.py`.*\n\n"
        f"**Archive:** [`archive/{run_date}/`](../archive/{run_date}/)\n\n"
        f"**Knockout results detected ({len(alive)} teams still alive):**\n"
        f"{played_block}\n\n"
        f"{model_line}"
        f"**Champion odds (top 5, 10,000 simulations):**\n\n"
        f"| Rank | Team | Win % |\n|------|------|-------|\n{odds_rows}\n\n---\n\n"
    )


def update_log(run_date: str) -> None:
    """Prepend today's entry to the update log (replacing any existing same-day entry)."""
    entry = build_log_entry(run_date)
    marker = "---\n"

    if not LOG_PATH.exists():
        LOG_PATH.write_text("# Update Log\n\n" + marker + "\n" + entry)
        print(f"  Created {LOG_PATH.relative_to(PROJECT_ROOT)}")
        return

    text = LOG_PATH.read_text()

    # Drop an existing entry for the same date so re-runs stay idempotent.
    today_header = f"## {run_date} —"
    if today_header in text:
        start = text.index(today_header)
        rest = text[start:]
        nxt = rest.find("\n## ", 1)
        end = start + nxt if nxt != -1 else len(text)
        text = text[:start] + text[end:]

    # Insert the fresh entry right after the intro's first horizontal rule.
    idx = text.find(marker)
    insert_at = idx + len(marker) if idx != -1 else len(text)
    new_text = text[:insert_at] + "\n" + entry + text[insert_at:]
    LOG_PATH.write_text(new_text)
    print(f"  Updated {LOG_PATH.relative_to(PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily refresh of the World Cup prediction.")
    parser.add_argument("--retrain", action="store_true", help="Also retrain the model.")
    parser.add_argument("--no-pull", action="store_true", help="Skip live data pulls.")
    args = parser.parse_args()

    run_date = date.today().isoformat()
    print(f"\n🔄 Daily update for {run_date}\n")

    if not args.no_pull:
        run_step("Pull World Cup fixtures", "src/data_collection/collect_world_cup_fixtures.py", required=False)
        run_step("Sync fixtures → results",  "src/data_processing/sync_fixtures_to_results.py", required=True)
        run_step("Pull FIFA rankings",     "src/data_collection/collect_fifa_rankings.py",       required=False)
        run_step("Pull Elo ratings",       "src/data_collection/collect_elo_ratings.py",         required=False)
    else:
        run_step("Sync fixtures → results (offline)", "src/data_processing/sync_fixtures_to_results.py", required=True)

    run_step("Rebuild V2 features", "src/feature_engineering/build_features_v2.py", required=True)

    if args.retrain:
        run_step("Retrain V2 model", "src/models/train_model_v2.py", required=True)
        run_step("Recompute evaluation artifacts", "src/models/evaluate_model.py", required=False)

    run_step("Run tournament simulation", "src/simulation/simulate_tournament_v2.py", required=True)

    print(f"\n{'=' * 70}\n▶ Archive + document\n{'=' * 70}")
    archive_run(run_date)
    update_log(run_date)

    print(f"\n✅ Daily update complete for {run_date}.")
    print("   Review the changes, then commit & push to update the live dashboard:")
    print('     git add -A && git commit -m "Daily prediction refresh {}" && git push'.format(run_date))


if __name__ == "__main__":
    main()
