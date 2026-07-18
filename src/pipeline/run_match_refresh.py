"""
Match-aware refresh runner.

Checks whether we are in a kickoff or post-match polling window, then runs the
daily update pipeline and commits only when bracket/fixture data changed.

Usage:
    python src/pipeline/run_match_refresh.py
    python src/pipeline/run_match_refresh.py --dry-run
    python src/pipeline/run_match_refresh.py --force
    python src/pipeline/run_match_refresh.py --status
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.match_refresh_scheduler import (  # noqa: E402
    decide_refresh,
    detect_new_results,
    mark_completed_from_bracket,
    save_state,
    snapshot_bracket_winners,
)

PY = sys.executable


def print_status() -> None:
    from src.pipeline.match_refresh_scheduler import build_watch_list

    watches = build_watch_list()
    if not watches:
        print("No knockout matches on the watch list.")
        return
    print("Match refresh watch list:")
    for watch in watches:
        status = "done" if watch.completed else "watching"
        poll_start = watch.kickoff_utc.replace(
            minute=watch.kickoff_utc.minute,
        )
        print(
            f"  - [{status}] {watch.label}\n"
            f"      key={watch.key}\n"
            f"      kickoff={watch.kickoff_utc.isoformat()}\n"
            f"      poll every {watch.poll_interval_minutes}m for "
            f"{watch.poll_duration_minutes}m after ~{watch.match_duration_minutes}m match"
        )


def run_pipeline() -> int:
    cmd = [PY, str(PROJECT_ROOT / "src/pipeline/run_daily_update.py")]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Match-aware World Cup refresh.")
    parser.add_argument("--dry-run", action="store_true", help="Print decision only.")
    parser.add_argument("--force", action="store_true", help="Run refresh regardless of schedule.")
    parser.add_argument("--check-only", action="store_true", help="Exit 2 if refresh would run, 0 if skip.")
    parser.add_argument("--status", action="store_true", help="Show watch list and exit.")
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    decision = decide_refresh()
    if args.force:
        decision = type(decision)(True, "Forced refresh.", decision.watch_key)

    print(f"Match refresh decision: {'RUN' if decision.should_refresh else 'SKIP'}")
    print(f"  Reason: {decision.reason}")

    if args.check_only:
        sys.exit(2 if decision.should_refresh else 0)

    if not decision.should_refresh:
        return

    if args.dry_run:
        print("Dry run — pipeline not executed.")
        return

    before = snapshot_bracket_winners()
    exit_code = run_pipeline()
    if exit_code != 0:
        sys.exit(exit_code)

    after = snapshot_bracket_winners()
    new_results = detect_new_results(before, after)
    mark_completed_from_bracket()
    save_state(last_reason=decision.reason)

    if new_results:
        print(f"  New knockout results detected: {', '.join(new_results)}")
    else:
        print("  No new knockout results yet — dashboard still refreshed from latest pull.")


if __name__ == "__main__":
    main()
