#!/usr/bin/env bash
# Resume runs that stopped before finishing: a usage limit longer than
# --max-pause-hours, repeated connection failures, or the harness itself dying.
# A run counts as stopped if it has no run_end event and no harness process is
# running it. Each is restarted with `run_agent --resume`, which restores its
# original settings. Meant for a timer just after the weekly limit resets:
#
#   systemd-run --user --on-calendar='2026-10-14 05:10' --unit bench-resume \
#     -E OPENROUTER_API_KEY -E PATH -E HOME --working-directory "$PWD" scripts/resume_stopped_runs.sh
#
# RUN_GLOB limits which runs are considered (default: leaderboard runs from today and yesterday).
set -u
cd "$(dirname "$0")/.."
RUN_GLOB=${RUN_GLOB:-"$(date -d yesterday +%Y%m%d)-*-lb-* $(date +%Y%m%d)-*-lb-*"}

pids=()
for dir in $RUN_GLOB; do
  dir=runs/$dir
  [ -d "$dir/logs" ] || continue
  id=$(basename "$dir")
  grep -q '"event": "run_end"' "$dir/logs/events.jsonl" 2>/dev/null && continue
  pgrep -f "run-id $id|resume $id" > /dev/null && continue
  echo "--- resume $id at $(date '+%F %T')"
  uv run --no-sync python -m harness.run_agent --resume "$id" >> "runs/$id.stdout.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "resume pass finished $(date '+%F %T')"
