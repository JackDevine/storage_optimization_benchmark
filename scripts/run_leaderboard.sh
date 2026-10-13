#!/usr/bin/env bash
# Leaderboard: each model's standard benchmark run (Python, 2 h), repeated.
# Claude models (subscription) and OpenRouter models run from separate queues,
# CLAUDE_SLOTS and OPENROUTER_SLOTS at a time, so Claude runs waiting out a
# usage limit don't hold up the others. Jobs are queued rep by rep, so each
# model has a first result before any has a second.
#
# Each job is "<model> <rep> [run_agent args]", comma-separated:
#   CLAUDE_JOBS="claude-opus-5-5 1 --effort max" OPENROUTER_JOBS="z-ai/glm-5.3 1 --effort low" \
#     scripts/run_leaderboard.sh
#
# Claude runs sit through usage limits up to MAX_PAUSE_HOURS; a run that stops
# for a longer one is picked up by scripts/resume_stopped_runs.sh.
#
# Run ids: <date>-lb-<model>-<effort>-r<k> (effort defaults to high).
set -u
cd "$(dirname "$0")/.."

CLAUDE_SLOTS=${CLAUDE_SLOTS:-2}
OPENROUTER_SLOTS=${OPENROUTER_SLOTS:-3}
MAX_HOURS=${MAX_HOURS:-2}
MAX_SUBMISSIONS=${MAX_SUBMISSIONS:-10}
MAX_COST=${MAX_COST:-20}
MAX_PAUSE_HOURS=${MAX_PAUSE_HOURS:-24}

run_job() {
  local model=$1 rep=$2 cost effort=high
  shift 2
  local args=("$@")
  for ((i = 0; i < ${#args[@]} - 1; i++)); do [ "${args[i]}" = --effort ] && effort=${args[i + 1]}; done
  case "$model" in */*) cost=$MAX_COST ;; *) cost=1000 ;; esac
  local run_id="$(date +%Y%m%d-%H%M%S)-lb-${model//\//_}-$effort-r$rep"
  echo "--- start $run_id at $(date '+%F %T')"
  if uv run --no-sync python -m harness.run_agent --model "$model" --run-id "$run_id" \
      --max-hours "$MAX_HOURS" --max-submissions "$MAX_SUBMISSIONS" --max-cost "$cost" \
      --max-pause-hours "$MAX_PAUSE_HOURS" "$@" > "runs/$run_id.stdout.log" 2>&1; then status=ok; else status="exit $?"; fi
  echo "--- $status $run_id at $(date '+%F %T')"
}
export -f run_job
export MAX_HOURS MAX_SUBMISSIONS MAX_COST MAX_PAUSE_HOURS

queue() {  # queue <slots> <comma-separated jobs>
  tr ',' '\n' <<< "$2" | sed '/^\s*$/d' | \
    xargs -P "$1" -L 1 bash -c 'sleep $((RANDOM % 20)); run_job "$@"' _
}

echo "Leaderboard started $(date '+%F %T')"
uv run --no-sync python -m harness.evaluator reference
queue "$CLAUDE_SLOTS" "${CLAUDE_JOBS:-}" &
queue "$OPENROUTER_SLOTS" "${OPENROUTER_JOBS:-}" &
wait
echo "Leaderboard finished $(date '+%F %T')"
