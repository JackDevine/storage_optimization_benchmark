#!/usr/bin/env bash
# Run the agent benchmark for several models, one after another.
#
#   nohup scripts/run_sweep.sh > runs/sweep.log 2>&1 &
#
# Each model gets the same budget: MAX_HOURS of active time (usage-limit pauses
# and submission scoring don't count) and MAX_SUBMISSIONS submissions. Cost is
# only reported (runs use the Claude subscription), so it's set high enough
# never to end a run. A run that fails doesn't stop the sweep; it can be
# continued later with `uv run python -m harness.run_agent --resume <run_id>`.
set -u
cd "$(dirname "$0")/.."

MODELS=(${MODELS:-claude-haiku-5-5 claude-sonnet-5-5 claude-opus-5-5})
MAX_HOURS=${MAX_HOURS:-2}
MAX_SUBMISSIONS=${MAX_SUBMISSIONS:-10}

echo "Sweep started $(date '+%F %T'): ${MODELS[*]}"
uv run python -m harness.evaluator reference  # baselines for this config (instant if cached)

for model in "${MODELS[@]}"; do
  echo "=== $model: started $(date '+%F %T') ==="
  if uv run python -m harness.run_agent --model "$model" \
      --max-hours "$MAX_HOURS" --max-submissions "$MAX_SUBMISSIONS" --max-cost 1000; then
    echo "=== $model: finished $(date '+%F %T') ==="
  else
    echo "=== $model: exited with status $? at $(date '+%F %T'); continuing ==="
  fi
done
echo "Sweep finished $(date '+%F %T')"
