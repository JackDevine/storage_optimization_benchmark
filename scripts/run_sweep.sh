#!/usr/bin/env bash
# Run the agent benchmark for several models, one after another.
#
#   nohup scripts/run_sweep.sh > runs/sweep.log 2>&1 &
#
# Each model gets the same budget: MAX_HOURS of active time (usage-limit pauses
# and submission scoring don't count) and MAX_SUBMISSIONS submissions. Models
# can be Claude names (subscription) or OpenRouter slugs (vendor/model):
#
#   MODELS="claude-opus-5-5 openai/gpt-5.1" scripts/run_sweep.sh
#
# For Claude runs cost is only reported, so it's set high enough never to end a
# run. OpenRouter runs spend real money and stop at MAX_COST (default $20 each).
# A run that fails doesn't stop the sweep; it can be continued later with
# `uv run python -m harness.run_agent --resume <run_id>`.
set -u
cd "$(dirname "$0")/.."

MODELS=(${MODELS:-claude-haiku-5-5 claude-sonnet-5-5 claude-opus-5-5})
MAX_HOURS=${MAX_HOURS:-2}
MAX_SUBMISSIONS=${MAX_SUBMISSIONS:-10}
MAX_COST=${MAX_COST:-20}  # per OpenRouter run, USD

echo "Sweep started $(date '+%F %T'): ${MODELS[*]}"
uv run python -m harness.evaluator reference  # baselines for this config (instant if cached)

for model in "${MODELS[@]}"; do
  case "$model" in
    */*) cost=$MAX_COST ;;  # OpenRouter: real spend
    *) cost=1000 ;;         # Claude subscription: reported only
  esac
  echo "=== $model: started $(date '+%F %T') ==="
  if uv run python -m harness.run_agent --model "$model" \
      --max-hours "$MAX_HOURS" --max-submissions "$MAX_SUBMISSIONS" --max-cost "$cost"; then
    echo "=== $model: finished $(date '+%F %T') ==="
  else
    echo "=== $model: exited with status $? at $(date '+%F %T'); continuing ==="
  fi
done
echo "Sweep finished $(date '+%F %T')"
