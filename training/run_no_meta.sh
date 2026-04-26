#!/bin/bash
# ARGUS ablation run — same as v34_full MINUS the metacognition stack.
# Drops: --capability-map, --causal-attribution, --most-learnable-cluster,
# --per-cluster-detectors, --proposer-validator, --adversarial-proposer.
# Keeps: ERCV, soft-rollback, epiplexity-weight, capacity-growth, SPSI
# (so we can still compute SPSI on this baseline run).
# Estimated wall time: ~6 hours.
#
# Why this run: isolates the contribution of the metacognition stack.
# If v34_full beats v34_no_meta on external pass@1, the metacognition
# layer is paying for itself. If they're tied, the +pp comes from
# capacity growth + chain-consensus alone.

set -euo pipefail
cd "$(dirname "$0")/.."

NAME=v34_no_meta
TS=$(date +%Y%m%d-%H%M%S)
LOG=logs/training/${NAME}_${TS}.log
mkdir -p logs/training

echo "[$(date)] Starting ${NAME} run -> outputs_${NAME}/, log=${LOG}" | tee -a "${LOG}"

.venv/Scripts/python.exe scripts/log_screenshot_monitor.py \
    --log "${LOG}" --name "${NAME}" --total-episodes 15 &
MONITOR_PID=$!
echo "[$(date)] log-screenshot monitor PID=${MONITOR_PID}" | tee -a "${LOG}"

PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 .venv/Scripts/python.exe scripts/run_self_improve_v3.py \
    --warmstart-adapter outputs_warmstart/adapters \
    --episodes 15 --steps-per-episode 64 --eval-n 100 \
    --solver-k 14 --solver-mnt 352 --capacity-growth 0 \
    --solver-k-max 14 --solver-mnt-max 352 \
    --epiplexity-samples 96 \
    --epiplexity-weight \
    --ercv-rollback --ercv-soft-rollback --ercv-zscore \
    --soft-replay-decay --spsi \
    --external-context-path outputs_self_improve_v3/gsm8k_train_seeds.jsonl \
    --external-context-rate 0.35 \
    --output-dir outputs_${NAME} \
    2>&1 | tee -a "${LOG}"

echo "[$(date)] Run complete; waiting for screenshot monitor to flush 4_end.png..." | tee -a "${LOG}"
wait_loop=0
while kill -0 ${MONITOR_PID} 2>/dev/null && [ ${wait_loop} -lt 30 ]; do
    sleep 1; wait_loop=$((wait_loop+1))
done
kill ${MONITOR_PID} 2>/dev/null || true

echo "[$(date)] Collecting artifacts..." | tee -a "${LOG}"
.venv/Scripts/python.exe scripts/collect_training_data.py \
    --run outputs_${NAME} --name ${NAME} 2>&1 | tee -a "${LOG}"
echo "[$(date)] Done. Snapshot at logs/snapshots/${NAME}/" | tee -a "${LOG}"
