#!/bin/bash
# ARGUS reproducibility run — same config as run_v34_full.sh but seed=1.
# Validates that the +7.0pp pass@1 result on seed=0 is not seed-luck.
# 15 episodes, eval n=100. Estimated wall: ~5h on RTX 5070 Ti Laptop.

set -euo pipefail
cd "$(dirname "$0")/.."

NAME=v34_full_seed1
TS=$(date +%Y%m%d-%H%M%S)
LOG=logs/training/${NAME}_${TS}.log
mkdir -p logs/training

echo "[$(date)] Starting ${NAME} run -> outputs_${NAME}/, log=${LOG}" | tee -a "${LOG}"

.venv/Scripts/python.exe scripts/log_screenshot_monitor.py \
    --log "${LOG}" --name "${NAME}" --total-episodes 15 &
MONITOR_PID=$!
echo "[$(date)] log-screenshot monitor PID=${MONITOR_PID}" | tee -a "${LOG}"

PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 .venv/Scripts/python.exe scripts/run_self_improve_v3.py \
    --warmstart-adapter outputs_warmstart/adapters --seed 1 \
    --episodes 15 --steps-per-episode 64 --eval-n 100 \
    --solver-k 14 --solver-mnt 352 --capacity-growth 0 \
    --solver-k-max 14 --solver-mnt-max 352 \
    --epiplexity-samples 96 \
    --capability-map --capability-map-k 10 \
    --causal-attribution --active-defense --epiplexity-weight \
    --ercv-rollback --ercv-soft-rollback --ercv-zscore \
    --most-learnable-cluster --per-cluster-detectors --soft-replay-decay \
    --proposer-validator --adversarial-proposer --spsi \
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
