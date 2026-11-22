#!/bin/bash
# Full multi-level product factorization analysis on classification-trained models.
#
# Runs analyze_full_product_factorization_agent_aware.py for each matched seed
# triplet (GNN/Transformer/Mamba) at epochs 0 and 500, then aggregates:
#   - Task-level asymmetric factorization (depth→agents and agents→depth R²)
#   - Agent-level factorization (per-agent R² for fixed n_agents)
#
# Usage:
# Run from anywhere; reads runs from $RUNS (default: <repo>/runs/exp1).
#   bash exp1_small_models/run_full_factorization_batch_classification.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNS="${RUNS:-$REPO/runs/exp1}"
MS="$RUNS/multi_seed_classification"
ANALYSIS_SCRIPT="$SCRIPT_DIR/analyze_full_product_factorization_agent_aware.py"
DATASET_PATH="$REPO/data/mindgames_orthogonal_v4_classification/val.json"
OUTPUT_DIR="$MS/factorization_analysis_full"


mkdir -p "$OUTPUT_DIR"

echo "========================================================================"
echo "Full Multi-Level Factorization: Classification-Trained Models"
echo "========================================================================"
echo "Script:  $ANALYSIS_SCRIPT"
echo "Dataset: $DATASET_PATH"
echo "Output:  $OUTPUT_DIR"
echo ""

# Matched seed triplets
declare -A GNN_DIRS=(
  [1]="$MS/gnn_seed1_results_8580377"
  [2]="$MS/gnn_seed2_results_8580377"
  [3]="$MS/gnn_seed3_results_8584516"
  [4]="$MS/gnn_seed4_results_8584516"
  [5]="$MS/gnn_seed5_results_8584516"
)
declare -A TRANSFORMER_DIRS=(
  [1]="$MS/transformer_seed1_results_8580377"
  [2]="$MS/transformer_seed2_results_8580377"
  [3]="$MS/transformer_seed3_results_8584516"
  [4]="$MS/transformer_seed4_results_8584516"
  [5]="$MS/transformer_seed5_results_8584516"
)
declare -A MAMBA_DIRS=(
  [1]="$MS/mamba_seed1_results_9403834"
  [2]="$MS/mamba_seed2_results_9403834"
  [3]="$MS/mamba_seed3_results_8584516"
  [4]="$MS/mamba_seed4_results_8584516"
  [5]="$MS/mamba_seed5_results_8584516"
)

SUCCESS=0
FAILED=0
SKIPPED=0

for EPOCH in 0 500; do
  echo "========================================================================"
  echo "Epoch $EPOCH"
  echo "========================================================================"

  for SEED in 1 2 3 4 5; do
    SEED_OUT="$OUTPUT_DIR/seed${SEED}_epoch${EPOCH}"
    RESULT_FILE="$SEED_OUT/multi_level_factorization_results.json"

    echo "  seed$SEED epoch$EPOCH..."

    if [ -f "$RESULT_FILE" ]; then
      echo "    SKIPPED (already analyzed)"
      SKIPPED=$((SKIPPED + 1))
      continue
    fi

    GNN_DIR="${GNN_DIRS[$SEED]}"
    TR_DIR="${TRANSFORMER_DIRS[$SEED]}"
    MB_DIR="${MAMBA_DIRS[$SEED]}"

    # Verify all three dirs have captured states at this epoch
    MISSING=0
    for DIR in "$GNN_DIR" "$TR_DIR" "$MB_DIR"; do
      if [ ! -f "$DIR/captured_states/states_epoch_${EPOCH}_val.pt" ]; then
        echo "    SKIPPED (missing states: $(basename $DIR) epoch $EPOCH)"
        MISSING=1
      fi
    done
    if [ $MISSING -eq 1 ]; then
      SKIPPED=$((SKIPPED + 1))
      continue
    fi

    mkdir -p "$SEED_OUT"

    if python3 "$ANALYSIS_SCRIPT" \
        --gnn_dir "$GNN_DIR" \
        --transformer_dir "$TR_DIR" \
        --mamba_dir "$MB_DIR" \
        --dataset_path "$DATASET_PATH" \
        --epoch "$EPOCH" \
        --split val \
        --n_components 10 \
        --output_dir "$SEED_OUT" 2>&1 | grep -E "R²|Mean|verdict|FACTORIZATION|✓|✗|Error|Traceback"; then
      echo "    ✓ Done"
      SUCCESS=$((SUCCESS + 1))
    else
      echo "    ✗ FAILED"
      FAILED=$((FAILED + 1))
    fi
  done
  echo ""
done

echo "========================================================================"
echo "Per-seed runs complete ($SUCCESS done, $FAILED failed, $SKIPPED skipped)"
echo "Aggregating..."
echo "========================================================================"

OUTPUT_DIR="$OUTPUT_DIR" python3 - <<'PYEOF'
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

import os
output_dir = Path(os.environ["OUTPUT_DIR"])

# Collect per-seed per-epoch results
# Structure: results[epoch][arch] = list of dicts
task = {0: defaultdict(list), 500: defaultdict(list)}
agent = {0: defaultdict(lambda: defaultdict(list)), 500: defaultdict(lambda: defaultdict(list))}

for epoch in [0, 500]:
    for seed in range(1, 6):
        rf = output_dir / f"seed{seed}_epoch{epoch}" / "multi_level_factorization_results.json"
        if not rf.exists():
            continue
        with open(rf) as f:
            data = json.load(f)

        for arch in ["GNN", "Transformer", "Mamba"]:
            tl = data["task_level"].get(arch)
            if tl:
                task[epoch][arch].append({
                    "r2_agents": tl["r2_agents"],
                    "r2_depth":  tl["r2_depth"],
                })

            al = data["agent_level"].get(arch, {})
            for n_str, res in al.items():
                agent[epoch][arch][n_str].append({
                    "mean_agent_r2": res["mean_agent_r2"],
                    "depth_r2":      res["depth_r2"],
                })

def fmt(vals, key):
    v = [x[key] for x in vals]
    m = np.mean(v)
    s = np.std(v) / np.sqrt(len(v))
    return m, s, len(v)

print("\n" + "=" * 72)
print("TASK-LEVEL FACTORIZATION (classification-trained models)")
print("=" * 72)
print(f"\n{'Architecture':<14} {'Direction':<22} {'Epoch 0':>18}  {'Epoch 500':>18}  {'Change':>8}")
print("-" * 82)

summary = {}
for arch in ["GNN", "Transformer", "Mamba"]:
    e0 = task[0][arch]
    e500 = task[500][arch]
    if not e0 or not e500:
        print(f"  {arch}: insufficient data")
        continue
    for key, label in [("r2_agents", "depth→n_agents"), ("r2_depth", "n_agents→depth")]:
        m0, s0, n0 = fmt(e0, key)
        m5, s5, n5 = fmt(e500, key)
        chg = (m5 - m0) / m0 * 100 if m0 > 0 else float("nan")
        print(f"  {arch:<12}  {label:<22}  {m0:.3f}±{s0:.3f} (n={n0})  {m5:.3f}±{s5:.3f} (n={n5})  {chg:+.1f}%")
    summary[arch] = {
        "epoch_0":   {k: fmt(e0, k)[:2] for k in ("r2_agents", "r2_depth")},
        "epoch_500": {k: fmt(e500, k)[:2] for k in ("r2_agents", "r2_depth")},
        "n_seeds":   len(e0),
    }

print("\n" + "=" * 72)
print("AGENT-LEVEL FACTORIZATION (classification-trained models)")
print("=" * 72)
for arch in ["GNN", "Transformer", "Mamba"]:
    print(f"\n  {arch}:")
    for epoch in [0, 500]:
        for n_str, vals in sorted(agent[epoch][arch].items()):
            if not vals:
                continue
            ma, sa, na = fmt(vals, "mean_agent_r2")
            md, sd, _  = fmt(vals, "depth_r2")
            print(f"    epoch={epoch}  n_agents={n_str}:  mean_agent_R²={ma:.3f}±{sa:.3f}  depth_R²={md:.3f}±{sd:.3f}  (n={na})")

# Save aggregated JSON
agg = {"task_level": summary, "agent_level": {}}
for arch in ["GNN", "Transformer", "Mamba"]:
    agg["agent_level"][arch] = {}
    for epoch in [0, 500]:
        agg["agent_level"][arch][str(epoch)] = {}
        for n_str, vals in agent[epoch][arch].items():
            if vals:
                ma, sa, na = fmt(vals, "mean_agent_r2")
                md, sd, _  = fmt(vals, "depth_r2")
                agg["agent_level"][arch][str(epoch)][n_str] = {
                    "mean_agent_r2_mean": float(ma), "mean_agent_r2_sem": float(sa),
                    "depth_r2_mean": float(md), "depth_r2_sem": float(sd),
                    "n_seeds": na,
                }

out = output_dir / "classification_full_factorization_aggregated.json"
with open(out, "w") as f:
    json.dump(agg, f, indent=2)
print(f"\nAggregated results saved: {out}")
PYEOF

echo ""
echo "========================================================================"
echo "Done."
echo "========================================================================"
