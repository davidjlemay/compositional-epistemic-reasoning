#!/bin/bash
# Batch product factorization analysis for classification-trained models.
#
# Measures both simplification AND factorization on the
# SAME classification-trained models (not separate generation-trained models).
#
# Runs analyze_temporal_factorization.py per seed at epochs 0 and 500, then
# aggregates R² across seeds within each architecture.
#
# Usage:
# Run from anywhere; reads runs from $RUNS (default: <repo>/runs/exp1).
#   bash exp1_small_models/run_factorization_batch_classification.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNS="${RUNS:-$REPO/runs/exp1}"
ANALYSIS_TOOLS="$SCRIPT_DIR"
DATA_DIR="$REPO/data/mindgames_orthogonal_v4_classification"
DATA_FILE="val.json"
OUTPUT_DIR="$RUNS/multi_seed_classification/factorization_analysis"

EPOCHS="0 500"


mkdir -p "$OUTPUT_DIR"

echo "========================================================================"
echo "Batch Factorization Analysis: Classification-Trained Models"
echo "========================================================================"
echo "Analysis tools:  $ANALYSIS_TOOLS"
echo "Data directory:  $DATA_DIR"
echo "Data file:       $DATA_FILE"
echo "Epochs:          $EPOCHS"
echo "Output:          $OUTPUT_DIR"
echo ""

SUCCESS=0
FAILED=0
SKIPPED=0

run_factorization() {
    local results_dir="$1"
    local seed_label="$2"
    local seed_output="$OUTPUT_DIR/$seed_label"

    echo "------------------------------------------------------------------------"
    echo "Analyzing: $seed_label"
    echo "  Path: $results_dir"

    if [[ "$results_dir" == *"FAILED"* ]]; then
        echo "  SKIPPED (marked as FAILED)"
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi

    if [ ! -d "$results_dir/captured_states" ]; then
        echo "  SKIPPED (no captured_states directory)"
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi

    if [ -f "$seed_output/temporal_factorization_val.json" ]; then
        echo "  SKIPPED (already analyzed)"
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi

    mkdir -p "$seed_output"

    if python3 "$ANALYSIS_TOOLS/analyze_temporal_factorization.py" \
        --arch_dirs "$results_dir" \
        --arch_names "$seed_label" \
        --data_dir "$DATA_DIR" \
        --data_file "$DATA_FILE" \
        --epochs $EPOCHS \
        --split val \
        --output_dir "$seed_output" 2>&1; then
        echo "  ✓ Done"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "  ✗ FAILED"
        FAILED=$((FAILED + 1))
    fi
    echo ""
}

for result_dir in "$RUNS/multi_seed_classification"/*/; do
    label=$(basename "$result_dir")
    # Skip the shell script itself and non-result directories
    [[ "$label" == *.sh ]] && continue
    [[ "$label" == "factorization_analysis" ]] && continue
    run_factorization "$result_dir" "$label"
done

echo "========================================================================"
echo "Per-seed analysis complete. Aggregating across seeds..."
echo "========================================================================"
echo ""

OUTPUT_DIR="$OUTPUT_DIR" python3 - <<'PYEOF'
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

import os
output_dir = Path(os.environ["OUTPUT_DIR"])

# Group seeds by architecture
arch_r2 = defaultdict(lambda: {'epoch_0': [], 'epoch_500': []})

for seed_dir in sorted(output_dir.iterdir()):
    if not seed_dir.is_dir():
        continue
    result_file = seed_dir / "temporal_factorization_val.json"
    if not result_file.exists():
        continue

    with open(result_file) as f:
        data = json.load(f)

    seed_label = seed_dir.name  # e.g. gnn_seed1_results_8580377
    # Determine architecture from seed label
    if seed_label.startswith("gnn"):
        arch = "GNN"
    elif seed_label.startswith("transformer"):
        arch = "Transformer"
    elif seed_label.startswith("mamba"):
        arch = "Mamba"
    else:
        continue

    for arch_name, arch_data in data["architectures"].items():
        epochs = arch_data["epochs"]
        mean_r2 = arch_data["mean_r2"]

        for ep, r2 in zip(epochs, mean_r2):
            if ep == 0:
                arch_r2[arch]["epoch_0"].append(r2)
            elif ep == 500:
                arch_r2[arch]["epoch_500"].append(r2)

print("=" * 70)
print("FACTORIZATION ON CLASSIFICATION-TRAINED MODELS")
print("(same models as Table 1 topology results)")
print("=" * 70)
print(f"\n{'Architecture':<15} {'Epoch 0 R²':>20}  {'Epoch 500 R²':>20}  {'Change':>10}")
print("-" * 70)

summary = {}
for arch in ["GNN", "Transformer", "Mamba"]:
    e0 = arch_r2[arch]["epoch_0"]
    e500 = arch_r2[arch]["epoch_500"]
    if not e0 or not e500:
        print(f"{arch:<15}  {'N/A':>20}  {'N/A':>20}  {'N/A':>10}")
        continue
    m0, s0 = np.mean(e0), np.std(e0) / np.sqrt(len(e0))
    m500, s500 = np.mean(e500), np.std(e500) / np.sqrt(len(e500))
    change = (m500 - m0) / m0 * 100
    n0, n500 = len(e0), len(e500)
    print(f"{arch:<15}  {m0:.3f}±{s0:.3f} (n={n0})  {m500:.3f}±{s500:.3f} (n={n500})  {change:+.1f}%")
    summary[arch] = {
        "epoch_0_mean": float(m0), "epoch_0_sem": float(s0), "epoch_0_n": n0,
        "epoch_500_mean": float(m500), "epoch_500_sem": float(s500), "epoch_500_n": n500,
        "change_pct": float(change),
        "epoch_0_values": [float(v) for v in e0],
        "epoch_500_values": [float(v) for v in e500],
    }

agg_file = output_dir / "classification_factorization_aggregated.json"
with open(agg_file, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nAggregated results saved: {agg_file}")
PYEOF

echo ""
echo "========================================================================"
echo "Batch Factorization Analysis Complete"
echo "========================================================================"
echo "  Completed: $SUCCESS"
echo "  Failed:    $FAILED"
echo "  Skipped:   $SKIPPED"
echo ""
echo "Per-seed results: $OUTPUT_DIR/<seed_label>/temporal_factorization_val.json"
echo "Aggregated:       $OUTPUT_DIR/classification_factorization_aggregated.json"
