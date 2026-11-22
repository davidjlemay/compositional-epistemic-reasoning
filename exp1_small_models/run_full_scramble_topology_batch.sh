#!/bin/bash
# Batch topology analysis for the full-scramble classification control results.
# Each example receives an independent uniform-random label (50/50 split).
#
# Run from anywhere; reads runs from $RUNS (default: <repo>/runs/exp1).
# Usage:
#   bash exp1_small_models/run_full_scramble_topology_batch.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNS="${RUNS:-$REPO/runs/exp1}"
FULL_SCRAMBLE_DIR="$RUNS/classification_full_scrambled"
DATA_DIR="$REPO/data/mindgames_orthogonal_v4_classification_full_scrambled"


echo "========================================================================"
echo "Batch Topology Analysis: Full-Scramble Classification Control"
echo "========================================================================"
echo "Analysis tools: $(pwd)"
echo "Results base:   $FULL_SCRAMBLE_DIR"
echo "Data directory: $DATA_DIR"
echo ""

SUCCESS=0
FAILED=0
SKIPPED=0

run_analysis() {
    local results_dir="$1"
    local label="$2"

    echo "------------------------------------------------------------------------"
    echo "Analyzing: $label"
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

    if [ -f "$results_dir/topology_analysis/topology_metrics.json" ]; then
        echo "  SKIPPED (topology_analysis already exists)"
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi

    if bash "$SCRIPT_DIR/run_analysis_auto.sh" "$results_dir" "$DATA_DIR" 2>&1; then
        echo "  Done"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "  FAILED"
        FAILED=$((FAILED + 1))
    fi
    echo ""
}

# GNN seeds
for seed in 1 2 3; do
    dir=$(ls -d "$FULL_SCRAMBLE_DIR/gnn_scramble_seed${seed}_results_"* 2>/dev/null | head -1)
    [ -n "$dir" ] && run_analysis "$dir" "gnn_scramble_seed${seed}"
done

# Transformer seeds
for seed in 1 2 3; do
    dir=$(ls -d "$FULL_SCRAMBLE_DIR/transformer_scramble_seed${seed}_results_"* 2>/dev/null | head -1)
    [ -n "$dir" ] && run_analysis "$dir" "transformer_scramble_seed${seed}"
done

# Mamba seeds
for seed in 1 2 3; do
    dir=$(ls -d "$FULL_SCRAMBLE_DIR/mamba_scramble_seed${seed}_results_"* 2>/dev/null | head -1)
    [ -n "$dir" ] && run_analysis "$dir" "mamba_scramble_seed${seed}"
done

echo "========================================================================"
echo "Batch Analysis Complete"
echo "========================================================================"
echo "  Completed: $SUCCESS"
echo "  Failed:    $FAILED"
echo "  Skipped:   $SKIPPED"
echo ""
echo "Topology results saved to topology_analysis/ in each result directory."
if [ $FAILED -gt 0 ]; then
    exit 1
fi
