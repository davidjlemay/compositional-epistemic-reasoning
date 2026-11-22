#!/bin/bash
# Batch topology analysis for the multi-seed classification runs and the
# 30%-label-noise control.
#
# Run from anywhere; reads runs from $RUNS (default: <repo>/runs/exp1).
# Usage:
#   bash exp1_small_models/run_classification_topology_batch.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNS="${RUNS:-$REPO/runs/exp1}"
DATA_DIR="$REPO/data/mindgames_orthogonal_v4_classification"

cd "$SCRIPT_DIR"

echo "========================================================================"
echo "Batch Topology Analysis: Classification + Scrambled Control"
echo "========================================================================"
echo "Analysis tools: $SCRIPT_DIR"
echo "Results base:   $RUNS"
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

    # Skip FAILED directories
    if [[ "$results_dir" == *"FAILED"* ]]; then
        echo "  SKIPPED (marked as FAILED)"
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi

    # Skip if no captured states
    if [ ! -d "$results_dir/captured_states" ]; then
        echo "  SKIPPED (no captured_states directory)"
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi

    # Skip if already analyzed
    if [ -f "$results_dir/topology_analysis/topology_metrics.json" ]; then
        echo "  SKIPPED (topology_analysis already exists)"
        SKIPPED=$((SKIPPED + 1))
        return 0
    fi

    if bash "$SCRIPT_DIR/run_analysis_auto.sh" "$results_dir" "$DATA_DIR" 2>&1; then
        echo "  ✓ Done"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "  ✗ FAILED"
        FAILED=$((FAILED + 1))
    fi
    echo ""
}

# --- Multi-seed classification runs ---
echo "=== Multi-Seed Classification (mindgames_orthogonal_v4_classification) ==="
echo ""

MULTI_SEED_DIR="$RUNS/multi_seed_classification"

for result_dir in "$MULTI_SEED_DIR"/*/; do
    label=$(basename "$result_dir")
    run_analysis "$result_dir" "$label"
done

# --- Scrambled-label control ---
echo "=== 30% Label-Noise Control ==="
echo ""

DATA_DIR="$REPO/data/mindgames_orthogonal_v4_classification_scrambled"
for result_dir in "$RUNS/classification_scrambled"/*/; do
    [ -d "$result_dir" ] || continue
    run_analysis "$result_dir" "$(basename "$result_dir")"
done

# --- Summary ---
echo "========================================================================"
echo "Batch Analysis Complete"
echo "========================================================================"
echo "  Completed: $SUCCESS"
echo "  Failed:    $FAILED"
echo "  Skipped:   $SKIPPED"
echo ""
echo "Topology results saved to topology_analysis/ in each result directory."
echo ""
if [ $FAILED -gt 0 ]; then
    exit 1
fi
