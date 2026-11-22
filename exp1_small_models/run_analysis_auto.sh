#!/bin/bash
# Automated topological analysis script for experiments with or without grokking
#
# Usage:
#   ./run_analysis_auto.sh <results_directory> [data_directory]
#
# Example:
#   ./run_analysis_auto.sh gnn_grokking_results_8067137 mindgames_grokking_split
#
# This script will:
# 1. Auto-detect transition epochs from the training curve
# 2. Run all four topological analyses using detected epochs
# 3. Save results to results_dir/topology_analysis and results_dir/modal_geometry_analysis

set -e  # Exit on error
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <results_directory> [data_directory]"
    echo ""
    echo "Example:"
    echo "  $0 gnn_grokking_results_8067137 mindgames_grokking_split"
    exit 1
fi

RESULTS_DIR="$1"
DATA_DIR="${2:-$SCRIPT_DIR/../data/mindgames_orthogonal_v4_classification}"

# Check if results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: Results directory not found: $RESULTS_DIR"
    exit 1
fi

# Check if checkpoint exists
if [ ! -f "$RESULTS_DIR/checkpoint_latest.pt" ]; then
    echo "Error: Checkpoint not found: $RESULTS_DIR/checkpoint_latest.pt"
    exit 1
fi

# Check if captured states exist
if [ ! -d "$RESULTS_DIR/captured_states" ]; then
    echo "Error: Captured states directory not found: $RESULTS_DIR/captured_states"
    exit 1
fi

echo "========================================================================"
echo "Automated Topological Analysis"
echo "========================================================================"
echo "Results directory: $RESULTS_DIR"
echo "Data directory: $DATA_DIR"
echo ""

# Step 1: Detect transition epochs
echo "========================================================================"
echo "Step 1: Detecting Transition Epochs"
echo "========================================================================"
python "$SCRIPT_DIR/detect_transition_epochs.py" \
    --checkpoint_path "$RESULTS_DIR/checkpoint_latest.pt" \
    --num_transitions 3

# Load detected epochs from JSON
EPOCH_FILE="$RESULTS_DIR/suggested_analysis_epochs.json"

if [ ! -f "$EPOCH_FILE" ]; then
    echo "Error: Failed to generate epoch suggestions"
    exit 1
fi

# Extract epochs using Python
read -r PRE_EPOCH POST_EPOCH TRANSITION_EPOCH GROKKING_DETECTED <<< $(python3 << EOF
import json
with open('$EPOCH_FILE', 'r') as f:
    data = json.load(f)

pre_epoch = data.get('early_epoch') or data['all_epochs'][0]
post_epoch = data.get('late_epoch') or data['all_epochs'][-1]

# Get main transition epoch
if data['grokking_detected'] and data['grokking_epoch']:
    transition_epoch = data['grokking_epoch']
elif data['suggested_epochs']:
    transition_epoch = data['suggested_epochs'][0]
else:
    transition_epoch = post_epoch // 2

print(f"{pre_epoch} {post_epoch} {transition_epoch} {data['grokking_detected']}")
EOF
)

echo ""
echo "Detected epochs:"
echo "  Pre-transition epoch:  $PRE_EPOCH"
echo "  Post-transition epoch: $POST_EPOCH"
echo "  Main transition epoch: $TRANSITION_EPOCH"
echo "  Grokking detected:     $GROKKING_DETECTED"
echo ""

# Step 2: Core topology analysis
echo "========================================================================"
echo "Step 2: Core Topological Analysis"
echo "========================================================================"
python "$SCRIPT_DIR/analyze_grokking_topology.py" \
    --results_dir "$RESULTS_DIR" \
    --split train \
    --epsilon_encoder 0.5 \
    --epsilon_decoder 0.3 \
    --auto_detect \
    --max_samples 200

echo ""
echo "Topology analysis complete!"
echo ""

# Step 3: Sample trajectory visualization
echo "========================================================================"
echo "Step 3: Sample Trajectory Visualization"
echo "========================================================================"
python "$SCRIPT_DIR/visualize_sample_trajectories.py" \
    --results_dir "$RESULTS_DIR" \
    --num_samples 15 \
    --transition_epoch "$TRANSITION_EPOCH" \
    --pre_epoch "$PRE_EPOCH" \
    --post_epoch "$POST_EPOCH"

echo ""
echo "Trajectory visualization complete!"
echo ""

# Step 4: Modal geometry analysis (requires data directory)
if [ -d "$DATA_DIR" ]; then
    echo "========================================================================"
    echo "Step 4: Modal Geometry Analysis (McKinsey-Tarski)"
    echo "========================================================================"
    python "$SCRIPT_DIR/visualize_modal_geometry.py" \
        --results_dir "$RESULTS_DIR" \
        --data_dir "$DATA_DIR" \
        --pre_grok_epoch "$PRE_EPOCH" \
        --post_grok_epoch "$POST_EPOCH" \
        --epsilon 0.5 \
        --max_samples 200

    echo ""
    echo "Modal geometry analysis complete!"
    echo ""

    # Step 5: Interior/closure analysis
    echo "========================================================================"
    echo "Step 5: Interior/Closure Analysis"
    echo "========================================================================"
    python "$SCRIPT_DIR/analyze_interior_closure.py" \
        --results_dir "$RESULTS_DIR" \
        --data_dir "$DATA_DIR" \
        --pre_epoch "$PRE_EPOCH" \
        --post_epoch "$POST_EPOCH" \
        --max_samples 200

    echo ""
    echo "Interior/closure analysis complete!"
    echo ""
else
    echo "========================================================================"
    echo "Steps 4-5: Skipped (data directory not found: $DATA_DIR)"
    echo "========================================================================"
    echo "Modal geometry and interior/closure analyses require the original dataset."
    echo ""
fi

# Summary
echo "========================================================================"
echo "Analysis Complete!"
echo "========================================================================"
echo ""
echo "Results saved to:"
echo "  - $RESULTS_DIR/topology_analysis/"
if [ -d "$DATA_DIR" ]; then
    echo "  - $RESULTS_DIR/modal_geometry_analysis/"
fi
echo ""
echo "Generated visualizations:"
echo "  1. topology_evolution.png         - Topological metrics over time"
echo "  2. sample_trajectories.png        - Individual sample paths through latent space"
echo "  3. distance_evolution.png         - Sample distances over training"
if [ -d "$DATA_DIR" ]; then
    echo "  4. modal_geometry_comparison.png  - Modal operator geometry comparison"
    echo "  5. interior_closure_analysis.png  - Interior vs closure properties"
    echo "  6. metric_comparison.png          - Geometric metric comparison"
fi
echo ""
echo "View the plots with:"
echo "  eog $RESULTS_DIR/topology_analysis/*.png"
if [ -d "$DATA_DIR" ]; then
    echo "  eog $RESULTS_DIR/modal_geometry_analysis/*.png"
fi
