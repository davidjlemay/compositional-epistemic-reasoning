#!/bin/bash
#SBATCH --job-name=nli_analyses
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/analyses_%j.out
#SBATCH --error=logs/analyses_%j.err
#
# Written for the Mila cluster (partition, GPU type and module names below).
# Adjust the #SBATCH lines and module loads for your cluster. Submit from
# this directory: sbatch slurm_run_analyses.sh

# ---------------------------------------------------------------------------
# Combined analysis script for the RoBERTa NLI results.
#
# Runs three analyses on a trained checkpoint directory:
#   1. probe_layer_representations.py   (CPU, ~4 GB,  ~15 min)
#   2. analyze_nli_generalization.py    (CPU, ~8 GB,  ~30 min)
#   3. analyze_attention_topology.py    (GPU, ~8 GB,  ~1 h)
#
# Usage:
#   sbatch slurm_run_analyses.sh <results_dir> [data_dir] [test_dir]
#
# Arguments:
#   results_dir  Path to training output directory (contains captured_states/)
#   data_dir     Path to NLI dataset (default: ../data/mindgames_orthogonal_v4_agent_aware)
#   test_dir     Path to held-out test set (default: ../data/mindgames_orthogonal_test_held_out)
#
# Example:
#   sbatch slurm_run_analyses.sh roberta_nli_results_9012676
# ---------------------------------------------------------------------------

set -euo pipefail

RESULTS_DIR="${1:?Usage: sbatch slurm_run_analyses.sh <results_dir> [data_dir] [test_dir]}"
DATA_DIR="${2:-../data/mindgames_orthogonal_v4_agent_aware}"
TEST_DIR="${3:-../data/mindgames_orthogonal_test_held_out}"

echo "========================================================================"
echo "NLI Analyses"
echo "========================================================================"
echo "Job started at : $(date)"
echo "Job ID         : $SLURM_JOB_ID"
echo "Node           : $SLURMD_NODENAME"
echo "Results dir    : $RESULTS_DIR"
echo "Data dir       : $DATA_DIR"

mkdir -p logs

# Load modules (Mila cluster)
module load anaconda/3
module load cuda/11.7

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

conda activate venv

echo "Python  : $(python --version)"
echo "PyTorch : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA    : $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU     : $(python -c 'import torch; print(torch.cuda.get_device_name() if torch.cuda.is_available() else "No GPU")')"

# ---------------------------------------------------------------------------
# Copy files to local node storage
# ---------------------------------------------------------------------------
echo ""
echo "Copying files to local storage..."

cp $SLURM_SUBMIT_DIR/probe_layer_representations.py    $SLURM_TMPDIR/
cp $SLURM_SUBMIT_DIR/analyze_nli_generalization.py     $SLURM_TMPDIR/
cp $SLURM_SUBMIT_DIR/analyze_attention_topology.py     $SLURM_TMPDIR/
cp $SLURM_SUBMIT_DIR/roberta_nli_model.py              $SLURM_TMPDIR/
cp $SLURM_SUBMIT_DIR/nli_dataset.py                    $SLURM_TMPDIR/

cp -r $SLURM_SUBMIT_DIR/$RESULTS_DIR                   $SLURM_TMPDIR/
cp -r $SLURM_SUBMIT_DIR/$DATA_DIR                      $SLURM_TMPDIR/

if [ -d "$SLURM_SUBMIT_DIR/$TEST_DIR" ]; then
    cp -r $SLURM_SUBMIT_DIR/$TEST_DIR $SLURM_TMPDIR/
fi

cd $SLURM_TMPDIR
# Everything was copied flat into $SLURM_TMPDIR; refer to it by basename.
RESULTS_DIR="$(basename "$RESULTS_DIR")"
DATA_DIR="$(basename "$DATA_DIR")"
TEST_DIR="$(basename "$TEST_DIR")"

# ---------------------------------------------------------------------------
# 1. Linear probing (CPU)
# ---------------------------------------------------------------------------
echo ""
echo "[$(date '+%H:%M:%S')] === Step 1: Linear probing ==="

python probe_layer_representations.py \
    --results_dir "$RESULTS_DIR" \
    --data_dir "$DATA_DIR" \
    --epoch best \
    --output_dir "$RESULTS_DIR/probe_analysis"

echo "[$(date '+%H:%M:%S')] Probe analysis complete."

# ---------------------------------------------------------------------------
# 2. Topology generalization analysis (CPU)
# ---------------------------------------------------------------------------
echo ""
echo "[$(date '+%H:%M:%S')] === Step 2: Topology generalization ==="

python analyze_nli_generalization.py \
    --results_dir "$RESULTS_DIR" \
    --output_dir "$RESULTS_DIR/topology_analysis" \
    --max_points 150 \
    --eps_percentile 20

echo "[$(date '+%H:%M:%S')] Topology generalization analysis complete."

# ---------------------------------------------------------------------------
# 3. Attention topology analysis (GPU)
# ---------------------------------------------------------------------------
echo ""
echo "[$(date '+%H:%M:%S')] === Step 3: Attention topology ==="

if [ -f "$RESULTS_DIR/checkpoint_best.pt" ]; then
    CHECKPOINT="$RESULTS_DIR/checkpoint_best.pt"
else
    CHECKPOINT="$RESULTS_DIR/checkpoint_latest.pt"
fi
echo "Using checkpoint: $CHECKPOINT"

python analyze_attention_topology.py \
    --checkpoint "$CHECKPOINT" \
    --data_dir "$DATA_DIR" \
    --split val \
    --output_dir "$RESULTS_DIR/attention_analysis" \
    --max_length 256 \
    --batch_size 16 \
    --max_points 300

echo "[$(date '+%H:%M:%S')] Attention topology analysis complete."

# ---------------------------------------------------------------------------
# Copy results back
# ---------------------------------------------------------------------------
echo ""
echo "Copying results back..."
cp -r $SLURM_TMPDIR/$RESULTS_DIR $SLURM_SUBMIT_DIR/${RESULTS_DIR}_analyses_$SLURM_JOB_ID

echo ""
echo "========================================================================"
echo "Job completed at: $(date)"
echo "Results: $SLURM_SUBMIT_DIR/${RESULTS_DIR}_analyses_$SLURM_JOB_ID"
echo "  Probe     : probe_analysis/"
echo "  Topology  : topology_analysis/"
echo "  Attention : attention_analysis/"
echo "========================================================================"
