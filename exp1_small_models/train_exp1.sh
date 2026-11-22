#!/bin/bash
# Train the Experiment 1 small models (GNN, Transformer, Mamba) on the
# MindGames 2x2 compositional split, plus the two label-scramble controls.
#
# Hyperparameters are the ones stored in the paper's checkpoints (see README).
# Run directories are named exactly as in the paper (the numeric suffix is the
# original SLURM job ID), so the analysis scripts, which list those directories,
# work without modification.
#
# Usage (from anywhere):
#   bash exp1_small_models/train_exp1.sh compositional [gnn transformer mamba]
#   bash exp1_small_models/train_exp1.sh noise30       [gnn transformer mamba]
#   bash exp1_small_models/train_exp1.sh full_scramble [gnn transformer mamba]
#
# Runs are sequential; on a cluster, launch one (condition, arch, seed) per job.
# Mamba needs the CUDA-only `mamba-ssm` package.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNS="${RUNS:-$REPO/runs/exp1}"

CONDITION="${1:?Usage: train_exp1.sh <compositional|noise30|full_scramble> [arch ...]}"
shift
ARCHS=("${@:-gnn transformer mamba}")
# shellcheck disable=SC2206
ARCHS=(${ARCHS[*]})

case "$CONDITION" in
  compositional)
    DATA_DIR="$REPO/data/mindgames_orthogonal_v4_classification"
    OUT_BASE="$RUNS/multi_seed_classification"
    SEEDS=(1 2 3 4 5) ;;
  noise30)
    DATA_DIR="$REPO/data/mindgames_orthogonal_v4_classification_scrambled"
    OUT_BASE="$RUNS/classification_scrambled"
    SEEDS=(1 2 3) ;;
  full_scramble)
    DATA_DIR="$REPO/data/mindgames_orthogonal_v4_classification_full_scrambled"
    OUT_BASE="$RUNS/classification_full_scrambled"
    SEEDS=(1 2 3) ;;
  *) echo "Unknown condition: $CONDITION" >&2; exit 1 ;;
esac

# Directory name used in the paper for (condition, arch, seed).
run_dir_name() {
  local arch=$1 seed=$2
  case "$CONDITION" in
    compositional)
      local job
      if [[ "$arch" == "mamba" && $seed -le 2 ]]; then job=9403834
      elif [[ $seed -le 2 ]]; then job=8580377
      else job=8584516; fi
      echo "${arch}_seed${seed}_results_${job}" ;;
    noise30)       echo "${arch}_seed${seed}_results_9424022" ;;
    full_scramble) echo "${arch}_scramble_seed${seed}_results_9474028" ;;
  esac
}

COMMON=(--num_epochs "${NUM_EPOCHS:-10000}" --batch_size 32 --learning_rate 1e-4 --weight_decay 1e-2
        --dropout 0.1 --eval_interval 1 --early_stopping_patience 500
        --num_encoder_layers 1 --num_decoder_layers 1 --disable_progress_bar)

mkdir -p "$OUT_BASE"
for arch in "${ARCHS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    out="$OUT_BASE/$(run_dir_name "$arch" "$seed")"
    if [ -f "$out/final_results.json" ]; then
      echo "skip $out (final_results.json exists)"; continue
    fi
    mkdir -p "$out"
    echo "=== $CONDITION / $arch / seed $seed -> $out"
    case "$arch" in
      gnn)
        python "$SCRIPT_DIR/train_gnn_grokking.py" --data_dir "$DATA_DIR" --output_dir "$out" \
          --embedding_dim 32 --hidden_size 64 --use_attention True \
          "${COMMON[@]}" --seed "$seed" --run_name "gnn_seed${seed}" ;;
      transformer)
        python "$SCRIPT_DIR/train_symbolic_grokking_transformer.py" --data_dir "$DATA_DIR" --output_dir "$out" \
          --d_model 32 --nhead 4 --dim_feedforward 64 \
          "${COMMON[@]}" --seed "$seed" --run_name "transformer_seed${seed}" ;;
      mamba)
        python "$SCRIPT_DIR/train_mamba_classification.py" --data_dir "$DATA_DIR" --output_dir "$out" \
          --d_model 64 \
          "${COMMON[@]}" --seed "$seed" --run_name "mamba_seed${seed}" ;;
      *) echo "Unknown arch: $arch" >&2; exit 1 ;;
    esac 2>&1 | tee "$out/training.log"
  done
done
