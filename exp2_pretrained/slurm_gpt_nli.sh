#!/bin/bash
#SBATCH --job-name=gpt_nli_epistemic
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/gpt_nli_%j.out
#SBATCH --error=logs/gpt_nli_%j.err
#
# Written for the Mila cluster (partition, GPU type and module names below).
# Adjust the #SBATCH lines and module loads for your cluster. Submit from
# this directory: sbatch slurm_gpt_nli.sh

# Decoder-only (GPT-style) replication of the RoBERTa experiment, checking whether the RoBERTa findings (topological simplification,
# attention encoding hierarchy) generalize beyond an encoder model.
#
# Default model is Qwen2.5-0.5B (fits comfortably in fp32 on one RTX8000).
# To run the 1.5B follow-up, override at submission time, e.g.:
#   sbatch --export=MODEL_NAME='Qwen/Qwen2.5-1.5B',TORCH_DTYPE=bfloat16,BATCH_SIZE=8 slurm_gpt_nli.sh
# (bfloat16 + smaller batch reduces AdamW memory footprint for the larger model;
# fp32 for 0.5B is fine at batch_size=16, matching the RoBERTa run's settings.)

echo "========================================================================"
echo "GPT-style NLI — Epistemic Entailment Classification (GPT-style replication)"
echo "========================================================================"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"

mkdir -p logs

# Load modules (Mila cluster)
module load anaconda/3
module load cuda/11.7

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# Cache downloaded weights on scratch rather than $HOME's quota-limited disk.
export HF_HOME=${HF_HOME:-$SCRATCH/huggingface_cache}

MODEL_NAME=${MODEL_NAME:-'Qwen/Qwen2.5-0.5B'}
TORCH_DTYPE=${TORCH_DTYPE:-'float32'}
BATCH_SIZE=${BATCH_SIZE:-16}

conda activate venv

echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name() if torch.cuda.is_available() else "No GPU")')"
echo "Model: $MODEL_NAME (dtype=$TORCH_DTYPE, batch_size=$BATCH_SIZE)"

# Copy code and data to local node storage
echo "Copying files to local storage..."
cp -r $SLURM_SUBMIT_DIR/train_gpt_nli.py $SLURM_TMPDIR/
cp -r $SLURM_SUBMIT_DIR/gpt_nli_model.py $SLURM_TMPDIR/
cp -r $SLURM_SUBMIT_DIR/nli_dataset_gpt.py $SLURM_TMPDIR/
cp -r $SLURM_SUBMIT_DIR/nli_dataset.py $SLURM_TMPDIR/
cp -r $SLURM_SUBMIT_DIR/../data/mindgames_orthogonal_v4_agent_aware $SLURM_TMPDIR/
cp -r $SLURM_SUBMIT_DIR/../data/mindgames_orthogonal_test_held_out $SLURM_TMPDIR/

cd $SLURM_TMPDIR
mkdir -p outputs_gpt

echo "========================================================================"
echo "Task: NLI classification — premise + hypothesis -> entailment / non_entailment"
echo "========================================================================"

python train_gpt_nli.py \
    --model_name "$MODEL_NAME" \
    --torch_dtype "$TORCH_DTYPE" \
    --data_dir ./mindgames_orthogonal_v4_agent_aware \
    --test_dir ./mindgames_orthogonal_test_held_out \
    --max_length 256 \
    --num_epochs 22 \
    --batch_size $BATCH_SIZE \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --eval_interval 1 \
    --state_capture_interval 1 \
    --early_stopping_patience 0 \
    --freeze_layers 0 \
    --output_dir ./outputs_gpt \
    --seed 42

echo ""
echo "Copying results back..."
SAFE_MODEL_NAME=$(echo "$MODEL_NAME" | tr '/' '_')
RESULT_DIR="gpt_nli_results_${SAFE_MODEL_NAME}_$SLURM_JOB_ID"
cp -r outputs_gpt $SLURM_SUBMIT_DIR/$RESULT_DIR

echo ""
echo "========================================================================"
echo "Job completed at: $(date)"
echo "Results: $SLURM_SUBMIT_DIR/$RESULT_DIR"
echo "========================================================================"

if [ -f outputs_gpt/test_results.json ]; then
    echo ""
    echo "=== TEST RESULTS (held-out agents) ==="
    python -c "
import json
with open('outputs_gpt/test_results.json') as f:
    r = json.load(f)
print(f'Test accuracy:  {r.get(\"accuracy\", \"N/A\"):.4f}')
print(f'Test F1 (macro): {r.get(\"f1\", \"N/A\"):.4f}')
if 'complexity_accuracy' in r:
    print('Accuracy by complexity:')
    for k, v in sorted(r['complexity_accuracy'].items()):
        print(f'  {k}: {v:.4f}')
"
fi
