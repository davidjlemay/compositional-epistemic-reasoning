#!/bin/bash
#SBATCH --job-name=roberta_nli_epistemic
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/roberta_nli_%j.out
#SBATCH --error=logs/roberta_nli_%j.err
#
# Written for the Mila cluster (partition, GPU type and module names below).
# Adjust the #SBATCH lines and module loads for your cluster. Submit from
# this directory: sbatch slurm_roberta_nli.sh

echo "========================================================================"
echo "RoBERTa NLI — Epistemic Entailment Classification"
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

conda activate venv

echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name() if torch.cuda.is_available() else "No GPU")')"

# Copy code and data to local node storage
echo "Copying files to local storage..."
cp -r $SLURM_SUBMIT_DIR/train_roberta_nli.py $SLURM_TMPDIR/
cp -r $SLURM_SUBMIT_DIR/roberta_nli_model.py $SLURM_TMPDIR/
cp -r $SLURM_SUBMIT_DIR/nli_dataset.py $SLURM_TMPDIR/
cp -r $SLURM_SUBMIT_DIR/../data/mindgames_orthogonal_v4_agent_aware $SLURM_TMPDIR/
cp -r $SLURM_SUBMIT_DIR/../data/mindgames_orthogonal_test_held_out $SLURM_TMPDIR/

cd $SLURM_TMPDIR
mkdir -p outputs

echo "========================================================================"
echo "Model: roberta-base (fine-tuned end-to-end)"
echo "Task: NLI classification — premise + hypothesis -> entailment / non_entailment"
echo "========================================================================"

python train_roberta_nli.py \
    --model_name roberta-base \
    --data_dir ./mindgames_orthogonal_v4_agent_aware \
    --test_dir ./mindgames_orthogonal_test_held_out \
    --max_length 256 \
    --num_epochs 50 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --eval_interval 1 \
    --state_capture_interval 1 \
    --early_stopping_patience 10 \
    --freeze_layers 0 \
    --output_dir ./outputs \
    --seed 42

echo ""
echo "Copying results back..."
RESULT_DIR="roberta_nli_results_$SLURM_JOB_ID"
cp -r outputs $SLURM_SUBMIT_DIR/$RESULT_DIR

echo ""
echo "========================================================================"
echo "Job completed at: $(date)"
echo "Results: $SLURM_SUBMIT_DIR/$RESULT_DIR"
echo "========================================================================"

# Print summary
if [ -f outputs/test_results.json ]; then
    echo ""
    echo "=== TEST RESULTS (held-out agents) ==="
    python -c "
import json
with open('outputs/test_results.json') as f:
    r = json.load(f)
print(f'Test accuracy:  {r.get(\"accuracy\", \"N/A\"):.4f}')
print(f'Test F1 (macro): {r.get(\"f1\", \"N/A\"):.4f}')
if 'complexity_accuracy' in r:
    print('Accuracy by complexity:')
    for k, v in sorted(r['complexity_accuracy'].items()):
        print(f'  {k}: {v:.4f}')
"
fi
