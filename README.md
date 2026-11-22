# Compositional Epistemic Reasoning: Geometric Traces of S5 Structure in Neural Networks

Code, data and result summaries for:

> **Formal Epistemic Structure Leaves Geometric Traces in LLMs: Evidence from S5 Multi-Agent Logic and RoBERTa.**
> David John Lemay, Emil Sayilov, Irina Rish.
> *IJCAI-ECAI 2026 Workshop on Logical and Symbolic Reasoning in Language Models (LogiSymb).*
> [PDF](paper/lemay2026_logisymb_epistemic_structure.pdf)

and its extended version, *Topological Structure in Compositional Epistemic Reasoning: From Small Models to Pretrained Transformers*, which adds the small-model experiments (Experiment 1).

We train models on binary entailment over S5 multi-agent epistemic logic problems ([MindGames](https://github.com/sileod/llm-theory-of-mind), SMCDEL format). A 2×2 factorial split trains on *n*_agents ∈ {2,3} × depth ∈ {0,1} and validates on *n*_agents ∈ {4,5} × depth ∈ {2,3}. We then ask whether hidden states and attention factorize along these two axes, as Van Benthem's product topology for S5 predicts. We also track how their Vietoris–Rips topology changes during learning.

| | Models | Code | Results |
|---|---|---|---|
| **Experiment 1** | GNN, Transformer, Mamba (~25K–91K params), 5 seeds, plus label-scramble controls | [`exp1_small_models/`](exp1_small_models/) | [`results/exp1/`](results/exp1/) |
| **Experiment 2** | RoBERTa-base (3 runs) and a Qwen2.5-0.5B replication | [`exp2_pretrained/`](exp2_pretrained/) | [`results/exp2/`](results/exp2/) |

## Repository layout

```
data/                         # the exact splits used in the papers
  mindgames_orthogonal_v4_classification/              # Exp 1: train.json / val.json (720 / 720)
  mindgames_orthogonal_v4_classification_scrambled/    # Exp 1: 30% label-noise control
  mindgames_orthogonal_v4_classification_full_scrambled/  # Exp 1: uniform-random-label control
  mindgames_orthogonal_v4_agent_aware/                 # Exp 2: train/val_generation.json (720 / 720)
  mindgames_orthogonal_test_held_out/                  # Exp 2: extra held-out test set (1,069)
data_generation/              # scripts that produced the splits (need the MindGames source)
exp1_small_models/            # training, topology and factorization analyses for Exp 1
exp2_pretrained/              # RoBERTa / Qwen fine-tuning and analyses (SLURM scripts)
kunneth_topology.py           # shared Künneth-formula persistent-homology helpers
results/                      # our outputs (JSON + aggregate figures), same layout as runs/
paper/                        # camera-ready IJCAI-ECAI 2026 LogiSymb paper
```

Everything you generate goes under `runs/` (git-ignored). `results/` holds our outputs in the same relative layout, for comparison. Model checkpoints and captured hidden states (~10 GB) are not included.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Mamba (Experiment 1) additionally needs a CUDA build of mamba-ssm:
pip install mamba-ssm
```

Run all commands from the repository root.

## Experiment 1: small models

**Training.** `train_exp1.sh` uses the hyperparameters stored in the paper's checkpoints: AdamW, lr 1e-4, weight decay 1e-2, batch 32, up to 10,000 epochs with early stopping (patience 500), evaluation and hidden-state capture every epoch.

```bash
bash exp1_small_models/train_exp1.sh compositional   # 3 archs x 5 seeds
bash exp1_small_models/train_exp1.sh noise30         # 30% label noise, 3 seeds
bash exp1_small_models/train_exp1.sh full_scramble   # uniform-random labels, 3 seeds
# a subset: bash exp1_small_models/train_exp1.sh compositional gnn transformer
```

Run directories are named exactly as in the paper (`gnn_seed1_results_8580377`, …), where the suffix is the original SLURM job ID. The analysis scripts list those directories explicitly, so they work unchanged.

**Analyses** (paper section → command):

| Result | Command | Output |
|---|---|---|
| Per-run VR topology (simplex counts, Betti numbers, trajectories) | `bash exp1_small_models/run_classification_topology_batch.sh` and `bash exp1_small_models/run_full_scramble_topology_batch.sh` | `<run>/topology_analysis/` |
| Topological simplification and correlation with accuracy (Table: simplification) | `python exp1_small_models/analyze_topology_classification.py --results_dir runs/exp1/multi_seed_classification --scrambled_dir runs/exp1/classification_full_scrambled` | `runs/exp1/topology_classification_analysis/` |
| Product factorization at init vs trained (Table: factorization) | `bash exp1_small_models/run_factorization_batch_classification.sh` | `runs/exp1/multi_seed_classification/factorization_analysis/` |
| Asymmetric / agent-level factorization (App.: asymmetry) | `bash exp1_small_models/run_full_factorization_batch_classification.sh` | `.../factorization_analysis_full/` |
| No-architecture (raw-input) control | `python exp1_small_models/analyze_raw_input_factorization.py --output_dir runs/exp1/raw_input` | |
| Nonlinear (MLP) probe control | `python exp1_small_models/analyze_nonlinear_factorization_probe.py` | `runs/exp1/nonlinear_factorization_probe_results.json` |
| Fixed-ε validation (App.: fixed ε) | `python exp1_small_models/analyze_topology_fixed_epsilon_classification.py` | `runs/exp1/topology_classification_analysis/fixed_epsilon/` |
| Percentile sensitivity | `python exp1_small_models/percentile_sensitivity_sweep.py --arch GNN` (also `Transformer`, `Mamba`) | stdout |
| Betti numbers across conditions (App.: Betti) | `python exp1_small_models/rerun_betti_numbers.py` | `runs/exp1/betti_numbers_corrected.json` |
| Künneth-formula test | `python exp1_small_models/analyze_kunneth_factorization_classification.py` | `runs/exp1/topology_classification_analysis/kunneth/` |

The Betti tables in the appendix come from `rerun_betti_numbers.py`. The per-run `topology_metrics.json` files in `results/exp1/` were produced before a fix to H0/H1 counting, so their Betti fields count every persistence birth rather than intervals alive at ε. Their simplex counts, the headline simplification metric, are unaffected.

## Experiment 2: RoBERTa and Qwen2.5

These are SLURM scripts written for the Mila cluster. Adjust the `#SBATCH` lines and module loads for your cluster, then submit from `exp2_pretrained/`:

```bash
cd exp2_pretrained
sbatch slurm_roberta_nli.sh                    # RoBERTa-base, lr 2e-5, bs 16, early stopping (patience 10), per-epoch states
sbatch slurm_run_analyses.sh <results_dir>     # linear probes, topology-vs-generalization, attention topology
sbatch slurm_gpt_nli.sh                        # Qwen2.5-0.5B, fixed 22 epochs, no early stopping
```

Other analyses:

```bash
python exp2_pretrained/analyze_kunneth_factorization.py --results_dir <roberta_results_dir>
python exp2_pretrained/analyze_attention_topology_gpt.py --checkpoint <qwen_results_dir>/checkpoint_best.pt --output_dir <qwen_results_dir>/attention_analysis_gpt
python exp2_pretrained/probe_layer_representations.py --results_dir <qwen_results_dir>
```

The RoBERTa run reported in the paper is `results/exp2/roberta_nli_results_9430862*` (best checkpoint at epoch 11, val acc 96.9%). The other two RoBERTa runs are `*_9490048_*` and `*_9490049_*`. The Qwen run is `gpt_nli_results_Qwen_Qwen2.5-0.5B_10203876`.

## Regenerating the data

The splits in `data/` are the ones used in the papers, so you don't need to regenerate them. To build new ones, use the MindGames generator, which in turn uses [SMCDEL](https://github.com/jrclogic/SMCDEL) to label problems:

```bash
git clone https://github.com/sileod/llm-theory-of-mind
export MINDGAMES_SRC=$PWD/llm-theory-of-mind/src
python data_generation/generate_orthogonal_disjoint_agent_aware.py --output_dir runs/data/v4_agent_aware --samples_per_combo 120 --seed 42
python data_generation/prepare_agent_aware_generation.py --input_dir runs/data/v4_agent_aware --output_dir runs/data/v4_generation
# label-scramble controls for Experiment 1:
python exp1_small_models/prepare_scrambled_classification.py
python exp1_small_models/prepare_full_scramble_classification.py
```

## Notes

- Comments and docstrings sometimes use internal experiment labels: `exp_3` is Experiment 1, `exp_6` is Experiment 2, and `exp_5` is the project that produced the dataset generator.
- Some script and argument names say "grokking" for historical reasons. The papers describe the dynamics as rapid compositional learning, not delayed generalization.
- Environment used for local analysis: Python 3.12, PyTorch 2.10, transformers 5.1, gudhi 3.11 (see `requirements.txt`). Training ran on single RTX8000 GPUs.

## Citation

```bibtex
@inproceedings{lemay2026epistemic,
  title     = {Formal Epistemic Structure Leaves Geometric Traces in {LLMs}: Evidence from {S5} Multi-Agent Logic and {RoBERTa}},
  author    = {Lemay, David John and Sayilov, Emil and Rish, Irina},
  booktitle = {IJCAI-ECAI 2026 Workshop on Logical and Symbolic Reasoning in Language Models (LogiSymb)},
  year      = {2026}
}
```

MindGames: Sileo & Lernould (2023), *MindGames: Targeting Theory of Mind in Large Language Models with Dynamic Epistemic Modal Logic*.

## License

MIT (see [LICENSE](LICENSE)). The MindGames generator is Apache-2.0 and is not vendored here.
