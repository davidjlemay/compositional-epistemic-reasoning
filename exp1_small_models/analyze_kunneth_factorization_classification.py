#!/usr/bin/env python3
"""
Kunneth-formula topological factorization test for the small-model
classification experiment (GNN / Transformer / Mamba, 5 seeds each).

Small-model analogue of exp2_pretrained/analyze_kunneth_factorization.py -- see
that script's docstring for the full motivation. In short: the paper's
existing evidence for product-topology factorization ("Table 3") is a linear
probe (fiber-conditional ICA + Ridge regression), which
tests an affine property, not a topological one. This script tests the
Kunneth formula instead: if the held-out 2x2 cell Z (n_agents in {4,5} x
hypothesis_depth in {2,3}) factors as X x Y (X varies n_agents at fixed
depth, Y varies depth at fixed n_agents), then
beta_k(Z) = sum_{i+j=k} beta_i(X) beta_j(Y) for k=0,1. This is invariant
under continuous, nonlinear deformation, unlike linear decodability.

Note on paper Definition 4: neither the existing task-level probe
(analyze_full_product_factorization_agent_aware.py, in-sample-pooled) nor the
fiber-aware nonlinear probe (analyze_nonlinear_factorization_probe.py,
within-fiber CV) implements the literal held-out cross-fiber train/test that
Definition 4 describes. This script's Kunneth check is a joint-vs-marginal
consistency test on the held-out cell, not a directional generalization
claim, so it does not inherit that discrepancy -- flagged here for future
reference, not fixed as part of this work.

Uncertainty comes from the existing 5 seeds per architecture (states already
captured, no retraining needed) rather than bootstrapping, since seed
replication is already the paper's standard for small models.

Usage:
    python analyze_kunneth_factorization_classification.py \
        --multi_seed_dir runs/exp1/multi_seed_classification \
        --val_path data/mindgames_orthogonal_v4_classification/val.json \
        --output_dir runs/exp1/topology_classification_analysis/kunneth \
        --epochs 0 500
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from kunneth_topology import kunneth_fiber_test

ARCH_PREFIXES = {
    'GNN': 'gnn_seed',
    'Transformer': 'transformer_seed',
    'Mamba': 'mamba_seed',
}


# ---------------------------------------------------------------------------
# I/O helpers (mirrors analyze_topology_fixed_epsilon_classification.py)
# ---------------------------------------------------------------------------

def find_seed_dirs(multi_seed_dir: Path, arch_prefix: str):
    """Return one valid (non-FAILED) dir per seed number for an architecture."""
    by_seed: dict[int, list[Path]] = {}
    for d in sorted(multi_seed_dir.iterdir()):
        if not (d.is_dir() and d.name.startswith(arch_prefix) and 'FAILED' not in d.name):
            continue
        if not (d / 'captured_states').exists():
            continue
        m = re.search(r'seed(\d+)', d.name)
        if not m:
            continue
        seed_num = int(m.group(1))
        by_seed.setdefault(seed_num, []).append(d)

    dirs = []
    for seed_num in sorted(by_seed):
        candidates = by_seed[seed_num]
        canonical = [c for c in candidates if (c / 'topology_analysis' / 'topology_metrics.json').exists()]
        dirs.append(canonical[0] if canonical else candidates[0])
    return dirs


def load_encoder_states(states_dir: Path, epoch: int):
    """Load encoder_states for a single epoch (no subsampling here -- the
    Kunneth test subsamples per fiber-cloud downstream)."""
    fpath = states_dir / f'states_epoch_{epoch}_val.pt'
    states = torch.load(fpath, map_location='cpu', weights_only=False)
    enc = states['encoder_states']
    return enc.numpy() if isinstance(enc, torch.Tensor) else np.asarray(enc)


def load_val_factors(val_path: Path):
    with open(val_path) as f:
        val = json.load(f)
    n_agents = np.array([e['n_agents'] for e in val])
    depth = np.array([e['hypothesis_depth'] for e in val])
    return n_agents, depth


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_seed_epoch(states_dir: Path, epoch: int, n_agents, depth,
                        max_points, eps_percentile, rng):
    embeddings = load_encoder_states(states_dir, epoch)
    if len(embeddings) != len(n_agents):
        n = min(len(embeddings), len(n_agents))
        embeddings, n_agents, depth = embeddings[:n], n_agents[:n], depth[:n]

    b_levels = sorted(np.unique(depth).tolist())
    a_levels = sorted(np.unique(n_agents).tolist())
    return kunneth_fiber_test(
        embeddings, n_agents, depth,
        b_ref=b_levels[0], a_ref=a_levels[0],
        max_points=max_points, eps_percentile=eps_percentile, rng=rng,
    )


def run_analysis(multi_seed_dir: Path, val_path: Path, output_dir: Path,
                  epochs: list, max_points: int, eps_percentile: int, seed: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    n_agents, depth = load_val_factors(val_path)
    print(f"Held-out cell: n_agents levels={sorted(set(n_agents.tolist()))}, "
          f"depth levels={sorted(set(depth.tolist()))}, N={len(n_agents)}")

    all_results = {}
    for arch_name, prefix in ARCH_PREFIXES.items():
        print(f"\n{'='*60}\nArchitecture: {arch_name}\n{'='*60}")
        seed_dirs = find_seed_dirs(multi_seed_dir, prefix)
        if not seed_dirs:
            print(f"  No valid seed dirs found for {arch_name}")
            continue

        arch_results = {epoch: [] for epoch in epochs}
        for sd in seed_dirs:
            states_dir = sd / 'captured_states'
            for epoch in epochs:
                fpath = states_dir / f'states_epoch_{epoch}_val.pt'
                if not fpath.exists():
                    print(f"  {sd.name}: missing epoch {epoch}, skipping")
                    continue
                rng = np.random.default_rng(seed)
                result = analyze_seed_epoch(
                    states_dir, epoch, n_agents, depth,
                    max_points, eps_percentile, rng,
                )
                arch_results[epoch].append(result)
                obs = result['observed']['Z']
                pred = result['predicted_z']
                print(f"  {sd.name} epoch {epoch}: obs H0={obs['h0']} pred H0={pred['h0']} | "
                      f"obs H1={obs['h1']} pred H1={pred['h1']}")

        all_results[arch_name] = arch_results

    def serializable(obj):
        if isinstance(obj, dict):
            return {str(k): serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [serializable(x) for x in obj]
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return obj

    with open(output_dir / 'kunneth_factorization_classification.json', 'w') as f:
        json.dump(serializable(all_results), f, indent=2)

    make_summary_figure(all_results, epochs, output_dir)
    print_summary_table(all_results, epochs)


def _aggregate(arch_results: list, key_path):
    """Mean +/- SEM across seeds for a nested observed/predicted field."""
    vals = []
    for r in arch_results:
        v = r
        for k in key_path:
            v = v[k]
        vals.append(v)
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    sem = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return float(np.mean(vals)), sem


def make_summary_figure(all_results: dict, epochs: list, output_dir: Path):
    archs = [a for a in ARCH_PREFIXES if a in all_results and all_results[a]]
    if not archs:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(archs))
    width = 0.18

    for ax_idx, (ax, hk) in enumerate(zip(axes, ['h0', 'h1'])):
        for e_idx, epoch in enumerate(epochs):
            obs_means, obs_sems, pred_means, pred_sems = [], [], [], []
            for arch in archs:
                seed_results = all_results[arch].get(epoch, [])
                if not seed_results:
                    obs_means.append(np.nan); obs_sems.append(0)
                    pred_means.append(np.nan); pred_sems.append(0)
                    continue
                om, os_ = _aggregate(seed_results, ['observed', 'Z', hk])
                pm, ps = _aggregate(seed_results, ['predicted_z', hk])
                obs_means.append(om); obs_sems.append(os_)
                pred_means.append(pm); pred_sems.append(ps)

            offset = (e_idx - (len(epochs) - 1) / 2) * width * 2
            ax.bar(x + offset - width / 2, obs_means, width, yerr=obs_sems,
                   label=f'Observed (epoch {epoch})', alpha=0.8)
            ax.bar(x + offset + width / 2, pred_means, width, yerr=pred_sems,
                   label=f'Predicted (epoch {epoch})', alpha=0.5, hatch='//')

        ax.set_xticks(x)
        ax.set_xticklabels(archs)
        ax.set_ylabel(f'β{hk[1]}')
        ax.set_title(f'{hk.upper()}: observed vs. Künneth-predicted')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Künneth Factorization Test — Classification (5 seeds/arch)', fontsize=13)
    plt.tight_layout()
    out = output_dir / 'kunneth_factorization_classification.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {out}")


def print_summary_table(all_results: dict, epochs: list):
    print(f"\n{'='*90}")
    print(f"{'Architecture':<14} {'Epoch':>6} {'Obs H0':>10} {'Pred H0':>10} {'Obs H1':>10} {'Pred H1':>10}")
    print(f"{'='*90}")
    for arch_name in ARCH_PREFIXES:
        if arch_name not in all_results:
            continue
        for epoch in epochs:
            seed_results = all_results[arch_name].get(epoch, [])
            if not seed_results:
                continue
            oh0, _ = _aggregate(seed_results, ['observed', 'Z', 'h0'])
            ph0, _ = _aggregate(seed_results, ['predicted_z', 'h0'])
            oh1, _ = _aggregate(seed_results, ['observed', 'Z', 'h1'])
            ph1, _ = _aggregate(seed_results, ['predicted_z', 'h1'])
            print(f"{arch_name:<14} {epoch:>6} {oh0:>10.1f} {ph0:>10.1f} {oh1:>10.1f} {ph1:>10.1f}")
    print(f"{'='*90}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--multi_seed_dir', type=str,
                        default='runs/exp1/multi_seed_classification')
    parser.add_argument('--val_path', type=str,
                        default='data/mindgames_orthogonal_v4_classification/val.json')
    parser.add_argument('--output_dir', type=str,
                        default='runs/exp1/topology_classification_analysis/kunneth')
    parser.add_argument('--epochs', type=int, nargs='+', default=[0, 500])
    parser.add_argument('--max_points', type=int, default=200)
    parser.add_argument('--eps_percentile', type=int, default=75,
                        help='Percentile of per-point k-NN distances (not all-pairs '
                             'distances) used to set the VR epsilon -- see '
                             'kunneth_topology.adaptive_eps_subsample docstring.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_analysis(
        multi_seed_dir=Path(args.multi_seed_dir),
        val_path=Path(args.val_path),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        max_points=args.max_points,
        eps_percentile=args.eps_percentile,
        seed=args.seed,
    )
