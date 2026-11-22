#!/usr/bin/env python3
"""
Kunneth-formula topological factorization test for RoBERTa/Qwen NLI hidden states.

Tests whether representations exhibit genuine product-topology factorization
over n_agents x hypothesis_depth (the Van Benthem product-topology claim,
"Key Result 1"), using persistent homology rather than linear decodability.
If the full held-out 2x2 cell Z factors as X x Y (X varies n_agents at fixed
depth, Y varies depth at fixed n_agents), the Kunneth formula predicts
beta_k(Z) = sum_{i+j=k} beta_i(X) beta_j(Y) for k=0,1. This holds under
continuous, nonlinear deformation, so it is a different failure mode than the
existing per-layer logistic probe (probe_layer_representations.py) -- which
is, as of this writing, the *only* factorization-adjacent test that exists at
LLM scale. There is no fiber-conditional ICA+Ridge test for RoBERTa/Qwen
anywhere in this repo; that test only exists for the small models
(exp1_small_models/analyze_full_product_factorization_agent_aware.py).

Note on paper Definition 4: the existing small-model fiber probes
(exp1_small_models/analyze_full_product_factorization_agent_aware.py
and exp1_small_models/analyze_nonlinear_factorization_probe.py)
do not implement the literal held-out cross-fiber train/test that Definition 4
describes -- they fit in-sample-pooled or within-fiber CV, never train-on-
fiber-A/test-on-fiber-B. This script's Kunneth check is a joint-vs-marginal
consistency test on the held-out cell, not a directional generalization claim,
so it does not inherit that discrepancy. Flagged here for future reference;
not fixed as part of this work.

Reads captured states from: <results_dir>/captured_states/states_epoch_<E>_val.pt
Format: {layer_idx: Tensor[N, hidden_size]}. Metadata (n_agents,
hypothesis_depth) is not saved alongside states -- it's reconstructed by
re-loading the val split and matching by order (states are saved in dataset
iteration order; the val loader uses shuffle=False, so this is safe). Train-
split states are NOT safe to use this way (train loader is shuffled) --
matches probe_layer_representations.py's documented reasoning.

Usage:
    python analyze_kunneth_factorization.py \
        --results_dir roberta_nli_results_9430862_analyses_9474787 \
        --data_dir mindgames_orthogonal_v4_agent_aware \
        --epoch best \
        --n_bootstrap 30 \
        --model_name roberta-base
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from nli_dataset import load_nli_dataset
from kunneth_topology import bootstrap_kunneth_test

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


# ---------------------------------------------------------------------------
# I/O helpers (mirrors probe_layer_representations.py's loading pattern)
# ---------------------------------------------------------------------------

def find_captured_epoch(results_dir: Path, target_epoch):
    """Find the captured epoch matching target_epoch ('best' or an int)."""
    states_dir = results_dir / 'captured_states'
    files = sorted(states_dir.glob('states_epoch_*_val.pt'))
    if not files:
        raise FileNotFoundError(f"No captured state files in {states_dir}")

    def parse_epoch(f):
        try:
            return int(f.stem.split('_')[2])
        except (IndexError, ValueError):
            return -1

    epochs = [(parse_epoch(f), f) for f in files]
    epochs = [(e, f) for e, f in epochs if e >= 0]

    if target_epoch == 'best':
        metrics_file = results_dir / 'val_metrics.json'
        if metrics_file.exists():
            with open(metrics_file) as fh:
                metrics = json.load(fh)
            best_epoch = max(metrics, key=lambda m: m['accuracy'])['epoch']
            return min(epochs, key=lambda ef: abs(ef[0] - best_epoch))
        return epochs[-1]

    target = int(target_epoch)
    return min(epochs, key=lambda ef: abs(ef[0] - target))


def load_states(states_file: Path) -> dict:
    """Load {layer_idx: np.ndarray[N, hidden]} from a states file."""
    states = torch.load(states_file, map_location='cpu', weights_only=False)
    return {
        int(k): (v.float().numpy() if isinstance(v, torch.Tensor) else np.asarray(v))
        for k, v in states.items()
    }


def get_factor_labels(examples: list):
    n_agents = np.array([e.get('n_agents', -1) for e in examples])
    depth = np.array([e.get('hypothesis_depth', -1) for e in examples])
    return n_agents, depth


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def run_layer_sweep(states, n_agents, depth, max_points, eps_percentile, n_bootstrap, seed):
    layer_results = {}
    for layer_idx in sorted(states.keys()):
        rng = np.random.default_rng(seed + layer_idx)
        embeddings = states[layer_idx]
        result = bootstrap_kunneth_test(
            embeddings, n_agents, depth,
            max_points=max_points, eps_percentile=eps_percentile,
            n_bootstrap=n_bootstrap, rng=rng,
        )
        layer_results[layer_idx] = result
        logging.info(
            f"  Layer {layer_idx:2d} | obs H0={result['observed_h0']['mean']:.1f} "
            f"pred H0={result['predicted_h0']['mean']:.1f} | "
            f"obs H1={result['observed_h1']['mean']:.1f} "
            f"pred H1={result['predicted_h1']['mean']:.1f}"
        )
    return layer_results


def plot_layer_sweep(layer_results: dict, output_dir: Path, model_name: str):
    layers = sorted(layer_results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, k in zip(axes, ['h0', 'h1']):
        obs_mean = [layer_results[l][f'observed_{k}']['mean'] for l in layers]
        obs_lo = [layer_results[l][f'observed_{k}']['ci_low'] for l in layers]
        obs_hi = [layer_results[l][f'observed_{k}']['ci_high'] for l in layers]
        pred_mean = [layer_results[l][f'predicted_{k}']['mean'] for l in layers]
        pred_lo = [layer_results[l][f'predicted_{k}']['ci_low'] for l in layers]
        pred_hi = [layer_results[l][f'predicted_{k}']['ci_high'] for l in layers]

        ax.plot(layers, obs_mean, 'o-', color='steelblue', label='Observed β(Z)')
        ax.fill_between(layers, obs_lo, obs_hi, color='steelblue', alpha=0.2)
        ax.plot(layers, pred_mean, 's--', color='tomato', label='Künneth-predicted')
        ax.fill_between(layers, pred_lo, pred_hi, color='tomato', alpha=0.2)
        ax.set_xlabel('Layer')
        ax.set_ylabel(f'β{k[1]}')
        ax.set_title(f'{k.upper()}: observed vs. Künneth-predicted')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Künneth Factorization Test — {model_name}', fontsize=13)
    plt.tight_layout()
    out = output_dir / f'kunneth_factorization_{model_name.replace("/", "_")}.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved plot: {out}")


def serializable(obj):
    if isinstance(obj, dict):
        return {str(k): serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serializable(x) for x in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    return obj


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Training output dir with captured_states/ and metrics')
    parser.add_argument('--data_dir', type=str, default='data/mindgames_orthogonal_v4_agent_aware')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Where to save results (default: results_dir/kunneth_analysis)')
    parser.add_argument('--epoch', type=str, default='best',
                        help='Epoch to analyze (int or "best")')
    parser.add_argument('--max_points', type=int, default=150)
    parser.add_argument('--eps_percentile', type=int, default=75,
                        help='Percentile of per-point k-NN distances (not all-pairs '
                             'distances) used to set the VR epsilon -- see '
                             'kunneth_topology.adaptive_eps_subsample docstring. '
                             'Validated against a positive control (kunneth_positive_control.py) '
                             'to recover known ground-truth Betti numbers at this default.')
    parser.add_argument('--n_bootstrap', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_name', type=str, default='roberta-base')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / 'kunneth_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    epoch, states_file = find_captured_epoch(results_dir, args.epoch)
    logging.info(f"Using captured states from epoch {epoch}: {states_file}")
    states = load_states(states_file)

    dataset = load_nli_dataset(Path(args.data_dir))
    val_examples = dataset['validation']

    n_states = next(iter(states.values())).shape[0]
    if n_states != len(val_examples):
        logging.warning(f"State count ({n_states}) != val examples ({len(val_examples)}). Truncating.")
        n = min(n_states, len(val_examples))
        val_examples = val_examples[:n]
        states = {k: v[:n] for k, v in states.items()}

    n_agents, depth = get_factor_labels(val_examples)
    mask = (n_agents >= 0) & (depth >= 0)
    if not mask.all():
        logging.warning(f"Dropping {(~mask).sum()} examples with missing factor labels.")
        n_agents, depth = n_agents[mask], depth[mask]
        states = {k: v[mask] for k, v in states.items()}

    logging.info(
        f"Held-out cell: n_agents levels={sorted(np.unique(n_agents).tolist())}, "
        f"depth levels={sorted(np.unique(depth).tolist())}, N={mask.sum()}"
    )

    layer_results = run_layer_sweep(
        states, n_agents, depth,
        max_points=args.max_points, eps_percentile=args.eps_percentile,
        n_bootstrap=args.n_bootstrap, seed=args.seed,
    )

    output = {
        'epoch': epoch,
        'model_name': args.model_name,
        'n_examples': int(mask.sum()),
        'max_points': args.max_points,
        'eps_percentile': args.eps_percentile,
        'n_bootstrap': args.n_bootstrap,
        'layer_results': layer_results,
        'note': (
            'Kunneth check is a joint-vs-marginal consistency test on the '
            'held-out 2x2 cell, independent of the cross-fiber-generalization '
            'gap in the existing linear probes (Definition 4 describes a '
            'held-out train/test split across depth fibers that is not '
            'literally implemented anywhere in this repo as of this analysis).'
        ),
    }
    out_file = output_dir / f'kunneth_factorization_epoch{epoch}.json'
    with open(out_file, 'w') as f:
        json.dump(serializable(output), f, indent=2)
    logging.info(f"Saved results: {out_file}")

    plot_layer_sweep(layer_results, output_dir, args.model_name)
    logging.info(f"Analysis complete. Results in: {output_dir}")


if __name__ == '__main__':
    main()
