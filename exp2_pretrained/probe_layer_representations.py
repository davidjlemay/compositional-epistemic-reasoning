#!/usr/bin/env python3
"""
Linear probing of per-layer [CLS] representations for epistemic variables.

For each RoBERTa layer, trains a logistic regression to decode:
  - constraint       (binary: entailment / non_entailment)
  - n_agents         (multiclass: 2, 3, 4 agents)
  - hypothesis_depth (multiclass: 0, 1, 2, 3)

Probe accuracy per layer reveals which layers encode each epistemic variable.

Reads captured states from: <results_dir>/captured_states/states_epoch_<E>_val.pt
Format: {layer_idx: Tensor[N, hidden_size]}

The dataset labels are reconstructed by re-loading the val split and matching
by order (states are saved in dataset iteration order).

Usage:
    python probe_layer_representations.py \
        --results_dir roberta_nli_results_<JOB_ID> \
        --epoch best \
        --data_dir mindgames_orthogonal_v4_agent_aware
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from nli_dataset import load_nli_dataset, load_test_set

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_best_epoch(results_dir: Path) -> int:
    """Return the epoch with the highest val accuracy from val_metrics.json."""
    metrics_file = results_dir / 'val_metrics.json'
    if not metrics_file.exists():
        raise FileNotFoundError(metrics_file)
    with open(metrics_file) as f:
        metrics = json.load(f)
    best = max(metrics, key=lambda m: m['accuracy'])
    return best['epoch']


def find_captured_epoch(results_dir: Path, target_epoch) -> int:
    """Find the closest captured epoch to target_epoch."""
    states_dir = results_dir / 'captured_states'
    files = sorted(states_dir.glob('states_epoch_*_val.pt'))
    if not files:
        raise FileNotFoundError(f"No captured state files in {states_dir}")

    def parse_epoch(f):
        try:
            return int(f.stem.split('_')[2])
        except (IndexError, ValueError):
            return -1

    epochs = [parse_epoch(f) for f in files]
    epochs = [(e, f) for e, f in zip(epochs, files) if e >= 0]

    if target_epoch == 'best':
        # Use last available (closest to best)
        return epochs[-1][0], epochs[-1][1]
    else:
        # Find closest
        target = int(target_epoch)
        closest = min(epochs, key=lambda ef: abs(ef[0] - target))
        return closest[0], closest[1]


def load_states(states_file: Path) -> dict:
    """Load {layer_idx: Tensor[N, hidden]} from a states file."""
    states = torch.load(states_file, map_location='cpu', weights_only=False)
    return {int(k): v.float().numpy() for k, v in states.items()}


def get_labels(examples: list) -> dict:
    """Extract probe targets from dataset examples."""
    constraint = [1 if e['constraint'] == 'entailment' else 0 for e in examples]
    n_agents = [e.get('n_agents', -1) for e in examples]
    depth = [e.get('hypothesis_depth', -1) for e in examples]
    return {
        'constraint': np.array(constraint),
        'n_agents': np.array(n_agents),
        'hypothesis_depth': np.array(depth),
    }


def probe_one_layer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_iter: int = 1000,
) -> dict:
    """Fit logistic regression probe and return accuracy metrics."""
    # Normalize
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_n = (X_train - mean) / std
    X_val_n = (X_val - mean) / std

    unique = np.unique(y_train)
    if len(unique) < 2:
        return {'val_acc': float('nan'), 'bal_acc': float('nan')}

    clf = LogisticRegression(
        max_iter=max_iter,
        C=1.0,
        solver='lbfgs',
        n_jobs=-1,
    )
    clf.fit(X_train_n, y_train)

    preds = clf.predict(X_val_n)
    return {
        'val_acc': float(accuracy_score(y_val, preds)),
        'bal_acc': float(balanced_accuracy_score(y_val, preds)),
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_probing(
    train_states: dict,
    val_states: dict,
    train_labels: dict,
    val_labels: dict,
    output_dir: Path,
    epoch: int,
):
    layers = sorted(train_states.keys())
    targets = list(train_labels.keys())
    results = {t: {} for t in targets}

    logging.info(f"Probing {len(layers)} layers × {len(targets)} targets")

    for layer in layers:
        X_tr = train_states[layer]
        X_va = val_states[layer]
        for target in targets:
            y_tr = train_labels[target]
            y_va = val_labels[target]
            # Filter out -1 (unknown) labels
            mask_tr = y_tr >= 0
            mask_va = y_va >= 0
            if mask_tr.sum() < 10 or mask_va.sum() < 5:
                continue
            r = probe_one_layer(X_tr[mask_tr], y_tr[mask_tr], X_va[mask_va], y_va[mask_va])
            results[target][layer] = r
            logging.info(
                f"  Layer {layer:2d} | {target:20s} | val_acc={r['val_acc']:.3f} "
                f"bal_acc={r['bal_acc']:.3f}"
            )

    # Baselines (majority class)
    baselines = {}
    for target in targets:
        y_va = val_labels[target]
        y_va = y_va[y_va >= 0]
        if len(y_va) > 0:
            majority = np.bincount(y_va).argmax()
            baselines[target] = float((y_va == majority).mean())

    # Save
    output = {
        'epoch': epoch,
        'results': {t: {str(l): v for l, v in r.items()} for t, r in results.items()},
        'baselines': baselines,
        'layers': [int(l) for l in layers],
    }
    out_file = output_dir / f'probe_results_epoch{epoch}.json'
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)
    logging.info(f"Saved probe results: {out_file}")

    # Plot
    _plot_probes(results, baselines, layers, epoch, output_dir)
    return output


def _plot_probes(results, baselines, layers, epoch, output_dir):
    targets = list(results.keys())
    fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 4), sharey=False)
    if len(targets) == 1:
        axes = [axes]

    colors = {'val_acc': 'steelblue', 'bal_acc': 'tomato'}

    for ax, target in zip(axes, targets):
        baseline = baselines.get(target, 0.0)
        layer_list = sorted(results[target].keys())
        val_accs = [results[target][l].get('val_acc', float('nan')) for l in layer_list]
        bal_accs = [results[target][l].get('bal_acc', float('nan')) for l in layer_list]

        ax.plot(layer_list, val_accs, 'o-', color=colors['val_acc'], label='Accuracy')
        ax.plot(layer_list, bal_accs, 's--', color=colors['bal_acc'], label='Balanced acc')
        ax.axhline(baseline, color='gray', linestyle=':', label=f'Majority ({baseline:.2f})')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Probe accuracy')
        ax.set_title(target.replace('_', ' ').title())
        ax.legend(fontsize=8)
        ax.set_xticks(layer_list)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Linear Probing — RoBERTa NLI (epoch {epoch})', fontsize=12)
    plt.tight_layout()
    out = output_dir / f'probe_curves_epoch{epoch}.png'
    plt.savefig(out, dpi=150)
    plt.close()
    logging.info(f"Saved plot: {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir', required=True,
                   help='Training output dir with captured_states/ and metrics')
    p.add_argument('--data_dir', default='data/mindgames_orthogonal_v4_agent_aware')
    p.add_argument('--test_dir', default='data/mindgames_orthogonal_test_held_out')
    p.add_argument('--epoch', default='best',
                   help='Epoch to probe (int or "best")')
    p.add_argument('--output_dir', default=None,
                   help='Where to save results (default: results_dir/probe_analysis)')
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / 'probe_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve epoch
    if args.epoch == 'best':
        target_epoch = find_best_epoch(results_dir)
        logging.info(f"Best epoch: {target_epoch}")
    else:
        target_epoch = int(args.epoch)

    epoch, states_file = find_captured_epoch(results_dir, target_epoch)
    logging.info(f"Using captured states from epoch {epoch}: {states_file}")

    # Load val states (captured during training eval)
    val_states = load_states(states_file)
    logging.info(f"Loaded states: {len(val_states)} layers, "
                 f"shape example: {next(iter(val_states.values())).shape}")

    # Load dataset labels
    dataset = load_nli_dataset(Path(args.data_dir))
    val_examples = dataset['validation']
    train_examples = dataset['train']

    N_val = next(iter(val_states.values())).shape[0]
    if N_val != len(val_examples):
        logging.warning(
            f"State count ({N_val}) != val examples ({len(val_examples)}). "
            f"Truncating to min."
        )
        n = min(N_val, len(val_examples))
        val_examples = val_examples[:n]
        val_states = {k: v[:n] for k, v in val_states.items()}

    val_labels = get_labels(val_examples)

    # Always probe using val states split 80/20.
    # Train states are captured during a shuffled DataLoader iteration, so their
    # order doesn't match the label list loaded from disk — using them would fit noise.
    # Val states use shuffle=False, so ordering is guaranteed to match val_examples.
    logging.info("Splitting val states 80/20 for probe train/eval.")
    n = len(val_examples)
    split = int(0.8 * n)
    train_states = {k: v[:split] for k, v in val_states.items()}
    val_states_probe = {k: v[split:] for k, v in val_states.items()}
    train_labels = {k: v[:split] for k, v in val_labels.items()}
    val_labels_probe = {k: v[split:] for k, v in val_labels.items()}
    run_probing(train_states, val_states_probe, train_labels, val_labels_probe, output_dir, epoch)


if __name__ == '__main__':
    main()
