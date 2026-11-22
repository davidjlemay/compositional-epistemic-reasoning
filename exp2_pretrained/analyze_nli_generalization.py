#!/usr/bin/env python3
"""
Analyze correlation between topological complexity and generalization for exp_6 NLI.

Adapted from exp_5/analyze_topology_generalization.py.

Key questions:
1. Does simplex count (per layer) predict generalization gap (val_acc - test_acc)?
2. Do specific layers have higher predictive power?
3. Does topology evolve differently pre/post grokking for NLI vs generation?

Topology is computed on per-layer [CLS] representations captured during evaluation.
States format: {layer_idx: tensor [N, hidden_size]} saved at each captured epoch.
"""

import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist
import argparse
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False
    print("Warning: gudhi not available. Topology computation disabled.")


def compute_vr_topology(points: np.ndarray, max_edge_length: float = 1.0, max_dim: int = 1):
    """
    Compute Vietoris-Rips persistent homology.

    Returns total simplices and Betti numbers (H0, H1).
    """
    if not GUDHI_AVAILABLE:
        return {'total_simplices': 0, 'h0': 0, 'h1': 0}

    rips = gudhi.RipsComplex(points=points, max_edge_length=max_edge_length)
    simplex_tree = rips.create_simplex_tree(max_dimension=max_dim + 1)
    simplex_tree.compute_persistence()

    total_simplices = simplex_tree.num_simplices()
    betti = simplex_tree.betti_numbers()
    h0 = betti[0] if len(betti) > 0 else 0
    h1 = betti[1] if len(betti) > 1 else 0

    return {'total_simplices': total_simplices, 'h0': h0, 'h1': h1}


def find_state_epochs(results_dir: Path):
    """Find all epochs with captured [CLS] state files."""
    states_dir = results_dir / 'captured_states'
    if not states_dir.exists():
        return []
    files = sorted(states_dir.glob('states_epoch_*_val.pt'))
    epochs = []
    for f in files:
        parts = f.stem.split('_')
        try:
            epoch = int(parts[2])
            if epoch >= 0:
                epochs.append(epoch)
        except (IndexError, ValueError):
            pass
    return sorted(epochs)


def load_metrics(results_dir: Path):
    """Load train and val metrics."""
    train_file = results_dir / 'train_metrics.json'
    val_file = results_dir / 'val_metrics.json'
    test_file = results_dir / 'test_results.json'

    train_metrics = json.load(open(train_file)) if train_file.exists() else []
    val_metrics = json.load(open(val_file)) if val_file.exists() else []
    test_results = json.load(open(test_file)) if test_file.exists() else {}

    return (
        {m['epoch']: m for m in train_metrics},
        {m['epoch']: m for m in val_metrics},
        test_results,
    )


def subsample_and_adaptive_eps(
    points: np.ndarray, max_points: int, eps_percentile: int
) -> tuple[np.ndarray, float]:
    """
    Subsample a point cloud and compute an adaptive ε for VR.

    ε is set to the eps_percentile-th percentile of pairwise distances in the
    original (non-reduced) space. This keeps graph density consistent across
    epochs regardless of how spread out representations are, avoiding the
    near-complete-graph problem that causes simplex explosion at fixed ε.

    No PCA: distances are computed in the original high-dimensional space,
    preserving the exact geometry and allowing consistent cross-epoch comparison.
    """
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]

    dists = pdist(points)  # condensed pairwise distances, O(N²·D) but N is small
    eps = float(np.percentile(dists, eps_percentile))
    return points, eps


def analyze_epoch(
    results_dir: Path,
    epoch: int,
    val_acc: float,
    train_acc: float,
    test_acc: float,
    max_points: int = 150,
    eps_percentile: int = 20,
):
    """Compute topology for each layer at a given epoch."""
    states_file = results_dir / 'captured_states' / f'states_epoch_{epoch}_val.pt'
    if not states_file.exists():
        return None

    states = torch.load(states_file, map_location='cpu', weights_only=False)
    # states: {layer_idx: [N, hidden_size]}

    layer_topo = {}
    for layer_idx, hidden in states.items():
        points = hidden.float().numpy() if isinstance(hidden, torch.Tensor) else np.array(hidden)
        points, eps = subsample_and_adaptive_eps(points, max_points=max_points, eps_percentile=eps_percentile)
        topo = compute_vr_topology(points, max_edge_length=eps)
        topo['eps'] = eps
        layer_topo[layer_idx] = topo

    return {
        'epoch': epoch,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'generalization_gap': val_acc - test_acc,
        'train_val_gap': train_acc - val_acc,
        'layer_topology': layer_topo,
        # Aggregate across all layers
        'total_simplices': sum(t['total_simplices'] for t in layer_topo.values()),
        'mean_h0': np.mean([t['h0'] for t in layer_topo.values()]),
        'mean_h1': np.mean([t['h1'] for t in layer_topo.values()]),
    }


def compute_correlations(data: list):
    """Compute Pearson/Spearman correlations between topology metrics and generalization."""
    if len(data) < 5:
        print("Too few data points for correlation analysis")
        return {}

    total_simplices = [d['total_simplices'] for d in data]
    gen_gap = [d['generalization_gap'] for d in data]
    val_acc = [d['val_acc'] for d in data]
    mean_h1 = [d['mean_h1'] for d in data]

    results = {}

    r, p = pearsonr(total_simplices, gen_gap)
    rho, _ = spearmanr(total_simplices, gen_gap)
    results['total_simplices_vs_gen_gap'] = {'pearson_r': r, 'pearson_p': p, 'spearman_rho': rho}

    r, p = pearsonr(total_simplices, val_acc)
    results['total_simplices_vs_val_acc'] = {'pearson_r': r, 'pearson_p': p}

    r, p = pearsonr(mean_h1, gen_gap)
    results['h1_vs_gen_gap'] = {'pearson_r': r, 'pearson_p': p}

    return results


def plot_results(data: list, output_dir: Path, model_name: str = 'roberta-base'):
    """Plot topology vs generalization evolution."""
    epochs = [d['epoch'] for d in data]
    val_acc = [d['val_acc'] for d in data]
    test_acc = [d['test_acc'] for d in data]
    gen_gap = [d['generalization_gap'] for d in data]
    total_simplices = [d['total_simplices'] for d in data]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Topology & Generalization — {model_name}', fontsize=13)

    axes[0, 0].plot(epochs, val_acc, label='Val acc', color='steelblue')
    axes[0, 0].plot(epochs, test_acc, label='Test acc (held-out)', color='tomato')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].set_title('Accuracy over training')

    axes[0, 1].plot(epochs, gen_gap, color='purple')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Val acc - Test acc')
    axes[0, 1].set_title('Generalization gap')

    axes[1, 0].plot(epochs, total_simplices, color='darkorange')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Total simplices (all layers)')
    axes[1, 0].set_title('Topology complexity')

    axes[1, 1].scatter(total_simplices, gen_gap, c=epochs, cmap='viridis', alpha=0.7)
    axes[1, 1].set_xlabel('Total simplices')
    axes[1, 1].set_ylabel('Generalization gap')
    axes[1, 1].set_title('Simplices vs generalization gap')
    cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
    cbar.set_label('Epoch')

    plt.tight_layout()
    out = output_dir / f'topology_generalization_{model_name.replace("/", "_")}.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved plot: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Path to training output directory (contains captured_states/)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Where to save analysis results (default: results_dir/topology_analysis)')
    parser.add_argument('--max_points', type=int, default=150,
                        help='Max points to subsample for VR computation')
    parser.add_argument('--eps_percentile', type=int, default=20,
                        help='Percentile of pairwise distances used as adaptive VR epsilon')
    parser.add_argument('--model_name', type=str, default='roberta-base')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / 'topology_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metrics
    train_by_epoch, val_by_epoch, test_results = load_metrics(results_dir)
    test_acc_global = test_results.get('accuracy', 0.0)
    print(f"Global test accuracy (held-out): {test_acc_global:.4f}")

    # Find epochs with captured states
    epochs = find_state_epochs(results_dir)
    print(f"Found {len(epochs)} epochs with captured states: {epochs[:10]}{'...' if len(epochs) > 10 else ''}")

    if not epochs:
        print("No captured states found. Run training with --state_capture_interval to generate states.")
        return

    # Analyze each epoch
    data = []
    for epoch in epochs:
        train_acc = train_by_epoch.get(epoch, {}).get('accuracy', 0.0)
        val_acc = val_by_epoch.get(epoch, {}).get('accuracy', 0.0)
        # Use global test acc (test was only run once at end); or load per-epoch if available
        result = analyze_epoch(
            results_dir, epoch,
            val_acc=val_acc,
            train_acc=train_acc,
            test_acc=test_acc_global,
            max_points=args.max_points,
            eps_percentile=args.eps_percentile,
        )
        if result:
            data.append(result)
            print(f"  Epoch {epoch:4d}: val={val_acc:.4f} gen_gap={result['generalization_gap']:.4f} "
                  f"simplices={result['total_simplices']}")

    if not data:
        print("No data collected.")
        return

    # Save raw data
    serializable = []
    for d in data:
        row = {k: v for k, v in d.items() if k != 'layer_topology'}
        serializable.append(row)
    with open(output_dir / 'aggregated_data.json', 'w') as f:
        json.dump(serializable, f, indent=2)

    # Correlations
    corr = compute_correlations(data)
    print("\nCorrelation results:")
    print(json.dumps(corr, indent=2))
    with open(output_dir / 'correlation_results.json', 'w') as f:
        json.dump(corr, f, indent=2)

    # Plots
    if len(data) >= 2:
        plot_results(data, output_dir, model_name=args.model_name)

    print(f"\nAnalysis complete. Results in: {output_dir}")


if __name__ == '__main__':
    main()
