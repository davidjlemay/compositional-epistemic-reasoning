#!/usr/bin/env python3
"""
Fixed-epsilon validation for the CLASSIFICATION experiment (exp_3).

Addresses the same concern as the generation experiment:
"If epsilon is the Nth percentile of distances and distances change
during training, does simplification just measure scale change?"

Runs over all available seeds for GNN, Transformer, and Mamba, then
produces per-architecture summaries and an aggregate LaTeX table.

Usage:
    python analyze_topology_fixed_epsilon_classification.py \
        --multi_seed_dir runs/exp1/multi_seed_classification \
        --output_dir runs/exp1/topology_classification_analysis/fixed_epsilon \
        --max_samples 200 \
        --n_epochs 20

The script subsamples n_epochs evenly across the full training run to
keep runtime manageable (full ~600-epoch runs at 200 samples each take
a few minutes per seed with the VR expansion step).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import re
from scipy.spatial.distance import cdist
import gudhi as gd
from gudhi import SimplexTree
import argparse


# ---------------------------------------------------------------------------
# Architecture config: which directory prefixes to look for
# ---------------------------------------------------------------------------
ARCH_PREFIXES = {
    'GNN': 'gnn_seed',
    'Transformer': 'transformer_seed',
    'Mamba': 'mamba_seed',
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def find_seed_dirs(multi_seed_dir: Path, arch_prefix: str):
    """Return one valid (non-FAILED) dir per seed number for an architecture.

    When multiple job IDs exist for the same seed (e.g. reruns), prefer the
    directory that already has a topology_analysis sub-folder (canonical run),
    then fall back to the first alphabetically.
    """
    import re
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
        # Prefer directories with existing topology_analysis
        canonical = [c for c in candidates if (c / 'topology_analysis' / 'topology_metrics.json').exists()]
        dirs.append(canonical[0] if canonical else candidates[0])
    return dirs


def list_val_epochs(states_dir: Path):
    """Return sorted list of epoch integers that have val state files."""
    epochs = []
    for f in states_dir.glob('states_epoch_*_val.pt'):
        m = re.search(r'epoch_(\d+)', f.name)
        if m:
            epochs.append(int(m.group(1)))
    return sorted(epochs)


def subsample_epochs(epochs, n_epochs):
    """Pick n_epochs evenly-spaced indices (always include first and last)."""
    if len(epochs) <= n_epochs:
        return epochs
    indices = np.linspace(0, len(epochs) - 1, n_epochs, dtype=int)
    return [epochs[i] for i in indices]


def load_representations(states_dir: Path, epoch: int, max_samples: int):
    """Load encoder_states for a single epoch and subsample."""
    fpath = states_dir / f'states_epoch_{epoch}_val.pt'
    states = torch.load(fpath, map_location='cpu', weights_only=False)
    enc = states['encoder_states']
    if isinstance(enc, torch.Tensor):
        arr = enc.numpy()
    else:
        # Fallback for list-of-batch format (generation experiment)
        rows = []
        for item in enc:
            if isinstance(item, list):
                h = item[-1]
                rows.append((h[-1, :] if h.dim() > 1 else h).numpy())
            else:
                rows.append((item[-1, :] if item.dim() > 1 else item).numpy())
        arr = np.array(rows)

    if max_samples and len(arr) > max_samples:
        rng = np.random.default_rng(seed=0)
        idx = rng.choice(len(arr), size=max_samples, replace=False)
        arr = arr[idx]
    return arr


# ---------------------------------------------------------------------------
# Distance and VR helpers
# ---------------------------------------------------------------------------

def pairwise_distances(data: np.ndarray):
    dm = cdist(data, data)
    return dm[np.triu_indices_from(dm, k=1)]


def distance_stats(data: np.ndarray):
    dists = pairwise_distances(data)
    return {
        'mean': float(np.mean(dists)),
        'percentile_20': float(np.percentile(dists, 20)),
        'percentile_30': float(np.percentile(dists, 30)),
        'median': float(np.median(dists)),
    }


def compute_vr_complex(points: np.ndarray, epsilon: float, max_dimension: int = 2):
    """Compute VR complex at a fixed epsilon; return simplex counts and Betti numbers."""
    st = SimplexTree()
    n = len(points)
    dm = cdist(points, points)

    for i in range(n):
        st.insert([i], filtration=0.0)
    for i in range(n):
        for j in range(i + 1, n):
            if dm[i, j] <= epsilon:
                st.insert([i, j], filtration=dm[i, j])

    st.expansion(max_dimension)

    counts = {0: 0, 1: 0, 2: 0}
    for simplex, _ in st.get_filtration():
        dim = len(simplex) - 1
        if dim in counts:
            counts[dim] += 1

    st.compute_persistence()
    persistence = st.persistence()
    h0 = sum(1 for d, _ in persistence if d == 0)
    h1 = sum(1 for d, _ in persistence if d == 1)

    return {
        'total_simplices': counts[0] + counts[1] + counts[2],
        'h0': h0,
        'h1': h1,
    }


# ---------------------------------------------------------------------------
# Per-seed analysis
# ---------------------------------------------------------------------------

def analyze_seed(seed_dir: Path, n_epochs: int, max_samples: int):
    """Run adaptive vs fixed-epsilon analysis for one seed. Returns dict."""
    states_dir = seed_dir / 'captured_states'
    all_epochs = list_val_epochs(states_dir)
    if not all_epochs:
        return None

    epochs = subsample_epochs(all_epochs, n_epochs)
    print(f"    Epochs: {epochs[0]}–{epochs[-1]} ({len(epochs)} sampled)")

    # Compute representations and epoch-0 stats
    repr0 = load_representations(states_dir, epochs[0], max_samples)
    stats0 = distance_stats(repr0)
    eps_fixed = stats0['percentile_20']  # Match the paper's 20th-percentile default

    print(f"    Fixed ε = {eps_fixed:.4f} (20th pct at epoch 0)")

    records = []
    for epoch in epochs:
        rep = load_representations(states_dir, epoch, max_samples)
        stats = distance_stats(rep)
        eps_adaptive = stats['percentile_20']

        topo_adaptive = compute_vr_complex(rep, eps_adaptive)
        topo_fixed = compute_vr_complex(rep, eps_fixed)

        records.append({
            'epoch': epoch,
            'mean_dist': stats['mean'],
            'eps_adaptive': eps_adaptive,
            'adaptive_simplices': topo_adaptive['total_simplices'],
            'adaptive_h1': topo_adaptive['h1'],
            'fixed_simplices': topo_fixed['total_simplices'],
            'fixed_h1': topo_fixed['h1'],
        })
        print(f"      epoch {epoch:4d}: adaptive={topo_adaptive['total_simplices']:,}  "
              f"fixed={topo_fixed['total_simplices']:,}  dist_mean={stats['mean']:.3f}")

    initial_adaptive = records[0]['adaptive_simplices']
    final_adaptive = records[-1]['adaptive_simplices']
    initial_fixed = records[0]['fixed_simplices']
    final_fixed = records[-1]['fixed_simplices']

    adaptive_reduction = (initial_adaptive - final_adaptive) / max(initial_adaptive, 1) * 100
    fixed_reduction = (initial_fixed - final_fixed) / max(initial_fixed, 1) * 100
    dist_change = (records[-1]['mean_dist'] - records[0]['mean_dist']) / max(records[0]['mean_dist'], 1e-9) * 100

    return {
        'seed_dir': str(seed_dir),
        'eps_fixed': eps_fixed,
        'epochs': [r['epoch'] for r in records],
        'records': records,
        'initial_mean_dist': records[0]['mean_dist'],
        'final_mean_dist': records[-1]['mean_dist'],
        'dist_change_pct': dist_change,
        'adaptive_reduction_pct': adaptive_reduction,
        'fixed_reduction_pct': fixed_reduction,
        'initial_adaptive': initial_adaptive,
        'final_adaptive': final_adaptive,
        'initial_fixed': initial_fixed,
        'final_fixed': final_fixed,
    }


# ---------------------------------------------------------------------------
# Main analysis loop
# ---------------------------------------------------------------------------

def run_analysis(multi_seed_dir: Path, output_dir: Path, n_epochs: int, max_samples: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for arch_name, prefix in ARCH_PREFIXES.items():
        print(f"\n{'='*60}")
        print(f"Architecture: {arch_name}")
        print(f"{'='*60}")

        seed_dirs = find_seed_dirs(multi_seed_dir, prefix)
        if not seed_dirs:
            print(f"  No valid seed dirs found for {arch_name}")
            continue

        arch_results = []
        for sd in seed_dirs:
            print(f"  Seed dir: {sd.name}")
            result = analyze_seed(sd, n_epochs, max_samples)
            if result is not None:
                arch_results.append(result)

        all_results[arch_name] = arch_results
        print(f"\n  {arch_name} summary ({len(arch_results)} seeds):")
        for r in arch_results:
            print(f"    {Path(r['seed_dir']).name}: "
                  f"adaptive={r['adaptive_reduction_pct']:.1f}%  "
                  f"fixed={r['fixed_reduction_pct']:.1f}%  "
                  f"dist_change={r['dist_change_pct']:+.1f}%")

    # Save JSON
    def serializable(obj):
        if isinstance(obj, dict):
            return {k: serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [serializable(x) for x in obj]
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return obj

    with open(output_dir / 'fixed_epsilon_classification.json', 'w') as f:
        json.dump(serializable(all_results), f, indent=2)

    # Generate plots and print summary table
    make_summary_figure(all_results, output_dir)
    print_summary_table(all_results)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def make_summary_figure(all_results: dict, output_dir: Path):
    """Per-architecture subplots: adaptive vs fixed simplices over epochs."""
    archs = [a for a in ARCH_PREFIXES if a in all_results and all_results[a]]
    if not archs:
        return

    fig, axes = plt.subplots(1, len(archs), figsize=(6 * len(archs), 5), sharey=False)
    if len(archs) == 1:
        axes = [axes]

    colors_adaptive = '#E74C3C'
    colors_fixed = '#3498DB'

    for ax, arch in zip(axes, archs):
        seed_results = all_results[arch]
        for i, r in enumerate(seed_results):
            epochs = r['epochs']
            adaptive = [rec['adaptive_simplices'] for rec in r['records']]
            fixed = [rec['fixed_simplices'] for rec in r['records']]
            label_a = 'Adaptive ε' if i == 0 else None
            label_f = 'Fixed ε (epoch 0)' if i == 0 else None
            ax.semilogy(epochs, adaptive, color=colors_adaptive, alpha=0.4,
                        linewidth=1.5, label=label_a)
            ax.semilogy(epochs, fixed, color=colors_fixed, alpha=0.4,
                        linewidth=1.5, label=label_f)

        # Mean across seeds
        min_len = min(len(r['records']) for r in seed_results)
        epochs_common = [r['records'][j]['epoch'] for j in range(min_len)]
        mean_adaptive = np.mean([[r['records'][j]['adaptive_simplices']
                                  for j in range(min_len)] for r in seed_results], axis=0)
        mean_fixed = np.mean([[r['records'][j]['fixed_simplices']
                                for j in range(min_len)] for r in seed_results], axis=0)
        ax.semilogy(epochs_common, mean_adaptive, color=colors_adaptive,
                    linewidth=2.5, linestyle='--', label='Mean adaptive')
        ax.semilogy(epochs_common, mean_fixed, color=colors_fixed,
                    linewidth=2.5, linestyle='--', label='Mean fixed')

        # Annotate reduction
        mean_fixed_red = np.mean([r['fixed_reduction_pct'] for r in seed_results])
        mean_dist_chg = np.mean([r['dist_change_pct'] for r in seed_results])
        ax.set_title(f'{arch}\nFixed ε: {mean_fixed_red:.0f}% reduction\n'
                     f'(dist change: {mean_dist_chg:+.0f}%)', fontsize=11)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Simplices')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Fixed-ε Validation: Classification Experiment\n'
                 'Adaptive vs Fixed Epsilon Topology', fontsize=13, fontweight='bold')
    plt.tight_layout()
    outpath = output_dir / 'fixed_epsilon_classification_comparison.png'
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved: {outpath}")


def print_summary_table(all_results: dict):
    """Print a summary table compatible with what we'll put in the paper."""
    print(f"\n{'='*72}")
    print(f"{'Architecture':<14} {'Adaptive Δ':>12} {'Fixed Δ':>10} {'Distance Δ':>12}")
    print(f"{'='*72}")
    for arch_name in ARCH_PREFIXES:
        if arch_name not in all_results or not all_results[arch_name]:
            continue
        results = all_results[arch_name]
        ad = np.mean([r['adaptive_reduction_pct'] for r in results])
        fd = np.mean([r['fixed_reduction_pct'] for r in results])
        dc = np.mean([r['dist_change_pct'] for r in results])
        n = len(results)
        print(f"{arch_name:<14} {ad:>11.1f}%  {fd:>9.1f}%  {dc:>+11.1f}%  (n={n})")
    print(f"{'='*72}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--multi_seed_dir', type=str,
                        default='runs/exp1/multi_seed_classification')
    parser.add_argument('--output_dir', type=str,
                        default='runs/exp1/topology_classification_analysis/fixed_epsilon')
    parser.add_argument('--max_samples', type=int, default=200,
                        help='Max samples per epoch for VR computation')
    parser.add_argument('--n_epochs', type=int, default=20,
                        help='Number of evenly-spaced epochs to sample')
    args = parser.parse_args()

    run_analysis(
        multi_seed_dir=Path(args.multi_seed_dir),
        output_dir=Path(args.output_dir),
        n_epochs=args.n_epochs,
        max_samples=args.max_samples,
    )
