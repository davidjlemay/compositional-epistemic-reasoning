#!/usr/bin/env python3
"""
Topology-generalization analysis for exp_3 multi-seed classification results.

Adapted from exp_5_orthogonal_dataset/analyze_topology_generalization.py.
Removes generation-specific test-set re-evaluation; adds scrambled-label
control overlay for the DFiv null-hypothesis comparison.

Key Questions:
1. Does simplicial complexity reduction accompany successful compositional learning?
2. Does topology differ between compositional (real labels) and scrambled control?
3. Do Betti numbers (H0, H1) collapse during learning for compositional tasks?

Usage:
    python exp1_small_models/analyze_topology_classification.py \
        --results_dir runs/exp1/multi_seed_classification \
        --scrambled_dir runs/exp1/classification_full_scrambled
"""

import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr
import argparse
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def find_available_epochs(results_dir):
    """Find all epochs with captured val states."""
    states_dir = Path(results_dir) / 'captured_states'
    if not states_dir.exists():
        return []
    epochs = []
    for f in sorted(states_dir.glob('states_epoch_*_val.pt')):
        try:
            epochs.append(int(f.stem.split('_')[2]))
        except ValueError:
            continue
    return sorted(epochs)


def load_topology_metrics(results_dir):
    """Load topology_metrics.json produced by run_analysis_auto.sh."""
    topology_file = Path(results_dir) / 'topology_analysis' / 'topology_metrics.json'
    if not topology_file.exists():
        return None
    with open(topology_file) as f:
        return json.load(f)


def load_training_metrics(results_dir):
    """Load per-epoch train/val accuracy from checkpoint_latest.pt."""
    ckpt_path = Path(results_dir) / 'checkpoint_latest.pt'
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    return {
        'train_metrics': ckpt.get('train_metrics', []),
        'val_metrics':   ckpt.get('val_metrics', []),
        'grokking_detected': ckpt.get('grokking_detected', False),
        'grokking_epoch':    ckpt.get('grokking_epoch', None),
        'total_epochs':      ckpt.get('epoch', None),
    }


def extract_topology_at_epochs(topology_metrics, epochs):
    """
    Return a dict mapping epoch -> topology dict for the requested epochs.
    Falls back gracefully if an epoch is missing.
    """
    result = {}
    if not topology_metrics:
        return result

    enc = topology_metrics.get('encoder_metrics', {})
    dec = topology_metrics.get('decoder_metrics', {})

    for epoch in epochs:
        key = str(epoch)
        enc_ep = enc.get(key, {})
        dec_ep = dec.get(key, {})

        if not enc_ep and not dec_ep:
            continue

        enc_simplices = enc_ep.get('total_simplices', 0)
        dec_simplices = dec_ep.get('total_simplices', 0)
        enc_h0 = enc_ep.get('h0_features', enc_ep.get('h0', 0))
        enc_h1 = enc_ep.get('h1_features', enc_ep.get('h1', 0))
        dec_h0 = dec_ep.get('h0_features', dec_ep.get('h0', 0))
        dec_h1 = dec_ep.get('h1_features', dec_ep.get('h1', 0))

        result[epoch] = {
            'encoder_simplices': enc_simplices,
            'decoder_simplices': dec_simplices,
            'total_simplices':   enc_simplices + dec_simplices,
            'encoder_h0': enc_h0,
            'encoder_h1': enc_h1,
            'decoder_h0': dec_h0,
            'decoder_h1': dec_h1,
            'avg_h0': (enc_h0 + dec_h0) / 2,
            'avg_h1': (enc_h1 + dec_h1) / 2,
        }
    return result


# ---------------------------------------------------------------------------
# Single-run analysis
# ---------------------------------------------------------------------------

def analyze_single_run(results_dir, architecture_name):
    """
    Combine topology + training metrics for one seed directory.
    Returns a list of per-epoch data-point dicts, or None if data is missing.
    """
    results_dir = Path(results_dir)

    training = load_training_metrics(results_dir)
    if not training:
        print(f"  ✗ No checkpoint: {results_dir.name}")
        return None

    topology_metrics = load_topology_metrics(results_dir)
    if not topology_metrics:
        print(f"  ✗ No topology_metrics.json: {results_dir.name}")
        return None

    available_epochs = find_available_epochs(results_dir)
    if not available_epochs:
        print(f"  ✗ No captured states: {results_dir.name}")
        return None

    topology_by_epoch = extract_topology_at_epochs(topology_metrics, available_epochs)

    train_m = training['train_metrics']
    val_m   = training['val_metrics']

    # Infer eval_interval
    total_epochs = training['total_epochs'] or len(train_m)
    n_val = len(val_m)
    eval_interval = max(1, total_epochs // max(n_val, 1))

    data_points = []
    for epoch in sorted(topology_by_epoch.keys()):
        topo = topology_by_epoch[epoch]

        val_idx   = min(epoch // eval_interval, n_val - 1) if eval_interval > 0 else 0
        train_idx = min(epoch, len(train_m) - 1)

        train_acc = train_m[train_idx]['accuracy'] if train_m else 0.0
        val_acc   = val_m[val_idx]['accuracy']     if val_m   else 0.0

        data_points.append({
            'epoch':               epoch,
            'architecture':        architecture_name,
            'run_dir':             str(results_dir),
            'train_acc':           train_acc,
            'val_acc':             val_acc,
            'generalization_gap':  train_acc - val_acc,
            'grokking_detected':   training['grokking_detected'],
            'grokking_epoch':      training['grokking_epoch'],
            **topo,
        })

    print(f"  ✓ {results_dir.name}: {len(data_points)} epochs, "
          f"peak val acc={max((d['val_acc'] for d in data_points), default=0):.3f}")
    return data_points


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------

def aggregate_multi_seed_results(base_dir, architecture_pattern):
    """Aggregate results for all seeds matching architecture_pattern."""
    base_dir = Path(base_dir)
    result_dirs = sorted(base_dir.glob(architecture_pattern))

    arch_name = architecture_pattern.split('_')[0].upper()
    print(f"\n  Architecture: {arch_name} (pattern: {architecture_pattern})")
    print(f"  Found {len(result_dirs)} directories")

    all_data = []
    for rd in result_dirs:
        if 'FAILED' in rd.name:
            print(f"  ⚠  Skipping FAILED: {rd.name}")
            continue
        data = analyze_single_run(rd, arch_name)
        if data:
            all_data.extend(data)

    return all_data


def load_scrambled_runs(scrambled_base_dir):
    """
    Load all scrambled-label control runs from scrambled_base_dir,
    grouped by architecture. Returns a flat list with 'architecture' field
    set to the matched architecture name (GNN, TRANSFORMER, MAMBA).
    """
    base = Path(scrambled_base_dir)
    patterns = [
        ('gnn_scramble_seed*',         'GNN'),
        ('transformer_scramble_seed*', 'TRANSFORMER'),
        ('mamba_scramble_seed*',        'MAMBA'),
    ]
    all_scrambled = []
    for pattern, arch_name in patterns:
        dirs = sorted(base.glob(pattern))
        print(f"\n  Scrambled {arch_name}: found {len(dirs)} seeds")
        for rd in dirs:
            data = analyze_single_run(rd, arch_name)
            if data:
                for d in data:
                    d['condition'] = 'scrambled'
                all_scrambled.extend(data)
    return all_scrambled


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_correlations(data_points):
    """Pearson/Spearman between topology metrics and generalization gap."""
    if len(data_points) < 3:
        return {}

    topology_keys = ['total_simplices', 'encoder_simplices', 'decoder_simplices',
                     'avg_h0', 'avg_h1', 'encoder_h1', 'decoder_h1']
    performance_keys = ['train_acc', 'val_acc', 'generalization_gap']

    correlations = {}
    for tk in topology_keys:
        topo_vals = [d[tk] for d in data_points if tk in d]
        if not topo_vals:
            continue
        for pk in performance_keys:
            perf_vals = [d[pk] for d in data_points if tk in d]
            if len(topo_vals) < 3:
                continue
            try:
                r, p = pearsonr(topo_vals, perf_vals)
                rho, _ = spearmanr(topo_vals, perf_vals)
                correlations[f"{tk}_vs_{pk}"] = {'pearson_r': r, 'p_value': p, 'spearman_rho': rho}
            except Exception:
                pass
    return correlations


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _epoch_stats(data_points, key):
    """Return (sorted_epochs, means, sems) for a given metric key."""
    groups = defaultdict(list)
    for d in data_points:
        groups[d['epoch']].append(d[key])
    epochs = sorted(groups)
    means = [np.mean(groups[e]) for e in epochs]
    sems  = [np.std(groups[e]) / np.sqrt(len(groups[e])) for e in epochs]
    return epochs, means, sems


def visualize_topology_evolution(all_runs, scrambled_runs, output_dir):
    """
    For each architecture, plot:
      Left:  train/val accuracy + total simplices (log) over epochs,
             with matched scrambled-label overlay (mean ± SEM).
      Right: simplices vs generalization gap (scatter, coloured by epoch).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    by_arch = defaultdict(list)
    for d in all_runs:
        by_arch[d['architecture']].append(d)

    sc_by_arch = defaultdict(list)
    for d in scrambled_runs:
        sc_by_arch[d['architecture']].append(d)

    architectures = sorted(by_arch.keys())
    n_arch = len(architectures)
    if n_arch == 0:
        print("No data to visualise.")
        return

    fig, axes = plt.subplots(n_arch, 2, figsize=(14, 4 * n_arch))
    if n_arch == 1:
        axes = [axes]

    for row, arch in enumerate(architectures):
        data = by_arch[arch]

        epochs, mean_train, _ = _epoch_stats(data, 'train_acc')
        _,      mean_val,   _ = _epoch_stats(data, 'val_acc')
        _,      mean_simp, sem_simp = _epoch_stats(data, 'total_simplices')

        grokking_epochs = [d['grokking_epoch'] for d in data
                           if d['grokking_detected'] and d['grokking_epoch']]

        # --- LEFT: accuracy + topology ---
        ax1 = axes[row][0]
        ax2 = ax1.twinx()

        ax1.plot(epochs, mean_train, 'b-', linewidth=2, label='Train acc (compositional)')
        ax1.plot(epochs, mean_val,   'g-', linewidth=2, label='Val acc (compositional)')
        ax1.set_ylim(0, 1.05)
        ax1.set_ylabel('Accuracy', color='black')
        ax1.set_xlabel('Epoch')

        simp_arr = np.maximum(np.array(mean_simp, dtype=float), 1)
        sem_arr  = np.array(sem_simp, dtype=float)
        ax2.semilogy(epochs, simp_arr, 'r-', linewidth=2, alpha=0.9,
                     label='Simplices (compositional, mean)')
        ax2.fill_between(epochs,
                         np.maximum(simp_arr - sem_arr, 1),
                         simp_arr + sem_arr,
                         alpha=0.15, color='red')
        ax2.set_ylabel('Total simplices (log)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # Matched scrambled overlay (mean ± SEM)
        sc_data = sc_by_arch.get(arch, [])
        if sc_data:
            sc_ep, sc_mean, sc_sem = _epoch_stats(sc_data, 'total_simplices')
            sc_arr = np.maximum(np.array(sc_mean, dtype=float), 1)
            sc_sem_arr = np.array(sc_sem, dtype=float)
            ax2.semilogy(sc_ep, sc_arr, color='grey', linestyle='--', linewidth=2,
                         alpha=0.8, label='Simplices (scrambled, mean)')
            ax2.fill_between(sc_ep,
                             np.maximum(sc_arr - sc_sem_arr, 1),
                             sc_arr + sc_sem_arr,
                             alpha=0.10, color='grey')

        if grokking_epochs:
            grok_epoch = max(set(grokking_epochs), key=grokking_epochs.count)
            ax1.axvline(grok_epoch, color='orange', linestyle='--', alpha=0.7,
                        label=f'Rapid learning (ep {grok_epoch})')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=8)
        ax1.set_title(f'{arch}: Accuracy + Topology Evolution')

        # --- RIGHT: simplices vs generalisation gap ---
        ax3 = axes[row][1]
        epoch_groups = defaultdict(list)
        for d in data:
            epoch_groups[d['epoch']].append(d)
        gap_vals  = [np.mean([x['generalization_gap'] for x in epoch_groups[e]]) for e in epochs]
        epoch_arr = np.array(epochs, dtype=float)
        norm_epochs = (epoch_arr - epoch_arr.min()) / max(epoch_arr.max() - epoch_arr.min(), 1)

        sc_plot = ax3.scatter(mean_simp, gap_vals, c=norm_epochs, cmap='viridis',
                              s=50, alpha=0.8, zorder=3)
        plt.colorbar(sc_plot, ax=ax3, label='Epoch (normalised)')

        if sc_data:
            sc_epoch_groups = defaultdict(list)
            for d in sc_data:
                sc_epoch_groups[d['epoch']].append(d)
            sc_ep2 = sorted(sc_epoch_groups)
            sc_simp2 = [np.mean([x['total_simplices'] for x in sc_epoch_groups[e]]) for e in sc_ep2]
            sc_gaps2 = [np.mean([x['generalization_gap'] for x in sc_epoch_groups[e]]) for e in sc_ep2]
            ax3.scatter(sc_simp2, sc_gaps2, c='grey', s=40, alpha=0.5, marker='x',
                        label='Scrambled', zorder=2)
            ax3.legend(fontsize=8)

        if len(mean_simp) > 2:
            try:
                r, p = pearsonr(mean_simp, gap_vals)
                ax3.set_title(f'{arch}: Simplices vs Gen. Gap  (r={r:.2f}, p={p:.3f})')
            except Exception:
                ax3.set_title(f'{arch}: Simplices vs Gen. Gap')
        else:
            ax3.set_title(f'{arch}: Simplices vs Gen. Gap')

        ax3.set_xlabel('Total simplices')
        ax3.set_ylabel('Generalisation gap (train − val)')
        ax3.set_xscale('log')

    plt.tight_layout()
    out_path = output_dir / 'topology_classification_evolution.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_path}")


def visualize_scrambled_comparison(all_runs, scrambled_runs, output_dir):
    """
    Summary figure comparing simplex reduction and H1 collapse for each
    architecture (compositional vs matched scrambled, mean ± SEM).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if not all_runs or not scrambled_runs:
        print("Skipping scrambled comparison (missing data).")
        return

    by_arch    = defaultdict(list)
    sc_by_arch = defaultdict(list)
    for d in all_runs:
        by_arch[d['architecture']].append(d)
    for d in scrambled_runs:
        sc_by_arch[d['architecture']].append(d)

    colour_map = {'GNN': '#1f77b4', 'TRANSFORMER': '#2ca02c', 'MAMBA': '#9467bd'}

    def norm_series(data_points):
        """Return (epochs, norm_mean, norm_sem) relative to epoch-0 mean."""
        ep, means, sems = _epoch_stats(data_points, 'total_simplices')
        init = means[0] if means[0] > 0 else 1
        return ep, [v / init for v in means], [s / init for s in sems]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Panel 1: normalised simplex reduction ---
    ax = axes[0]
    for arch in sorted(by_arch):
        colour = colour_map.get(arch, 'steelblue')
        ep, nm, ns = norm_series(by_arch[arch])
        ax.plot(ep, nm, color=colour, linewidth=2, label=f'{arch} (compositional)')
        ax.fill_between(ep,
                        np.maximum(np.array(nm) - np.array(ns), 1e-4),
                        np.array(nm) + np.array(ns),
                        alpha=0.15, color=colour)
        sc_data = sc_by_arch.get(arch, [])
        if sc_data:
            sc_ep, sc_nm, sc_ns = norm_series(sc_data)
            ax.plot(sc_ep, sc_nm, color=colour, linestyle='--', linewidth=1.5,
                    alpha=0.7, label=f'{arch} (scrambled)')
            ax.fill_between(sc_ep,
                            np.maximum(np.array(sc_nm) - np.array(sc_ns), 1e-4),
                            np.array(sc_nm) + np.array(sc_ns),
                            alpha=0.08, color=colour)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Simplices / initial (mean ± SEM)')
    ax.set_title('Topological Simplification:\nCompositional vs Scrambled')
    ax.legend(fontsize=8)
    ax.set_yscale('log')

    # --- Panel 2: H1 evolution ---
    ax2 = axes[1]
    for arch in sorted(by_arch):
        colour = colour_map.get(arch, 'steelblue')
        ep, h1_mean, h1_sem = _epoch_stats(by_arch[arch], 'avg_h1')
        ax2.plot(ep, h1_mean, color=colour, linewidth=2, label=f'{arch} (compositional)')
        ax2.fill_between(ep,
                         np.maximum(np.array(h1_mean) - np.array(h1_sem), 0),
                         np.array(h1_mean) + np.array(h1_sem),
                         alpha=0.15, color=colour)
        sc_data = sc_by_arch.get(arch, [])
        if sc_data:
            sc_ep, sc_h1, sc_h1s = _epoch_stats(sc_data, 'avg_h1')
            ax2.plot(sc_ep, sc_h1, color=colour, linestyle='--', linewidth=1.5,
                     alpha=0.7, label=f'{arch} (scrambled)')
            ax2.fill_between(sc_ep,
                             np.maximum(np.array(sc_h1) - np.array(sc_h1s), 0),
                             np.array(sc_h1) + np.array(sc_h1s),
                             alpha=0.08, color=colour)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('H1 (loops/holes) — mean encoder+decoder')
    ax2.set_title('H1 Collapse:\nCompositional vs Scrambled')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out_path = output_dir / 'topology_scrambled_comparison.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def create_correlation_heatmap(all_runs, output_dir):
    """Correlation heatmap between topology and performance metrics."""
    output_dir = Path(output_dir)
    by_arch = defaultdict(list)
    for d in all_runs:
        by_arch[d['architecture']].append(d)

    n_arch = len(by_arch)
    if n_arch == 0:
        return

    fig, axes = plt.subplots(1, n_arch, figsize=(6 * n_arch, 5))
    if n_arch == 1:
        axes = [axes]

    metric_keys = ['total_simplices', 'encoder_simplices', 'encoder_h0', 'encoder_h1',
                   'decoder_h0', 'decoder_h1', 'train_acc', 'val_acc', 'generalization_gap']

    for ax, (arch, data) in zip(axes, sorted(by_arch.items())):
        mat = {}
        for k in metric_keys:
            vals = [d.get(k, 0) for d in data]
            if vals:
                mat[k] = vals

        if len(mat) < 2:
            continue

        keys = list(mat.keys())
        corr = np.zeros((len(keys), len(keys)))
        for i, ki in enumerate(keys):
            for j, kj in enumerate(keys):
                try:
                    r, _ = pearsonr(mat[ki], mat[kj])
                    corr[i, j] = r
                except Exception:
                    corr[i, j] = 0.0

        sns.heatmap(corr, xticklabels=keys, yticklabels=keys,
                    annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                    vmin=-1, vmax=1, ax=ax, annot_kws={'size': 7})
        ax.set_title(f'{arch}: Topology–Performance Correlations')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)

    plt.tight_layout()
    out_path = output_dir / 'correlation_heatmap.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Topology-generalization analysis for classification task (exp_3)')

    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing multi-seed classification results')
    parser.add_argument('--scrambled_dir', type=str, default=None,
                        help='Directory for scrambled-label control run (optional)')
    parser.add_argument('--output_dir', type=str,
                        default='runs/exp1/topology_classification_analysis',
                        help='Output directory for plots and JSON')
    parser.add_argument('--architectures', type=str, nargs='+',
                        default=['gnn_seed*', 'transformer_seed*', 'mamba_seed*'],
                        help='Glob patterns for architecture result directories')

    args = parser.parse_args()

    print("=" * 70)
    print("TOPOLOGY-GENERALIZATION ANALYSIS: Classification Task")
    print("=" * 70)
    print(f"Results dir:  {args.results_dir}")
    print(f"Scrambled:    {args.scrambled_dir or 'not provided'}")
    print(f"Output dir:   {args.output_dir}")
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # --- Compositional multi-seed runs ---
    all_runs = []
    print("Loading compositional classification runs...")
    for pattern in args.architectures:
        runs = aggregate_multi_seed_results(args.results_dir, pattern)
        all_runs.extend(runs)

    if not all_runs:
        print("\n⚠  No valid runs found. Check --results_dir and --architectures.")
        return

    print(f"\n✓ Loaded {len(all_runs)} data points from {len(args.architectures)} architecture(s)")

    # --- Scrambled control ---
    scrambled_runs = []
    if args.scrambled_dir:
        print("\nLoading scrambled-label controls (all architectures)...")
        scrambled_runs = load_scrambled_runs(args.scrambled_dir)
        print(f"  ✓ Loaded {len(scrambled_runs)} scrambled data points total")

    # --- Correlations ---
    print("\nComputing correlations...")
    correlations = {}
    by_arch = defaultdict(list)
    for d in all_runs:
        by_arch[d['architecture']].append(d)

    for arch, data in by_arch.items():
        correlations[arch] = compute_correlations(data)
        simp_vals = [d['total_simplices'] for d in data]
        gap_vals  = [d['generalization_gap'] for d in data]
        if len(simp_vals) >= 3:
            try:
                r, p = pearsonr(simp_vals, gap_vals)
                print(f"  {arch}: simplices vs gen-gap  r={r:.3f}, p={p:.4f}")
            except Exception:
                pass

    sc_by_arch = defaultdict(list)
    for d in scrambled_runs:
        sc_by_arch[d['architecture']].append(d)

    for arch, data in sc_by_arch.items():
        simp_vals = [d['total_simplices'] for d in data]
        gap_vals  = [d['generalization_gap'] for d in data]
        if len(simp_vals) >= 3:
            try:
                r, p = pearsonr(simp_vals, gap_vals)
                print(f"  {arch} (scrambled): simplices vs gen-gap  r={r:.3f}, p={p:.4f}")
            except Exception:
                pass

    # --- Save correlation results ---
    corr_path = output_dir / 'correlation_results.json'
    with open(corr_path, 'w') as f:
        json.dump(correlations, f, indent=2, default=float)
    print(f"\nSaved correlations: {corr_path}")

    # --- Summary statistics with central tendencies ---
    def _reduction_stats(data):
        """Return mean and SEM of per-seed simplex reduction.

        Each seed (run_dir) is treated independently: its own epoch-0 value
        is the baseline and its own final epoch is the endpoint.
        """
        by_seed = defaultdict(list)
        for d in data:
            by_seed[d['run_dir']].append(d)

        reductions = []
        all_inits, all_finals = [], []
        for seed_data in by_seed.values():
            if not seed_data:
                continue
            seed_ep_min = min(d['epoch'] for d in seed_data)
            seed_ep_max = max(d['epoch'] for d in seed_data)
            init_vals  = [d['total_simplices'] for d in seed_data if d['epoch'] == seed_ep_min]
            final_vals = [d['total_simplices'] for d in seed_data if d['epoch'] == seed_ep_max]
            if init_vals and final_vals:
                i = np.mean(init_vals)
                f = np.mean(final_vals)
                all_inits.append(i)
                all_finals.append(f)
                if i > 0:
                    reductions.append((i - f) / i * 100)

        init_m  = np.mean(all_inits)  if all_inits  else 0
        final_m = np.mean(all_finals) if all_finals else 0
        if not reductions:
            return None, None, init_m, final_m
        mean_r = np.mean(reductions)
        sem_r  = np.std(reductions, ddof=1) / np.sqrt(len(reductions)) if len(reductions) > 1 else 0.0
        return mean_r, sem_r, init_m, final_m

    print("\n" + "=" * 70)
    print("CENTRAL TENDENCIES: Simplex Reduction (mean ± SEM across seeds)")
    print("=" * 70)
    print(f"{'Architecture':<22} {'Condition':<14} {'Init':<10} {'Final':<10} {'Reduction'}")
    print("-" * 70)

    summary = {}
    for arch, data in sorted(by_arch.items()):
        mean_r, sem_r, init_m, final_m = _reduction_stats(data)
        peak_val = max(d['val_acc'] for d in data)
        grokking = [d['grokking_detected'] for d in data]
        summary[arch] = {
            'n_data_points':           len(data),
            'peak_val_acc':            peak_val,
            'grokking_pct':            sum(grokking) / len(grokking) if grokking else 0,
            'mean_init_simplices':     float(init_m),
            'mean_final_simplices':    float(final_m),
            'mean_reduction_pct':      float(mean_r) if mean_r is not None else None,
            'sem_reduction_pct':       float(sem_r)  if sem_r  is not None else None,
        }
        red_str = f"{mean_r:.1f}% ± {sem_r:.1f}%" if mean_r is not None else "n/a"
        print(f"  {arch:<20} {'compositional':<14} {init_m:<10.0f} {final_m:<10.0f} {red_str}")

    for arch, data in sorted(sc_by_arch.items()):
        mean_r, sem_r, init_m, final_m = _reduction_stats(data)
        peak_val = max(d['val_acc'] for d in data)
        key = f"{arch}_scrambled"
        summary[key] = {
            'n_data_points':        len(data),
            'peak_val_acc':         peak_val,
            'mean_init_simplices':  float(init_m),
            'mean_final_simplices': float(final_m),
            'mean_reduction_pct':   float(mean_r) if mean_r is not None else None,
            'sem_reduction_pct':    float(sem_r)  if sem_r  is not None else None,
        }
        red_str = f"{mean_r:.1f}% ± {sem_r:.1f}%" if mean_r is not None else "n/a"
        print(f"  {arch:<20} {'scrambled':<14} {init_m:<10.0f} {final_m:<10.0f} {red_str}")

    summary_path = output_dir / 'summary_statistics.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"Saved summary: {summary_path}")

    # --- Visualisations ---
    print("\nGenerating plots...")
    visualize_topology_evolution(all_runs, scrambled_runs, output_dir)
    visualize_scrambled_comparison(all_runs, scrambled_runs, output_dir)
    create_correlation_heatmap(all_runs, output_dir)

    print("\n" + "=" * 70)
    print("Analysis complete.")
    print(f"Outputs in: {output_dir}/")
    print("  topology_classification_evolution.png")
    print("  topology_scrambled_comparison.png")
    print("  correlation_heatmap.png")
    print("  correlation_results.json")
    print("  summary_statistics.json")
    print("=" * 70)


if __name__ == '__main__':
    main()
