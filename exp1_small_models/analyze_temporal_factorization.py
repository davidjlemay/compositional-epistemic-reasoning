#!/usr/bin/env python3
"""
Temporal Factorization Analysis - Track factorization evolution through training

This script measures product factorization (depth→n_agents R²) at multiple epochs
to test whether factorization emerges during training or exists from initialization.

Key question: Is factorization strongest at initialization (architectural inductive bias)
or does it emerge during learning?
"""

import torch
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import argparse
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


def load_states(states_path):
    """Load hidden states from checkpoint."""
    if not states_path.exists():
        return None

    data = torch.load(states_path, map_location='cpu')

    if 'encoder_states' in data:
        states = data['encoder_states']
    elif 'hidden_states' in data:
        states = data['hidden_states']
    elif 'encoder_hidden' in data:
        states = data['encoder_hidden']
    else:
        return None

    if isinstance(states, torch.Tensor):
        states = states.cpu().numpy()

    if len(states.shape) == 3:
        batch, seq_len, hidden_dim = states.shape
        states = states.reshape(-1, hidden_dim)

    return states


def load_metadata(data_path):
    """Load metadata and discover factor structure."""
    with open(data_path, 'r') as f:
        data = json.load(f)

    features = {
        'n_agents': [],
        'hypothesis_depth': [],
        'interaction_score': [],
        'constraint': [],
    }

    for item in data:
        features['n_agents'].append(item['n_agents'])
        features['hypothesis_depth'].append(item['hypothesis_depth'])
        features['interaction_score'].append(item['interaction_score'])
        label_field = item.get('constraint', item.get('label', 'non-entailment'))
        features['constraint'].append(1 if label_field == 'entailment' else 0)

    return features


def compute_fiber_factorization(states, features, depth_value):
    """
    Compute factorization R² for predicting n_agents within depth fiber.

    This is the key metric: fixing depth, how well can we predict n_agents?
    High R² indicates strong product factorization X ≈ X_agents × X_depth.
    """
    feature_values = np.array(features['hypothesis_depth'])
    n_samples_per_problem = states.shape[0] // len(feature_values)
    feature_values_expanded = np.repeat(feature_values, n_samples_per_problem)

    min_len = min(len(states), len(feature_values_expanded))
    states = states[:min_len]
    feature_values_expanded = feature_values_expanded[:min_len]

    fiber_mask = (feature_values_expanded == depth_value)

    if fiber_mask.sum() < 10:
        return None

    fiber_states = states[fiber_mask]

    # Get n_agents values in this fiber
    target_values = np.array(features['n_agents'])
    target_values_expanded = np.repeat(target_values, n_samples_per_problem)
    target_values_expanded = target_values_expanded[:min_len]
    target_in_fiber = target_values_expanded[fiber_mask]

    # Check variation
    unique_vals = np.unique(target_in_fiber)
    if len(unique_vals) <= 1:
        return None

    # PCA + Ridge regression
    n_components = min(10, fiber_states.shape[0] - 1, fiber_states.shape[1])
    if n_components < 1:
        return None

    try:
        pca = PCA(n_components=n_components)
        fiber_pca = pca.fit_transform(fiber_states)

        reg = Ridge(alpha=1.0)
        reg.fit(fiber_pca, target_in_fiber)
        y_pred = reg.predict(fiber_pca)
        r2 = r2_score(target_in_fiber, y_pred)

        return float(r2)
    except:
        return None


def analyze_epoch(arch_dir, data_dir, epoch, split, arch_name, data_file=None):
    """
    Analyze factorization at a single epoch.

    Returns dict with R² scores for each testable fiber.
    """
    # Load states
    states_path = Path(arch_dir) / 'captured_states' / f'states_epoch_{epoch}_{split}.pt'

    if not states_path.exists():
        return None

    states = load_states(states_path)
    if states is None:
        return None

    # Load metadata
    filename = data_file if data_file else f'{split}_generation.json'
    data_path = Path(data_dir) / filename
    features = load_metadata(data_path)

    # Discover available depth values
    depth_values = sorted(set(features['hypothesis_depth']))

    # Compute R² for each depth fiber
    fiber_r2 = {}
    for depth_val in depth_values:
        r2 = compute_fiber_factorization(states, features, depth_val)
        if r2 is not None:
            fiber_r2[f'depth_{depth_val}'] = r2

    if len(fiber_r2) == 0:
        return None

    return {
        'epoch': epoch,
        'split': split,
        'architecture': arch_name,
        'fiber_r2': fiber_r2,
        'mean_r2': float(np.mean(list(fiber_r2.values()))),
        'std_r2': float(np.std(list(fiber_r2.values()))),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Temporal factorization analysis - track R² evolution'
    )
    parser.add_argument('--arch_dirs', type=str, nargs='+', required=True,
                       help='Architecture result directories')
    parser.add_argument('--arch_names', type=str, nargs='+', required=True,
                       help='Architecture names (must match arch_dirs order)')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Dataset directory')
    parser.add_argument('--data_file', type=str, default=None,
                       help='Dataset filename within data_dir (default: {split}_generation.json)')
    parser.add_argument('--epochs', type=int, nargs='+',
                       default=[0, 500, 1000, 1500, 2000, 3000, 5000, 7000, 10000],
                       help='Epochs to analyze')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val'],
                       help='Which split to analyze')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory')

    args = parser.parse_args()

    if len(args.arch_dirs) != len(args.arch_names):
        raise ValueError("Number of arch_dirs must match arch_names")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("="*80)
    print("TEMPORAL FACTORIZATION ANALYSIS")
    print("="*80)
    print(f"Dataset: {args.data_dir}")
    print(f"Split: {args.split}")
    print(f"Epochs: {args.epochs}")
    print(f"Architectures: {args.arch_names}")
    print()

    # Analyze each architecture across all epochs
    all_results = {
        'dataset': args.data_dir,
        'split': args.split,
        'epochs': args.epochs,
        'architectures': {},
    }

    for arch_dir, arch_name in zip(args.arch_dirs, args.arch_names):
        print(f"\n{'='*80}")
        print(f"ANALYZING {arch_name}")
        print(f"{'='*80}")

        arch_results = {
            'directory': arch_dir,
            'epochs': [],
            'mean_r2': [],
            'std_r2': [],
            'fiber_details': [],
        }

        for epoch in args.epochs:
            print(f"\nEpoch {epoch}...", end=" ")

            result = analyze_epoch(arch_dir, args.data_dir, epoch, args.split, arch_name,
                                   data_file=args.data_file)

            if result is not None:
                arch_results['epochs'].append(epoch)
                arch_results['mean_r2'].append(result['mean_r2'])
                arch_results['std_r2'].append(result['std_r2'])
                arch_results['fiber_details'].append(result)

                print(f"✓ Mean R²={result['mean_r2']:.3f} ± {result['std_r2']:.3f}")
            else:
                print("✗ No data")

        all_results['architectures'][arch_name] = arch_results

        # Print summary for this architecture
        if len(arch_results['epochs']) > 0:
            print(f"\n{arch_name} Summary:")
            print(f"  Epochs analyzed: {len(arch_results['epochs'])}")
            print(f"  R² range: {min(arch_results['mean_r2']):.3f} - {max(arch_results['mean_r2']):.3f}")

            # Check if factorization is strongest at init
            if arch_results['epochs'][0] == 0:
                init_r2 = arch_results['mean_r2'][0]
                max_r2 = max(arch_results['mean_r2'])
                if init_r2 == max_r2:
                    print(f"  ⚠️  FACTORIZATION STRONGEST AT INITIALIZATION! R²={init_r2:.3f}")
                else:
                    print(f"  Factorization increases during training (init={init_r2:.3f}, max={max_r2:.3f})")

    # Save results
    results_path = output_dir / f'temporal_factorization_{args.split}.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {results_path}")
    print(f"{'='*80}")

    # Create visualization
    create_temporal_plot(all_results, output_dir, args.split)


def create_temporal_plot(results, output_dir, split):
    """Create publication-quality temporal evolution plot."""
    print("\nGenerating temporal evolution plot...")

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'GNN': 'steelblue', 'Transformer': 'coral', 'Mamba': 'mediumseagreen'}
    markers = {'GNN': 's', 'Transformer': 'o', 'Mamba': '^'}

    for arch_name, arch_data in results['architectures'].items():
        if len(arch_data['epochs']) == 0:
            continue

        epochs = arch_data['epochs']
        mean_r2 = arch_data['mean_r2']
        std_r2 = arch_data['std_r2']

        color = colors.get(arch_name, 'gray')
        marker = markers.get(arch_name, 'o')

        # Plot with error bands
        ax.plot(epochs, mean_r2, marker=marker, label=arch_name,
               linewidth=2, markersize=8, color=color)
        ax.fill_between(epochs,
                        np.array(mean_r2) - np.array(std_r2),
                        np.array(mean_r2) + np.array(std_r2),
                        alpha=0.2, color=color)

    # Formatting
    ax.set_xlabel('Training Epoch', fontsize=13)
    ax.set_ylabel('Product Factorization R²\n(depth → n_agents)', fontsize=13)
    ax.set_title(f'Factorization Evolution Through Training ({split.upper()} set)',
                fontsize=14, weight='bold')
    ax.legend(loc='best', frameon=True, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.0])

    # Add grokking region annotation (around 2000-4000 epochs)
    ax.axvspan(2000, 4000, alpha=0.1, color='gray', label='Typical grokking region')

    # Add annotation if factorization is strongest at init
    arch_with_init = []
    for arch_name, arch_data in results['architectures'].items():
        if len(arch_data['epochs']) > 0 and arch_data['epochs'][0] == 0:
            init_r2 = arch_data['mean_r2'][0]
            max_r2 = max(arch_data['mean_r2'])
            if init_r2 >= max_r2 * 0.95:  # Within 5% of max
                arch_with_init.append(arch_name)

    if arch_with_init:
        ax.annotate('Factorization strongest\nat initialization!',
                   xy=(0, 0.85), xytext=(1500, 0.90),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2),
                   fontsize=11, color='red', weight='bold',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    plot_path = output_dir / f'temporal_factorization_{split}.pdf'
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    print(f"  Saved to {plot_path}")

    plot_path_png = output_dir / f'temporal_factorization_{split}.png'
    plt.savefig(plot_path_png, bbox_inches='tight', dpi=300)
    print(f"  Saved to {plot_path_png}")

    plt.close()


if __name__ == '__main__':
    main()
