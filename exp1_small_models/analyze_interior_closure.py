#!/usr/bin/env python3
"""
Deep analysis of McKinsey-Tarski interior vs closure operators.

□φ (necessity) → interior(φ) - points in the "core" of the region
◇φ (possibility) → closure(φ) - points including the "boundary"

Key geometric signatures:
- Interior points: higher local density, fewer boundary connections
- Closure points: lower local density, more boundary connections, include extremal points
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.neighbors import NearestNeighbors
import json
import sys

import gudhi as gd
from gudhi import SimplexTree

# Add parent directory to path to import data_format_utils
sys.path.insert(0, str(Path(__file__).parent))
from data_format_utils import load_dataset, get_smcdel_formula, parse_modal_operators as parse_ops


def compute_vietoris_rips_complex(points, epsilon, max_dimension=2):
    """Compute VR complex."""
    simplex_tree = SimplexTree()
    for i in range(len(points)):
        simplex_tree.insert([i], filtration=0.0)

    dm = cdist(points, points)
    n_points = len(points)

    for i in range(n_points):
        for j in range(i+1, n_points):
            if dm[i, j] <= epsilon:
                simplex_tree.insert([i, j], filtration=dm[i, j])

    simplex_tree.expansion(max_dimension)
    return simplex_tree


# parse_modal_operators is now imported from data_format_utils as parse_ops
# Keeping a local wrapper for backwards compatibility
def parse_modal_operators(formula: str):
    """Extract modal operator information."""
    return parse_ops(formula)


def load_and_classify(states_dir: Path, data_dir: Path, epoch: int, max_samples: int = 200):
    """Load states and classify by operator type.

    Args:
        states_dir: Directory containing captured states
        data_dir: Directory containing dataset
        epoch: Epoch to load
        max_samples: Maximum number of samples to load (None = all)
    """
    # Load states
    state_file = states_dir / f'states_epoch_{epoch}_train.pt'
    states = torch.load(state_file, map_location='cpu', weights_only=False)

    # Extract encoder representations
    # Handle both GRU and Transformer formats
    encoder_states = states['encoder_states']

    # Detect format based on tensor shape
    if isinstance(encoder_states, torch.Tensor):
        # Transformer format: direct [batch, hidden_dim] tensor
        if encoder_states.dim() == 2:
            representations = encoder_states.numpy()
        # GRU format: [batch, seq_len, hidden_dim] tensor - take last timestep
        elif encoder_states.dim() == 3:
            representations = encoder_states[:, -1, :].numpy()
        else:
            raise ValueError(f"Unexpected encoder_states tensor shape: {encoder_states.shape}")
    elif isinstance(encoder_states, list):
        # GRU format: list of [seq_len, hidden_dim] tensors
        representations = []
        for batch_item in encoder_states:
            if isinstance(batch_item, torch.Tensor):
                if batch_item.dim() == 2:
                    # [seq_len, hidden_dim] - take last timestep
                    final_hidden = batch_item[-1, :]
                elif batch_item.dim() == 1:
                    # Already just [hidden_dim]
                    final_hidden = batch_item
                else:
                    raise ValueError(f"Unexpected batch_item shape: {batch_item.shape}")
                representations.append(final_hidden.numpy())
            else:
                raise TypeError(f"Unexpected batch_item type: {type(batch_item)}")
        representations = np.array(representations)
    else:
        raise TypeError(f"Unexpected encoder_states type: {type(encoder_states)}")

    # Apply sampling if requested
    if max_samples is not None and len(representations) > max_samples:
        indices = np.random.choice(len(representations), max_samples, replace=False)
        indices = np.sort(indices)  # Keep in original order
        representations = representations[indices]
    else:
        indices = None

    # Load dataset using unified loader
    dataset, data_format = load_dataset(data_dir, 'train')

    # Classify
    classified = {
        'necessity': {'indices': [], 'repr': []},
        'possibility': {'indices': [], 'repr': []},
        'mixed': {'indices': [], 'repr': []}
    }

    # Use sampled indices if available
    if indices is not None:
        dataset_indices = indices
    else:
        dataset_indices = range(len(representations))

    for repr_idx, dataset_idx in enumerate(dataset_indices):
        example = dataset[dataset_idx]
        # Use unified formula extraction
        formula = get_smcdel_formula(example, data_format)
        info = parse_modal_operators(formula)
        op_type = info['operator_type']

        if op_type in classified:
            classified[op_type]['indices'].append(repr_idx)
            classified[op_type]['repr'].append(representations[repr_idx])

    for op_type in classified:
        if classified[op_type]['repr']:
            classified[op_type]['repr'] = np.array(classified[op_type]['repr'])

    return representations, classified


def compute_interior_closure_metrics(points: np.ndarray, k: int = 5):
    """
    Compute metrics distinguishing interior vs closure/boundary.

    Interior signature (□):
    - High local density
    - Low variance in neighbor distances
    - Central position relative to neighbors

    Closure signature (◇):
    - Includes boundary points with lower local density
    - Higher variance in neighbor distances
    - More extremal positions
    """
    if len(points) < k:
        return None

    # Fit k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=min(k+1, len(points)), metric='euclidean')
    nbrs.fit(points)
    distances, indices = nbrs.kneighbors(points)

    # Remove self (distance 0)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    # Compute metrics
    metrics = {}

    # Local density (inverse of mean k-NN distance)
    mean_knn_dist = np.mean(distances, axis=1)
    metrics['local_density'] = 1.0 / (mean_knn_dist + 1e-6)

    # Neighbor distance variance (uniformity)
    metrics['neighbor_variance'] = np.var(distances, axis=1)

    # Centrality (how central is point relative to its neighbors)
    centrality = []
    for i in range(len(points)):
        point = points[i]
        neighbors = points[indices[i]]
        neighbor_centroid = np.mean(neighbors, axis=0)
        dist_to_centroid = np.linalg.norm(point - neighbor_centroid)
        centrality.append(dist_to_centroid)
    metrics['centrality'] = np.array(centrality)

    # Boundary score (combination of low density + high variance + high centrality)
    # Normalize each component
    density_norm = (metrics['local_density'] - metrics['local_density'].min()) / \
                   (metrics['local_density'].max() - metrics['local_density'].min() + 1e-6)
    variance_norm = (metrics['neighbor_variance'] - metrics['neighbor_variance'].min()) / \
                    (metrics['neighbor_variance'].max() - metrics['neighbor_variance'].min() + 1e-6)
    centrality_norm = (metrics['centrality'] - metrics['centrality'].min()) / \
                      (metrics['centrality'].max() - metrics['centrality'].min() + 1e-6)

    # Boundary score: high when density is low, variance is high, centrality is high
    metrics['boundary_score'] = (1 - density_norm + variance_norm + centrality_norm) / 3.0

    # Interior score: opposite
    metrics['interior_score'] = 1 - metrics['boundary_score']

    return metrics


def analyze_interior_closure(
    results_dir: Path,
    data_dir: Path,
    pre_epoch: int,
    post_epoch: int,
    max_samples: int = 200,
    output_dir: Path = None
):
    """
    Main analysis comparing interior (□) vs closure (◇) properties.
    """
    results_dir = Path(results_dir)
    data_dir = Path(data_dir)

    if output_dir is None:
        output_dir = results_dir / 'modal_geometry_analysis'
    output_dir.mkdir(exist_ok=True)

    print("="*70)
    print("McKinsey-Tarski Interior/Closure Analysis")
    print("="*70)

    states_dir = results_dir / 'captured_states'

    # Load data
    print(f"\nLoading epoch {pre_epoch} (pre-grokking, max {max_samples} samples)...")
    pre_repr, pre_classified = load_and_classify(states_dir, data_dir, pre_epoch, max_samples=max_samples)

    print(f"Loading epoch {post_epoch} (post-grokking, max {max_samples} samples)...")
    post_repr, post_classified = load_and_classify(states_dir, data_dir, post_epoch, max_samples=max_samples)

    # Compute metrics for each operator type
    print("\nComputing interior/closure metrics...")

    pre_metrics = {}
    post_metrics = {}

    for op_type in ['necessity', 'possibility']:
        print(f"\n{op_type.upper()}:")

        pre_points = pre_classified[op_type]['repr']
        post_points = post_classified[op_type]['repr']

        if len(pre_points) > 0:
            pre_metrics[op_type] = compute_interior_closure_metrics(pre_points, k=5)
            print(f"  Pre-grokking:  {len(pre_points)} samples")
            if pre_metrics[op_type]:
                print(f"    Avg interior score:  {np.mean(pre_metrics[op_type]['interior_score']):.3f}")
                print(f"    Avg boundary score:  {np.mean(pre_metrics[op_type]['boundary_score']):.3f}")
                print(f"    Avg local density:   {np.mean(pre_metrics[op_type]['local_density']):.3f}")

        if len(post_points) > 0:
            post_metrics[op_type] = compute_interior_closure_metrics(post_points, k=5)
            print(f"  Post-grokking: {len(post_points)} samples")
            if post_metrics[op_type]:
                print(f"    Avg interior score:  {np.mean(post_metrics[op_type]['interior_score']):.3f}")
                print(f"    Avg boundary score:  {np.mean(post_metrics[op_type]['boundary_score']):.3f}")
                print(f"    Avg local density:   {np.mean(post_metrics[op_type]['local_density']):.3f}")

    # Create visualization
    create_interior_closure_viz(
        pre_repr, post_repr,
        pre_classified, post_classified,
        pre_metrics, post_metrics,
        pre_epoch, post_epoch,
        output_dir
    )


def create_interior_closure_viz(
    pre_repr, post_repr,
    pre_classified, post_classified,
    pre_metrics, post_metrics,
    pre_epoch, post_epoch,
    output_dir
):
    """Create comprehensive visualization."""

    # PCA projection
    all_data = np.vstack([pre_repr, post_repr])
    pca = PCA(n_components=2)
    pca.fit(all_data)

    pre_proj = pca.transform(pre_repr)
    post_proj = pca.transform(post_repr)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Color maps
    cmap_interior = plt.cm.Blues
    cmap_closure = plt.cm.Reds

    # ========== Pre-grokking: Necessity (□) ==========
    ax = axes[0, 0]
    indices = pre_classified['necessity']['indices']
    if indices and pre_metrics.get('necessity'):
        scores = pre_metrics['necessity']['interior_score']
        scatter = ax.scatter(pre_proj[indices, 0], pre_proj[indices, 1],
                           c=scores, cmap=cmap_interior, s=100,
                           edgecolors='black', linewidth=0.5,
                           vmin=0, vmax=1)
        plt.colorbar(scatter, ax=ax, label='Interior Score')

    ax.set_title(f'□ Necessity (Pre-Grokking)\nEpoch {pre_epoch}\nExpected: Interior/Core Points',
                fontsize=10, fontweight='bold')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.grid(True, alpha=0.3)

    # ========== Pre-grokking: Possibility (◇) ==========
    ax = axes[0, 1]
    indices = pre_classified['possibility']['indices']
    if indices and pre_metrics.get('possibility'):
        scores = pre_metrics['possibility']['boundary_score']
        scatter = ax.scatter(pre_proj[indices, 0], pre_proj[indices, 1],
                           c=scores, cmap=cmap_closure, s=100,
                           edgecolors='black', linewidth=0.5,
                           vmin=0, vmax=1)
        plt.colorbar(scatter, ax=ax, label='Boundary Score')

    ax.set_title(f'◇ Possibility (Pre-Grokking)\nEpoch {pre_epoch}\nExpected: Closure/Boundary Points',
                fontsize=10, fontweight='bold')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.grid(True, alpha=0.3)

    # ========== Pre-grokking: Comparison ==========
    ax = axes[0, 2]
    nec_idx = pre_classified['necessity']['indices']
    pos_idx = pre_classified['possibility']['indices']

    if nec_idx:
        ax.scatter(pre_proj[nec_idx, 0], pre_proj[nec_idx, 1],
                  c='#2E86AB', marker='o', s=80, alpha=0.7,
                  label='□ Necessity', edgecolors='black', linewidth=0.5)
    if pos_idx:
        ax.scatter(pre_proj[pos_idx, 0], pre_proj[pos_idx, 1],
                  c='#A23B72', marker='^', s=80, alpha=0.7,
                  label='◇ Possibility', edgecolors='black', linewidth=0.5)

    ax.set_title(f'Combined (Pre-Grokking)\nEpoch {pre_epoch}',
                fontsize=10, fontweight='bold')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ========== Post-grokking: Necessity (□) ==========
    ax = axes[1, 0]
    indices = post_classified['necessity']['indices']
    if indices and post_metrics.get('necessity'):
        scores = post_metrics['necessity']['interior_score']
        scatter = ax.scatter(post_proj[indices, 0], post_proj[indices, 1],
                           c=scores, cmap=cmap_interior, s=100,
                           edgecolors='black', linewidth=0.5,
                           vmin=0, vmax=1)
        plt.colorbar(scatter, ax=ax, label='Interior Score')

    ax.set_title(f'□ Necessity (Post-Grokking)\nEpoch {post_epoch}\nExpected: Interior/Core Points',
                fontsize=10, fontweight='bold')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.grid(True, alpha=0.3)

    # ========== Post-grokking: Possibility (◇) ==========
    ax = axes[1, 1]
    indices = post_classified['possibility']['indices']
    if indices and post_metrics.get('possibility'):
        scores = post_metrics['possibility']['boundary_score']
        scatter = ax.scatter(post_proj[indices, 0], post_proj[indices, 1],
                           c=scores, cmap=cmap_closure, s=100,
                           edgecolors='black', linewidth=0.5,
                           vmin=0, vmax=1)
        plt.colorbar(scatter, ax=ax, label='Boundary Score')

    ax.set_title(f'◇ Possibility (Post-Grokking)\nEpoch {post_epoch}\nExpected: Closure/Boundary Points',
                fontsize=10, fontweight='bold')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.grid(True, alpha=0.3)

    # ========== Post-grokking: Comparison ==========
    ax = axes[1, 2]
    nec_idx = post_classified['necessity']['indices']
    pos_idx = post_classified['possibility']['indices']

    if nec_idx:
        ax.scatter(post_proj[nec_idx, 0], post_proj[nec_idx, 1],
                  c='#2E86AB', marker='o', s=80, alpha=0.7,
                  label='□ Necessity', edgecolors='black', linewidth=0.5)
    if pos_idx:
        ax.scatter(post_proj[pos_idx, 0], post_proj[pos_idx, 1],
                  c='#A23B72', marker='^', s=80, alpha=0.7,
                  label='◇ Possibility', edgecolors='black', linewidth=0.5)

    ax.set_title(f'Combined (Post-Grokking)\nEpoch {post_epoch}',
                fontsize=10, fontweight='bold')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('McKinsey-Tarski Interpretation: □ (Interior) vs ◇ (Closure/Boundary)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_file = output_dir / 'interior_closure_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    plt.close()

    # Create metric comparison plot
    create_metric_comparison(pre_metrics, post_metrics, pre_epoch, post_epoch, output_dir)


def create_metric_comparison(pre_metrics, post_metrics, pre_epoch, post_epoch, output_dir):
    """Create bar plot comparing metrics."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    op_types = ['necessity', 'possibility']
    x = np.arange(len(op_types))
    width = 0.35

    metrics_to_plot = [
        ('interior_score', 'Interior Score', '(Higher = more interior-like)'),
        ('local_density', 'Local Density', '(Higher = tighter clustering)'),
        ('neighbor_variance', 'Neighbor Distance Variance', '(Lower = more uniform)')
    ]

    for idx, (metric, title, subtitle) in enumerate(metrics_to_plot):
        ax = axes[idx]

        pre_vals = []
        post_vals = []

        for op_type in op_types:
            if op_type in pre_metrics and pre_metrics[op_type]:
                pre_vals.append(np.mean(pre_metrics[op_type][metric]))
            else:
                pre_vals.append(0)

            if op_type in post_metrics and post_metrics[op_type]:
                post_vals.append(np.mean(post_metrics[op_type][metric]))
            else:
                post_vals.append(0)

        ax.bar(x - width/2, pre_vals, width, label=f'Pre (epoch {pre_epoch})',
               color='lightblue', edgecolor='black')
        ax.bar(x + width/2, post_vals, width, label=f'Post (epoch {post_epoch})',
               color='lightcoral', edgecolor='black')

        ax.set_ylabel(title)
        ax.set_title(f'{title}\n{subtitle}', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(['□ Necessity\n(Interior)', '◇ Possibility\n(Closure)'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('McKinsey-Tarski Geometric Metrics Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()

    output_file = output_dir / 'metric_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Metric comparison saved to: {output_file}")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--pre_epoch', type=int, default=490)
    parser.add_argument('--post_epoch', type=int, default=900)
    parser.add_argument('--max_samples', type=int, default=200,
                       help='Maximum samples to use (to reduce memory usage)')

    args = parser.parse_args()

    analyze_interior_closure(
        Path(args.results_dir),
        Path(args.data_dir),
        args.pre_epoch,
        args.post_epoch,
        max_samples=args.max_samples
    )
