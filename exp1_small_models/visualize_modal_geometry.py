#!/usr/bin/env python3
"""
Visualize simplicial complexes before/after grokking with McKinsey-Tarski interpretation.

Connects the geometric structure to modal operators:
- □ (necessity) → interior operator (contractive)
- ◇ (possibility) → closure operator (expansive)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import re
from collections import defaultdict
import json
import sys

# Import topology functions
import gudhi as gd
from gudhi import SimplexTree
from scipy.spatial.distance import cdist

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
            distance = dm[i, j]
            if distance <= epsilon:
                simplex_tree.insert([i, j], filtration=distance)

    simplex_tree.expansion(max_dimension)
    return simplex_tree


# parse_modal_operators is now imported from data_format_utils as parse_ops
# Keeping a local wrapper for backwards compatibility
def parse_modal_operators(formula: str):
    """
    Extract modal operator information from SMCDEL formula.

    Returns:
        dict with counts and positions of operators
    """
    return parse_ops(formula)


def load_dataset_with_formulas(data_dir: Path):
    """Load the original dataset to get formulas."""
    try:
        dataset, data_format = load_dataset(data_dir, 'train')
        return dataset
    except Exception as e:
        print(f"Warning: Could not load dataset from {data_dir}: {e}")
        print(f"Will use dummy data for visualization")
        return None


def load_and_classify_states(states_dir: Path, data_dir: Path, epoch: int, split: str = 'train', max_samples: int = None):
    """
    Load states and classify by modal operator type.

    Args:
        states_dir: Directory containing captured states
        data_dir: Directory containing dataset
        epoch: Epoch to load
        split: Data split ('train' or 'val')
        max_samples: Maximum number of samples to load (None = all)

    Returns:
        dict with representations and metadata classified by operator type
    """
    # Load hidden states
    state_file = states_dir / f'states_epoch_{epoch}_{split}.pt'
    if not state_file.exists():
        raise FileNotFoundError(f"State file not found: {state_file}")

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

    # Load dataset to get formulas
    dataset = load_dataset_with_formulas(data_dir)

    # Classify by operator type
    classified = {
        'necessity': {'indices': [], 'repr': []},
        'possibility': {'indices': [], 'repr': []},
        'mixed': {'indices': [], 'repr': []},
        'none': {'indices': [], 'repr': []}
    }

    operator_info = []

    if dataset is not None:
        # Detect data format
        data_format = None
        try:
            _, data_format = load_dataset(data_dir, 'train')
        except:
            pass

        # Use actual formulas
        # If we sampled, we need to use the original indices from the dataset
        if indices is not None:
            dataset_indices = indices
        else:
            dataset_indices = range(len(representations))

        for repr_idx, dataset_idx in enumerate(dataset_indices):
            example = dataset[dataset_idx]
            # Use unified formula extraction
            if data_format:
                formula = get_smcdel_formula(example, data_format)
            else:
                # Fallback for legacy code
                formula = example.get('smcdel_problem', example.get('hypothesis', ''))

            info = parse_modal_operators(formula)
            operator_info.append(info)

            op_type = info['operator_type']
            classified[op_type]['indices'].append(repr_idx)
            classified[op_type]['repr'].append(representations[repr_idx])
    else:
        # Fallback: random classification for visualization
        print("Using random operator classification (dataset not available)")
        for idx in range(len(representations)):
            op_type = np.random.choice(['necessity', 'possibility', 'mixed', 'none'],
                                      p=[0.35, 0.35, 0.2, 0.1])
            classified[op_type]['indices'].append(idx)
            classified[op_type]['repr'].append(representations[idx])
            operator_info.append({'operator_type': op_type})

    # Convert lists to arrays
    for op_type in classified:
        if classified[op_type]['repr']:
            classified[op_type]['repr'] = np.array(classified[op_type]['repr'])

    return {
        'all_repr': representations,
        'classified': classified,
        'operator_info': operator_info
    }


def compute_geometric_properties(points: np.ndarray, epsilon: float):
    """
    Compute geometric properties relevant to McKinsey-Tarski interpretation.

    Returns:
        dict with interior, boundary, and closure properties
    """
    if len(points) == 0:
        return {
            'centroid': None,
            'spread': 0,
            'density': 0,
            'avg_neighbor_dist': 0
        }

    # Compute centroid
    centroid = np.mean(points, axis=0)

    # Compute spread (average distance from centroid)
    spread = np.mean(np.linalg.norm(points - centroid, axis=1))

    # Compute density (inverse of average pairwise distance)
    if len(points) > 1:
        pairwise_dists = pdist(points)
        avg_dist = np.mean(pairwise_dists)
        density = 1.0 / (avg_dist + 1e-6)
        avg_neighbor_dist = avg_dist
    else:
        density = 0
        avg_neighbor_dist = 0

    return {
        'centroid': centroid,
        'spread': spread,
        'density': density,
        'avg_neighbor_dist': avg_neighbor_dist,
        'num_points': len(points)
    }


def visualize_modal_simplicial_complex(
    results_dir: Path,
    data_dir: Path,
    pre_grok_epoch: int,
    post_grok_epoch: int,
    epsilon: float = 0.5,
    max_samples: int = 200,
    output_dir: Path = None
):
    """
    Main visualization: simplicial complexes colored by modal operator type.
    """
    results_dir = Path(results_dir)
    data_dir = Path(data_dir)

    if output_dir is None:
        output_dir = results_dir / 'modal_geometry_analysis'
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("="*70)
    print("McKinsey-Tarski Modal Geometry Visualization")
    print("="*70)

    states_dir = results_dir / 'captured_states'

    # Load and classify states
    print(f"\nLoading pre-grokking epoch {pre_grok_epoch} (max {max_samples} samples)...")
    pre_data = load_and_classify_states(states_dir, data_dir, pre_grok_epoch, 'train', max_samples=max_samples)

    print(f"Loading post-grokking epoch {post_grok_epoch} (max {max_samples} samples)...")
    post_data = load_and_classify_states(states_dir, data_dir, post_grok_epoch, 'train', max_samples=max_samples)

    # Print operator distribution
    print(f"\nOperator Distribution:")
    for op_type in ['necessity', 'possibility', 'mixed', 'none']:
        pre_count = len(pre_data['classified'][op_type]['indices'])
        post_count = len(post_data['classified'][op_type]['indices'])
        print(f"  {op_type:12s}: {pre_count:3d} samples")

    # Compute PCA on combined data for consistent projection
    print("\nFitting PCA on combined data...")
    all_data = np.vstack([pre_data['all_repr'], post_data['all_repr']])
    pca = PCA(n_components=3)
    pca.fit(all_data)

    print(f"PCA variance explained: {pca.explained_variance_ratio_}")

    # Project both epochs
    pre_proj = pca.transform(pre_data['all_repr'])
    post_proj = pca.transform(post_data['all_repr'])

    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))

    # Define colors for operator types (McKinsey-Tarski inspired)
    colors = {
        'necessity': '#2E86AB',     # Blue - interior/closed/contractive
        'possibility': '#A23B72',   # Purple - closure/boundary/expansive
        'mixed': '#F18F01',         # Orange - combination
        'none': '#C73E1D'           # Red - base/atomic
    }

    # Define markers
    markers = {
        'necessity': 'o',
        'possibility': '^',
        'mixed': 's',
        'none': 'D'
    }

    # ========== Plot 1: Pre-grokking 2D scatter with edges ==========
    ax1 = fig.add_subplot(2, 3, 1)
    plot_simplicial_2d(ax1, pre_proj, pre_data, epsilon, colors, markers,
                       f'Pre-Grokking (Epoch {pre_grok_epoch})\nSimplicial Complex')

    # ========== Plot 2: Post-grokking 2D scatter with edges ==========
    ax2 = fig.add_subplot(2, 3, 2)
    plot_simplicial_2d(ax2, post_proj, post_data, epsilon, colors, markers,
                       f'Post-Grokking (Epoch {post_grok_epoch})\nSimplicial Complex')

    # ========== Plot 3: Overlay comparison ==========
    ax3 = fig.add_subplot(2, 3, 3)
    plot_overlay_comparison(ax3, pre_proj, post_proj, pre_data, post_data,
                           colors, markers)

    # ========== Plot 4: 3D Pre-grokking ==========
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    plot_3d_scatter(ax4, pre_proj, pre_data, colors, markers,
                   f'Pre-Grokking 3D\n(Epoch {pre_grok_epoch})')

    # ========== Plot 5: 3D Post-grokking ==========
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    plot_3d_scatter(ax5, post_proj, post_data, colors, markers,
                   f'Post-Grokking 3D\n(Epoch {post_grok_epoch})')

    # ========== Plot 6: Geometric properties comparison ==========
    ax6 = fig.add_subplot(2, 3, 6)
    plot_geometric_properties(ax6, pre_data, post_data, epsilon, colors)

    plt.suptitle('McKinsey-Tarski Modal Geometry: □ (Necessity/Interior) vs ◇ (Possibility/Closure)',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    output_file = output_dir / 'modal_geometry_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    plt.close()

    # Compute and save detailed statistics
    compute_modal_statistics(pre_data, post_data, epsilon, output_dir,
                            pre_grok_epoch, post_grok_epoch)


def plot_simplicial_2d(ax, proj, data, epsilon, colors, markers, title):
    """Plot 2D scatter with simplicial edges."""
    # Compute VR complex on 2D projection
    st = compute_vietoris_rips_complex(proj[:, :2], epsilon)

    # Draw edges
    edges = []
    for simplex, _ in st.get_filtration():
        if len(simplex) == 2:
            edges.append(simplex)

    for edge in edges:
        i, j = edge
        ax.plot([proj[i, 0], proj[j, 0]], [proj[i, 1], proj[j, 1]],
               'gray', alpha=0.15, linewidth=0.5, zorder=1)

    # Plot points colored by operator type
    for op_type in ['necessity', 'possibility', 'mixed', 'none']:
        indices = data['classified'][op_type]['indices']
        if indices:
            op_proj = proj[indices]
            ax.scatter(op_proj[:, 0], op_proj[:, 1],
                      c=colors[op_type], marker=markers[op_type],
                      s=80, alpha=0.7, edgecolors='black', linewidth=0.5,
                      label=f'{op_type.capitalize()} (□)' if op_type == 'necessity'
                            else f'{op_type.capitalize()} (◇)' if op_type == 'possibility'
                            else op_type.capitalize(),
                      zorder=2)

    ax.set_xlabel('PC1', fontsize=10)
    ax.set_ylabel('PC2', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)


def plot_overlay_comparison(ax, pre_proj, post_proj, pre_data, post_data, colors, markers):
    """Overlay pre and post to show movement."""
    # Plot necessity operators
    pre_nec_idx = pre_data['classified']['necessity']['indices']
    post_nec_idx = post_data['classified']['necessity']['indices']

    if pre_nec_idx:
        ax.scatter(pre_proj[pre_nec_idx, 0], pre_proj[pre_nec_idx, 1],
                  c=colors['necessity'], marker='o', s=60, alpha=0.3,
                  label='Pre: Necessity (□)', edgecolors='none')

    if post_nec_idx:
        ax.scatter(post_proj[post_nec_idx, 0], post_proj[post_nec_idx, 1],
                  c=colors['necessity'], marker='*', s=100, alpha=0.8,
                  label='Post: Necessity (□)', edgecolors='black', linewidth=0.5)

    # Plot possibility operators
    pre_pos_idx = pre_data['classified']['possibility']['indices']
    post_pos_idx = post_data['classified']['possibility']['indices']

    if pre_pos_idx:
        ax.scatter(pre_proj[pre_pos_idx, 0], pre_proj[pre_pos_idx, 1],
                  c=colors['possibility'], marker='^', s=60, alpha=0.3,
                  label='Pre: Possibility (◇)', edgecolors='none')

    if post_pos_idx:
        ax.scatter(post_proj[post_pos_idx, 0], post_proj[post_pos_idx, 1],
                  c=colors['possibility'], marker='^', s=100, alpha=0.8,
                  label='Post: Possibility (◇)', edgecolors='black', linewidth=0.5)

    ax.set_xlabel('PC1', fontsize=10)
    ax.set_ylabel('PC2', fontsize=10)
    ax.set_title('Overlay: Movement Through Grokking\n(Faded=Pre, Bold=Post)',
                fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)


def plot_3d_scatter(ax, proj, data, colors, markers, title):
    """3D scatter plot."""
    for op_type in ['necessity', 'possibility', 'mixed', 'none']:
        indices = data['classified'][op_type]['indices']
        if indices:
            op_proj = proj[indices]
            ax.scatter(op_proj[:, 0], op_proj[:, 1], op_proj[:, 2],
                      c=colors[op_type], marker=markers[op_type],
                      s=50, alpha=0.6, edgecolors='black', linewidth=0.5,
                      label=op_type.capitalize())

    ax.set_xlabel('PC1', fontsize=9)
    ax.set_ylabel('PC2', fontsize=9)
    ax.set_zlabel('PC3', fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)


def plot_geometric_properties(ax, pre_data, post_data, epsilon, colors):
    """Compare geometric properties by operator type."""
    op_types = ['necessity', 'possibility', 'mixed']

    # Compute properties
    pre_props = {}
    post_props = {}

    for op_type in op_types:
        pre_repr = pre_data['classified'][op_type]['repr']
        post_repr = post_data['classified'][op_type]['repr']

        pre_props[op_type] = compute_geometric_properties(pre_repr, epsilon)
        post_props[op_type] = compute_geometric_properties(post_repr, epsilon)

    # Plot spread comparison
    x = np.arange(len(op_types))
    width = 0.35

    pre_spreads = [pre_props[op]['spread'] for op in op_types]
    post_spreads = [post_props[op]['spread'] for op in op_types]

    ax.bar(x - width/2, pre_spreads, width, label='Pre-grokking',
           color='lightblue', edgecolor='black', linewidth=1)
    ax.bar(x + width/2, post_spreads, width, label='Post-grokking',
           color='lightcoral', edgecolor='black', linewidth=1)

    ax.set_ylabel('Geometric Spread\n(Avg distance from centroid)', fontsize=10)
    ax.set_title('McKinsey-Tarski Properties:\nContraction (□) vs Expansion (◇)',
                fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['□ Necessity\n(Interior)', '◇ Possibility\n(Closure)', 'Mixed'],
                      fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Add text annotations
    for i, op in enumerate(op_types):
        pre_val = pre_spreads[i]
        post_val = post_spreads[i]
        change = ((post_val - pre_val) / (pre_val + 1e-6)) * 100
        ax.text(i, max(pre_val, post_val) * 1.05, f'{change:+.1f}%',
               ha='center', fontsize=8, fontweight='bold')


def compute_modal_statistics(pre_data, post_data, epsilon, output_dir,
                             pre_epoch, post_epoch):
    """Compute and save detailed statistics."""
    stats = {
        'pre_grokking_epoch': pre_epoch,
        'post_grokking_epoch': post_epoch,
        'epsilon': epsilon,
        'operator_types': {}
    }

    for op_type in ['necessity', 'possibility', 'mixed', 'none']:
        pre_repr = pre_data['classified'][op_type]['repr']
        post_repr = post_data['classified'][op_type]['repr']

        pre_props = compute_geometric_properties(pre_repr, epsilon)
        post_props = compute_geometric_properties(post_repr, epsilon)

        stats['operator_types'][op_type] = {
            'pre_grokking': pre_props,
            'post_grokking': post_props,
            'change': {
                'spread': post_props['spread'] - pre_props['spread'],
                'density': post_props['density'] - pre_props['density']
            }
        }

    # Save statistics
    stats_file = output_dir / 'modal_geometry_stats.json'
    with open(stats_file, 'w') as f:
        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            return obj

        json.dump(stats, f, indent=2, default=convert)

    print(f"Statistics saved to: {stats_file}")

    # Print summary
    print("\n" + "="*70)
    print("McKinsey-Tarski Geometric Analysis Summary")
    print("="*70)

    for op_type in ['necessity', 'possibility']:
        print(f"\n{op_type.upper()} (□)" if op_type == 'necessity' else f"\n{op_type.upper()} (◇)")
        print("-" * 40)

        pre_props = stats['operator_types'][op_type]['pre_grokking']
        post_props = stats['operator_types'][op_type]['post_grokking']

        print(f"  Pre-grokking spread:  {pre_props['spread']:.4f}")
        print(f"  Post-grokking spread: {post_props['spread']:.4f}")
        print(f"  Change: {post_props['spread'] - pre_props['spread']:.4f}")
        print(f"  Interpretation: {'CONTRACTION (interior)' if post_props['spread'] < pre_props['spread'] else 'EXPANSION (closure)'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Visualize modal geometry with McKinsey-Tarski interpretation'
    )
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory with captured states')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory with original dataset (train.json)')
    parser.add_argument('--pre_grok_epoch', type=int, default=490,
                       help='Epoch before grokking')
    parser.add_argument('--post_grok_epoch', type=int, default=900,
                       help='Epoch after grokking')
    parser.add_argument('--epsilon', type=float, default=0.5,
                       help='VR complex epsilon')
    parser.add_argument('--max_samples', type=int, default=200,
                       help='Maximum samples to use (to reduce memory usage)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory')

    args = parser.parse_args()

    visualize_modal_simplicial_complex(
        results_dir=Path(args.results_dir),
        data_dir=Path(args.data_dir),
        pre_grok_epoch=args.pre_grok_epoch,
        post_grok_epoch=args.post_grok_epoch,
        epsilon=args.epsilon,
        max_samples=args.max_samples,
        output_dir=Path(args.output_dir) if args.output_dir else None
    )
