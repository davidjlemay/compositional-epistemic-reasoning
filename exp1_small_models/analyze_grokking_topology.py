#!/usr/bin/env python3
"""
Topological analysis of grokking experiment using VR_trajectories.
Analyzes how the latent space topology evolves through the grokking transition.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import json
from collections import defaultdict

# Import topology libraries directly
import gudhi as gd
from gudhi import SimplexTree
from scipy.spatial.distance import cdist

import warnings
warnings.filterwarnings('ignore')


def compute_vietoris_rips_complex(points, epsilon, max_dimension=2):
    """
    Compute the Vietoris-Rips complex for a given point cloud at a fixed epsilon value.
    Adapted from VR_trajectories.py
    """
    simplex_tree = SimplexTree()

    # Add vertices
    for i in range(len(points)):
        simplex_tree.insert([i], filtration=0.0)

    # Compute pairwise distances
    dm = cdist(points, points)
    n_points = len(points)

    for i in range(n_points):
        for j in range(i+1, n_points):
            distance = dm[i, j]
            if distance <= epsilon:
                simplex_tree.insert([i, j], filtration=distance)

    # Expand complex to higher dimensions
    simplex_tree.expansion(max_dimension)

    return simplex_tree


def load_checkpoint_info(checkpoint_path: Path) -> Dict:
    """Load checkpoint and extract grokking information."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    info = {
        'epoch': checkpoint['epoch'],
        'grokking_detected': checkpoint.get('grokking_detected', False),
        'grokking_epoch': checkpoint.get('grokking_epoch', None),
        'train_metrics': checkpoint.get('train_metrics', {}),
        'val_metrics': checkpoint.get('val_metrics', {})
    }

    return info


def load_captured_states(states_dir: Path, split: str = 'train') -> Dict[int, Dict]:
    """
    Load captured hidden states from all epochs.

    Returns:
        Dictionary mapping epoch -> state data
    """
    states_dir = Path(states_dir)
    states_by_epoch = {}

    for state_file in sorted(states_dir.glob(f"states_epoch_*_{split}.pt")):
        # Parse filename: states_epoch_X_split.pt
        parts = state_file.stem.split('_')
        epoch = int(parts[2])

        states = torch.load(state_file, map_location='cpu', weights_only=False)
        states_by_epoch[epoch] = states

    print(f"Loaded {len(states_by_epoch)} state captures for {split} split")
    return states_by_epoch


def extract_encoder_representations(states_dict: Dict) -> np.ndarray:
    """
    Extract encoder hidden states and convert to numpy array.

    Handles two formats:
    1. GRU/RNN: List[List[Tensor]] where each item is batch of sequences
    2. GNN: Single Tensor of shape [num_samples, hidden_dim]

    Returns:
        Array of shape [num_samples, hidden_dim]
    """
    encoder_states = states_dict['encoder_states']

    # Check if it's already a tensor (GNN format)
    if isinstance(encoder_states, torch.Tensor):
        return encoder_states.numpy()

    # Otherwise handle GRU/RNN format
    representations = []
    for batch_item in encoder_states:
        if isinstance(batch_item, list):
            # batch_item is List[Tensor], take the last layer's output
            hidden_state = batch_item[-1]  # Shape: [seq_len, hidden_dim]
            # Use the final time step as the representation
            final_hidden = hidden_state[-1, :]  # Shape: [hidden_dim]
        else:
            # Single tensor per sample
            final_hidden = batch_item if batch_item.dim() == 1 else batch_item[-1, :]

        representations.append(final_hidden.numpy())

    return np.array(representations)


def extract_decoder_representations(states_dict: Dict) -> np.ndarray:
    """
    Extract decoder hidden states.

    Handles two formats:
    1. GRU/RNN: List[List[Tensor]] where each item is batch of sequences
    2. GNN: Single Tensor of shape [num_samples, hidden_dim]

    Returns:
        Array of shape [num_samples, hidden_dim]
    """
    decoder_states = states_dict['decoder_states']

    # Check if it's already a tensor (GNN format)
    if isinstance(decoder_states, torch.Tensor):
        return decoder_states.numpy()

    # Otherwise handle GRU/RNN format
    representations = []
    for batch_item in decoder_states:
        if isinstance(batch_item, list):
            hidden_state = batch_item[-1]  # Last layer
            final_hidden = hidden_state[-1, :]  # Final time step
        else:
            # Single tensor per sample
            final_hidden = batch_item if batch_item.dim() == 1 else batch_item[-1, :]

        representations.append(final_hidden.numpy())

    return np.array(representations)


def compute_topology_metrics(data: np.ndarray, epsilon: float, max_dimension: int = 2) -> Dict:
    """
    Compute topological metrics for a given representation.

    Returns:
        Dictionary with topology metrics
    """
    # Compute VR complex
    st = compute_vietoris_rips_complex(data, epsilon, max_dimension=max_dimension)

    # Get simplex counts
    num_0_simplices = 0
    num_1_simplices = 0
    num_2_simplices = 0

    for simplex, _ in st.get_filtration():
        dim = len(simplex) - 1
        if dim == 0:
            num_0_simplices += 1
        elif dim == 1:
            num_1_simplices += 1
        elif dim == 2:
            num_2_simplices += 1

    # Compute Betti numbers at the complex's own filtration value (i.e., the
    # number of persistence intervals still alive at epsilon), matching the
    # method used in analyze_attention_topology.py. The previous implementation
    # here (`sum(1 for dim, (birth, death) in persistence if dim == 0)`) counted
    # every dimension-0 birth event ever created across the whole filtration,
    # which is always exactly len(points) regardless of connectivity -- a bug,
    # not a "sample cap artifact".
    st.compute_persistence()
    betti = st.betti_numbers()
    h0_features = int(betti[0]) if len(betti) > 0 else 0
    h1_features = int(betti[1]) if len(betti) > 1 else 0

    metrics = {
        'num_0_simplices': num_0_simplices,
        'num_1_simplices': num_1_simplices,
        'num_2_simplices': num_2_simplices,
        'h0_features': h0_features,
        'h1_features': h1_features,
        'total_simplices': num_0_simplices + num_1_simplices + num_2_simplices
    }

    return metrics


def analyze_grokking_topology(
    results_dir: Path,
    split: str = 'train',
    epsilon_encoder: float = 0.5,
    epsilon_decoder: float = 0.3,
    output_dir: Path = None,
    transition_epochs: List[int] = None,
    max_samples: int = None
):
    """
    Main analysis function: tracks topological evolution through grokking.

    Args:
        results_dir: Directory with checkpoint and captured states
        split: 'train' or 'val'
        epsilon_encoder: VR epsilon for encoder
        epsilon_decoder: VR epsilon for decoder
        output_dir: Where to save results
        transition_epochs: Additional epochs to highlight (optional)
        max_samples: Maximum number of samples to analyze (None = all)
    """
    results_dir = Path(results_dir)

    if output_dir is None:
        output_dir = results_dir / 'topology_analysis'
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Grokking Topology Analysis")
    print("=" * 60)

    # Load checkpoint info
    checkpoint_path = results_dir / 'checkpoint_latest.pt'
    info = load_checkpoint_info(checkpoint_path)

    print(f"\nExperiment Info:")
    print(f"  Total epochs: {info['epoch']}")
    print(f"  Grokking detected: {info['grokking_detected']}")
    print(f"  Grokking epoch: {info['grokking_epoch']}")

    # Handle both list and dict formats for metrics
    train_metrics = info['train_metrics']
    val_metrics = info['val_metrics']

    if isinstance(train_metrics, list) and train_metrics:
        train_acc = train_metrics[-1].get('accuracy', 'N/A') if isinstance(train_metrics[-1], dict) else 'N/A'
    elif isinstance(train_metrics, dict):
        train_acc = train_metrics.get('accuracy', 'N/A')
    else:
        train_acc = 'N/A'

    if isinstance(val_metrics, list) and val_metrics:
        val_acc = val_metrics[-1].get('accuracy', 'N/A') if isinstance(val_metrics[-1], dict) else 'N/A'
    elif isinstance(val_metrics, dict):
        val_acc = val_metrics.get('accuracy', 'N/A')
    else:
        val_acc = 'N/A'

    if train_acc != 'N/A':
        print(f"  Final train accuracy: {train_acc:.4f}")
    if val_acc != 'N/A':
        print(f"  Final val accuracy: {val_acc:.4f}")

    # Load captured states
    states_dir = results_dir / 'captured_states'
    states_by_epoch = load_captured_states(states_dir, split=split)

    if not states_by_epoch:
        print("No captured states found!")
        return

    # Analyze topology evolution
    print(f"\n{'='*60}")
    print(f"Analyzing topology evolution across {len(states_by_epoch)} epochs")
    print(f"{'='*60}")

    epochs = sorted(states_by_epoch.keys())
    encoder_metrics_by_epoch = {}
    decoder_metrics_by_epoch = {}

    for epoch in epochs:
        print(f"\nEpoch {epoch}:", end=" ")

        states = states_by_epoch[epoch]

        # Extract representations
        encoder_repr = extract_encoder_representations(states)
        decoder_repr = extract_decoder_representations(states)

        # Subsample if max_samples is specified
        if max_samples is not None and len(encoder_repr) > max_samples:
            print(f"Subsampling {max_samples} from {len(encoder_repr)} samples")
            indices = np.random.choice(len(encoder_repr), size=max_samples, replace=False)
            encoder_repr = encoder_repr[indices]
            decoder_repr = decoder_repr[indices]

        print(f"Encoder: {encoder_repr.shape}, Decoder: {decoder_repr.shape}")

        # Compute topology metrics
        encoder_metrics = compute_topology_metrics(encoder_repr, epsilon_encoder)
        decoder_metrics = compute_topology_metrics(decoder_repr, epsilon_decoder)

        encoder_metrics_by_epoch[epoch] = encoder_metrics
        decoder_metrics_by_epoch[epoch] = decoder_metrics

        print(f"  Encoder: {encoder_metrics['total_simplices']} simplices, "
              f"H0={encoder_metrics['h0_features']}, H1={encoder_metrics['h1_features']}")
        print(f"  Decoder: {decoder_metrics['total_simplices']} simplices, "
              f"H0={decoder_metrics['h0_features']}, H1={decoder_metrics['h1_features']}")

    # Plot topology evolution
    plot_topology_evolution(
        encoder_metrics_by_epoch,
        decoder_metrics_by_epoch,
        info['grokking_epoch'],
        output_dir,
        transition_epochs=transition_epochs
    )

    # Save metrics
    metrics_file = output_dir / 'topology_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump({
            'info': info,
            'encoder_metrics': {int(k): v for k, v in encoder_metrics_by_epoch.items()},
            'decoder_metrics': {int(k): v for k, v in decoder_metrics_by_epoch.items()},
            'epsilon_encoder': epsilon_encoder,
            'epsilon_decoder': epsilon_decoder
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Analysis complete! Results saved to {output_dir}")
    print(f"{'='*60}")

    return encoder_metrics_by_epoch, decoder_metrics_by_epoch


def plot_topology_evolution(
    encoder_metrics: Dict[int, Dict],
    decoder_metrics: Dict[int, Dict],
    grokking_epoch: int,
    output_dir: Path,
    transition_epochs: List[int] = None
):
    """
    Plot how topological features evolve through grokking.

    Args:
        encoder_metrics: Metrics for encoder by epoch
        decoder_metrics: Metrics for decoder by epoch
        grokking_epoch: Primary grokking/transition epoch (can be None)
        output_dir: Where to save plots
        transition_epochs: Additional transition epochs to highlight
    """

    epochs = sorted(encoder_metrics.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Topological Evolution Through Grokking', fontsize=16, fontweight='bold')

    # Extract metrics for plotting
    enc_h0 = [encoder_metrics[e]['h0_features'] for e in epochs]
    enc_h1 = [encoder_metrics[e]['h1_features'] for e in epochs]
    dec_h0 = [decoder_metrics[e]['h0_features'] for e in epochs]
    dec_h1 = [decoder_metrics[e]['h1_features'] for e in epochs]

    enc_total = [encoder_metrics[e]['total_simplices'] for e in epochs]
    dec_total = [decoder_metrics[e]['total_simplices'] for e in epochs]

    # Plot H0 (connected components)
    ax = axes[0, 0]
    ax.plot(epochs, enc_h0, 'o-', label='Encoder', linewidth=2, markersize=6)
    ax.plot(epochs, dec_h0, 's-', label='Decoder', linewidth=2, markersize=6)
    if grokking_epoch:
        ax.axvline(grokking_epoch, color='red', linestyle='--', linewidth=2, label='Grokking')
    if transition_epochs:
        for i, t_epoch in enumerate(transition_epochs):
            label = f'Transition {i+1}' if i == 0 else None
            ax.axvline(t_epoch, color='orange', linestyle=':', linewidth=1.5, label=label, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('H0 Features (Connected Components)', fontsize=12)
    ax.set_title('Topological Feature: H0', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot H1 (loops/holes)
    ax = axes[0, 1]
    ax.plot(epochs, enc_h1, 'o-', label='Encoder', linewidth=2, markersize=6)
    ax.plot(epochs, dec_h1, 's-', label='Decoder', linewidth=2, markersize=6)
    if grokking_epoch:
        ax.axvline(grokking_epoch, color='red', linestyle='--', linewidth=2, label='Grokking')
    if transition_epochs:
        for i, t_epoch in enumerate(transition_epochs):
            label = f'Transition {i+1}' if i == 0 else None
            ax.axvline(t_epoch, color='orange', linestyle=':', linewidth=1.5, label=label, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('H1 Features (Loops)', fontsize=12)
    ax.set_title('Topological Feature: H1', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot total simplices
    ax = axes[1, 0]
    ax.plot(epochs, enc_total, 'o-', label='Encoder', linewidth=2, markersize=6)
    ax.plot(epochs, dec_total, 's-', label='Decoder', linewidth=2, markersize=6)
    if grokking_epoch:
        ax.axvline(grokking_epoch, color='red', linestyle='--', linewidth=2, label='Grokking')
    if transition_epochs:
        for i, t_epoch in enumerate(transition_epochs):
            label = f'Transition {i+1}' if i == 0 else None
            ax.axvline(t_epoch, color='orange', linestyle=':', linewidth=1.5, label=label, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total Simplices', fontsize=12)
    ax.set_title('Simplicial Complex Size', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot ratio of features
    ax = axes[1, 1]
    enc_ratio = [enc_h1[i] / max(enc_h0[i], 1) for i in range(len(epochs))]
    dec_ratio = [dec_h1[i] / max(dec_h0[i], 1) for i in range(len(epochs))]
    ax.plot(epochs, enc_ratio, 'o-', label='Encoder', linewidth=2, markersize=6)
    ax.plot(epochs, dec_ratio, 's-', label='Decoder', linewidth=2, markersize=6)
    if grokking_epoch:
        ax.axvline(grokking_epoch, color='red', linestyle='--', linewidth=2, label='Grokking')
    if transition_epochs:
        for i, t_epoch in enumerate(transition_epochs):
            label = f'Transition {i+1}' if i == 0 else None
            ax.axvline(t_epoch, color='orange', linestyle=':', linewidth=1.5, label=label, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('H1/H0 Ratio', fontsize=12)
    ax.set_title('Topological Complexity Ratio', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_file = output_dir / 'topology_evolution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nTopology evolution plot saved to: {output_file}")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Analyze grokking topology')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing checkpoint and captured states')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val'],
                        help='Which split to analyze')
    parser.add_argument('--epsilon_encoder', type=float, default=0.5,
                        help='Epsilon for encoder VR complex')
    parser.add_argument('--epsilon_decoder', type=float, default=0.3,
                        help='Epsilon for decoder VR complex')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: results_dir/topology_analysis)')
    parser.add_argument('--transition_epochs', type=int, nargs='+', default=None,
                        help='Additional transition epochs to highlight')
    parser.add_argument('--auto_detect', action='store_true',
                        help='Auto-detect transition epochs from checkpoint')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to analyze (default: all)')

    args = parser.parse_args()

    # Auto-detect epochs if requested
    transition_epochs = args.transition_epochs

    if args.auto_detect:
        try:
            from detect_transition_epochs import suggest_analysis_epochs
            checkpoint_path = Path(args.results_dir) / 'checkpoint_latest.pt'
            result = suggest_analysis_epochs(checkpoint_path, verbose=True)

            if result['transition_epochs']:
                transition_epochs = [e for e, _ in result['transition_epochs']]
                print(f"\nAuto-detected transition epochs: {transition_epochs}")
        except Exception as e:
            print(f"Warning: Auto-detection failed: {e}")
            print("Continuing without transition epochs")

    analyze_grokking_topology(
        results_dir=Path(args.results_dir),
        split=args.split,
        epsilon_encoder=args.epsilon_encoder,
        epsilon_decoder=args.epsilon_decoder,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        transition_epochs=transition_epochs,
        max_samples=args.max_samples
    )
