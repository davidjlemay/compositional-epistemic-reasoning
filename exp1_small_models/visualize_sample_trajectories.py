#!/usr/bin/env python3
"""
Visualize how individual sample points move through latent space during grokking.
Uses PCA to project high-dimensional trajectories to 2D/3D.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
import seaborn as sns

def load_captured_states(states_dir: Path, split: str = 'train'):
    """Load captured hidden states from all epochs."""
    states_dir = Path(states_dir)
    states_by_epoch = {}

    for state_file in sorted(states_dir.glob(f"states_epoch_*_{split}.pt")):
        parts = state_file.stem.split('_')
        epoch = int(parts[2])
        states = torch.load(state_file, map_location='cpu', weights_only=False)
        states_by_epoch[epoch] = states

    return states_by_epoch

def extract_encoder_representations(states_dict):
    """
    Extract encoder final hidden states.

    Handles two formats:
    1. GRU/RNN: List[List[Tensor]]
    2. GNN: Single Tensor of shape [num_samples, hidden_dim]
    """
    encoder_states = states_dict['encoder_states']

    # Check if it's already a tensor (GNN format)
    if isinstance(encoder_states, torch.Tensor):
        return encoder_states.numpy()

    # Otherwise handle GRU/RNN format
    representations = []
    for batch_item in encoder_states:
        if isinstance(batch_item, list):
            hidden_state = batch_item[-1]
            final_hidden = hidden_state[-1, :]
        else:
            final_hidden = batch_item if batch_item.dim() == 1 else batch_item[-1, :]
        representations.append(final_hidden.numpy())
    return np.array(representations)

def visualize_trajectories_pca(
    results_dir: Path,
    sample_indices=None,
    output_dir=None,
    transition_epoch=None,
    pre_epoch=None,
    post_epoch=None
):
    """
    Visualize how sample points move through latent space.

    Args:
        results_dir: Directory with captured states
        sample_indices: Which samples to track (default: first 10)
        output_dir: Where to save plots
        transition_epoch: Epoch to mark as key transition (e.g., grokking)
        pre_epoch: Epoch to use for pre-transition comparison
        post_epoch: Epoch to use for post-transition comparison
    """
    results_dir = Path(results_dir)
    if output_dir is None:
        output_dir = results_dir / 'topology_analysis'
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("Loading captured states...")
    states_by_epoch = load_captured_states(results_dir / 'captured_states', split='train')
    epochs = sorted(states_by_epoch.keys())

    print(f"Loaded {len(epochs)} epochs")

    # Extract representations for each epoch
    representations_by_epoch = {}
    for epoch in epochs:
        representations_by_epoch[epoch] = extract_encoder_representations(states_by_epoch[epoch])

    num_samples = representations_by_epoch[epochs[0]].shape[0]
    hidden_dim = representations_by_epoch[epochs[0]].shape[1]

    print(f"Samples: {num_samples}, Hidden dim: {hidden_dim}")

    # Select samples to track
    if sample_indices is None:
        sample_indices = list(range(min(10, num_samples)))

    # Collect all representations for PCA
    all_representations = []
    for epoch in epochs:
        all_representations.append(representations_by_epoch[epoch])
    all_representations = np.vstack(all_representations)

    print(f"Fitting PCA on {all_representations.shape[0]} total representations...")

    # Fit PCA on all data
    pca = PCA(n_components=3)
    pca.fit(all_representations)

    print(f"PCA explained variance: {pca.explained_variance_ratio_}")

    # Project each epoch's representations
    projected_by_epoch = {}
    for epoch in epochs:
        projected_by_epoch[epoch] = pca.transform(representations_by_epoch[epoch])

    # Create visualization
    fig = plt.figure(figsize=(16, 6))

    # 2D trajectory plot
    ax1 = fig.add_subplot(131)
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min(epochs), vmax=max(epochs))

    for sample_idx in sample_indices:
        trajectory_x = [projected_by_epoch[e][sample_idx, 0] for e in epochs]
        trajectory_y = [projected_by_epoch[e][sample_idx, 1] for e in epochs]

        # Plot trajectory with color gradient
        for i in range(len(epochs) - 1):
            color = cmap(norm(epochs[i]))
            ax1.plot(trajectory_x[i:i+2], trajectory_y[i:i+2],
                    color=color, alpha=0.6, linewidth=1.5)

        # Mark start and end
        ax1.scatter(trajectory_x[0], trajectory_y[0], c='green', s=100,
                   marker='o', edgecolors='black', linewidths=1.5, zorder=10)
        ax1.scatter(trajectory_x[-1], trajectory_y[-1], c='red', s=100,
                   marker='*', edgecolors='black', linewidths=1.5, zorder=10)

    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)', fontsize=11)
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)', fontsize=11)
    ax1.set_title('Sample Trajectories Through Training\n(Green=Start, Red=End)',
                 fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Colorbar for epochs
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1)
    cbar.set_label('Epoch', fontsize=10)

    # Before/After comparison
    ax2 = fig.add_subplot(132)

    # Find epochs closest to key transitions
    # Use provided epochs or auto-detect
    if pre_epoch is not None:
        pre_grok_epoch = min(epochs, key=lambda e: abs(e - pre_epoch))
    else:
        pre_grok_epoch = max([e for e in epochs if e < epochs[-1] // 2], default=epochs[0])

    if post_epoch is not None:
        grok_epoch = min(epochs, key=lambda e: abs(e - post_epoch))
    elif transition_epoch is not None:
        grok_epoch = min(epochs, key=lambda e: abs(e - transition_epoch))
    else:
        grok_epoch = epochs[-1]

    pre_proj = projected_by_epoch[pre_grok_epoch]
    grok_proj = projected_by_epoch[grok_epoch]

    ax2.scatter(pre_proj[:, 0], pre_proj[:, 1], c='blue', alpha=0.5,
               s=30, label=f'Epoch {pre_grok_epoch} (early)')
    ax2.scatter(grok_proj[:, 0], grok_proj[:, 1], c='red', alpha=0.5,
               s=30, label=f'Epoch {grok_epoch} (late)')

    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)', fontsize=11)
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)', fontsize=11)
    title = 'Latent Space Before/After Transition' if transition_epoch else 'Latent Space Early/Late Training'
    ax2.set_title(title, fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3D view
    ax3 = fig.add_subplot(133, projection='3d')

    for sample_idx in sample_indices[:5]:  # Fewer samples for 3D clarity
        trajectory_x = [projected_by_epoch[e][sample_idx, 0] for e in epochs]
        trajectory_y = [projected_by_epoch[e][sample_idx, 1] for e in epochs]
        trajectory_z = [projected_by_epoch[e][sample_idx, 2] for e in epochs]

        for i in range(len(epochs) - 1):
            color = cmap(norm(epochs[i]))
            ax3.plot(trajectory_x[i:i+2], trajectory_y[i:i+2], trajectory_z[i:i+2],
                    color=color, alpha=0.6, linewidth=1.5)

        ax3.scatter(trajectory_x[0], trajectory_y[0], trajectory_z[0],
                   c='green', s=50, marker='o', edgecolors='black')
        ax3.scatter(trajectory_x[-1], trajectory_y[-1], trajectory_z[-1],
                   c='red', s=50, marker='*', edgecolors='black')

    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=9)
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=9)
    ax3.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})', fontsize=9)
    ax3.set_title('3D Trajectory View', fontsize=12, fontweight='bold')

    plt.tight_layout()

    output_file = output_dir / 'sample_trajectories.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nTrajectory visualization saved to: {output_file}")
    plt.close()

    # Create distance evolution plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Track distance from origin
    ax = axes[0]
    for sample_idx in sample_indices[:5]:
        distances = [np.linalg.norm(projected_by_epoch[e][sample_idx, :2])
                    for e in epochs]
        ax.plot(epochs, distances, marker='o', markersize=3, linewidth=1.5, alpha=0.7)

    if transition_epoch is not None:
        ax.axvline(transition_epoch, color='red', linestyle='--', linewidth=2, label='Transition')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Distance from Origin (L2)', fontsize=12)
    ax.set_title('Sample Movement from Origin', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Track pairwise distances
    ax = axes[1]
    sample_pairs = [(0, 1), (0, 2), (1, 2)] if len(sample_indices) >= 3 else [(0, 1)]

    for i, j in sample_pairs:
        if i < len(sample_indices) and j < len(sample_indices):
            distances = []
            for e in epochs:
                proj = projected_by_epoch[e]
                dist = np.linalg.norm(proj[sample_indices[i]] - proj[sample_indices[j]])
                distances.append(dist)
            ax.plot(epochs, distances, marker='o', markersize=3,
                   linewidth=1.5, alpha=0.7, label=f'Sample {i}-{j}')

    if transition_epoch is not None:
        ax.axvline(transition_epoch, color='red', linestyle='--', linewidth=2, label='Transition')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Pairwise Distance (L2)', fontsize=12)
    ax.set_title('Sample Separation Over Time', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = output_dir / 'distance_evolution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Distance evolution saved to: {output_file}")
    plt.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Visualize sample trajectories through grokking')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing captured states')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to track')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--transition_epoch', type=int, default=None,
                        help='Epoch to mark as key transition (e.g., grokking)')
    parser.add_argument('--pre_epoch', type=int, default=None,
                        help='Epoch for pre-transition comparison')
    parser.add_argument('--post_epoch', type=int, default=None,
                        help='Epoch for post-transition comparison')
    parser.add_argument('--auto_detect', action='store_true',
                        help='Auto-detect transition epochs from checkpoint')

    args = parser.parse_args()

    sample_indices = list(range(args.num_samples))

    # Auto-detect epochs if requested
    transition_epoch = args.transition_epoch
    pre_epoch = args.pre_epoch
    post_epoch = args.post_epoch

    if args.auto_detect:
        try:
            from detect_transition_epochs import suggest_analysis_epochs
            checkpoint_path = Path(args.results_dir) / 'checkpoint_latest.pt'
            result = suggest_analysis_epochs(checkpoint_path, verbose=True)

            if result['grokking_detected']:
                transition_epoch = result['grokking_epoch']
            elif result['transition_epochs']:
                transition_epoch = result['transition_epochs'][0][0]

            if result['all_epochs']:
                pre_epoch = result['all_epochs'][0]
                post_epoch = result['all_epochs'][-1]

            print(f"\nAuto-detected epochs:")
            print(f"  Transition: {transition_epoch}")
            print(f"  Pre-transition: {pre_epoch}")
            print(f"  Post-transition: {post_epoch}")
        except Exception as e:
            print(f"Warning: Auto-detection failed: {e}")
            print("Using default epoch selection")

    visualize_trajectories_pca(
        results_dir=Path(args.results_dir),
        sample_indices=sample_indices,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        transition_epoch=transition_epoch,
        pre_epoch=pre_epoch,
        post_epoch=post_epoch
    )
