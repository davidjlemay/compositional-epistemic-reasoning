#!/usr/bin/env python3
"""
Full Product Space Factorization Analysis: Multi-Level Van Benthem Test

Tests factorization at TWO levels:

LEVEL 1: Task-level (original test)
    X ≈ X_n_agents × X_depth

LEVEL 2: Agent-level (Van Benthem's true structure)
    For fixed n_agents, test:
    X ≈ X_agent0 × X_agent1 × ... × X_agentn × X_depth

This addresses the critical review concern that we were only testing
scalar factorization (n_agents × depth) rather than true multi-agent
product topology (agent beliefs × ... × depth).

Key Questions:
1. TASK LEVEL: Do ICA components separate n_agents and depth?
2. AGENT LEVEL: Within fixed n_agents, do components separate individual agents?
3. Is agent-level structure present at initialization?
4. Is it consistent across GNN, Transformer, and Mamba?
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import re
from scipy.stats import pearsonr, spearmanr, f_oneway
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from collections import defaultdict
import argparse
import seaborn as sns


def load_embeddings_and_metadata(states_dir: Path, dataset_path: Path,
                                   epoch: int, split: str = 'val'):
    """
    Load embeddings and extract task-level AND agent-level metadata.

    Returns:
        embeddings: np.ndarray [n_samples, hidden_dim]
        metadata: dict with 'n_agents', 'hypothesis_depth', 'agent_observations', etc.
    """
    # Load hidden states
    state_file = states_dir / f'states_epoch_{epoch}_{split}.pt'
    if not state_file.exists():
        raise FileNotFoundError(f"State file not found: {state_file}")

    print(f"Loading states from: {state_file}")
    states_data = torch.load(state_file, map_location='cpu', weights_only=False)

    # Extract encoder states
    if 'hidden_states' in states_data:
        states = states_data['hidden_states'].cpu().numpy()
    elif 'encoder_states' in states_data:
        encoder_states = states_data['encoder_states']
        if isinstance(encoder_states, torch.Tensor):
            states = encoder_states.cpu().numpy()
        else:
            states = np.array([s.cpu().numpy() if isinstance(s, torch.Tensor) else s
                              for s in encoder_states])
    else:
        raise ValueError(f"Unknown states format. Keys: {list(states_data.keys())}")

    # Average over sequence if needed
    if states.ndim == 3:
        states = states.mean(axis=1)

    embeddings = states
    print(f"Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")

    # Load dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    # Extract metadata
    metadata = {
        'n_agents': [],
        'hypothesis_depth': [],
        'agent_observations': [],  # NEW: list of per-agent observations
        'agent_names': [],  # NEW: actual agent names
    }

    for i in range(len(embeddings)):
        item = dataset[i]

        # Get task-level metadata
        n_agents = item.get('n_agents', 0)
        depth = item.get('hypothesis_depth', 0)

        metadata['n_agents'].append(n_agents)
        metadata['hypothesis_depth'].append(depth)

        # Get agent-level metadata (from V4 format)
        agent_obs = item.get('agent_observations', [])
        agent_names = item.get('agent_names', [])

        metadata['agent_observations'].append(agent_obs)
        metadata['agent_names'].append(agent_names)

    # Convert to arrays
    metadata['n_agents'] = np.array(metadata['n_agents'])
    metadata['hypothesis_depth'] = np.array(metadata['hypothesis_depth'])

    print(f"  n_agents range: {metadata['n_agents'].min()}-{metadata['n_agents'].max()}")
    print(f"  hypothesis_depth range: {metadata['hypothesis_depth'].min()}-{metadata['hypothesis_depth'].max()}")

    # Check if agent-level metadata is available
    has_agent_data = any(len(obs) > 0 for obs in metadata['agent_observations'])
    print(f"  Agent-level data available: {has_agent_data}")

    return embeddings, metadata


def apply_ica_full(embeddings: np.ndarray, n_components: int, random_state: int = 42):
    """Apply ICA to discover independent factors."""
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    print(f"\nApplying ICA with {n_components} components...")
    ica = FastICA(n_components=n_components, random_state=random_state, max_iter=2000, tol=1e-4)
    components = ica.fit_transform(embeddings_scaled)

    print(f"  ICA converged in {ica.n_iter_} iterations")

    return components, ica


def test_task_level_factorization(components: np.ndarray, metadata: dict,
                                    architecture: str) -> dict:
    """
    Test TASK-LEVEL factorization: X ≈ X_n_agents × X_depth

    Returns:
        results: dict with correlations and R² scores
    """
    n_agents = metadata['n_agents']
    hypothesis_depth = metadata['hypothesis_depth']

    print(f"\n{'='*70}")
    print(f"{architecture}: TASK-LEVEL Factorization (n_agents × depth)")
    print(f"{'='*70}")

    component_results = []

    for comp_idx in range(components.shape[1]):
        comp_values = components[:, comp_idx]

        # Correlation with n_agents
        if len(np.unique(n_agents)) > 1:
            r_agents, p_agents = pearsonr(comp_values, n_agents)
            r_agents_spearman, _ = spearmanr(comp_values, n_agents)
        else:
            r_agents, p_agents = 0.0, 1.0
            r_agents_spearman = 0.0

        # Correlation with hypothesis_depth
        if len(np.unique(hypothesis_depth)) > 1:
            r_depth, p_depth = pearsonr(comp_values, hypothesis_depth)
            r_depth_spearman, _ = spearmanr(comp_values, hypothesis_depth)
        else:
            r_depth, p_depth = 0.0, 1.0
            r_depth_spearman = 0.0

        # Classify component
        abs_r_agents = abs(r_agents)
        abs_r_depth = abs(r_depth)

        if abs_r_agents > 0.3 and abs_r_depth < 0.15:
            component_type = "N_AGENTS-SPECIFIC"
        elif abs_r_depth > 0.3 and abs_r_agents < 0.15:
            component_type = "DEPTH-SPECIFIC"
        elif abs_r_agents > 0.3 and abs_r_depth > 0.3:
            component_type = "MIXED"
        else:
            component_type = "UNCLEAR"

        result = {
            'component': comp_idx + 1,
            'agents_correlation': float(r_agents),
            'agents_p': float(p_agents),
            'depth_correlation': float(r_depth),
            'depth_p': float(p_depth),
            'type': component_type
        }

        component_results.append(result)

        print(f"\nComponent {comp_idx + 1}:")
        print(f"  n_agents: r={r_agents:+.3f} (p={p_agents:.2e})")
        print(f"  depth:    r={r_depth:+.3f} (p={p_depth:.2e})")
        print(f"  → Type: {component_type}")

    # Ridge regression quality
    ridge_agents = Ridge(alpha=1.0)
    ridge_agents.fit(components, n_agents)
    r2_agents = r2_score(n_agents, ridge_agents.predict(components))

    ridge_depth = Ridge(alpha=1.0)
    ridge_depth.fit(components, hypothesis_depth)
    r2_depth = r2_score(hypothesis_depth, ridge_depth.predict(components))

    print(f"\nTask-level factorization quality:")
    print(f"  R²(n_agents): {r2_agents:.4f}")
    print(f"  R²(depth): {r2_depth:.4f}")

    return {
        'architecture': architecture,
        'level': 'task',
        'component_results': component_results,
        'r2_agents': float(r2_agents),
        'r2_depth': float(r2_depth),
    }


def encode_agent_observations(agent_obs_list):
    """
    Encode agent observations as numeric features.

    agent_obs_list: list of sets/lists, e.g., [[0, 1], [2], [0, 1, 2]]
    Returns: array of encoded features (e.g., sum of observed variables)
    """
    features = []
    for obs in agent_obs_list:
        if isinstance(obs, (list, tuple, set)):
            # Simple encoding: sum of observed variable indices
            feature = sum(obs) if len(obs) > 0 else 0
        else:
            feature = 0
        features.append(feature)
    return np.array(features)


def test_agent_level_factorization(components: np.ndarray, metadata: dict,
                                     architecture: str, n_agents_fixed: int) -> dict:
    """
    Test AGENT-LEVEL factorization for fixed n_agents.

    For problems with n_agents=n_agents_fixed, test:
        X ≈ X_agent0 × X_agent1 × ... × X_depth

    Returns:
        results: dict with per-agent correlations and R² scores
    """
    # Filter to fixed n_agents
    mask = metadata['n_agents'] == n_agents_fixed
    if mask.sum() == 0:
        print(f"\n  No samples with n_agents={n_agents_fixed}")
        return None

    print(f"\n{'='*70}")
    print(f"{architecture}: AGENT-LEVEL Factorization (n_agents={n_agents_fixed})")
    print(f"{'='*70}")
    print(f"  Testing on {mask.sum()} samples")

    # Get subset
    components_subset = components[mask]
    depth_subset = metadata['hypothesis_depth'][mask]
    agent_obs_subset = [metadata['agent_observations'][i] for i in range(len(mask)) if mask[i]]

    # Check if we have agent-level data
    if not agent_obs_subset or all(len(obs) == 0 for obs in agent_obs_subset):
        print(f"  ⚠️  No agent-level observation data available")
        return None

    # Extract per-agent observation features
    agent_features = []
    for agent_idx in range(n_agents_fixed):
        agent_obs = [obs[agent_idx] if agent_idx < len(obs) else []
                     for obs in agent_obs_subset]
        agent_feature = encode_agent_observations(agent_obs)
        agent_features.append(agent_feature)

    # Test component correlations with each agent
    component_agent_correlations = []

    for comp_idx in range(components_subset.shape[1]):
        comp_values = components_subset[:, comp_idx]

        agent_corrs = []
        for agent_idx, agent_feature in enumerate(agent_features):
            if len(np.unique(agent_feature)) > 1:
                r, p = pearsonr(comp_values, agent_feature)
            else:
                r, p = 0.0, 1.0
            agent_corrs.append({'agent': agent_idx, 'r': float(r), 'p': float(p)})

        # Correlation with depth
        if len(np.unique(depth_subset)) > 1:
            r_depth, p_depth = pearsonr(comp_values, depth_subset)
        else:
            r_depth, p_depth = 0.0, 1.0

        component_agent_correlations.append({
            'component': comp_idx + 1,
            'agent_correlations': agent_corrs,
            'depth_correlation': float(r_depth),
            'depth_p': float(p_depth)
        })

        # Print summary
        print(f"\nComponent {comp_idx + 1}:")
        for ac in agent_corrs:
            print(f"  Agent{ac['agent']}: r={ac['r']:+.3f} (p={ac['p']:.2e})")
        print(f"  Depth:    r={r_depth:+.3f} (p={p_depth:.2e})")

    # Test ridge regression for each agent
    agent_r2_scores = []
    for agent_idx, agent_feature in enumerate(agent_features):
        ridge = Ridge(alpha=1.0)
        ridge.fit(components_subset, agent_feature)
        r2 = r2_score(agent_feature, ridge.predict(components_subset))
        agent_r2_scores.append(r2)
        print(f"\nAgent{agent_idx} prediction: R² = {r2:.4f}")

    # Test depth prediction
    ridge_depth = Ridge(alpha=1.0)
    ridge_depth.fit(components_subset, depth_subset)
    r2_depth = r2_score(depth_subset, ridge_depth.predict(components_subset))
    print(f"Depth prediction: R² = {r2_depth:.4f}")

    # Joint prediction test
    targets = np.column_stack(agent_features + [depth_subset])
    ridge_joint = Ridge(alpha=1.0)
    ridge_joint.fit(components_subset, targets)
    preds_joint = ridge_joint.predict(components_subset)

    r2_joint = []
    for i in range(n_agents_fixed):
        r2 = r2_score(agent_features[i], preds_joint[:, i])
        r2_joint.append(r2)
    r2_joint_depth = r2_score(depth_subset, preds_joint[:, -1])

    print(f"\nJoint prediction:")
    for i, r2 in enumerate(r2_joint):
        print(f"  Agent{i}: R² = {r2:.4f}")
    print(f"  Depth: R² = {r2_joint_depth:.4f}")

    # Verdict
    mean_agent_r2 = np.mean(agent_r2_scores)
    if mean_agent_r2 > 0.5 and r2_depth > 0.5:
        verdict = "✓ AGENT-LEVEL FACTORIZATION CONFIRMED"
    elif mean_agent_r2 > 0.3 or r2_depth > 0.3:
        verdict = "⚠ PARTIAL AGENT-LEVEL FACTORIZATION"
    else:
        verdict = "✗ NO AGENT-LEVEL FACTORIZATION"

    print(f"\n{verdict}")
    print(f"  Mean agent R²: {mean_agent_r2:.4f}")
    print(f"  Depth R²: {r2_depth:.4f}")

    return {
        'architecture': architecture,
        'level': 'agent',
        'n_agents_fixed': n_agents_fixed,
        'n_samples': int(mask.sum()),
        'component_agent_correlations': component_agent_correlations,
        'agent_r2_scores': [float(r2) for r2 in agent_r2_scores],
        'depth_r2': float(r2_depth),
        'joint_agent_r2': [float(r2) for r2 in r2_joint],
        'joint_depth_r2': float(r2_joint_depth),
        'mean_agent_r2': float(mean_agent_r2),
        'verdict': verdict
    }


def visualize_multi_level_factorization(
    gnn_task_results, transformer_task_results, mamba_task_results,
    gnn_agent_results_by_n, transformer_agent_results_by_n, mamba_agent_results_by_n,
    output_dir: Path
):
    """Create visualization of both task-level and agent-level factorization."""

    fig = plt.figure(figsize=(20, 14))

    # TASK-LEVEL VISUALIZATION (top row)
    architectures_task = [
        ('GNN', gnn_task_results),
        ('Transformer', transformer_task_results),
        ('Mamba', mamba_task_results)
    ]

    for arch_idx, (name, task_results) in enumerate(architectures_task):
        ax = fig.add_subplot(3, 4, arch_idx + 1)

        # Plot factorization quality
        r2_agents = task_results['r2_agents']
        r2_depth = task_results['r2_depth']

        metrics = ['R²(n_agents)', 'R²(depth)']
        values = [r2_agents, r2_depth]
        colors = ['#2E86AB', '#A23B72']

        bars = ax.bar(range(len(metrics)), values, color=colors)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics)
        ax.set_ylabel('R² Score')
        ax.set_title(f'{name}: Task-Level\n(n_agents × depth)', fontweight='bold')
        ax.set_ylim([0, 1.0])
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)

    # AGENT-LEVEL VISUALIZATION (middle and bottom rows)
    # For each architecture, show agent-level results for different n_agents values

    all_agent_results = [
        ('GNN', gnn_agent_results_by_n),
        ('Transformer', transformer_agent_results_by_n),
        ('Mamba', mamba_agent_results_by_n)
    ]

    for arch_idx, (name, agent_results_dict) in enumerate(all_agent_results):
        # Plot 1: Mean agent R² across n_agents values
        ax1 = fig.add_subplot(3, 4, arch_idx + 5)

        if agent_results_dict:
            n_agents_vals = sorted(agent_results_dict.keys())
            mean_r2s = [agent_results_dict[n]['mean_agent_r2'] for n in n_agents_vals]
            depth_r2s = [agent_results_dict[n]['depth_r2'] for n in n_agents_vals]

            x = np.arange(len(n_agents_vals))
            width = 0.35

            ax1.bar(x - width/2, mean_r2s, width, label='Mean Agent R²', color='#F18F01')
            ax1.bar(x + width/2, depth_r2s, width, label='Depth R²', color='#C73E1D')

            ax1.set_xlabel('n_agents')
            ax1.set_ylabel('R² Score')
            ax1.set_title(f'{name}: Agent-Level R²', fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(n_agents_vals)
            ax1.set_ylim([0, 1.0])
            ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No agent-level data',
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title(f'{name}: Agent-Level', fontweight='bold')

        # Plot 2: Per-agent breakdown for first n_agents value
        ax2 = fig.add_subplot(3, 4, arch_idx + 9)

        if agent_results_dict:
            first_n = min(agent_results_dict.keys())
            agent_r2s = agent_results_dict[first_n]['agent_r2_scores']

            agents = [f'Agent{i}' for i in range(len(agent_r2s))]
            ax2.bar(range(len(agents)), agent_r2s, color='#2E86AB')
            ax2.set_xlabel('Agent')
            ax2.set_ylabel('R² Score')
            ax2.set_title(f'{name}: Per-Agent R² (n={first_n})', fontweight='bold')
            ax2.set_xticks(range(len(agents)))
            ax2.set_xticklabels(agents)
            ax2.set_ylim([0, 1.0])
            ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            ax2.grid(axis='y', alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No agent data',
                    ha='center', va='center', transform=ax2.transAxes)

    plt.suptitle('Multi-Level Van Benthem Product Space Analysis\n' +
                 'Top: Task-Level (n_agents × depth) | Middle/Bottom: Agent-Level (agent₀ × agent₁ × ... × depth)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_file = output_dir / 'multi_level_factorization_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Multi-level product space factorization analysis')
    parser.add_argument('--gnn_dir', type=str, required=True)
    parser.add_argument('--transformer_dir', type=str, required=True)
    parser.add_argument('--mamba_dir', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--epoch', type=int, default=0,
                        help='Epoch to analyze (default: 0 for initialization)')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'])
    parser.add_argument('--n_components', type=int, default=10,
                        help='Number of ICA components')
    parser.add_argument('--output_dir', type=str, default='multi_level_factorization')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("="*70)
    print("Multi-Level Van Benthem Product Space Analysis")
    print("="*70)
    print(f"Epoch: {args.epoch}")
    print(f"Components: {args.n_components}")

    # Load all architectures
    print("\n" + "="*70)
    print("LOADING EMBEDDINGS")
    print("="*70)

    gnn_embeddings, gnn_metadata = load_embeddings_and_metadata(
        Path(args.gnn_dir) / 'captured_states', Path(args.dataset_path),
        args.epoch, args.split
    )

    transformer_embeddings, transformer_metadata = load_embeddings_and_metadata(
        Path(args.transformer_dir) / 'captured_states', Path(args.dataset_path),
        args.epoch, args.split
    )

    mamba_embeddings, mamba_metadata = load_embeddings_and_metadata(
        Path(args.mamba_dir) / 'captured_states', Path(args.dataset_path),
        args.epoch, args.split
    )

    # Apply ICA
    print("\n" + "="*70)
    print("APPLYING ICA")
    print("="*70)

    gnn_components, _ = apply_ica_full(gnn_embeddings, args.n_components)
    transformer_components, _ = apply_ica_full(transformer_embeddings, args.n_components)
    mamba_components, _ = apply_ica_full(mamba_embeddings, args.n_components)

    # LEVEL 1: Task-level factorization
    print("\n" + "="*70)
    print("LEVEL 1: TASK-LEVEL FACTORIZATION (n_agents × depth)")
    print("="*70)

    gnn_task_results = test_task_level_factorization(gnn_components, gnn_metadata, 'GNN')
    transformer_task_results = test_task_level_factorization(transformer_components, transformer_metadata, 'Transformer')
    mamba_task_results = test_task_level_factorization(mamba_components, mamba_metadata, 'Mamba')

    # LEVEL 2: Agent-level factorization
    print("\n" + "="*70)
    print("LEVEL 2: AGENT-LEVEL FACTORIZATION (agent₀ × agent₁ × ... × depth)")
    print("="*70)

    # Test for each n_agents value present in data
    unique_n_agents = sorted(set(gnn_metadata['n_agents']))
    print(f"\nTesting agent-level factorization for n_agents: {unique_n_agents}")

    gnn_agent_results_by_n = {}
    transformer_agent_results_by_n = {}
    mamba_agent_results_by_n = {}

    for n_agents_val in unique_n_agents:
        # Convert numpy int64 to Python int for JSON serialization
        n_agents_val = int(n_agents_val)
        print(f"\n{'='*70}")
        print(f"Testing n_agents = {n_agents_val}")
        print(f"{'='*70}")

        gnn_agent = test_agent_level_factorization(gnn_components, gnn_metadata, 'GNN', n_agents_val)
        transformer_agent = test_agent_level_factorization(transformer_components, transformer_metadata, 'Transformer', n_agents_val)
        mamba_agent = test_agent_level_factorization(mamba_components, mamba_metadata, 'Mamba', n_agents_val)

        if gnn_agent:
            gnn_agent_results_by_n[n_agents_val] = gnn_agent
        if transformer_agent:
            transformer_agent_results_by_n[n_agents_val] = transformer_agent
        if mamba_agent:
            mamba_agent_results_by_n[n_agents_val] = mamba_agent

    # Save all results
    all_results = {
        'epoch': args.epoch,
        'split': args.split,
        'n_components': args.n_components,
        'task_level': {
            'GNN': gnn_task_results,
            'Transformer': transformer_task_results,
            'Mamba': mamba_task_results
        },
        'agent_level': {
            'GNN': gnn_agent_results_by_n,
            'Transformer': transformer_agent_results_by_n,
            'Mamba': mamba_agent_results_by_n
        }
    }

    results_file = output_dir / 'multi_level_factorization_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {results_file}")

    # Visualize
    visualize_multi_level_factorization(
        gnn_task_results, transformer_task_results, mamba_task_results,
        gnn_agent_results_by_n, transformer_agent_results_by_n, mamba_agent_results_by_n,
        output_dir
    )

    # Final summary
    print("\n" + "="*70)
    print("MULTI-LEVEL VAN BENTHEM VALIDATION SUMMARY")
    print("="*70)

    print("\nTASK-LEVEL (n_agents × depth):")
    for name, results in [('GNN', gnn_task_results),
                          ('Transformer', transformer_task_results),
                          ('Mamba', mamba_task_results)]:
        print(f"  {name}: R²(n_agents)={results['r2_agents']:.3f}, R²(depth)={results['r2_depth']:.3f}")

    print("\nAGENT-LEVEL (agent₀ × agent₁ × ... × depth):")
    for n_agents_val in unique_n_agents:
        print(f"\n  n_agents={n_agents_val}:")
        for name, results_dict in [('GNN', gnn_agent_results_by_n),
                                   ('Transformer', transformer_agent_results_by_n),
                                   ('Mamba', mamba_agent_results_by_n)]:
            if n_agents_val in results_dict:
                r = results_dict[n_agents_val]
                print(f"    {name}: Mean Agent R²={r['mean_agent_r2']:.3f}, {r['verdict']}")

    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
