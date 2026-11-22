#!/usr/bin/env python3
"""
Utility to detect interesting transition epochs from training curves.

Useful when grokking is not formally detected, but we still want to identify
key learning phases for topological analysis.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from scipy.ndimage import gaussian_filter1d


def load_metrics_from_checkpoint(checkpoint_path: Path) -> Dict:
    """Load training metrics from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    return {
        'epoch': checkpoint['epoch'],
        'grokking_detected': checkpoint.get('grokking_detected', False),
        'grokking_epoch': checkpoint.get('grokking_epoch', None),
        'train_metrics': checkpoint.get('train_metrics', []),
        'val_metrics': checkpoint.get('val_metrics', [])
    }


def compute_gradient_magnitude(values: List[float], smooth_sigma: float = 2.0) -> np.ndarray:
    """
    Compute the gradient magnitude of a metric curve.

    Args:
        values: List of metric values over epochs
        smooth_sigma: Gaussian smoothing sigma (higher = smoother)

    Returns:
        Array of gradient magnitudes
    """
    if len(values) < 3:
        return np.zeros(len(values))

    values_array = np.array(values)

    # Smooth the curve to reduce noise
    smoothed = gaussian_filter1d(values_array, sigma=smooth_sigma)

    # Compute gradient (rate of change)
    gradient = np.gradient(smoothed)

    # Return absolute magnitude of gradient
    return np.abs(gradient)


def find_steepest_transitions(
    metrics: Dict,
    metric_name: str = 'loss',
    split: str = 'val',
    num_transitions: int = 3,
    smooth_sigma: float = 2.0,
    min_epoch_gap: int = 50
) -> List[Tuple[int, float]]:
    """
    Find epochs with steepest changes in a given metric.

    Args:
        metrics: Dictionary with train_metrics and val_metrics
        metric_name: Which metric to analyze ('loss', 'accuracy', etc.)
        split: 'train' or 'val'
        num_transitions: How many transition points to return
        smooth_sigma: Smoothing parameter for gradient computation
        min_epoch_gap: Minimum epoch separation between transitions

    Returns:
        List of (epoch, gradient_magnitude) tuples sorted by magnitude (descending)
    """
    metrics_list = metrics[f'{split}_metrics']

    if not metrics_list:
        return []

    # Extract the specific metric values
    # Handle both list of dicts and dict of lists formats
    if isinstance(metrics_list, list):
        values = [m.get(metric_name, 0) for m in metrics_list]
    elif isinstance(metrics_list, dict):
        values = metrics_list.get(metric_name, [])
    else:
        print(f"Warning: Unexpected metrics format: {type(metrics_list)}")
        return []

    if len(values) < 3:
        print(f"Warning: Not enough data points ({len(values)}) for transition detection")
        return []

    # Compute gradient magnitude
    gradient_mag = compute_gradient_magnitude(values, smooth_sigma=smooth_sigma)

    # Find peaks in gradient (steepest changes)
    epochs = list(range(len(values)))

    # Create list of (epoch, gradient) pairs
    candidates = [(epoch, grad) for epoch, grad in zip(epochs, gradient_mag)]

    # Sort by gradient magnitude (descending)
    candidates.sort(key=lambda x: x[1], reverse=True)

    # Select top transitions with minimum gap constraint
    selected = []
    for epoch, grad in candidates:
        # Check if this epoch is far enough from already selected ones
        if all(abs(epoch - sel_epoch) >= min_epoch_gap for sel_epoch, _ in selected):
            selected.append((epoch, grad))
            if len(selected) >= num_transitions:
                break

    # Sort by epoch for easier interpretation
    selected.sort(key=lambda x: x[0])

    return selected


def suggest_analysis_epochs(
    checkpoint_path: Path,
    num_epochs: int = 3,
    include_endpoints: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Suggest epochs for topological analysis based on training dynamics.

    Strategy:
    1. If grokking detected, use grokking_epoch
    2. Otherwise, find steepest transitions in validation loss/accuracy
    3. Include early and late training epochs for comparison

    Args:
        checkpoint_path: Path to checkpoint file
        num_epochs: Number of transition epochs to identify
        include_endpoints: Whether to include first/last epochs
        verbose: Print detailed information

    Returns:
        Dictionary with suggested epochs and metadata
    """
    checkpoint_path = Path(checkpoint_path)

    if verbose:
        print("=" * 70)
        print("Detecting Transition Epochs for Topological Analysis")
        print("=" * 70)

    # Load metrics
    metrics = load_metrics_from_checkpoint(checkpoint_path)

    total_epochs = metrics['epoch']
    grokking_detected = metrics['grokking_detected']
    grokking_epoch = metrics['grokking_epoch']

    if verbose:
        print(f"\nTraining Summary:")
        print(f"  Total epochs: {total_epochs}")
        print(f"  Grokking detected: {grokking_detected}")
        if grokking_detected:
            print(f"  Grokking epoch: {grokking_epoch}")

    result = {
        'total_epochs': total_epochs,
        'grokking_detected': grokking_detected,
        'grokking_epoch': grokking_epoch,
        'suggested_epochs': [],
        'transition_epochs': [],
        'early_epoch': None,
        'late_epoch': None
    }

    # Strategy 1: Use grokking epoch if available
    if grokking_detected and grokking_epoch is not None:
        if verbose:
            print(f"\nUsing detected grokking epoch: {grokking_epoch}")

        # Pre-grokking: ~100 epochs before or 1/2 way to grokking
        pre_grok = max(0, grokking_epoch - 100)
        if pre_grok < 100:
            pre_grok = grokking_epoch // 2

        # Post-grokking: end of training
        post_grok = total_epochs

        result['suggested_epochs'] = [pre_grok, grokking_epoch, post_grok]
        result['transition_epochs'] = [(grokking_epoch, 'grokking')]

        if include_endpoints:
            result['early_epoch'] = 0
            result['late_epoch'] = total_epochs

    else:
        # Strategy 2: Find steepest transitions
        if verbose:
            print(f"\nNo grokking detected. Analyzing training curves...")

        # Try validation loss first (most common indicator)
        transitions_loss = find_steepest_transitions(
            metrics, metric_name='loss', split='val',
            num_transitions=num_epochs, smooth_sigma=3.0, min_epoch_gap=50
        )

        # Try validation accuracy as backup
        transitions_acc = find_steepest_transitions(
            metrics, metric_name='accuracy', split='val',
            num_transitions=num_epochs, smooth_sigma=3.0, min_epoch_gap=50
        )

        if verbose:
            print(f"\nSteepest validation loss transitions:")
            for epoch, grad in transitions_loss:
                print(f"  Epoch {epoch}: gradient magnitude = {grad:.6f}")

            if transitions_acc:
                print(f"\nSteepest validation accuracy transitions:")
                for epoch, grad in transitions_acc:
                    print(f"  Epoch {epoch}: gradient magnitude = {grad:.6f}")

        # Combine and deduplicate
        all_transitions = transitions_loss + transitions_acc
        # Remove duplicates within 20 epochs
        unique_transitions = []
        for epoch, grad in sorted(all_transitions, key=lambda x: x[0]):
            if not any(abs(epoch - e) < 20 for e, _ in unique_transitions):
                unique_transitions.append((epoch, grad))

        # Sort by epoch
        unique_transitions.sort(key=lambda x: x[0])

        # Take top num_epochs transitions
        result['transition_epochs'] = unique_transitions[:num_epochs]
        result['suggested_epochs'] = [e for e, _ in result['transition_epochs']]

        # Add early and late epochs for comparison
        if include_endpoints:
            early_epoch = min(50, total_epochs // 10)
            late_epoch = total_epochs

            result['early_epoch'] = early_epoch
            result['late_epoch'] = late_epoch

            # Ensure we have at least 2-3 comparison points
            if len(result['suggested_epochs']) < 2:
                result['suggested_epochs'] = [early_epoch, late_epoch]

    # Ensure epochs are sorted and unique
    all_epochs = result['suggested_epochs'].copy()
    if include_endpoints:
        if result['early_epoch'] is not None:
            all_epochs.append(result['early_epoch'])
        if result['late_epoch'] is not None:
            all_epochs.append(result['late_epoch'])

    all_epochs = sorted(list(set(all_epochs)))
    result['all_epochs'] = all_epochs

    if verbose:
        print(f"\n{'='*70}")
        print("Suggested Epochs for Analysis:")
        print(f"{'='*70}")
        print(f"All epochs: {all_epochs}")
        print(f"\nRecommended usage:")
        if len(all_epochs) >= 2:
            print(f"  Pre-transition epoch:  {all_epochs[0]}")
            print(f"  Post-transition epoch: {all_epochs[-1]}")
        if result['transition_epochs']:
            print(f"\nKey transition epochs:")
            for epoch, info in result['transition_epochs']:
                print(f"  Epoch {epoch}")

    return result


def save_suggested_epochs(checkpoint_path: Path, output_path: Optional[Path] = None):
    """
    Detect and save suggested epochs to a JSON file.

    Args:
        checkpoint_path: Path to checkpoint
        output_path: Where to save results (default: same dir as checkpoint)
    """
    checkpoint_path = Path(checkpoint_path)

    if output_path is None:
        output_path = checkpoint_path.parent / 'suggested_analysis_epochs.json'

    result = suggest_analysis_epochs(checkpoint_path, verbose=True)

    # Convert to JSON-serializable format
    result_json = {
        'total_epochs': int(result['total_epochs']),
        'grokking_detected': bool(result['grokking_detected']),
        'grokking_epoch': int(result['grokking_epoch']) if result['grokking_epoch'] else None,
        'all_epochs': [int(e) for e in result['all_epochs']],
        'suggested_epochs': [int(e) for e in result['suggested_epochs']],
        'transition_epochs': [(int(e), float(g)) if isinstance(g, (int, float)) else (int(e), str(g))
                              for e, g in result['transition_epochs']],
        'early_epoch': int(result['early_epoch']) if result['early_epoch'] is not None else None,
        'late_epoch': int(result['late_epoch']) if result['late_epoch'] is not None else None
    }

    with open(output_path, 'w') as f:
        json.dump(result_json, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return result_json


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Detect transition epochs for topological analysis'
    )
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to checkpoint file (checkpoint_latest.pt)')
    parser.add_argument('--num_transitions', type=int, default=3,
                        help='Number of transition epochs to identify')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Where to save results JSON')
    parser.add_argument('--no_endpoints', action='store_true',
                        help='Do not include first/last epochs')

    args = parser.parse_args()

    save_suggested_epochs(
        checkpoint_path=Path(args.checkpoint_path),
        output_path=Path(args.output_path) if args.output_path else None
    )
