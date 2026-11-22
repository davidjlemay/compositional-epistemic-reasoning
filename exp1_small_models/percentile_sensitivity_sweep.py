#!/usr/bin/env python3
"""
Percentile-sensitivity sweep for the fixed-epsilon validation (exp_3
classification), answering pm7p Q1: "How sensitive are the topology results
to the choice of epsilon percentile, sample size, and dimensionality
reduction?"

Reuses analyze_topology_fixed_epsilon_classification.py's exact methodology
(freeze epsilon at its value at epoch 0, apply the same threshold at later
epochs, encoder representations only, max_samples=200) but sweeps the
percentile used to set that frozen epsilon across {10, 20, 30, 40, 50}
instead of hardcoding the 20th, on one architecture across all 5 canonical
seeds.

Usage:
    python percentile_sensitivity_sweep.py --arch GNN
"""

import argparse
from pathlib import Path

import numpy as np

from analyze_topology_fixed_epsilon_classification import (
    ARCH_PREFIXES,
    find_seed_dirs,
    list_val_epochs,
    subsample_epochs,
    load_representations,
    pairwise_distances,
    compute_vr_complex,
)

PERCENTILES = [10, 20, 30, 40, 50]
N_EPOCHS = 20
MAX_SAMPLES = 200


def analyze_seed_at_percentile(seed_dir: Path, percentile: int):
    states_dir = seed_dir / 'captured_states'
    all_epochs = list_val_epochs(states_dir)
    if not all_epochs:
        return None
    epochs = subsample_epochs(all_epochs, N_EPOCHS)

    repr0 = load_representations(states_dir, epochs[0], MAX_SAMPLES)
    eps_fixed = float(np.percentile(pairwise_distances(repr0), percentile))

    init_topo = compute_vr_complex(repr0, eps_fixed)
    repr_final = load_representations(states_dir, epochs[-1], MAX_SAMPLES)
    final_topo = compute_vr_complex(repr_final, eps_fixed)

    init_s = init_topo['total_simplices']
    final_s = final_topo['total_simplices']
    reduction = (init_s - final_s) / max(init_s, 1) * 100
    return {
        'eps_fixed': eps_fixed,
        'init_simplices': init_s,
        'final_simplices': final_s,
        'reduction_pct': reduction,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='GNN', choices=list(ARCH_PREFIXES))
    parser.add_argument('--multi_seed_dir', type=str, default='runs/exp1/multi_seed_classification')
    args = parser.parse_args()

    multi_seed_dir = Path(args.multi_seed_dir)
    seed_dirs = find_seed_dirs(multi_seed_dir, ARCH_PREFIXES[args.arch])
    print(f"Architecture: {args.arch}  ({len(seed_dirs)} seeds)")

    results = {}
    for pct in PERCENTILES:
        print(f"\n--- Percentile {pct} ---")
        reductions = []
        for sd in seed_dirs:
            r = analyze_seed_at_percentile(sd, pct)
            if r is None:
                continue
            reductions.append(r['reduction_pct'])
            print(f"  {sd.name}: eps={r['eps_fixed']:.4f}  "
                  f"{r['init_simplices']}->{r['final_simplices']}  "
                  f"reduction={r['reduction_pct']:.1f}%")
        mean_r = float(np.mean(reductions))
        sem_r = float(np.std(reductions, ddof=1) / np.sqrt(len(reductions))) if len(reductions) > 1 else 0.0
        results[pct] = (mean_r, sem_r, len(reductions))
        print(f"  Mean: {mean_r:.1f}% +/- {sem_r:.1f}%  (n={len(reductions)})")

    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.arch} fixed-epsilon reduction vs. percentile choice")
    print(f"{'='*60}")
    for pct, (mean_r, sem_r, n) in results.items():
        print(f"  {pct:3d}th percentile: {mean_r:.1f}% +/- {sem_r:.1f}%  (n={n})")


if __name__ == '__main__':
    main()
