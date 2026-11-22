#!/usr/bin/env python3
"""
One-off rerun of Appendix B's Betti-number analysis after fixing the h0/h1
computation bug in analyze_grokking_topology.py (the old code counted every dimension-0 persistence birth event, which
is always exactly N regardless of connectivity, rather than filtering for
intervals still alive at epsilon).

Recomputes encoder/decoder beta_1 at init (epoch 0) and final epoch, for the
same canonical seed directories used elsewhere in the paper's analysis
(matching run_full_factorization_batch_classification.sh's seed mapping),
using the SAME fixed epsilon convention (encoder=0.5, decoder=0.3) and the
SAME max_samples=200 subsampling as the original committed topology_metrics.json
files, so results are a like-for-like bug-fixed replacement, not a
methodology change.
"""

import sys
import glob
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from analyze_grokking_topology import (
    extract_encoder_representations,
    extract_decoder_representations,
    compute_topology_metrics,
)

MAX_SAMPLES = 200
EPS_ENCODER = 0.5
EPS_DECODER = 0.3
SEED = 42

REPO = Path(__file__).resolve().parent.parent
BASE = REPO / 'runs' / 'exp1'  # where train_exp1.sh writes run directories

CANONICAL_SEEDS = {
    'GNN': [
        'multi_seed_classification/gnn_seed1_results_8580377',
        'multi_seed_classification/gnn_seed2_results_8580377',
        'multi_seed_classification/gnn_seed3_results_8584516',
        'multi_seed_classification/gnn_seed4_results_8584516',
        'multi_seed_classification/gnn_seed5_results_8584516',
    ],
    'Transformer': [
        'multi_seed_classification/transformer_seed1_results_8580377',
        'multi_seed_classification/transformer_seed2_results_8580377',
        'multi_seed_classification/transformer_seed3_results_8584516',
        'multi_seed_classification/transformer_seed4_results_8584516',
        'multi_seed_classification/transformer_seed5_results_8584516',
    ],
    'Mamba': [
        'multi_seed_classification/mamba_seed1_results_9403834',
        'multi_seed_classification/mamba_seed2_results_9403834',
        'multi_seed_classification/mamba_seed3_results_8584516',
        'multi_seed_classification/mamba_seed4_results_8584516',
        'multi_seed_classification/mamba_seed5_results_8584516',
    ],
}

SCRAMBLE_30PCT_SEEDS = {
    'GNN': [f'classification_scrambled/gnn_seed{i}_results_9424022' for i in (1, 2, 3)],
    'Transformer': [f'classification_scrambled/transformer_seed{i}_results_9424022' for i in (1, 2, 3)],
    'Mamba': [f'classification_scrambled/mamba_seed{i}_results_9424022' for i in (1, 2, 3)],
}

FULL_SCRAMBLE_SEEDS = {
    'GNN': [f'classification_full_scrambled/gnn_scramble_seed{i}_results_9474028' for i in (1, 2, 3)],
    'Transformer': [f'classification_full_scrambled/transformer_scramble_seed{i}_results_9474028' for i in (1, 2, 3)],
    'Mamba': [f'classification_full_scrambled/mamba_scramble_seed{i}_results_9474028' for i in (1, 2, 3)],
}


def load_epoch_states(seed_dir: Path, epoch: int):
    f = seed_dir / 'captured_states' / f'states_epoch_{epoch}_val.pt'
    if not f.exists():
        return None
    return torch.load(f, map_location='cpu', weights_only=False)


def available_epochs(seed_dir: Path):
    files = sorted((seed_dir / 'captured_states').glob('states_epoch_*_val.pt'))
    epochs = []
    for f in files:
        try:
            epochs.append(int(f.stem.split('_')[2]))
        except (IndexError, ValueError):
            continue
    return sorted(epochs)


def analyze_seed(seed_dir: Path, rng: np.random.Generator):
    epochs = available_epochs(seed_dir)
    if not epochs:
        return None
    init_epoch, final_epoch = epochs[0], epochs[-1]

    result = {}
    for tag, epoch in [('init', init_epoch), ('final', final_epoch)]:
        states = load_epoch_states(seed_dir, epoch)
        if states is None:
            return None
        enc = extract_encoder_representations(states)
        dec = extract_decoder_representations(states)

        if enc.shape[0] > MAX_SAMPLES:
            idx = rng.choice(enc.shape[0], size=MAX_SAMPLES, replace=False)
            enc, dec = enc[idx], dec[idx]

        enc_m = compute_topology_metrics(enc, epsilon=EPS_ENCODER)
        dec_m = compute_topology_metrics(dec, epsilon=EPS_DECODER)
        result[tag] = {'epoch': epoch, 'encoder': enc_m, 'decoder': dec_m}
    return result


def aggregate(condition_name: str, seed_map: dict):
    print(f"\n{'='*70}\n{condition_name}\n{'='*70}")
    out = {}
    for arch, seed_dirs in seed_map.items():
        enc_init, enc_final, dec_init, dec_final = [], [], [], []
        for sd in seed_dirs:
            seed_dir = BASE / sd
            rng = np.random.default_rng(SEED)
            r = analyze_seed(seed_dir, rng)
            if r is None:
                print(f"  ✗ {sd}: no captured states")
                continue
            enc_init.append(r['init']['encoder']['h1_features'])
            enc_final.append(r['final']['encoder']['h1_features'])
            dec_init.append(r['init']['decoder']['h1_features'])
            dec_final.append(r['final']['decoder']['h1_features'])
            print(f"  ✓ {sd}: encoder beta1 {r['init']['encoder']['h1_features']}"
                  f"->{r['final']['encoder']['h1_features']} (epoch {r['init']['epoch']}->{r['final']['epoch']}), "
                  f"decoder beta1 {r['init']['decoder']['h1_features']}->{r['final']['decoder']['h1_features']}")

        def m_sem(vals):
            if not vals:
                return None, None
            return float(np.mean(vals)), float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0

        out[arch] = {
            'n_seeds': len(enc_init),
            'encoder_beta1_init':  m_sem(enc_init),
            'encoder_beta1_final': m_sem(enc_final),
            'decoder_beta1_init':  m_sem(dec_init),
            'decoder_beta1_final': m_sem(dec_final),
        }
    return out


def main():
    results = {
        'compositional': aggregate('COMPOSITIONAL (n=5 seeds)', CANONICAL_SEEDS),
        'scrambled_30pct': aggregate('30%-LABEL-NOISE (n=3 seeds)', SCRAMBLE_30PCT_SEEDS),
        'full_scramble': aggregate('FULL-SCRAMBLE (n=3 seeds)', FULL_SCRAMBLE_SEEDS),
    }

    print(f"\n{'='*70}\nSUMMARY (corrected beta_1, fixed epsilon=0.5/0.3, max_samples=200)\n{'='*70}")
    for condition, by_arch in results.items():
        print(f"\n-- {condition} --")
        for arch, d in by_arch.items():
            ei, ei_s = d['encoder_beta1_init']
            ef, ef_s = d['encoder_beta1_final']
            di, di_s = d['decoder_beta1_init']
            df, df_s = d['decoder_beta1_final']
            print(f"  {arch:12s} encoder: {ei:.1f}±{ei_s:.1f} -> {ef:.1f}±{ef_s:.1f}   "
                  f"decoder: {di:.1f}±{di_s:.1f} -> {df:.1f}±{df_s:.1f}  (n={d['n_seeds']})")

    out_path = BASE / 'betti_numbers_corrected.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
