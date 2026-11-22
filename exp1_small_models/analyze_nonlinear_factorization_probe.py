#!/usr/bin/env python3
"""
Nonlinear-probe control for product factorization: checks whether a nonlinear
(MLP) probe recovers structure that the linear cross-fiber probe misses.

Reuses the exact fiber-conditional factorization test that produced Table 3
(fix hypothesis_depth, apply FastICA to encoder states, predict n_agents),
matching analyze_full_product_factorization_agent_aware.py's methodology
precisely, but compares TWO regressors instead of only Ridge:
  - Ridge (linear) -- the original method
  - MLPRegressor (small 1-hidden-layer network) -- nonlinear

(An RBF KernelRidge probe was also tried and dropped: with default
hyperparameters it scored consistently and substantially negative R^2 across
every condition, indicating a poorly-specified probe -- alpha/gamma would
need a proper grid search to be a fair comparison -- rather than a genuine
"nonlinearity doesn't help" result, so it is not included here.)

Unlike the original script (which reports in-sample R^2: fit and score on the
same fiber), this uses 5-fold cross-validated R^2 for all three regressors.
In-sample R^2 would let a flexible nonlinear model win via memorization on a
~360-example fiber; cross-validation is the fair way to ask "does a nonlinear
probe recover more genuine, generalizing factorization signal than ridge
regression does," which is what the AC's critique is actually about.

Runs on the same 5 canonical seeds per architecture (GNN/Transformer/Mamba),
at epoch 0 and epoch 500, on the classification validation split (720
examples, depth in {2,3} fibers, n_agents in {4,5} target) -- identical scope
to Table 3.

Usage:
    python analyze_nonlinear_factorization_probe.py
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, KFold

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

REPO = Path(__file__).resolve().parent.parent
BASE = REPO / 'runs' / 'exp1'  # where train_exp1.sh writes run directories
N_COMPONENTS = 10
CV_FOLDS = 5
CV_SEED = 42

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

DATASET_PATH = REPO / 'data' / 'mindgames_orthogonal_v4_classification' / 'val.json'


def load_embeddings(seed_dir: Path, epoch: int) -> np.ndarray:
    """Load encoder states, mean-pool over sequence dim if needed (matches
    analyze_full_product_factorization_agent_aware.py's load logic)."""
    f = seed_dir / 'captured_states' / f'states_epoch_{epoch}_val.pt'
    data = torch.load(f, map_location='cpu', weights_only=False)
    if 'hidden_states' in data:
        states = data['hidden_states']
    elif 'encoder_states' in data:
        states = data['encoder_states']
    else:
        raise KeyError(f"Could not find hidden states in {list(data.keys())}")
    states = states.cpu().numpy() if isinstance(states, torch.Tensor) else np.array(states)
    if states.ndim == 3:
        states = states.mean(axis=1)
    return states


def load_metadata(dataset_path: Path):
    with open(dataset_path) as f:
        data = json.load(f)
    n_agents = np.array([item['n_agents'] for item in data])
    depth = np.array([item['hypothesis_depth'] for item in data])
    return n_agents, depth


def cv_r2(model_factory, X, y):
    """Mean 5-fold CV R^2 for a given sklearn regressor factory."""
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_SEED)
    scores = cross_val_score(model_factory(), X, y, cv=kf, scoring='r2')
    return float(np.mean(scores)), float(np.std(scores))


def factorization_r2_by_fiber(embeddings, n_agents, depth):
    """ICA + {Ridge, MLP, KernelRidge}, cross-validated R^2 per depth fiber,
    then averaged across fibers -- matches Table 3's task-level r2_agents
    aggregation, but with CV instead of in-sample scoring, and three probes."""
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    ica = FastICA(n_components=N_COMPONENTS, random_state=CV_SEED, max_iter=2000, tol=1e-4)
    components = ica.fit_transform(embeddings_scaled)

    results = {'ridge': [], 'mlp': []}
    for depth_val in sorted(np.unique(depth)):
        mask = depth == depth_val
        X, y = components[mask], n_agents[mask]
        if len(np.unique(y)) < 2:
            continue

        r_ridge, _ = cv_r2(lambda: Ridge(alpha=1.0), X, y)
        r_mlp, _ = cv_r2(lambda: MLPRegressor(hidden_layer_sizes=(32,), alpha=1.0,
                                              max_iter=2000, random_state=CV_SEED), X, y)

        results['ridge'].append(r_ridge)
        results['mlp'].append(r_mlp)

    return {k: float(np.mean(v)) for k, v in results.items() if v}


def main():
    all_results = {}
    for arch, seed_dirs in CANONICAL_SEEDS.items():
        logging.info(f"\n{'='*70}\n{arch}\n{'='*70}")
        n_agents, depth = load_metadata(DATASET_PATH)

        by_epoch = {}
        for epoch in [0, 500]:
            ridge_vals, mlp_vals = [], []
            for sd in seed_dirs:
                seed_dir = BASE / sd
                emb = load_embeddings(seed_dir, epoch)
                r = factorization_r2_by_fiber(emb, n_agents, depth)
                ridge_vals.append(r['ridge'])
                mlp_vals.append(r['mlp'])
                logging.info(f"  epoch {epoch} {sd.split('/')[-1]}: "
                             f"ridge={r['ridge']:.3f} mlp={r['mlp']:.3f}")

            def m_sem(vals):
                return float(np.mean(vals)), float(np.std(vals, ddof=1) / np.sqrt(len(vals)))

            by_epoch[epoch] = {
                'ridge': m_sem(ridge_vals),
                'mlp': m_sem(mlp_vals),
            }
        all_results[arch] = by_epoch

    print(f"\n{'='*70}\nSUMMARY: Cross-validated R^2, linear vs. nonlinear probe\n{'='*70}")
    print(f"{'Arch':<12}{'Epoch':<8}{'Ridge (linear)':<20}{'MLP (nonlinear)'}")
    for arch, by_epoch in all_results.items():
        for epoch, r in by_epoch.items():
            print(f"{arch:<12}{epoch:<8}"
                  f"{r['ridge'][0]:.3f}±{r['ridge'][1]:.3f}        "
                  f"{r['mlp'][0]:.3f}±{r['mlp'][1]:.3f}")

    out_path = BASE / 'nonlinear_factorization_probe_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
