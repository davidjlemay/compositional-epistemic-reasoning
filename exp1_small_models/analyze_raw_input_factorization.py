#!/usr/bin/env python3
"""
No-architecture control for product factorization.

Question addressed: "Is it possible that the product factorization is a property of the
data, not the architecture, and the reason the Gaussian baseline doesn't work
is it is a bad prior?"

The existing Gaussian-vector baseline (analyze_random_baseline.py) rules out
one alternative explanation: that ridge regression on *any* sufficiently
high-dimensional vector space trivially inflates R^2 regardless of structure.
It does NOT rule out a different alternative: that the *input data's*
categorical structure (n_agents, hypothesis_depth are literally listed in the
"VARS ... OBS ... " problem string) is what's linearly recoverable, and that
any reasonably-behaved feature map -- not specifically a continuity-preserving
neural architecture -- would preserve it just as well as a randomly
initialized GNN/Transformer/Mamba does.

This script closes that gap directly: it runs the *exact same*
fiber-conditional PCA + Ridge test used to produce Table 3/paper.tex (and
analyze_temporal_factorization.py, which produced those numbers), but on a
feature representation with NO architecture involved at all -- not even an
embedding lookup:

  --feature_type bow          Bag-of-tokens count vector (order-free, the
                               cleanest "zero architecture" representation:
                               literally just "how many times does each
                               vocabulary token appear").
  --feature_type padded_ids   Flattened, zero-padded raw token-ID sequence
                               (preserves token order/position, still no
                               learned transform of any kind).

Interpretation:
  - If raw-input R^2 is comparable to network-initialization R^2 (paper:
    0.75-0.89), the factorization signal is substantially inherited from the
    input encoding itself, not specifically from architecture.
  - If raw-input R^2 is comparable to the Gaussian baseline (paper: 0.03-0.05)
    despite n_agents being a deterministic property of the input string, that
    is evidence FOR the architectural claim: some transformation -- even a
    random-init one -- is needed to make this structure linearly decodable
    via fiber-conditional PCA/Ridge, and architecture (not data) supplies it.

Usage:
    python analyze_raw_input_factorization.py \
        --data_dir mindgames_orthogonal_v4_classification \
        --feature_type bow \
        --output_dir raw_input_factorization_control
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from epistemic_logic_tokenizer import EpistemicLogicTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


# ---------------------------------------------------------------------------
# Data loading and raw feature extraction
# ---------------------------------------------------------------------------

def load_split(data_dir: Path, filename: str) -> list:
    path = data_dir / filename
    with open(path) as f:
        return json.load(f)


def build_tokenizer(examples_for_vocab: list) -> EpistemicLogicTokenizer:
    """Build vocabulary the same way the classification models do: from the
    `smcdel_problem` field (see train_mamba_classification.py, which tokenizes
    exactly this field as network input)."""
    tokenizer = EpistemicLogicTokenizer()
    formulas = [e['smcdel_problem'] for e in examples_for_vocab]
    tokenizer.build_vocabulary_from_data(formulas)
    return tokenizer


def bow_features(examples: list, tokenizer: EpistemicLogicTokenizer) -> np.ndarray:
    """Bag-of-tokens count vector per example: [N, vocab_size]. Order-free,
    no embedding, no network -- the minimal-assumption 'raw data' baseline."""
    vocab_size = tokenizer.vocab_size
    feats = np.zeros((len(examples), vocab_size), dtype=np.float64)
    for i, ex in enumerate(examples):
        ids = tokenizer.encode(ex['smcdel_problem'], add_special_tokens=False)
        for tid in ids:
            feats[i, tid] += 1
    return feats


def padded_id_features(examples: list, tokenizer: EpistemicLogicTokenizer) -> np.ndarray:
    """Flattened, zero-padded raw token-ID sequence per example: [N, max_len].
    Preserves token order/position but still applies no learned transform."""
    all_ids = [tokenizer.encode(ex['smcdel_problem'], add_special_tokens=False)
               for ex in examples]
    max_len = max(len(ids) for ids in all_ids)
    feats = np.zeros((len(examples), max_len), dtype=np.float64)
    for i, ids in enumerate(all_ids):
        feats[i, :len(ids)] = ids
    return feats


def gaussian_baseline(n_samples: int, n_features: int, seed: int) -> np.ndarray:
    """Matched-dimensionality Gaussian control, for a three-way comparison
    against both the raw-input features and the paper's network-init R^2."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_samples, n_features))


def permuted_id_features(examples: list, tokenizer: EpistemicLogicTokenizer, seed: int) -> np.ndarray:
    """Flattened, zero-padded token-ID sequence with each example's own tokens
    randomly SHUFFLED before padding. This exactly preserves per-example
    sequence length and the exact multiset of tokens (including whatever
    tokens encode agent count/identity) -- the AC's specific concern that the
    Gaussian baseline "does not preserve input statistics such as sequence
    length or agent token count." Any R^2 gap between this and the
    order-preserving padded_ids control isolates the contribution of token
    ORDER specifically, beyond raw per-example token statistics."""
    rng = np.random.default_rng(seed)
    all_ids = [tokenizer.encode(ex['smcdel_problem'], add_special_tokens=False)
               for ex in examples]
    max_len = max(len(ids) for ids in all_ids)
    feats = np.zeros((len(examples), max_len), dtype=np.float64)
    for i, ids in enumerate(all_ids):
        shuffled = list(ids)
        rng.shuffle(shuffled)
        feats[i, :len(shuffled)] = shuffled
    return feats


# ---------------------------------------------------------------------------
# Fiber-conditional factorization test (mirrors analyze_temporal_factorization.py
# and analyze_random_baseline.py exactly: PCA(<=10 comps) + Ridge(alpha=1.0),
# fit and scored on the same fiber -- matching the methodology that produced
# Table 3, for a directly comparable number.)
# ---------------------------------------------------------------------------

def compute_fiber_factorization(
    features: np.ndarray,
    fiber_values: np.ndarray,
    target_values: np.ndarray,
    n_components: int = 10,
) -> dict:
    fiber_results = {}
    all_r2 = []

    for fiber_val in sorted(np.unique(fiber_values)):
        mask = fiber_values == fiber_val
        fiber_feats = features[mask]
        fiber_targets = target_values[mask]

        if len(np.unique(fiber_targets)) < 2:
            logging.info(f"  fiber={fiber_val}: SKIP (no variation in target)")
            continue

        n_comp = min(n_components, fiber_feats.shape[0] - 1, fiber_feats.shape[1])
        if n_comp < 1:
            continue

        pca = PCA(n_components=n_comp)
        reduced = pca.fit_transform(fiber_feats)

        ridge = Ridge(alpha=1.0)
        ridge.fit(reduced, fiber_targets)
        preds = ridge.predict(reduced)
        r2 = float(r2_score(fiber_targets, preds))

        fiber_results[f'fiber={fiber_val}'] = {
            'r2': r2,
            'n_samples': int(mask.sum()),
            'explained_variance': float(pca.explained_variance_ratio_.sum()),
        }
        all_r2.append(r2)
        logging.info(f"  fiber={fiber_val}: n={mask.sum()}, R2={r2:.4f}")

    if not all_r2:
        return {'fiber_results': {}, 'mean_r2': float('nan'), 'std_r2': float('nan'), 'n_fibers': 0}

    return {
        'fiber_results': fiber_results,
        'mean_r2': float(np.mean(all_r2)),
        'std_r2': float(np.std(all_r2)),
        'n_fibers': len(all_r2),
    }


def run_direction(features, n_agents, depth, direction: str, n_components: int) -> dict:
    if direction == 'depth_to_agents':
        fiber_values, target_values = depth, n_agents
    elif direction == 'agents_to_depth':
        fiber_values, target_values = n_agents, depth
    else:
        raise ValueError(direction)

    logging.info(f"\n--- Direction: {direction} ---")
    return compute_fiber_factorization(features, fiber_values, target_values, n_components)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_dir', type=str, default='data/mindgames_orthogonal_v4_classification')
    p.add_argument('--split_file', type=str, default='val.json',
                   help='Which split to measure factorization on (paper Table 3 uses '
                        'the 720-example validation split).')
    p.add_argument('--vocab_files', type=str, nargs='+', default=['train.json', 'val.json'],
                   help='Files used to build tokenizer vocabulary, matching how the '
                        'classification models build their vocab (train+val union).')
    p.add_argument('--feature_type', type=str, default='bow',
                   choices=['bow', 'padded_ids', 'permuted_ids'])
    p.add_argument('--permute_seed', type=int, default=0,
                   help='Seed for the per-example token shuffle used by --feature_type permuted_ids.')
    p.add_argument('--n_components', type=int, default=10)
    p.add_argument('--gaussian_seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46],
                   help='Seeds for the matched-dimensionality Gaussian control run '
                        'alongside the raw-input control, for a three-way comparison.')
    p.add_argument('--output_dir', type=str, default='raw_input_factorization_control')
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("=" * 80)
    logging.info("NO-ARCHITECTURE CONTROL: Fiber-Conditional Factorization on Raw Input")
    logging.info("=" * 80)

    vocab_examples = []
    for fname in args.vocab_files:
        vocab_examples.extend(load_split(data_dir, fname))
    tokenizer = build_tokenizer(vocab_examples)

    examples = load_split(data_dir, args.split_file)
    logging.info(f"Loaded {len(examples)} examples from {args.split_file}")

    n_agents = np.array([e['n_agents'] for e in examples])
    depth = np.array([e['hypothesis_depth'] for e in examples])

    if args.feature_type == 'bow':
        features = bow_features(examples, tokenizer)
    elif args.feature_type == 'permuted_ids':
        features = permuted_id_features(examples, tokenizer, seed=args.permute_seed)
    else:
        features = padded_id_features(examples, tokenizer)
    logging.info(f"Raw feature matrix ({args.feature_type}): {features.shape}")

    results = {
        'data_dir': str(data_dir),
        'split_file': args.split_file,
        'feature_type': args.feature_type,
        'feature_dim': int(features.shape[1]),
        'n_examples': int(features.shape[0]),
        'n_components': args.n_components,
    }

    # --- Raw-input control: both factorization directions ---
    logging.info("\n" + "=" * 80)
    logging.info("RAW-INPUT CONTROL (no architecture, no embedding)")
    logging.info("=" * 80)
    results['raw_input'] = {
        'depth_to_agents': run_direction(features, n_agents, depth, 'depth_to_agents', args.n_components),
        'agents_to_depth': run_direction(features, n_agents, depth, 'agents_to_depth', args.n_components),
    }

    # --- Matched-dimensionality Gaussian control (multi-seed, for a
    #     three-way comparison against raw-input and paper Table 2/3 numbers) ---
    logging.info("\n" + "=" * 80)
    logging.info("GAUSSIAN CONTROL (matched dimensionality, multi-seed)")
    logging.info("=" * 80)
    gaussian_runs = []
    for seed in args.gaussian_seeds:
        g_feats = gaussian_baseline(features.shape[0], features.shape[1], seed)
        gaussian_runs.append({
            'seed': seed,
            'depth_to_agents': run_direction(g_feats, n_agents, depth, 'depth_to_agents', args.n_components),
        })
    gaussian_mean_r2 = [r['depth_to_agents']['mean_r2'] for r in gaussian_runs
                        if not np.isnan(r['depth_to_agents']['mean_r2'])]
    results['gaussian_control'] = {
        'per_seed': gaussian_runs,
        'mean_r2': float(np.mean(gaussian_mean_r2)) if gaussian_mean_r2 else float('nan'),
        'std_r2': float(np.std(gaussian_mean_r2)) if gaussian_mean_r2 else float('nan'),
    }

    # --- Summary ---
    raw_r2 = results['raw_input']['depth_to_agents']['mean_r2']
    gauss_r2 = results['gaussian_control']['mean_r2']

    logging.info("\n" + "=" * 80)
    logging.info("SUMMARY (depth -> n_agents, matching Table 3's primary direction)")
    logging.info("=" * 80)
    logging.info(f"  Raw-input ({args.feature_type}) R^2:  {raw_r2:.4f}")
    logging.info(f"  Gaussian control R^2:              {gauss_r2:.4f} +/- {results['gaussian_control']['std_r2']:.4f}")
    logging.info("  Paper Table 2/3 (network init) R^2: 0.748-0.894 (GNN/Transformer/Mamba)")
    logging.info("  Paper Gaussian baseline R^2:        0.03-0.05")

    if raw_r2 > 0.5:
        verdict = ("Raw input alone substantially reproduces the factorization signal -- "
                   "the effect is largely inherited from input encoding, not specifically "
                   "from network architecture.")
    elif raw_r2 <= gauss_r2 + 0.1:
        verdict = ("Raw input performs no better than the Gaussian control despite n_agents "
                   "being a deterministic property of the input string -- some transformation "
                   "(even a random-init network) is needed to make this structure linearly "
                   "decodable via fiber-conditional PCA/Ridge. This supports the architectural "
                   "claim.")
    else:
        verdict = "Intermediate result: raw input shows partial but not full recovery of the signal."

    results['verdict'] = verdict
    logging.info(f"\nVerdict: {verdict}")

    out_file = output_dir / f'raw_input_factorization_{args.feature_type}.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    logging.info(f"\nResults saved to: {out_file}")


if __name__ == '__main__':
    main()
