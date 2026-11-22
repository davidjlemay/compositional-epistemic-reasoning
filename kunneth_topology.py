"""
Kunneth-formula persistent-homology factorization test.

Tests whether a representation manifold believed to factor as a product
Z = X x Y over two independent variation axes (e.g. n_agents and
hypothesis_depth) actually does so topologically. If it does, the Kunneth
formula predicts

    beta_k(Z) = sum_{i+j=k} beta_i(X) * beta_j(Y)

This holds under continuous, including nonlinear, deformation of the
representation space -- unlike linear cross-fiber decodability (ICA + Ridge
regression), which an affine change of coordinates can create or destroy
without touching the underlying topology. It is therefore a different failure
mode than a linear probe: it can fail where probe R^2 is high, and hold where
probe R^2 is mediocre.

Reused across exp1_small_models (small models, single pooled
embedding per example) and exp2_pretrained (RoBERTa/Qwen, per-layer [CLS] /
last-token states) via thin driver scripts in each experiment directory.
"""

import numpy as np
from scipy.spatial.distance import cdist
from gudhi import SimplexTree


def compute_vietoris_rips_complex(points, epsilon, max_dimension=2):
    """Vietoris-Rips complex at a fixed epsilon.

    Copied from VR_trajectories.py:compute_vietoris_rips_complex rather than
    imported: that module unconditionally imports tensorflow at the top level
    (for unrelated visualization helpers), which is not installed in this
    project's venv. Every other analysis script in this repo (exp_3/exp_5/
    exp_6) works around the same issue by inlining this function rather than
    importing it -- matching that convention here instead of adding a new,
    heavier import path.
    """
    simplex_tree = SimplexTree()
    for i in range(len(points)):
        simplex_tree.insert([i], filtration=0.0)

    dm = cdist(points, points)
    n_points = len(points)
    for i in range(n_points):
        for j in range(i + 1, n_points):
            distance = dm[i, j]
            if distance <= epsilon:
                simplex_tree.insert([i, j], filtration=distance)

    simplex_tree.expansion(max_dimension)
    return simplex_tree


def adaptive_eps_subsample(points, max_points, eps_percentile, rng,
                            k_neighbors=3, mult=1.5):
    """Subsample a point cloud and compute a density-adaptive VR epsilon.

    Epsilon is derived from k-nearest-neighbor distances (a LOCAL, per-point
    statistic), not from a percentile of all pairwise distances. This
    matters: the naive "eps = percentile of all pairwise distances"
    approach (used elsewhere in this repo, e.g.
    exp2_pretrained/analyze_nli_generalization.py's subsample_and_adaptive_eps)
    is structurally unable to resolve more than roughly 100/eps_percentile
    genuine clusters, *regardless of how well-separated those clusters are*
    -- once within-cluster pairs fall below eps_percentile% of all pairs
    (which happens automatically once cluster count exceeds ~5 at the
    conventional 20th-percentile default), the percentile threshold
    necessarily lands in the between-cluster distance range and merges
    everything. Verified directly on a synthetic case with clusters spaced
    1000 apart and 0.01 internal spread: the 20th-percentile epsilon jumps
    from 0.04 (correctly separating) at k=2 clusters to ~1000 (merging
    everything) at k=5+ clusters, with no dependence on actual separation.

    k-NN distance avoids this because each point's k-th-nearest-neighbor
    distance only depends on its own local neighborhood, not on how many
    other far-away clusters exist elsewhere in the cloud -- confirmed to
    stay in a stable, correctly-scaled range (~0.03-0.065 on the same
    synthetic case) across k=1 to k=24 clusters.

    eps_percentile here means "percentile of the per-point k-NN distances"
    (a local density statistic), not "percentile of all pairwise distances"
    -- a different quantity than the pipeline-wide convention despite the
    shared parameter name, kept for CLI consistency across driver scripts.
    """
    points = np.asarray(points)
    if len(points) > max_points:
        idx = rng.choice(len(points), size=max_points, replace=False)
        points = points[idx]
    n = len(points)
    if n < 2:
        return points, 0.0

    dm = cdist(points, points)
    np.fill_diagonal(dm, np.inf)
    k_eff = min(k_neighbors, n - 1)
    knn_dist = np.partition(dm, k_eff - 1, axis=1)[:, k_eff - 1]
    eps = float(np.percentile(knn_dist, eps_percentile) * mult)
    return points, eps


def compute_betti(points, epsilon, max_dimension=2):
    """VR persistent homology Betti numbers (H0, H1) at a fixed epsilon.

    Uses the corrected compute_persistence() -> betti_numbers() sequence.
    Never count persistence pairs manually (e.g. summing entries with
    dimension==0 from st.persistence()) -- that counts every birth event
    across the whole filtration, which for H0 always equals the point count
    regardless of connectivity.
    """
    points = np.asarray(points)
    if len(points) < 2:
        return {'h0': len(points), 'h1': 0, 'total_simplices': len(points)}
    st = compute_vietoris_rips_complex(points, epsilon, max_dimension=max_dimension)
    st.compute_persistence()
    betti = st.betti_numbers()
    return {
        'h0': int(betti[0]) if len(betti) > 0 else 0,
        'h1': int(betti[1]) if len(betti) > 1 else 0,
        'total_simplices': st.num_simplices(),
    }


def kunneth_predicted_betti(betti_x, betti_y):
    """Kunneth-formula prediction for beta_k(X x Y) from marginal Betti numbers.

        beta_0(X x Y) = beta_0(X) * beta_0(Y)
        beta_1(X x Y) = beta_0(X) * beta_1(Y) + beta_1(X) * beta_0(Y)
    """
    h0_pred = betti_x['h0'] * betti_y['h0']
    h1_pred = betti_x['h0'] * betti_y['h1'] + betti_x['h1'] * betti_y['h0']
    return {'h0': h0_pred, 'h1': h1_pred}


def build_fiber_clouds(embeddings, factor_a, factor_b, b_ref, a_ref):
    """Partition a point cloud into X/Y/Z fiber clouds for the Kunneth test.

    X: varies factor_a, factor_b held fixed at b_ref
    Y: varies factor_b, factor_a held fixed at a_ref
    Z: full joint cloud (all examples)

    In this repo's usage, factor_a = n_agents, factor_b = hypothesis_depth.
    """
    x_mask = factor_b == b_ref
    y_mask = factor_a == a_ref
    return {'X': embeddings[x_mask], 'Y': embeddings[y_mask], 'Z': embeddings}


def kunneth_fiber_test(embeddings, factor_a, factor_b, b_ref, a_ref,
                        max_points, eps_percentile, rng, max_dimension=2):
    """Single (non-bootstrapped) Kunneth check: observed vs predicted beta_k(Z)."""
    clouds = build_fiber_clouds(embeddings, factor_a, factor_b, b_ref, a_ref)
    observed = {}
    for key, pts in clouds.items():
        pts_sub, eps = adaptive_eps_subsample(pts, max_points, eps_percentile, rng)
        betti = compute_betti(pts_sub, eps, max_dimension=max_dimension)
        betti['eps'] = eps
        betti['n_points'] = len(pts_sub)
        observed[key] = betti

    predicted = kunneth_predicted_betti(observed['X'], observed['Y'])
    return {
        'observed': observed,
        'predicted_z': predicted,
        'b_ref': b_ref.item() if hasattr(b_ref, 'item') else b_ref,
        'a_ref': a_ref.item() if hasattr(a_ref, 'item') else a_ref,
    }


def _ci(vals):
    vals = np.asarray(vals, dtype=float)
    return {
        'mean': float(np.mean(vals)),
        'std': float(np.std(vals)),
        'ci_low': float(np.percentile(vals, 2.5)),
        'ci_high': float(np.percentile(vals, 97.5)),
    }


def bootstrap_kunneth_test(embeddings, factor_a, factor_b, max_points, eps_percentile,
                            n_bootstrap, rng, max_dimension=2,
                            a_levels=None, b_levels=None):
    """Bootstrap the Kunneth check over point-cloud subsample draws.

    factor_a, factor_b are the two independent variation axes (n_agents,
    hypothesis_depth). Uncertainty comes from repeated subsampling within a
    single fixed embedding array, not from retraining -- appropriate when
    embeddings come from one saved checkpoint (RoBERTa/Qwen). For small
    models with multiple trained seeds, call this once per seed instead and
    aggregate mean +/- SEM across seeds at the call site.

    Also runs a reference-invariance secondary check (single draw, not
    bootstrapped): repeats the marginal fiber computation at the other
    reference level of each factor. True product structure implies a fiber's
    topology should not depend on which level of the *other* factor it was
    sliced at.
    """
    a_levels = sorted(np.unique(factor_a).tolist()) if a_levels is None else a_levels
    b_levels = sorted(np.unique(factor_b).tolist()) if b_levels is None else b_levels
    if len(a_levels) < 2 or len(b_levels) < 2:
        raise ValueError('Need at least 2 levels of each factor for a Kunneth check')

    primary_b_ref, alt_b_ref = b_levels[0], b_levels[1]
    primary_a_ref, alt_a_ref = a_levels[0], a_levels[1]

    draws = [
        kunneth_fiber_test(
            embeddings, factor_a, factor_b, primary_b_ref, primary_a_ref,
            max_points, eps_percentile, rng, max_dimension=max_dimension,
        )
        for _ in range(n_bootstrap)
    ]

    summary = {
        'observed_h0': _ci([d['observed']['Z']['h0'] for d in draws]),
        'observed_h1': _ci([d['observed']['Z']['h1'] for d in draws]),
        'predicted_h0': _ci([d['predicted_z']['h0'] for d in draws]),
        'predicted_h1': _ci([d['predicted_z']['h1'] for d in draws]),
        'n_bootstrap': n_bootstrap,
        'b_ref': primary_b_ref,
        'a_ref': primary_a_ref,
    }

    alt_x = kunneth_fiber_test(
        embeddings, factor_a, factor_b, alt_b_ref, primary_a_ref,
        max_points, eps_percentile, rng, max_dimension=max_dimension,
    )
    alt_y = kunneth_fiber_test(
        embeddings, factor_a, factor_b, primary_b_ref, alt_a_ref,
        max_points, eps_percentile, rng, max_dimension=max_dimension,
    )
    summary['reference_invariance'] = {
        'X_at_primary_b_ref': draws[0]['observed']['X'],
        'X_at_alt_b_ref': alt_x['observed']['X'],
        'Y_at_primary_a_ref': draws[0]['observed']['Y'],
        'Y_at_alt_a_ref': alt_y['observed']['Y'],
    }

    return summary
