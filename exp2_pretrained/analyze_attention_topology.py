#!/usr/bin/env python3
"""
Attention-based topological analysis for exp_6 RoBERTa NLI.

For each (layer, head), extracts the [CLS] token's attention distribution over
the input sequence. These N attention vectors (one per example) form a point
cloud in R^seq_len. The VR topology of this cloud characterizes how the head
organizes examples.

Hypotheses:
  - Heads specializing in epistemic reasoning cluster examples by n_agents or
    hypothesis_depth → lower H0 (fewer components = tighter clusters by class)
  - Heads with lower topological complexity (fewer simplices) better predict
    generalization: same as the hidden-state finding in exp_5

Additional scalar metrics per head:
  - Attention entropy: -sum(p * log(p)) — low = focused, high = diffuse
  - Hypothesis mass: total attention weight on hypothesis tokens
  - Cross-attention ratio: hypothesis mass / total mass (premise: hypothesis split)

Outputs:
  - attention_topology.json: per-(layer, head) topology + scalar metrics
  - head_entropy_heatmap.png: entropy across heads and layers
  - head_hypothesis_mass_heatmap.png: attention on hypothesis tokens
  - head_topology_heatmap.png: VR simplex count per head
  - correlation_epistemic.json: per-head correlation with n_agents, depth

Usage:
    python analyze_attention_topology.py \
        --checkpoint roberta_nli_results_<JOB_ID>/checkpoint_best.pt \
        --data_dir mindgames_orthogonal_v4_agent_aware \
        --output_dir attention_analysis
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast

sys.path.insert(0, str(Path(__file__).parent))
from nli_dataset import NLIDataset, collate_fn, load_nli_dataset, load_test_set
from roberta_nli_model import RobertaNLI

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False
    logging.warning("gudhi not available — topology computation disabled")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_vr_topology(points: np.ndarray, max_edge_length: float = 1.0) -> dict:
    """VR persistent homology on a point cloud. Returns simplex count and Betti numbers."""
    if not GUDHI_AVAILABLE or len(points) < 3:
        return {'total_simplices': 0, 'h0': 0, 'h1': 0}

    rips = gudhi.RipsComplex(points=points, max_edge_length=max_edge_length)
    st = rips.create_simplex_tree(max_dimension=2)
    st.compute_persistence()
    betti = st.betti_numbers()
    return {
        'total_simplices': st.num_simplices(),
        'h0': int(betti[0]) if len(betti) > 0 else 0,
        'h1': int(betti[1]) if len(betti) > 1 else 0,
    }


def attention_entropy(attn: np.ndarray) -> float:
    """Shannon entropy of an attention distribution (higher = more diffuse)."""
    p = np.clip(attn, 1e-9, 1.0)
    return float(-np.sum(p * np.log(p)))


def locate_hypothesis_tokens(input_ids: torch.Tensor, sep_token_id: int) -> list:
    """
    Find token index ranges for the hypothesis in RoBERTa's encoding:
        <s> premise </s> </s> hypothesis </s>
    RoBERTa uses TWO consecutive </s> tokens at the segment boundary, so the
    hypothesis starts after the second </s> (seps[1] + 1), not the first.
    Returns list of (start, end) index tuples (exclusive end).
    """
    ranges = []
    for ids in input_ids:
        ids_list = ids.tolist()
        seps = [i for i, t in enumerate(ids_list) if t == sep_token_id]
        if len(seps) >= 3:
            # Normal case: </s> premise </s> </s> hypothesis </s>
            # seps[0] = end of premise, seps[1] = second boundary SEP,
            # seps[2] = final SEP after hypothesis
            ranges.append((seps[1] + 1, seps[2]))
        elif len(seps) == 2:
            # Two SEPs: treat second as boundary, rest is hypothesis
            ranges.append((seps[1] + 1, len(ids_list)))
        elif len(seps) == 1:
            ranges.append((seps[0] + 1, len(ids_list)))
        else:
            ranges.append((1, len(ids_list)))
    return ranges


# ---------------------------------------------------------------------------
# Streaming inference + per-head analysis (single pass, no full attn storage)
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_and_analyze_heads(
    model, loader, device, sep_token_id, max_samples=None, max_points=300, eps_percentile=20
):
    """
    Single-pass streaming collection and analysis.

    Instead of storing full [layers, heads, seq, seq] tensors (7.5 GB for N=200),
    extracts only the [CLS] row (attn[:, :, :, 0, :]) per batch, shape
    [batch, layers, heads, seq].  Entropy and hyp_mass are accumulated online.

    Memory: ~29 MB for 200 examples (vs 7.5 GB for the naive approach).

    Returns:
        head_data:  dict[(layer, head)] -> {topology, mean_entropy, ..., per_example_*}
        metadata:   list of dicts with n_agents, hypothesis_depth, constraint, label
    """
    model.eval()

    # Will be populated on first batch
    cls_by_head = None      # dict[(l, h)] -> list of np.ndarray shape [seq]
    entropy_by_head = None  # dict[(l, h)] -> list of float
    hyp_mass_by_head = None # dict[(l, h)] -> list of float
    metadata = []
    total = 0

    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )

        # outputs.attentions: tuple of num_layers tensors, each [batch, heads, seq, seq]
        # Stack to [batch, layers, heads, seq, seq], then take CLS row → [batch, layers, heads, seq]
        batch_attn_full = torch.stack(outputs.attentions, dim=1)  # on device
        cls_rows = batch_attn_full[:, :, :, 0, :].float().cpu()   # [batch, layers, heads, seq]
        del batch_attn_full

        batch_size, num_layers, num_heads, _ = cls_rows.shape

        # Initialise accumulators on first batch
        if cls_by_head is None:
            cls_by_head = {(l, h): [] for l in range(num_layers) for h in range(num_heads)}
            entropy_by_head = {(l, h): [] for l in range(num_layers) for h in range(num_heads)}
            hyp_mass_by_head = {(l, h): [] for l in range(num_layers) for h in range(num_heads)}

        # Hypothesis token ranges for this batch
        hyp_ranges = locate_hypothesis_tokens(input_ids.cpu(), sep_token_id)

        for i in range(batch_size):
            seq_len = int(attention_mask[i].sum().item())
            hyp_start, hyp_end = hyp_ranges[i]
            hyp_end = min(hyp_end, seq_len)

            for l in range(num_layers):
                for h in range(num_heads):
                    cls_vec = cls_rows[i, l, h, :seq_len].numpy()  # [seq_len] trimmed
                    cls_by_head[(l, h)].append(cls_vec)
                    entropy_by_head[(l, h)].append(attention_entropy(cls_vec))
                    hyp_mass = float(cls_vec[hyp_start:hyp_end].sum()) if hyp_end > hyp_start else 0.0
                    hyp_mass_by_head[(l, h)].append(hyp_mass)

            metadata.append({
                'n_agents': batch['n_agents'][i].item(),
                'hypothesis_depth': batch['hypothesis_depth'][i].item(),
                'constraint': batch['constraint'][i],
                'label': batch['labels'][i].item(),
            })

        total += batch_size
        if max_samples and total >= max_samples:
            break

    logging.info(f"Collected streaming attention data for {total} examples")

    # Compute per-head VR topology
    head_data = {}
    for l in range(num_layers):
        for h in range(num_heads):
            vecs = cls_by_head[(l, h)]
            max_len = max(v.shape[0] for v in vecs)
            padded = np.zeros((len(vecs), max_len))
            for i, v in enumerate(vecs):
                padded[i, :v.shape[0]] = v

            if len(padded) > max_points:
                idx = np.random.choice(len(padded), max_points, replace=False)
                padded = padded[idx]

            dists = pdist(padded)
            eps = float(np.percentile(dists, eps_percentile))
            topo = compute_vr_topology(padded, max_edge_length=eps)

            head_data[(l, h)] = {
                'layer': l,
                'head': h,
                'topology': topo,
                'mean_entropy': float(np.mean(entropy_by_head[(l, h)])),
                'std_entropy': float(np.std(entropy_by_head[(l, h)])),
                'mean_hyp_mass': float(np.mean(hyp_mass_by_head[(l, h)])),
                'std_hyp_mass': float(np.std(hyp_mass_by_head[(l, h)])),
                'per_example_entropy': entropy_by_head[(l, h)],
                'per_example_hyp_mass': hyp_mass_by_head[(l, h)],
            }

    return head_data, metadata


def compute_epistemic_correlations(head_data, metadata):
    """
    For each (layer, head), correlate per-example attention metrics
    (entropy, hyp_mass) with epistemic variables (n_agents, hypothesis_depth).

    Returns: dict[(layer, head)] -> {target: {pearson_r, pearson_p}}
    """
    n_agents = np.array([m['n_agents'] for m in metadata])
    depth = np.array([m['hypothesis_depth'] for m in metadata])
    label = np.array([m['label'] for m in metadata])

    correlations = {}
    for (l, h), data in head_data.items():
        entropy = np.array(data['per_example_entropy'])
        hyp_mass = np.array(data['per_example_hyp_mass'])
        corr = {}
        for name, x in [('entropy', entropy), ('hyp_mass', hyp_mass)]:
            for target_name, y in [
                ('n_agents', n_agents),
                ('hypothesis_depth', depth),
                ('label', label),
            ]:
                mask = y >= 0
                if mask.sum() > 10:
                    r, p = pearsonr(x[mask], y[mask])
                    corr[f'{name}_vs_{target_name}'] = {'pearson_r': float(r), 'pearson_p': float(p)}
        correlations[(l, h)] = corr

    return correlations


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_head_heatmap(values: np.ndarray, title: str, out_path: Path, fmt='.2f'):
    """Plot (layers × heads) heatmap."""
    fig, ax = plt.subplots(figsize=(max(6, values.shape[1]), max(4, values.shape[0] // 2)))
    sns.heatmap(values, ax=ax, cmap='viridis', annot=(values.shape[0] * values.shape[1] <= 144),
                fmt=fmt, linewidths=0.3)
    ax.set_xlabel('Head')
    ax.set_ylabel('Layer')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logging.info(f"Saved: {out_path}")


def plot_head_correlation_heatmap(correlations, num_layers, num_heads, metric, target, out_path):
    """Heatmap of Pearson r between a head metric and an epistemic variable."""
    mat = np.zeros((num_layers, num_heads))
    for l in range(num_layers):
        for h in range(num_heads):
            key = f'{metric}_vs_{target}'
            r = correlations.get((l, h), {}).get(key, {}).get('pearson_r', 0.0)
            mat[l, h] = r
    plot_head_heatmap(
        mat,
        title=f'Pearson r: {metric} vs {target}',
        out_path=out_path,
        fmt='.2f',
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True,
                   help='Path to checkpoint_best.pt or checkpoint_latest.pt')
    p.add_argument('--data_dir', default='data/mindgames_orthogonal_v4_agent_aware')
    p.add_argument('--test_dir', default=None,
                   help='If set, analyze held-out test set instead of val')
    p.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    p.add_argument('--output_dir', default='attention_analysis')
    p.add_argument('--max_length', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--max_samples', type=int, default=None,
                   help='Limit number of examples (for speed)')
    p.add_argument('--max_points', type=int, default=300,
                   help='Max examples to subsample for VR computation')
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: {device}")

    # Load checkpoint
    ckpt_path = Path(args.checkpoint)
    logging.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    ckpt_args = ckpt.get('args', {})
    model_name = ckpt_args.get('model_name', 'roberta-base')

    model = RobertaNLI(model_name=model_name, num_labels=2)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    num_layers = model.num_layers
    num_heads = model.roberta.config.num_attention_heads
    logging.info(f"Model: {model_name}, {num_layers} layers, {num_heads} heads")

    # Tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    sep_token_id = tokenizer.sep_token_id

    # Dataset
    dataset = load_nli_dataset(Path(args.data_dir))
    if args.split == 'test' and args.test_dir:
        examples = load_test_set(Path(args.test_dir))
    elif args.split == 'train':
        examples = dataset['train']
    else:
        examples = dataset['validation']

    ds = NLIDataset(examples, tokenizer, max_length=args.max_length, use_natural_language=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Single-pass streaming collection + per-head analysis (no full attn storage)
    logging.info("Streaming attention data and computing per-head topology...")
    head_data, metadata = collect_and_analyze_heads(
        model, loader, device,
        sep_token_id=sep_token_id,
        max_samples=args.max_samples,
        max_points=args.max_points,
    )

    # Correlations with epistemic variables
    logging.info("Computing correlations with epistemic variables...")
    correlations = compute_epistemic_correlations(head_data, metadata)

    # Build matrices for heatmaps
    entropy_mat = np.zeros((num_layers, num_heads))
    hyp_mass_mat = np.zeros((num_layers, num_heads))
    simplices_mat = np.zeros((num_layers, num_heads))
    h0_mat = np.zeros((num_layers, num_heads))
    h1_mat = np.zeros((num_layers, num_heads))

    for l in range(num_layers):
        for h in range(num_heads):
            d = head_data[(l, h)]
            entropy_mat[l, h] = d['mean_entropy']
            hyp_mass_mat[l, h] = d['mean_hyp_mass']
            simplices_mat[l, h] = d['topology']['total_simplices']
            h0_mat[l, h] = d['topology']['h0']
            h1_mat[l, h] = d['topology']['h1']

    # Plots
    plot_head_heatmap(entropy_mat, 'Mean [CLS] Attention Entropy',
                      output_dir / 'head_entropy_heatmap.png')
    plot_head_heatmap(hyp_mass_mat, 'Mean [CLS] → Hypothesis Attention Mass',
                      output_dir / 'head_hypothesis_mass_heatmap.png')
    plot_head_heatmap(np.log1p(simplices_mat), 'log(1 + VR Total Simplices) — [CLS] Attention Cloud',
                      output_dir / 'head_topology_heatmap.png')
    plot_head_heatmap(h0_mat, 'VR H0 (Connected Components) — [CLS] Attention Cloud',
                      output_dir / 'head_h0_heatmap.png')

    for target in ['n_agents', 'hypothesis_depth', 'label']:
        plot_head_correlation_heatmap(
            correlations, num_layers, num_heads,
            metric='hyp_mass', target=target,
            out_path=output_dir / f'corr_hyp_mass_vs_{target}.png',
        )
        plot_head_correlation_heatmap(
            correlations, num_layers, num_heads,
            metric='entropy', target=target,
            out_path=output_dir / f'corr_entropy_vs_{target}.png',
        )

    # Save JSON
    serializable_head = {}
    for (l, h), d in head_data.items():
        key = f'L{l}_H{h}'
        serializable_head[key] = {
            'layer': l, 'head': h,
            'topology': d['topology'],
            'mean_entropy': d['mean_entropy'],
            'std_entropy': d['std_entropy'],
            'mean_hyp_mass': d['mean_hyp_mass'],
            'std_hyp_mass': d['std_hyp_mass'],
        }

    serializable_corr = {
        f'L{l}_H{h}': v for (l, h), v in correlations.items()
    }

    with open(output_dir / 'attention_topology.json', 'w') as f:
        json.dump(serializable_head, f, indent=2)
    with open(output_dir / 'correlation_epistemic.json', 'w') as f:
        json.dump(serializable_corr, f, indent=2)

    # Print top-5 heads by hypothesis mass and by n_agents correlation
    logging.info("\nTop 5 heads by mean [CLS] → hypothesis attention mass:")
    ranked = sorted(serializable_head.items(), key=lambda x: x[1]['mean_hyp_mass'], reverse=True)
    for k, v in ranked[:5]:
        logging.info(f"  {k}: hyp_mass={v['mean_hyp_mass']:.4f} "
                     f"entropy={v['mean_entropy']:.4f} "
                     f"simplices={v['topology']['total_simplices']}")

    logging.info("\nTop 5 heads by |r| with n_agents (entropy):")
    ranked_corr = sorted(
        [(k, v) for k, v in serializable_corr.items() if 'entropy_vs_n_agents' in v],
        key=lambda x: abs(x[1]['entropy_vs_n_agents']['pearson_r']),
        reverse=True,
    )
    for k, v in ranked_corr[:5]:
        r = v['entropy_vs_n_agents']['pearson_r']
        p = v['entropy_vs_n_agents']['pearson_p']
        logging.info(f"  {k}: r={r:.4f} p={p:.4f}")

    logging.info(f"\nDone. Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
