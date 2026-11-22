#!/usr/bin/env python3
"""
Attention-based topological analysis for exp_6 GPT-style NLI (decoder-only
analogue of analyze_attention_topology.py).

For each (layer, head), extracts the LAST non-pad token's attention distribution
over the input sequence — the decoder-only analogue of RoBERTa's [CLS] (position 0)
row. Causal masking means position 0 only ever attends to itself, so it carries no
sequence-summary information; the last token is the only position that has
attended to the full sequence, matching how GPTNLI pools for classification
(gpt_nli_model.py). Hypothesis token spans come directly from the dataset's
character-offset mapping (nli_dataset_gpt.py), rather than the SEP-counting
heuristic used for RoBERTa's two-segment encoding.

Outputs mirror analyze_attention_topology.py exactly (same filenames/JSON schema)
so results are directly comparable across the RoBERTa and GPT experiments.

Usage:
    python analyze_attention_topology_gpt.py \
        --checkpoint outputs_gpt/checkpoint_best.pt \
        --data_dir mindgames_orthogonal_v4_agent_aware \
        --output_dir attention_analysis_gpt
"""

import argparse
import json
import logging
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from nli_dataset import load_nli_dataset, load_test_set
from nli_dataset_gpt import GPTNLIDataset, collate_fn_gpt
from gpt_nli_model import GPTNLI

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


# ---------------------------------------------------------------------------
# Streaming inference + per-head analysis (single pass, no full attn storage)
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_and_analyze_heads(
    model, loader, device, max_samples=None, max_points=300, eps_percentile=20
):
    """
    Single-pass streaming collection and analysis, mirroring the RoBERTa version
    but rows are indexed at each example's LAST valid token position instead of
    a fixed position 0.
    """
    model.eval()

    entropy_by_head = None
    hyp_mass_by_head = None
    metadata = []
    total = 0
    num_layers = num_heads = None

    # Reservoir of full per-example attention rows ([layers, heads, seq_len]),
    # capped at max_points via reservoir sampling so RAM stays bounded regardless
    # of validation set size. Only entropy/hyp_mass scalars (cheap) are kept for
    # every example, since compute_epistemic_correlations needs the full set.
    reservoir = []
    seen = 0
    rng = np.random.default_rng()

    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )

        # outputs.attentions: tuple of num_layers tensors, each [batch, heads, seq, seq]
        batch_attn_full = torch.stack(outputs.attentions, dim=1).float().cpu()
        # [batch, layers, heads, seq_q, seq_k]
        del outputs

        batch_size, layers_dim, heads_dim = (
            batch_attn_full.shape[0], batch_attn_full.shape[1], batch_attn_full.shape[2]
        )
        if num_layers is None:
            num_layers, num_heads = layers_dim, heads_dim
            entropy_by_head = {(l, h): [] for l in range(num_layers) for h in range(num_heads)}
            hyp_mass_by_head = {(l, h): [] for l in range(num_layers) for h in range(num_heads)}

        attn_mask_cpu = attention_mask.cpu()

        for i in range(batch_size):
            seq_len = int(attn_mask_cpu[i].sum().item())
            last_idx = seq_len - 1
            hyp_start = int(batch['hyp_token_start'][i])
            hyp_end = min(int(batch['hyp_token_end'][i]), seq_len)

            # [layers, heads, seq_len]: last-token row, trimmed to real sequence length
            row = batch_attn_full[i, :, :, last_idx, :seq_len].numpy()

            for l in range(num_layers):
                for h in range(num_heads):
                    vec = row[l, h]
                    entropy_by_head[(l, h)].append(attention_entropy(vec))
                    hyp_mass = float(vec[hyp_start:hyp_end].sum()) if hyp_end > hyp_start else 0.0
                    hyp_mass_by_head[(l, h)].append(hyp_mass)

            # Reservoir sampling (Algorithm R): keeps an unbiased max_points-sized
            # sample of full rows across all heads, without ever storing all N.
            if len(reservoir) < max_points:
                reservoir.append(row)
            else:
                j = rng.integers(0, seen + 1)
                if j < max_points:
                    reservoir[j] = row
            seen += 1

            metadata.append({
                'n_agents': batch['n_agents'][i].item(),
                'hypothesis_depth': batch['hypothesis_depth'][i].item(),
                'constraint': batch['constraint'][i],
                'label': batch['labels'][i].item(),
            })

        total += batch_size
        if max_samples and total >= max_samples:
            break

    logging.info(f"Collected streaming attention data for {total} examples "
                 f"(topology reservoir: {len(reservoir)} points)")

    head_data = {}
    for l in range(num_layers):
        for h in range(num_heads):
            vecs = [r[l, h] for r in reservoir]
            max_len = max(v.shape[0] for v in vecs)
            padded = np.zeros((len(vecs), max_len))
            for i, v in enumerate(vecs):
                padded[i, :v.shape[0]] = v

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

    return head_data, metadata, num_layers, num_heads


def compute_epistemic_correlations(head_data, metadata):
    """For each (layer, head), correlate per-example attention metrics with
    epistemic variables (n_agents, hypothesis_depth, label)."""
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
    p.add_argument('--test_dir', default=None)
    p.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    p.add_argument('--output_dir', default='attention_analysis_gpt')
    p.add_argument('--max_length', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--max_samples', type=int, default=None)
    p.add_argument('--max_points', type=int, default=300)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: {device}")

    ckpt_path = Path(args.checkpoint)
    logging.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    ckpt_args = ckpt.get('args', {})
    model_name = ckpt_args.get('model_name')
    torch_dtype = ckpt_args.get('torch_dtype', 'float32')

    model = GPTNLI(model_name=model_name, num_labels=2, torch_dtype=torch_dtype)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    logging.info(f"Model: {model_name}, {model.num_layers} layers, {model.num_heads} heads")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    collate = partial(collate_fn_gpt, pad_token_id=tokenizer.pad_token_id)

    dataset = load_nli_dataset(Path(args.data_dir))
    if args.split == 'test' and args.test_dir:
        examples = load_test_set(Path(args.test_dir))
    elif args.split == 'train':
        examples = dataset['train']
    else:
        examples = dataset['validation']

    ds = GPTNLIDataset(examples, tokenizer, max_length=args.max_length, use_natural_language=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    logging.info("Streaming attention data and computing per-head topology...")
    head_data, metadata, num_layers, num_heads = collect_and_analyze_heads(
        model, loader, device,
        max_samples=args.max_samples,
        max_points=args.max_points,
    )

    logging.info("Computing correlations with epistemic variables...")
    correlations = compute_epistemic_correlations(head_data, metadata)

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

    plot_head_heatmap(entropy_mat, 'Mean Last-Token Attention Entropy',
                      output_dir / 'head_entropy_heatmap.png')
    plot_head_heatmap(hyp_mass_mat, 'Mean Last-Token → Hypothesis Attention Mass',
                      output_dir / 'head_hypothesis_mass_heatmap.png')
    plot_head_heatmap(np.log1p(simplices_mat), 'log(1 + VR Total Simplices) — Last-Token Attention Cloud',
                      output_dir / 'head_topology_heatmap.png')
    plot_head_heatmap(h0_mat, 'VR H0 (Connected Components) — Last-Token Attention Cloud',
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

    logging.info("\nTop 5 heads by mean last-token → hypothesis attention mass:")
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
