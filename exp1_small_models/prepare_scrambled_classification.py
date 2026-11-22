#!/usr/bin/env python3
"""
Prepare scrambled-label control dataset for topological analysis (classification task).

Shuffles the label field across training examples while keeping all other fields
(premise, hypothesis, n_agents, hypothesis_depth, etc.) intact.
The validation set is left unmodified.

This creates a non-compositional control: the model must memorize random
input→label mappings rather than learning compositional structure.
Used to test whether topological simplification is specific to compositional
learning or accompanies any supervised learning.

Output: mindgames_orthogonal_v4_classification_scrambled/
  train.json     -- scrambled labels
  val.json       -- original (unchanged)
  metadata.json  -- records scrambling seed, source, and label balance
"""

import json
import random
import argparse
import shutil
from pathlib import Path


def scramble_labels(examples, seed):
    """Return new list with label fields shuffled across examples."""
    rng = random.Random(seed)
    labels = [ex['label'] for ex in examples]
    rng.shuffle(labels)
    return [{**ex, 'label': label} for ex, label in zip(examples, labels)]


def label_counts(examples):
    counts = {}
    for ex in examples:
        counts[ex['label']] = counts.get(ex['label'], 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str,
                        default='data/mindgames_orthogonal_v4_classification',
                        help='Source dataset directory')
    parser.add_argument('--output_dir', type=str,
                        default='data/mindgames_orthogonal_v4_classification_scrambled',
                        help='Output directory for scrambled dataset')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    train_path = source_dir / 'train.json'
    with open(train_path) as f:
        train_examples = json.load(f)

    print(f"Loaded {len(train_examples)} training examples from {train_path}")
    print(f"Original label distribution: {label_counts(train_examples)}")

    scrambled_train = scramble_labels(train_examples, args.seed)

    original_labels = [ex['label'] for ex in train_examples]
    scrambled_labels = [ex['label'] for ex in scrambled_train]
    n_changed = sum(o != s for o, s in zip(original_labels, scrambled_labels))
    print(f"Labels changed: {n_changed}/{len(train_examples)} "
          f"({100*n_changed/len(train_examples):.1f}%)")
    print(f"Scrambled label distribution: {label_counts(scrambled_train)}")

    val_path = source_dir / 'val.json'
    shutil.copy(val_path, output_dir / 'val.json')
    print(f"Copied val set unchanged ({val_path})")

    out_train = output_dir / 'train.json'
    with open(out_train, 'w') as f:
        json.dump(scrambled_train, f, indent=2)
    print(f"Wrote scrambled training set to {out_train}")

    meta = {
        'source_dir': str(source_dir),
        'scramble_seed': args.seed,
        'n_train': len(scrambled_train),
        'n_labels_changed': n_changed,
        'original_label_distribution': label_counts(train_examples),
        'scrambled_label_distribution': label_counts(scrambled_train),
        'description': (
            'Scrambled-label control: classification labels are randomly '
            'permuted across training examples. Val set is unmodified. '
            'Used to test whether topological simplification is specific '
            'to compositional learning or accompanies learning in general.'
        )
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote metadata to {output_dir}/metadata.json")
    print("Done.")


if __name__ == '__main__':
    main()
