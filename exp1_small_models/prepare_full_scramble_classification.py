#!/usr/bin/env python3
"""
Prepare fully-scrambled label control dataset for topological analysis.

Assigns each training example a label drawn independently and uniformly at
random from {'entailment', 'non_entailment'}, breaking all correlation between
input and label.  This differs from the partial-scramble version
(prepare_scrambled_classification.py), which permutes labels and therefore
preserves the original class distribution — leaving ~70% of labels unchanged
when the dataset is class-imbalanced (80/20), so models can still learn.

Here every example gets an independent coin flip, so:
  - ~50% of training labels are changed regardless of class balance
  - ~80% of labels are changed (since ~80% were originally 'entailment')
  - Models cannot do better than ~50% val accuracy (no learnable signal)

The validation set is left unmodified.

Output: mindgames_orthogonal_v4_classification_full_scrambled/
  train.json     -- fully randomised labels (50/50)
  val.json       -- original (unchanged)
  metadata.json  -- records scrambling seed, source, and label balance
"""

import json
import random
import argparse
import shutil
from pathlib import Path

LABEL_SET = ['entailment', 'non_entailment']


def fully_scramble_labels(examples, seed):
    """Assign each example an independent uniform-random label."""
    rng = random.Random(seed)
    return [{**ex, 'label': rng.choice(LABEL_SET)} for ex in examples]


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
                        default='data/mindgames_orthogonal_v4_classification_full_scrambled',
                        help='Output directory for fully-scrambled dataset')
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

    scrambled_train = fully_scramble_labels(train_examples, args.seed)

    original_labels = [ex['label'] for ex in train_examples]
    scrambled_labels_list = [ex['label'] for ex in scrambled_train]
    n_changed = sum(o != s for o, s in zip(original_labels, scrambled_labels_list))
    print(f"Labels changed: {n_changed}/{len(train_examples)} "
          f"({100*n_changed/len(train_examples):.1f}%)")
    print(f"Scrambled label distribution: {label_counts(scrambled_train)}")

    val_path = source_dir / 'val.json'
    shutil.copy(val_path, output_dir / 'val.json')
    print(f"Copied val set unchanged ({val_path})")

    out_train = output_dir / 'train.json'
    with open(out_train, 'w') as f:
        json.dump(scrambled_train, f, indent=2)
    print(f"Wrote fully-scrambled training set to {out_train}")

    meta = {
        'source_dir': str(source_dir),
        'scramble_seed': args.seed,
        'scramble_method': 'independent_uniform',
        'n_train': len(scrambled_train),
        'n_labels_changed': n_changed,
        'original_label_distribution': label_counts(train_examples),
        'scrambled_label_distribution': label_counts(scrambled_train),
        'description': (
            'Full-scramble control: each training label is drawn independently '
            'and uniformly at random from {entailment, non_entailment}, '
            'breaking all input-label correlation. Val set is unmodified. '
            'Models trained on this data cannot exceed ~50% val accuracy. '
            'Used as a null hypothesis control: does near-complete topological '
            'simplification require a learnable, compositionally structured task?'
        )
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote metadata to {output_dir}/metadata.json")
    print("Done.")


if __name__ == '__main__':
    main()
