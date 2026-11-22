#!/usr/bin/env python3
"""
Unified data loading utilities for both Experiment 3 and Experiment 4 formats.

Exp 3 format:
  - Field: "smcdel_problem" (full premise + hypothesis)
  - Files: train.json, val.json

Exp 4 format:
  - Fields: "premise", "hypothesis", "constraint"
  - Files: train_generation.json, val_generation.json
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


def detect_data_format(data_dir: Path) -> str:
    """
    Detect whether data_dir contains Exp 3 or Exp 4 format.

    Returns:
        'exp3' or 'exp4'
    """
    data_dir = Path(data_dir)

    # Check for Exp 4 format files
    if (data_dir / 'train_generation.json').exists():
        return 'exp4'

    # Check for Exp 3 format files
    if (data_dir / 'train.json').exists():
        return 'exp3'

    raise ValueError(f"Could not detect data format in {data_dir}. "
                     f"Expected either train.json (exp3) or train_generation.json (exp4)")


def get_smcdel_formula(example: Dict, data_format: str) -> str:
    """
    Extract the full SMCDEL formula from an example.

    For Exp 3: return the 'smcdel_problem' field
    For Exp 4: reconstruct from 'premise' and 'hypothesis'

    Args:
        example: Dataset example dictionary
        data_format: 'exp3' or 'exp4'

    Returns:
        Full SMCDEL formula string for modal operator analysis
    """
    if data_format == 'exp3':
        return example['smcdel_problem']

    elif data_format == 'exp4':
        # For Exp 4, the hypothesis contains the modal operators we care about
        # The premise is the setup, but the modal reasoning is in the hypothesis
        return example['hypothesis']

    else:
        raise ValueError(f"Unknown data format: {data_format}")


def load_dataset(data_dir: Path, split: str = 'train') -> tuple[List[Dict], str]:
    """
    Load dataset in either Exp 3 or Exp 4 format.

    Args:
        data_dir: Directory containing the dataset
        split: 'train' or 'val'

    Returns:
        (dataset, data_format) where:
        - dataset is a list of examples
        - data_format is 'exp3' or 'exp4'
    """
    data_dir = Path(data_dir)
    data_format = detect_data_format(data_dir)

    if data_format == 'exp3':
        filename = f'{split}.json'
    else:  # exp4
        filename = f'{split}_generation.json'

    filepath = data_dir / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    with open(filepath, 'r') as f:
        dataset = json.load(f)

    return dataset, data_format


def get_split_filename(split: str, data_format: str) -> str:
    """
    Get the appropriate filename for a split based on data format.

    Args:
        split: 'train' or 'val'
        data_format: 'exp3' or 'exp4'

    Returns:
        Filename string
    """
    if data_format == 'exp3':
        return f'{split}.json'
    else:  # exp4
        return f'{split}_generation.json'


def parse_modal_operators(formula: str) -> Dict:
    """
    Extract modal operator information from a formula.

    Works for both full SMCDEL formulas (Exp 3) and hypothesis-only strings (Exp 4).

    Returns:
        dict with counts and operator type classification
    """
    # Count modal operators
    box_count = formula.count('knows that')  # □ - necessity
    diamond_count = formula.count('knows whether')  # ◇ - possibility

    # Classify formula type
    if box_count > 0 and diamond_count == 0:
        operator_type = 'necessity'
    elif diamond_count > 0 and box_count == 0:
        operator_type = 'possibility'
    elif box_count > 0 and diamond_count > 0:
        operator_type = 'mixed'
    else:
        operator_type = 'none'

    return {
        'box_count': box_count,
        'diamond_count': diamond_count,
        'operator_type': operator_type,
        'total_operators': box_count + diamond_count
    }


if __name__ == "__main__":
    # Test the utilities
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_format_utils.py <data_dir>")
        sys.exit(1)

    data_dir = Path(sys.argv[1])

    print(f"Testing data format detection for: {data_dir}")
    print("-" * 70)

    try:
        data_format = detect_data_format(data_dir)
        print(f"✓ Detected format: {data_format}")

        dataset, fmt = load_dataset(data_dir, 'train')
        print(f"✓ Loaded {len(dataset)} training examples")

        if dataset:
            example = dataset[0]
            formula = get_smcdel_formula(example, fmt)
            print(f"\n✓ First example formula:\n  {formula[:100]}...")

            op_info = parse_modal_operators(formula)
            print(f"\n✓ Modal operators:")
            print(f"  Type: {op_info['operator_type']}")
            print(f"  Necessity (□): {op_info['box_count']}")
            print(f"  Possibility (◇): {op_info['diamond_count']}")

        print("\n" + "=" * 70)
        print("✓ All tests passed!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
