#!/usr/bin/env python3
"""
NLI dataset for exp_6: RoBERTa cross-encoder on epistemic entailment classification.

Input: premise + hypothesis (concatenated as cross-encoder input)
Output: binary label (entailment=1, non_entailment=0)

Adapted from exp_5/natural_language_dataset.py.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset


LABEL_MAP = {'entailment': 1, 'non_entailment': 0}


class NLIDataset(Dataset):
    """
    Cross-encoder NLI dataset for epistemic reasoning.

    Each example is a (premise, hypothesis) pair with an entailment label.
    The tokenizer concatenates them as [CLS] premise [SEP] hypothesis [SEP].
    """

    def __init__(
        self,
        examples: List[Dict],
        tokenizer,
        max_length: int = 256,
        use_natural_language: bool = True,
    ):
        """
        Args:
            examples: List of dicts with 'premise'/'premise_natural', 'hypothesis'/'hypothesis_natural', 'constraint'
            tokenizer: HuggingFace tokenizer (e.g., RoBERTa)
            max_length: Max total sequence length for cross-encoder input
            use_natural_language: Use NL fields (premise_natural/hypothesis_natural) if available
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_natural_language = use_natural_language
        self.examples = examples

    def _get_texts(self, example: Dict):
        """Extract premise and hypothesis text, preferring NL variants."""
        if self.use_natural_language:
            premise = example.get('premise_natural') or example.get('premise', '')
            hypothesis = example.get('hypothesis_natural') or example.get('hypothesis', '')
        else:
            premise = example.get('premise', '')
            hypothesis = example.get('hypothesis', '')
        return premise, hypothesis

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        premise, hypothesis = self._get_texts(example)
        label = LABEL_MAP[example['constraint']]

        # Cross-encoder tokenization: [CLS] premise [SEP] hypothesis [SEP]
        encoding = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(
                encoding.get('token_type_ids', [0] * len(encoding['input_ids'])),
                dtype=torch.long
            ),
            'label': torch.tensor(label, dtype=torch.long),
            'premise': premise,
            'hypothesis': hypothesis,
            'constraint': example['constraint'],
            'n_agents': example.get('n_agents', -1),
            'hypothesis_depth': example.get('hypothesis_depth', -1),
        }


def collate_fn(batch):
    """Pad sequences to the same length within a batch."""
    max_len = max(item['input_ids'].shape[0] for item in batch)

    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    token_type_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.stack([item['label'] for item in batch])

    for i, item in enumerate(batch):
        length = item['input_ids'].shape[0]
        input_ids[i, :length] = item['input_ids']
        attention_mask[i, :length] = item['attention_mask']
        token_type_ids[i, :length] = item['token_type_ids']

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        'labels': labels,
        'premise': [item['premise'] for item in batch],
        'hypothesis': [item['hypothesis'] for item in batch],
        'constraint': [item['constraint'] for item in batch],
        'n_agents': torch.tensor([item['n_agents'] for item in batch], dtype=torch.long),
        'hypothesis_depth': torch.tensor([item['hypothesis_depth'] for item in batch], dtype=torch.long),
    }


def load_nli_dataset(data_dir: Path) -> Dict[str, List[Dict]]:
    """
    Load train/val splits from agent_aware generation directory.
    Uses train_generation.json and val_generation.json (same format as exp_5).
    """
    data_dir = Path(data_dir)
    train_file = data_dir / 'train_generation.json'
    val_file = data_dir / 'val_generation.json'

    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not val_file.exists():
        raise FileNotFoundError(f"Validation file not found: {val_file}")

    with open(train_file) as f:
        train_examples = json.load(f)
    with open(val_file) as f:
        val_examples = json.load(f)

    logging.info(f"Loaded {len(train_examples)} train, {len(val_examples)} val examples")
    return {'train': train_examples, 'validation': val_examples}


def load_test_set(test_dir: Path) -> List[Dict]:
    """Load held-out test set (unseen agent names, higher complexity)."""
    test_dir = Path(test_dir)
    test_file = test_dir / 'test_generation.json'
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    with open(test_file) as f:
        examples = json.load(f)
    logging.info(f"Loaded {len(examples)} held-out test examples")
    return examples
