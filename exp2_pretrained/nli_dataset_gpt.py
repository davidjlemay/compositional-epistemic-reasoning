#!/usr/bin/env python3
"""
NLI dataset for exp_6 GPT-style replication: decoder-only LM (Qwen2.5 etc.) on
epistemic entailment classification.

Decoder-only models have no [CLS]/[SEP] two-segment convention, so premise and
hypothesis are combined into a single explicit prompt string and tokenized once.
The hypothesis token span is recovered via the fast tokenizer's character-offset
mapping, so the attention-topology analysis can locate hypothesis tokens exactly
(the decoder-only analogue of `locate_hypothesis_tokens` in nli_dataset.py, which
relies on RoBERTa's `</s></s>` segment-boundary convention).
"""

import logging
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from nli_dataset import LABEL_MAP  # noqa: F401  (re-exported for callers)


class GPTNLIDataset(Dataset):
    """Single-sequence prompt dataset for decoder-only NLI fine-tuning."""

    def __init__(
        self,
        examples: List[Dict],
        tokenizer,
        max_length: int = 256,
        use_natural_language: bool = True,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_natural_language = use_natural_language

    def _get_texts(self, example: Dict):
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

        prefix = f"Premise: {premise}\nHypothesis: "
        suffix = "\nAnswer:"
        hyp_char_start = len(prefix)
        hyp_char_end = hyp_char_start + len(hypothesis)
        full_text = prefix + hypothesis + suffix

        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_offsets_mapping=True,
        )
        offsets = encoding['offset_mapping']

        hyp_token_start, hyp_token_end = None, None
        for i, (start, end) in enumerate(offsets):
            if start == end:
                continue  # special/added token with no character span
            if hyp_token_start is None and end > hyp_char_start:
                hyp_token_start = i
            if start < hyp_char_end:
                hyp_token_end = i + 1
        if hyp_token_start is None:
            hyp_token_start = len(offsets) - 1
        if hyp_token_end is None or hyp_token_end <= hyp_token_start:
            hyp_token_end = len(offsets)

        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'premise': premise,
            'hypothesis': hypothesis,
            'constraint': example['constraint'],
            'n_agents': example.get('n_agents', -1),
            'hypothesis_depth': example.get('hypothesis_depth', -1),
            'hyp_token_start': hyp_token_start,
            'hyp_token_end': hyp_token_end,
        }


def collate_fn_gpt(batch, pad_token_id: int):
    """Right-pad a batch. Padded positions are masked via attention_mask, so the
    filler id doesn't affect model behavior; pad_token_id is only used so decoded
    debug output doesn't show a misleading real token."""
    max_len = max(item['input_ids'].shape[0] for item in batch)

    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.stack([item['label'] for item in batch])

    for i, item in enumerate(batch):
        length = item['input_ids'].shape[0]
        input_ids[i, :length] = item['input_ids']
        attention_mask[i, :length] = item['attention_mask']

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'premise': [item['premise'] for item in batch],
        'hypothesis': [item['hypothesis'] for item in batch],
        'constraint': [item['constraint'] for item in batch],
        'n_agents': torch.tensor([item['n_agents'] for item in batch], dtype=torch.long),
        'hypothesis_depth': torch.tensor(
            [item['hypothesis_depth'] for item in batch], dtype=torch.long
        ),
        'hyp_token_start': torch.tensor(
            [item['hyp_token_start'] for item in batch], dtype=torch.long
        ),
        'hyp_token_end': torch.tensor(
            [item['hyp_token_end'] for item in batch], dtype=torch.long
        ),
    }
