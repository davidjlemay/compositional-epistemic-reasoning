#!/usr/bin/env python3
"""
Epistemic Logic Tokenizer for MindGames dataset.
Handles tokenization of SMCDEL symbolic logic formulas.
"""

import torch
from torch.nn.utils.rnn import pad_sequence
import re
from typing import List, Dict, Tuple, Optional, Union
import json

class EpistemicLogicTokenizer:
    """
    Tokenizer for epistemic logic formulas in SMCDEL format.
    Handles agents, knowledge operators, propositional variables, and logical connectives.
    """

    def __init__(self):
        # Define the core vocabulary for epistemic logic
        self.special_tokens = {
            '<PAD>': 0,
            '<SOS>': 1,  # Start of sequence
            '<EOS>': 2,  # End of sequence
            '<UNK>': 3,  # Unknown token
        }

        # Core SMCDEL tokens
        self.smcdel_keywords = ['VARS', 'LAW', 'Top', 'OBS', 'VALID?']
        self.modal_operators = ['knows', 'whether']
        self.logical_operators = ['&', '|', '~', '!']
        self.punctuation = ['(', ')', '[', ']', ':', ',', '?']
        self.boolean_operators = ['that']

        # Agent patterns and variables will be extracted from data
        self.agent_tokens = []
        self.variable_tokens = []
        self.compound_tokens = []

        # Build base vocabulary
        self.vocab = {}
        self.idx = len(self.special_tokens)

        # Add special tokens
        for token, idx in self.special_tokens.items():
            self.vocab[token] = idx

        # Add core tokens
        for token_list in [self.smcdel_keywords, self.modal_operators,
                          self.logical_operators, self.punctuation, self.boolean_operators]:
            for token in token_list:
                if token not in self.vocab:
                    self.vocab[token] = self.idx
                    self.idx += 1

        print(f"Base vocabulary size: {len(self.vocab)}")

    def build_vocabulary_from_data(self, symbolic_formulas: List[str]):
        """Build vocabulary from the actual symbolic formulas in the dataset."""

        print("Building vocabulary from symbolic formulas...")

        # Collect all unique tokens
        token_counts = {}

        for formula in symbolic_formulas:
            if formula:
                tokens = self.tokenize_smcdel_formula(formula, vocab_building=True)
                for token in tokens:
                    token_counts[token] = token_counts.get(token, 0) + 1

        # Add frequent tokens to vocabulary
        for token, count in token_counts.items():
            if token not in self.vocab and count >= 2:  # Only add tokens that appear at least twice
                self.vocab[token] = self.idx
                self.idx += 1

        # Create reverse mapping
        self.idx_to_token = {idx: token for token, idx in self.vocab.items()}

        print(f"Final vocabulary size: {len(self.vocab)}")
        print(f"Sample tokens: {list(self.vocab.keys())[:20]}")

        # Categorize tokens for analysis
        self.agent_tokens = [t for t in self.vocab.keys() if 'Agent' in t]
        self.variable_tokens = [t for t in self.vocab.keys() if t.isdigit()]
        self.compound_tokens = [t for t in self.vocab.keys() if ('&' in t or '|' in t) and len(t) > 1]

        print(f"Agent tokens: {len(self.agent_tokens)}")
        print(f"Variable tokens: {self.variable_tokens}")
        print(f"Sample compound tokens: {self.compound_tokens[:10]}")

    def tokenize_smcdel_formula(self, formula: str, vocab_building: bool = False) -> List[str]:
        """
        Tokenize an SMCDEL formula into meaningful tokens.
        Uses regex patterns to identify multi-character tokens.
        """
        if not formula.strip():
            return []

        # Define patterns for multi-character tokens
        patterns = [
            (r'VALID\?', 'VALID?'),
            (r'VARS', 'VARS'),
            (r'LAW', 'LAW'),
            (r'Top', 'Top'),
            (r'OBS', 'OBS'),
            (r'Agent[a-z]', None),  # Agent names
            (r'knows\s+whether', 'knows_whether'),  # Combined operator
            (r'knows\s+that', 'knows_that'),  # Combined operator
            (r'knows', 'knows'),
            (r'whether', 'whether'),
            (r'that', 'that'),
            (r'\d+(?:[&|]\d+)*', None),  # Variable combinations like "1&2&3"
            (r'Agent[a-z]:\d+(?:,\d+)*', None),  # Agent observations like "Agenta:1,2,3"
            (r'[~!]', None),  # Negation operators
            (r'[&|]', None),  # Logical operators
            (r'[()[\]]', None),  # Parentheses and brackets
            (r'[,:?]', None),  # Punctuation
            (r'\d+', None),  # Individual numbers
            (r'\s+', 'SPACE'),  # Whitespace (we'll filter this out)
        ]

        tokens = []
        remaining = formula

        while remaining.strip():
            matched = False

            for pattern, replacement in patterns:
                match = re.match(pattern, remaining)
                if match:
                    token = replacement if replacement else match.group(0)
                    if token != 'SPACE':  # Skip whitespace
                        tokens.append(token)
                    remaining = remaining[match.end():].strip()
                    matched = True
                    break

            if not matched:
                # Take the first character if no pattern matches
                if remaining[0] not in [' ', '\t', '\n']:
                    tokens.append(remaining[0])
                remaining = remaining[1:].strip()

        return tokens

    def encode(self, text: str, add_special_tokens: bool = True, max_length: Optional[int] = None) -> List[int]:
        """Encode text into token IDs."""
        tokens = self.tokenize_smcdel_formula(text)

        if add_special_tokens:
            tokens = ['<SOS>'] + tokens + ['<EOS>']

        # Convert to IDs
        token_ids = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]

        # Truncate if necessary
        if max_length is not None and len(token_ids) > max_length:
            if add_special_tokens:
                # Keep SOS and EOS
                token_ids = token_ids[:max_length-1] + [self.vocab['<EOS>']]
            else:
                token_ids = token_ids[:max_length]

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        tokens = [self.idx_to_token.get(idx, '<UNK>') for idx in token_ids]

        if skip_special_tokens:
            tokens = [token for token in tokens if token not in self.special_tokens]

        # Join tokens with appropriate spacing
        result_parts = []
        for i, token in enumerate(tokens):
            if i > 0 and not token.startswith(':') and not tokens[i-1].endswith(':'):
                if token not in ['(', ')', '[', ']', ',', ':', '?', '&', '|', '~', '!']:
                    result_parts.append(' ')
            result_parts.append(token)

        return ''.join(result_parts)

    def batch_encode(self, texts: List[str], max_length: Optional[int] = None,
                    padding: bool = True) -> Dict[str, torch.Tensor]:
        """Batch encode multiple texts."""
        encoded_texts = [self.encode(text, max_length=max_length) for text in texts]

        if padding:
            # Find max length in batch
            if max_length is None:
                max_length = max(len(encoded) for encoded in encoded_texts) if encoded_texts else 1

            # Pad sequences
            padded = []
            attention_masks = []

            for encoded in encoded_texts:
                # Pad or truncate
                if len(encoded) < max_length:
                    attention_mask = [1] * len(encoded) + [0] * (max_length - len(encoded))
                    encoded = encoded + [self.vocab['<PAD>']] * (max_length - len(encoded))
                else:
                    encoded = encoded[:max_length]
                    attention_mask = [1] * max_length

                padded.append(encoded)
                attention_masks.append(attention_mask)

            return {
                'input_ids': torch.tensor(padded, dtype=torch.long),
                'attention_mask': torch.tensor(attention_masks, dtype=torch.long)
            }
        else:
            return {
                'input_ids': [torch.tensor(encoded, dtype=torch.long) for encoded in encoded_texts]
            }

    def save_vocab(self, path: str):
        """Save vocabulary to file."""
        vocab_data = {
            'vocab': self.vocab,
            'agent_tokens': self.agent_tokens,
            'variable_tokens': self.variable_tokens,
            'compound_tokens': self.compound_tokens
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)

    def load_vocab(self, path: str):
        """Load vocabulary from file."""
        with open(path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)

        self.vocab = vocab_data['vocab']
        self.agent_tokens = vocab_data.get('agent_tokens', [])
        self.variable_tokens = vocab_data.get('variable_tokens', [])
        self.compound_tokens = vocab_data.get('compound_tokens', [])
        self.idx_to_token = {int(idx): token for token, idx in self.vocab.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        return self.vocab['<PAD>']

    @property
    def sos_token_id(self) -> int:
        return self.vocab['<SOS>']

    @property
    def eos_token_id(self) -> int:
        return self.vocab['<EOS>']

    @property
    def unk_token_id(self) -> int:
        return self.vocab['<UNK>']


def test_epistemic_tokenizer():
    """Test the tokenizer with actual SMCDEL formulas."""

    # Load a small sample of symbolic formulas to build vocabulary
    from datasets import load_dataset

    print("Loading MindGames dataset for tokenizer testing...")
    dataset = load_dataset("sileod/mindgames")

    # Collect sample formulas
    sample_formulas = []
    for split_data in dataset.values():
        for i, example in enumerate(split_data):
            if i < 100:  # Use first 100 examples from each split
                sample_formulas.append(example['smcdel_problem'])

    # Create and build tokenizer
    tokenizer = EpistemicLogicTokenizer()
    tokenizer.build_vocabulary_from_data(sample_formulas)

    # Test with actual examples
    test_formulas = [
        "VARS 1,2,3,4 LAW Top OBS Agenta:1 Agentb:2 Agentc:3 Agentd:4 VALID? [ ! (1|2|3|4) ] Agentd knows whether (Agenta knows whether 3)",
        "VARS 1,2 LAW Top OBS Agenta:2 Agentb:1 VALID? [ ! (1|2) ] (Agenta knows whether 2)",
        "VARS 1,2,3 LAW Top OBS Agenta:3 Agentb:3 Agentc:1,2 VALID? [ ! (1|2|3) ] (Agenta knows that (~(1|2|3)))"
    ]

    print(f"\n=== TOKENIZER TESTING ===")

    for i, formula in enumerate(test_formulas):
        print(f"\nTest {i+1}:")
        print(f"Formula: {formula}")

        # Tokenize
        tokens = tokenizer.tokenize_smcdel_formula(formula)
        print(f"Tokens ({len(tokens)}): {tokens}")

        # Encode
        encoded = tokenizer.encode(formula)
        print(f"Encoded ({len(encoded)}): {encoded}")

        # Decode
        decoded = tokenizer.decode(encoded)
        print(f"Decoded: {decoded}")

    # Test batch encoding
    print(f"\n=== BATCH ENCODING TEST ===")
    batch_result = tokenizer.batch_encode(test_formulas, max_length=50)
    print(f"Batch input_ids shape: {batch_result['input_ids'].shape}")
    print(f"Batch attention_mask shape: {batch_result['attention_mask'].shape}")

    # Save tokenizer
    tokenizer.save_vocab('epistemic_logic_tokenizer.json')
    print(f"\nTokenizer saved to epistemic_logic_tokenizer.json")

    return tokenizer

if __name__ == "__main__":
    test_epistemic_tokenizer()