#!/usr/bin/env python3
"""
Compact Transformer model for epistemic logic grokking experiments.
Designed to have similar parameter count (~13-15k) to the GRU baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            x with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CompactTransformer(nn.Module):
    """
    Compact Transformer encoder-decoder for sequence-to-sequence learning.
    Designed to be parameter-constrained to encourage grokking.
    """

    def __init__(
        self,
        vocab_size,
        d_model=32,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=64,
        dropout=0.1,
        max_seq_len=100
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Shared embedding layer (for both encoder and decoder)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )

        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_padding_mask(self, seq, pad_idx=0):
        """
        Create padding mask for attention.
        Args:
            seq: [batch, seq_len]
        Returns:
            mask: [batch, seq_len] with True for padding positions
        """
        return (seq == pad_idx)

    def create_causal_mask(self, seq_len, device):
        """
        Create causal (look-ahead) mask for decoder.
        Args:
            seq_len: length of sequence
        Returns:
            mask: [seq_len, seq_len] upper triangular matrix
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask

    def encode(self, src, src_padding_mask=None):
        """
        Encode source sequence.
        Args:
            src: [batch, src_len] token indices
            src_padding_mask: [batch, src_len] padding mask
        Returns:
            memory: [batch, src_len, d_model]
        """
        # Embed and add positional encoding
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)

        # Encode
        memory = self.encoder(src_emb, src_key_padding_mask=src_padding_mask)

        return memory

    def decode(self, tgt, memory, tgt_causal_mask=None, tgt_padding_mask=None,
               memory_padding_mask=None, return_hidden=False):
        """
        Decode target sequence given encoder memory.
        Args:
            tgt: [batch, tgt_len] token indices
            memory: [batch, src_len, d_model] encoder outputs
            tgt_causal_mask: [tgt_len, tgt_len] causal mask
            tgt_padding_mask: [batch, tgt_len] padding mask
            memory_padding_mask: [batch, src_len] source padding mask
            return_hidden: If True, return both output and hidden states
        Returns:
            output: [batch, tgt_len, vocab_size]
            decoder_output: [batch, tgt_len, d_model] (only if return_hidden=True)
        """
        # Embed and add positional encoding
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)

        # Decode
        decoder_output = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_causal_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_padding_mask
        )

        # Project to vocabulary
        output = self.output_projection(decoder_output)

        if return_hidden:
            return output, decoder_output
        return output

    def forward(self, src, tgt, src_padding_mask=None, tgt_padding_mask=None):
        """
        Forward pass for training.
        Args:
            src: [batch, src_len] source sequences
            tgt: [batch, tgt_len] target sequences (with <sos> prepended)
            src_padding_mask: [batch, src_len] source padding mask
            tgt_padding_mask: [batch, tgt_len] target padding mask
        Returns:
            output: [batch, tgt_len, vocab_size] logits
        """
        # Encode
        memory = self.encode(src, src_padding_mask)

        # Create causal mask for decoder
        tgt_len = tgt.size(1)
        tgt_causal_mask = self.create_causal_mask(tgt_len, tgt.device)

        # Decode
        output = self.decode(
            tgt,
            memory,
            tgt_causal_mask=tgt_causal_mask,
            tgt_padding_mask=tgt_padding_mask,
            memory_padding_mask=src_padding_mask
        )

        return output

    def generate(self, src, max_length=20, sos_token=1, eos_token=2, pad_token=0):
        """
        Autoregressive generation.
        Args:
            src: [batch, src_len] source sequences
            max_length: maximum generation length
            sos_token: start-of-sequence token
            eos_token: end-of-sequence token
            pad_token: padding token
        Returns:
            predictions: [batch, max_length] generated sequences
        """
        device = src.device
        batch_size = src.size(0)

        # Create source padding mask
        src_padding_mask = self.create_padding_mask(src, pad_token)

        # Encode source
        memory = self.encode(src, src_padding_mask)

        # Initialize target with <sos> token
        tgt = torch.full((batch_size, 1), sos_token, dtype=torch.long, device=device)

        # Generate tokens autoregressively
        for _ in range(max_length - 1):
            # Create causal mask
            tgt_causal_mask = self.create_causal_mask(tgt.size(1), device)

            # Decode
            output = self.decode(
                tgt,
                memory,
                tgt_causal_mask=tgt_causal_mask,
                memory_padding_mask=src_padding_mask
            )

            # Get next token (greedy)
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)

            # Append to target
            tgt = torch.cat([tgt, next_token], dim=1)

            # Check if all sequences have generated <eos>
            if (next_token == eos_token).all():
                break

        return tgt


class HiddenStateCapture(nn.Module):
    """
    Module to capture hidden states from transformer for topological analysis.
    Wraps the encoder and decoder to save intermediate representations.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.captured_encoder_states = []
        self.captured_decoder_states = []
        self.capture_enabled = False

    def enable_capture(self):
        """Enable hidden state capture."""
        self.capture_enabled = True
        self.captured_encoder_states = []
        self.captured_decoder_states = []

    def disable_capture(self):
        """Disable hidden state capture."""
        self.capture_enabled = False

    def get_captured_states(self):
        """Return captured states."""
        return {
            'encoder_states': self.captured_encoder_states,
            'decoder_states': self.captured_decoder_states
        }

    def forward(self, src, tgt, src_padding_mask=None, tgt_padding_mask=None):
        """
        Forward pass with optional state capture.
        """
        # Encode
        memory = self.model.encode(src, src_padding_mask)

        if self.capture_enabled:
            # Capture encoder output (mean over sequence for each sample)
            # [batch, src_len, d_model] -> [batch, d_model]
            encoder_repr = memory.mean(dim=1)
            self.captured_encoder_states.append(encoder_repr.detach().cpu())

        # Create causal mask
        tgt_len = tgt.size(1)
        tgt_causal_mask = self.model.create_causal_mask(tgt_len, tgt.device)

        # Decode (request hidden states if capturing)
        if self.capture_enabled:
            output, decoder_hidden = self.model.decode(
                tgt,
                memory,
                tgt_causal_mask=tgt_causal_mask,
                tgt_padding_mask=tgt_padding_mask,
                memory_padding_mask=src_padding_mask,
                return_hidden=True
            )
            # Capture decoder hidden states (mean over sequence)
            # [batch, tgt_len, d_model] -> [batch, d_model]
            decoder_repr = decoder_hidden.mean(dim=1)
            self.captured_decoder_states.append(decoder_repr.detach().cpu())
        else:
            output = self.model.decode(
                tgt,
                memory,
                tgt_causal_mask=tgt_causal_mask,
                tgt_padding_mask=tgt_padding_mask,
                memory_padding_mask=src_padding_mask,
                return_hidden=False
            )

        return output


def create_epistemic_transformer(vocab_size, d_model=32, nhead=2,
                                num_encoder_layers=1, num_decoder_layers=1,
                                dim_feedforward=64, dropout=0.1):
    """
    Factory function to create a compact transformer for epistemic logic.

    Args:
        vocab_size: size of vocabulary
        d_model: embedding dimension (default: 32)
        nhead: number of attention heads (default: 2)
        num_encoder_layers: number of encoder layers (default: 1)
        num_decoder_layers: number of decoder layers (default: 1)
        dim_feedforward: feedforward dimension (default: 64, 2x d_model)
        dropout: dropout rate

    Returns:
        model: CompactTransformer model
    """
    model = CompactTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    )

    return model


if __name__ == "__main__":
    # Test the model
    vocab_size = 59
    batch_size = 8
    src_len = 20
    tgt_len = 5

    model = create_epistemic_transformer(vocab_size)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Test forward pass
    src = torch.randint(0, vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))

    output = model(src, tgt)
    print(f"Output shape: {output.shape}")  # Should be [batch, tgt_len, vocab_size]

    # Test generation
    generated = model.generate(src, max_length=10)
    print(f"Generated shape: {generated.shape}")  # Should be [batch, max_length]

    # Test with hidden state capture
    wrapper = HiddenStateCapture(model)
    wrapper.enable_capture()
    output = wrapper(src, tgt)
    states = wrapper.get_captured_states()
    print(f"Captured encoder states: {len(states['encoder_states'])} batches")
    print(f"Captured decoder states: {len(states['decoder_states'])} batches")
