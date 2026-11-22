#!/usr/bin/env python3
"""
Mamba-based model for epistemic logic grokking experiments.
Uses selective state space models as an alternative to attention.

Mamba is a continuous map (state space transition + selection mechanism)
so our product topology theory should still apply!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("WARNING: mamba-ssm not installed. Install with: pip install mamba-ssm")


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (optional for Mamba, but helps)."""

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


class MambaEncoder(nn.Module):
    """
    Mamba-based encoder using selective state space models.
    Replaces Transformer encoder with Mamba layers.
    """

    def __init__(self, d_model, num_layers=1, dropout=0.1):
        super().__init__()

        if not MAMBA_AVAILABLE:
            raise ImportError("mamba-ssm package required. Install with: pip install mamba-ssm")

        self.d_model = d_model
        self.num_layers = num_layers

        # Stack of Mamba layers
        self.layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=16,  # State dimension (default)
                d_conv=4,    # Convolution kernel size
                expand=2     # Expansion factor
            )
            for _ in range(num_layers)
        ])

        # Layer norms
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            output: [batch, seq_len, d_model]
        """
        # Pass through Mamba layers with residual connections
        for layer, norm in zip(self.layers, self.norms):
            # Mamba layer
            x_residual = x
            x = layer(x) + x_residual
            x = norm(x)
            x = self.dropout(x)

        return x


class MambaDecoder(nn.Module):
    """
    Mamba-based decoder with cross-attention to encoder outputs.
    Uses Mamba for self-processing and linear cross-attention to encoder.
    """

    def __init__(self, d_model, num_layers=1, dropout=0.1):
        super().__init__()

        if not MAMBA_AVAILABLE:
            raise ImportError("mamba-ssm package required. Install with: pip install mamba-ssm")

        self.d_model = d_model
        self.num_layers = num_layers

        # Mamba layers for self-processing
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=16,
                d_conv=4,
                expand=2
            )
            for _ in range(num_layers)
        ])

        # Cross-attention to encoder (simplified linear attention)
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads=2, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])

        # Layer norms
        self.mamba_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers)
        ])

        self.cross_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None):
        """
        Args:
            x: [batch, tgt_len, d_model] - decoder input
            memory: [batch, src_len, d_model] - encoder output
            tgt_mask: [tgt_len, tgt_len] - causal mask (optional, Mamba handles this)
        Returns:
            output: [batch, tgt_len, d_model]
        """
        for mamba_layer, cross_attn, mamba_norm, cross_norm in zip(
            self.mamba_layers, self.cross_attn_layers, self.mamba_norms, self.cross_norms
        ):
            # Self-processing with Mamba (handles causality automatically)
            x_residual = x
            x = mamba_layer(x) + x_residual
            x = mamba_norm(x)
            x = self.dropout(x)

            # Cross-attention to encoder memory
            x_residual = x
            x_attn, _ = cross_attn(x, memory, memory)
            x = x_attn + x_residual
            x = cross_norm(x)
            x = self.dropout(x)

        return x


class CompactMamba(nn.Module):
    """
    Compact Mamba encoder-decoder for sequence-to-sequence learning.
    Uses selective state space models instead of attention.

    Key theoretical property: Mamba is also a CONTINUOUS MAP!
    - State transitions: continuous
    - Selection mechanism: continuous (gating)
    - Linear projections: continuous

    Therefore, product topology preservation should still hold!
    """

    def __init__(
        self,
        vocab_size,
        d_model=32,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dropout=0.1,
        max_seq_len=100
    ):
        super().__init__()

        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba-ssm package required for Mamba model.\n"
                "Install with: pip install mamba-ssm\n"
                "Requires CUDA and causal-conv1d."
            )

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Shared embedding layer (for both encoder and decoder)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # Positional encoding (optional for Mamba, but can help)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        # Mamba encoder
        self.encoder = MambaEncoder(
            d_model=d_model,
            num_layers=num_encoder_layers,
            dropout=dropout
        )

        # Mamba decoder
        self.decoder = MambaDecoder(
            d_model=d_model,
            num_layers=num_decoder_layers,
            dropout=dropout
        )

        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for name, p in self.named_parameters():
            # Skip Mamba internal parameters (they have their own initialization)
            if 'mamba_layers' in name or 'layers' in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_causal_mask(self, seq_len, device):
        """
        Create causal (look-ahead) mask for decoder.
        Note: Mamba handles causality internally, so this is mostly for compatibility.
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask

    def encode(self, src):
        """
        Encode source sequence using Mamba.
        Args:
            src: [batch, src_len] token indices
        Returns:
            memory: [batch, src_len, d_model]
        """
        # Embed and add positional encoding
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)

        # Encode with Mamba
        memory = self.encoder(src_emb)

        return memory

    def decode(self, tgt, memory, return_hidden=False):
        """
        Decode target sequence given encoder memory using Mamba.
        Args:
            tgt: [batch, tgt_len] token indices
            memory: [batch, src_len, d_model] encoder outputs
            return_hidden: If True, return both output and hidden states
        Returns:
            output: [batch, tgt_len, vocab_size]
            decoder_output: [batch, tgt_len, d_model] (only if return_hidden=True)
        """
        # Embed and add positional encoding
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)

        # Decode with Mamba (causality handled internally)
        decoder_output = self.decoder(tgt_emb, memory)

        # Project to vocabulary
        output = self.output_projection(decoder_output)

        if return_hidden:
            return output, decoder_output
        return output

    def forward(self, src, tgt):
        """
        Forward pass for training.
        Args:
            src: [batch, src_len] source sequences
            tgt: [batch, tgt_len] target sequences (with <sos> prepended)
        Returns:
            output: [batch, tgt_len, vocab_size] logits
        """
        # Encode
        memory = self.encode(src)

        # Decode
        output = self.decode(tgt, memory)

        return output

    def generate(self, src, max_length=20, sos_token=1, eos_token=2):
        """
        Autoregressive generation using Mamba.
        Args:
            src: [batch, src_len] source sequences
            max_length: maximum generation length
            sos_token: start-of-sequence token
            eos_token: end-of-sequence token
        Returns:
            predictions: [batch, max_length] generated sequences
        """
        device = src.device
        batch_size = src.size(0)

        # Encode source
        memory = self.encode(src)

        # Initialize target with <sos> token
        tgt = torch.full((batch_size, 1), sos_token, dtype=torch.long, device=device)

        # Generate tokens autoregressively
        for _ in range(max_length - 1):
            # Decode (Mamba handles causality)
            output = self.decode(tgt, memory)

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
    Module to capture hidden states from Mamba for topological analysis.
    Wraps the encoder and decoder to save intermediate representations.

    Tests theoretical prediction: Mamba (state space model) should also
    preserve product topology since it's a continuous map!
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

    def forward(self, src, tgt):
        """
        Forward pass with optional state capture.
        """
        # Encode
        memory = self.model.encode(src)

        if self.capture_enabled:
            # Capture encoder output (mean over sequence for each sample)
            # [batch, src_len, d_model] -> [batch, d_model]
            encoder_repr = memory.mean(dim=1)
            self.captured_encoder_states.append(encoder_repr.detach().cpu())

        # Decode (request hidden states if capturing)
        if self.capture_enabled:
            output, decoder_hidden = self.model.decode(
                tgt,
                memory,
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
                return_hidden=False
            )

        return output


def create_epistemic_mamba(vocab_size, d_model=32,
                           num_encoder_layers=1, num_decoder_layers=1,
                           dropout=0.1):
    """
    Factory function to create a compact Mamba for epistemic logic.

    Args:
        vocab_size: size of vocabulary
        d_model: embedding dimension (default: 32)
        num_encoder_layers: number of Mamba encoder layers (default: 1)
        num_decoder_layers: number of Mamba decoder layers (default: 1)
        dropout: dropout rate

    Returns:
        model: CompactMamba model
    """
    model = CompactMamba(
        vocab_size=vocab_size,
        d_model=d_model,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dropout=dropout
    )

    return model


if __name__ == "__main__":
    # Test the model
    if not MAMBA_AVAILABLE:
        print("Mamba not available. Install with: pip install mamba-ssm")
        print("Requires CUDA and causal-conv1d.")
        exit(1)

    vocab_size = 59
    batch_size = 8
    src_len = 20
    tgt_len = 5

    model = create_epistemic_mamba(vocab_size)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Test forward pass
    src = torch.randint(0, vocab_size, (batch_size, src_len)).cuda()
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len)).cuda()
    model = model.cuda()

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

    print("\n✓ Mamba model tests passed!")
    print("✓ Ready for product topology validation experiments")
