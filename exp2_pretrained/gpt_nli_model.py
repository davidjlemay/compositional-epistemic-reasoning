#!/usr/bin/env python3
"""
Decoder-only (GPT-style) analogue of roberta_nli_model.py for epistemic NLI
classification (generalization check beyond a bidirectional encoder).

Architecture:
  - Any causal-LM backbone with Llama-family module naming (`.embed_tokens`,
    `.layers`) — this covers Qwen2/Qwen2.5, Llama, Mistral, Gemma/Gemma2.
  - Input: single prompt "Premise: ... \nHypothesis: ... \nAnswer:" (no [CLS]/[SEP]
    convention available for decoder-only models; see nli_dataset_gpt.py).
  - Classification head on the LAST non-padding token's hidden state, since causal
    attention means only the final token has attended to the full sequence —
    the decoder-only analogue of RoBERTa's [CLS]-token (position-0) pooling.
  - Per-layer hidden state capture at the same pooling position, for the VR
    topology / probing analyses used in the small-model and RoBERTa experiments.
"""

import logging
from typing import Dict, List

import torch
import torch.nn as nn
from transformers import AutoModel


class GPTNLI(nn.Module):
    """Causal-LM backbone + last-token classification head for NLI."""

    def __init__(
        self,
        model_name: str,
        num_labels: int = 2,
        dropout: float = 0.1,
        freeze_layers: int = 0,
        torch_dtype: str = 'float32',
    ):
        """
        Args:
            model_name: HuggingFace model identifier (e.g. 'Qwen/Qwen2.5-0.5B')
            num_labels: Number of output classes (2: entailment / non_entailment)
            dropout: Dropout on classification head
            freeze_layers: Number of bottom transformer layers to freeze (0 = fine-tune all)
            torch_dtype: 'float32' or 'bfloat16' — use bfloat16 for larger models
                (e.g. 1.5B) if fp32 AdamW state doesn't fit in GPU memory.
        """
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels

        dtype = getattr(torch, torch_dtype)
        logging.info(f"Loading {model_name} for NLI classification (dtype={torch_dtype})")
        # Use eager attention so output_attentions=True works for topology analysis.
        # sdpa/flash-attn (the transformers default for these models) do not return
        # attention weights.
        self.transformer = AutoModel.from_pretrained(
            model_name,
            attn_implementation='eager',
            torch_dtype=dtype,
        )

        self.layers = self._find_layers()
        self.embed_tokens = self._find_embed_tokens()

        if freeze_layers > 0:
            self._freeze_bottom_layers(freeze_layers)

        hidden_size = self.transformer.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        ).to(dtype)

        self.num_layers = self.transformer.config.num_hidden_layers
        self.num_heads = self.transformer.config.num_attention_heads
        logging.info(
            f"GPTNLI: {self.num_layers} layers, {self.num_heads} heads, "
            f"hidden_size={hidden_size}, freeze_layers={freeze_layers}"
        )

    def _find_layers(self):
        if hasattr(self.transformer, 'layers'):
            return self.transformer.layers
        raise AttributeError(
            f"Could not locate decoder layers on {self.model_name} "
            f"(expected `.layers`, Llama-family naming). Inspect "
            f"`AutoModel.from_pretrained('{self.model_name}')` and adapt _find_layers()."
        )

    def _find_embed_tokens(self):
        if hasattr(self.transformer, 'embed_tokens'):
            return self.transformer.embed_tokens
        raise AttributeError(
            f"Could not locate token embeddings on {self.model_name} "
            f"(expected `.embed_tokens`). Inspect the model structure and adapt "
            f"_find_embed_tokens()."
        )

    def _freeze_bottom_layers(self, n: int):
        for param in self.embed_tokens.parameters():
            param.requires_grad = False
        for layer in self.layers[:n]:
            for param in layer.parameters():
                param.requires_grad = False
        logging.info(f"Frozen embeddings + bottom {n} transformer layers")

    @staticmethod
    def _last_token_indices(attention_mask: torch.Tensor) -> torch.Tensor:
        """Index of the last non-pad token per example (batches are right-padded)."""
        return attention_mask.sum(dim=1) - 1

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        idx = self._last_token_indices(attention_mask)
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        last_hidden = outputs.hidden_states[-1]
        pooled = last_hidden[batch_idx, idx]  # [batch, hidden]
        logits = self.classifier(pooled.to(self.classifier[1].weight.dtype))

        result = {'logits': logits}
        if output_hidden_states:
            result['hidden_states'] = outputs.hidden_states
        return result


class HiddenStateCapture:
    """
    Captures per-layer last-token representations during evaluation for topology
    analysis — the decoder-only analogue of RoBERTa's per-layer [CLS] capture.

    Because the pooling position varies per example (last non-pad token, not a
    fixed index 0 as with RoBERTa's [CLS]), the batch's attention_mask must be
    registered via `set_batch_context` immediately before each forward pass.

    Usage:
        capture = HiddenStateCapture(model)
        capture.start()
        for batch in loader:
            capture.set_batch_context(batch['attention_mask'])
            model(batch['input_ids'], batch['attention_mask'])
        states = capture.get_states()
        capture.stop()
    """

    def __init__(self, model: GPTNLI):
        self.model = model
        self.captured: Dict[int, List[torch.Tensor]] = {}
        self._active = False
        self._hooks = []
        self._last_idx = None

    def set_batch_context(self, attention_mask: torch.Tensor):
        self._last_idx = (attention_mask.sum(dim=1) - 1).cpu()

    def start(self):
        """Register forward hooks on the embedding layer and each decoder layer."""
        self._active = True
        self.captured = {i: [] for i in range(self.model.num_layers + 1)}

        def make_hook(layer_idx):
            def hook(module, input, output):
                if self._active and self._last_idx is not None:
                    hidden = output[0] if isinstance(output, tuple) else output
                    hidden = hidden.detach().float().cpu()
                    batch_idx = torch.arange(hidden.size(0))
                    self.captured[layer_idx].append(hidden[batch_idx, self._last_idx])
            return hook

        h = self.model.embed_tokens.register_forward_hook(make_hook(0))
        self._hooks.append(h)
        for i, layer in enumerate(self.model.layers):
            h = layer.register_forward_hook(make_hook(i + 1))
            self._hooks.append(h)

    def stop(self):
        """Remove all hooks."""
        self._active = False
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def get_states(self) -> Dict[int, torch.Tensor]:
        """Return concatenated last-token states per layer: {layer_idx: [N, hidden_size]}."""
        return {
            layer_idx: torch.cat(tensors, dim=0)
            for layer_idx, tensors in self.captured.items()
            if tensors
        }
