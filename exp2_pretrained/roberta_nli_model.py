#!/usr/bin/env python3
"""
RoBERTa cross-encoder for epistemic NLI classification (exp_6).

Architecture:
  - roberta-base (fine-tuned end-to-end, not frozen)
  - Input: [CLS] premise [SEP] hypothesis [SEP]
  - Classification head on [CLS] token representation
  - Outputs: binary logits (entailment / non_entailment)
  - Hidden state capture at each layer for topological analysis
"""

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig
from typing import Optional, Dict, List
import logging


class RobertaNLI(nn.Module):
    """
    RoBERTa-based cross-encoder for epistemic entailment classification.

    Exposes per-layer hidden states for topological analysis.
    """

    def __init__(
        self,
        model_name: str = 'roberta-base',
        num_labels: int = 2,
        dropout: float = 0.1,
        freeze_layers: int = 0,
    ):
        """
        Args:
            model_name: HuggingFace model identifier
            num_labels: Number of output classes (2: entailment / non_entailment)
            dropout: Dropout on classification head
            freeze_layers: Number of bottom transformer layers to freeze (0 = fine-tune all)
        """
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels

        logging.info(f"Loading {model_name} for NLI classification")
        # Use eager attention so output_attentions=True works for topology analysis.
        # sdpa (transformers>=4.36 default) does not return attention weights.
        self.roberta = RobertaModel.from_pretrained(
            model_name,
            add_pooling_layer=True,
            attn_implementation='eager',
        )

        # Optionally freeze bottom N layers to reduce compute / prevent catastrophic forgetting
        if freeze_layers > 0:
            self._freeze_bottom_layers(freeze_layers)

        hidden_size = self.roberta.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

        self.num_layers = self.roberta.config.num_hidden_layers
        logging.info(
            f"RobertaNLI: {self.num_layers} layers, hidden_size={hidden_size}, "
            f"freeze_layers={freeze_layers}"
        )

    def _freeze_bottom_layers(self, n: int):
        """Freeze embeddings and the bottom n transformer layers."""
        for param in self.roberta.embeddings.parameters():
            param.requires_grad = False
        for layer in self.roberta.encoder.layer[:n]:
            for param in layer.parameters():
                param.requires_grad = False
        logging.info(f"Frozen embeddings + bottom {n} transformer layers")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            token_type_ids: [batch, seq_len] (ignored by RoBERTa but kept for API compat)
            output_hidden_states: If True, return all layer hidden states

        Returns:
            dict with 'logits' and optionally 'hidden_states'
        """
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )

        # Use pooler output ([CLS] representation after dense + tanh)
        cls_repr = outputs.pooler_output  # [batch, hidden_size]
        logits = self.classifier(cls_repr)  # [batch, num_labels]

        result = {'logits': logits}
        if output_hidden_states:
            # outputs.hidden_states: tuple of (num_layers+1) tensors, each [batch, seq_len, hidden]
            result['hidden_states'] = outputs.hidden_states
        return result

    def get_cls_representations(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Return [CLS] token representation at each layer for topology analysis.

        Returns:
            List of [batch, hidden_size] tensors, one per layer (including embedding layer)
        """
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Extract [CLS] (position 0) from each layer
        return [h[:, 0, :] for h in outputs.hidden_states]


class HiddenStateCapture:
    """
    Captures per-layer [CLS] representations during evaluation for topology analysis.

    Usage:
        capture = HiddenStateCapture(model)
        capture.start()
        for batch in loader:
            model(...)  # capture hooks collect states automatically
        states = capture.get_states()
        capture.stop()
    """

    def __init__(self, model: RobertaNLI):
        self.model = model
        self.captured: Dict[int, List[torch.Tensor]] = {}
        self._active = False
        self._hooks = []

    def start(self):
        """Register forward hooks on each transformer layer."""
        self._active = True
        self.captured = {i: [] for i in range(self.model.num_layers + 1)}

        # Hook on embedding output (layer 0)
        def make_hook(layer_idx):
            def hook(module, input, output):
                if self._active:
                    # output is a tuple; first element is hidden states [batch, seq, hidden]
                    hidden = output[0] if isinstance(output, tuple) else output
                    self.captured[layer_idx].append(hidden[:, 0, :].detach().cpu())
            return hook

        # Embedding layer
        h = self.model.roberta.embeddings.register_forward_hook(make_hook(0))
        self._hooks.append(h)

        # Transformer layers
        for i, layer in enumerate(self.model.roberta.encoder.layer):
            h = layer.register_forward_hook(make_hook(i + 1))
            self._hooks.append(h)

    def stop(self):
        """Remove all hooks."""
        self._active = False
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def get_states(self) -> Dict[int, torch.Tensor]:
        """Return concatenated [CLS] states per layer: {layer_idx: [N, hidden_size]}."""
        return {
            layer_idx: torch.cat(tensors, dim=0)
            for layer_idx, tensors in self.captured.items()
            if tensors
        }
