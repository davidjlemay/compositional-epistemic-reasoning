#!/usr/bin/env python3
"""
GNN-based sequence-to-sequence model for epistemic logic reasoning.
Converts token sequences to graphs and uses graph convolutions to exploit modal logic structure.
Designed for grokking experiments with similar parameter count to GRU/Transformer baselines.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class TokenToGraphConverter:
    """
    Converts token sequences into graphs for GNN processing.
    Creates edges based on:
    1. Sequential adjacency (token i connects to i+1)
    2. Structural relationships (operators connect to their arguments)
    3. Positional proximity (nearby tokens are connected)
    """

    def __init__(self, window_size: int = 3):
        """
        Args:
            window_size: Size of sliding window for creating edges
        """
        self.window_size = window_size

    def convert_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Convert batch of token sequences to graph representation.

        Args:
            input_ids: [batch_size, seq_len] token IDs
            attention_mask: [batch_size, seq_len] attention mask

        Returns:
            node_features: [total_nodes, 1] flattened node features (token IDs)
            edge_index: [2, total_edges] edge list in COO format
            batch_assignment: [total_nodes] assigns each node to its batch
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        all_node_features = []
        all_edge_indices = []
        batch_assignment = []
        node_offset = 0

        for b in range(batch_size):
            # Get valid tokens (non-padding)
            mask = attention_mask[b].bool()
            valid_tokens = input_ids[b][mask]
            num_nodes = valid_tokens.size(0)

            if num_nodes == 0:
                continue

            # Node features are just token IDs (will be embedded later)
            node_features = valid_tokens.unsqueeze(1).float()
            all_node_features.append(node_features)

            # Create edges
            edge_list = []

            # 1. Sequential edges (bi-directional)
            for i in range(num_nodes - 1):
                edge_list.append([i, i + 1])
                edge_list.append([i + 1, i])  # Bi-directional

            # 2. Window-based edges (connect tokens within window)
            for i in range(num_nodes):
                for j in range(i + 2, min(i + self.window_size + 1, num_nodes)):
                    edge_list.append([i, j])
                    edge_list.append([j, i])

            # 3. Self-loops for each node
            for i in range(num_nodes):
                edge_list.append([i, i])

            if edge_list:
                edges = torch.tensor(edge_list, dtype=torch.long, device=device).t()
                # Add offset for batch processing
                edges = edges + node_offset
                all_edge_indices.append(edges)

            # Track batch assignment
            batch_assignment.extend([b] * num_nodes)
            node_offset += num_nodes

        # Concatenate all batches
        if all_node_features:
            node_features = torch.cat(all_node_features, dim=0)
            edge_index = torch.cat(all_edge_indices, dim=1) if all_edge_indices else torch.zeros((2, 0), dtype=torch.long, device=device)
            batch_assignment = torch.tensor(batch_assignment, dtype=torch.long, device=device)
        else:
            node_features = torch.zeros((0, 1), device=device)
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            batch_assignment = torch.zeros(0, dtype=torch.long, device=device)

        return node_features, edge_index, batch_assignment


class GraphConvLayer(nn.Module):
    """
    Simple graph convolution layer (similar to GCN).
    Message passing: h_i' = σ(W * mean(h_j for j in N(i)))
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, in_features] node features
            edge_index: [2, num_edges] edge list

        Returns:
            x_out: [num_nodes, out_features] updated node features
        """
        num_nodes = x.size(0)

        # Message passing: aggregate neighbors
        row, col = edge_index

        # Compute messages (just use neighbor features)
        messages = x[col]  # [num_edges, in_features]

        # Aggregate by averaging (mean aggregation)
        aggregated = torch.zeros(num_nodes, x.size(1), device=x.device)
        aggregated.index_add_(0, row, messages)

        # Normalize by degree
        degree = torch.zeros(num_nodes, device=x.device)
        degree.index_add_(0, row, torch.ones(messages.size(0), device=x.device))
        degree = torch.clamp(degree, min=1.0).unsqueeze(1)

        aggregated = aggregated / degree

        # Apply linear transformation and activation
        out = self.linear(aggregated)
        out = self.activation(out)

        return out


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Network (GAT) layer.
    Uses attention mechanism to weight neighbor contributions.
    """

    def __init__(self, in_features: int, out_features: int, num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        assert out_features % num_heads == 0
        self.head_dim = out_features // num_heads

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.zeros(1, num_heads, 2 * self.head_dim))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

        nn.init.xavier_uniform_(self.a)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, in_features]
            edge_index: [2, num_edges]

        Returns:
            out: [num_nodes, out_features]
        """
        num_nodes = x.size(0)

        # Linear transformation
        h = self.W(x)  # [num_nodes, out_features]
        h = h.view(num_nodes, self.num_heads, self.head_dim)  # [num_nodes, num_heads, head_dim]

        # Compute attention coefficients
        row, col = edge_index
        h_i = h[row]  # [num_edges, num_heads, head_dim]
        h_j = h[col]  # [num_edges, num_heads, head_dim]

        # Concatenate for attention
        h_cat = torch.cat([h_i, h_j], dim=-1)  # [num_edges, num_heads, 2*head_dim]

        # Attention scores
        e = (h_cat * self.a).sum(dim=-1)  # [num_edges, num_heads]
        e = self.leaky_relu(e)

        # Normalize attention scores using softmax per source node
        attention = torch.zeros(num_nodes, self.num_heads, device=x.device)
        attention = attention.index_add(0, row, torch.exp(e))
        attention = torch.clamp(attention, min=1e-8)

        alpha = torch.exp(e) / attention[row]  # [num_edges, num_heads]
        alpha = self.dropout(alpha)

        # Apply attention weights
        h_j_weighted = h_j * alpha.unsqueeze(-1)  # [num_edges, num_heads, head_dim]

        # Aggregate
        out = torch.zeros(num_nodes, self.num_heads, self.head_dim, device=x.device)
        out = out.index_add(0, row, h_j_weighted)

        # Concatenate heads
        out = out.view(num_nodes, self.out_features)

        return out


class GlobalGraphPooling(nn.Module):
    """
    Global graph pooling to get fixed-size representation from variable-size graphs.
    Combines mean, max, and sum pooling.
    """

    def __init__(self, mode: str = 'mean'):
        super().__init__()
        self.mode = mode

    def forward(self, x: torch.Tensor, batch_assignment: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, features] node features
            batch_assignment: [num_nodes] batch index for each node
            batch_size: number of graphs in batch

        Returns:
            pooled: [batch_size, features] pooled graph representations
        """
        features = x.size(1)
        pooled = torch.zeros(batch_size, features, device=x.device)

        if self.mode == 'mean':
            # Mean pooling
            pooled = pooled.index_add(0, batch_assignment, x)
            count = torch.zeros(batch_size, device=x.device)
            count = count.index_add(0, batch_assignment, torch.ones(x.size(0), device=x.device))
            count = torch.clamp(count, min=1.0).unsqueeze(1)
            pooled = pooled / count
        elif self.mode == 'max':
            # Max pooling
            for i in range(batch_size):
                mask = batch_assignment == i
                if mask.any():
                    pooled[i] = x[mask].max(dim=0)[0]
        elif self.mode == 'sum':
            # Sum pooling
            pooled = pooled.index_add(0, batch_assignment, x)

        return pooled


class GNNEncoder(nn.Module):
    """
    GNN encoder that converts token sequences to graph representations.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int,
                 num_layers: int = 2, use_attention: bool = True, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        # Graph to sequence converter
        self.graph_converter = TokenToGraphConverter(window_size=3)

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = embedding_dim if i == 0 else hidden_size
            if use_attention:
                self.gnn_layers.append(GraphAttentionLayer(in_dim, hidden_size, num_heads=2, dropout=dropout))
            else:
                self.gnn_layers.append(GraphConvLayer(in_dim, hidden_size))

        # Global pooling
        self.pooling = GlobalGraphPooling(mode='mean')

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict:
        """
        Args:
            input_ids: [batch_size, seq_len] token IDs
            attention_mask: [batch_size, seq_len] attention mask

        Returns:
            dict with graph_embedding: [batch_size, hidden_size]
        """
        batch_size = input_ids.size(0)

        # Convert sequences to graphs
        node_features, edge_index, batch_assignment = self.graph_converter.convert_batch(
            input_ids, attention_mask
        )

        # Embed tokens
        if node_features.size(0) > 0:
            node_ids = node_features.squeeze(1).long()
            x = self.embedding(node_ids)  # [num_nodes, embedding_dim]
            x = self.dropout(x)

            # Apply GNN layers
            for gnn_layer in self.gnn_layers:
                x = gnn_layer(x, edge_index)
                x = self.dropout(x)

            # Global pooling to get graph-level representation
            graph_embedding = self.pooling(x, batch_assignment, batch_size)
        else:
            # Handle empty batch
            graph_embedding = torch.zeros(batch_size, self.hidden_size, device=input_ids.device)

        return {
            'graph_embedding': graph_embedding,
            'node_features': x if node_features.size(0) > 0 else None,
            'attention_mask': attention_mask
        }


class GRUDecoder(nn.Module):
    """
    GRU decoder for generating output sequences from graph embeddings.
    Similar to the GRU model's decoder but takes graph embedding as initial state.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int,
                 num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        # GRU layers
        self.gru = nn.GRU(
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output projection
        self.output_projection = nn.Linear(hidden_size, vocab_size)

    def forward(self, target_ids: torch.Tensor, encoder_embedding: torch.Tensor,
                teacher_forcing_ratio: float = 1.0) -> Dict:
        """
        Args:
            target_ids: [batch_size, target_seq_len] target token IDs
            encoder_embedding: [batch_size, hidden_size] graph embedding from encoder
            teacher_forcing_ratio: probability of using teacher forcing

        Returns:
            dict with logits: [batch_size, target_seq_len-1, vocab_size]
        """
        batch_size, target_seq_len = target_ids.shape

        # Initialize hidden state with encoder embedding
        # Expand to [num_layers, batch_size, hidden_size]
        hidden = encoder_embedding.unsqueeze(0).repeat(self.num_layers, 1, 1)

        outputs = []

        # Start with SOS token
        current_input = target_ids[:, 0]  # [batch_size]

        for t in range(target_seq_len - 1):
            # Embed current input
            embedded = self.embedding(current_input).unsqueeze(1)  # [batch_size, 1, embedding_dim]
            embedded = self.dropout(embedded)

            # GRU forward
            output, hidden = self.gru(embedded, hidden)

            # Project to vocabulary
            logits = self.output_projection(output.squeeze(1))  # [batch_size, vocab_size]
            outputs.append(logits)

            # Teacher forcing
            if torch.rand(1).item() < teacher_forcing_ratio and t + 1 < target_seq_len:
                current_input = target_ids[:, t + 1]
            else:
                current_input = logits.argmax(dim=-1)

        # Stack outputs
        logits = torch.stack(outputs, dim=1)  # [batch_size, target_seq_len-1, vocab_size]

        return {
            'logits': logits,
            'hidden_states': hidden
        }


class EpistemicSeq2SeqGNN(nn.Module):
    """
    Complete GNN-based sequence-to-sequence model for epistemic logic reasoning.
    Uses GNN encoder to exploit graph structure and GRU decoder for generation.
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 32, hidden_size: int = 128,
                 num_encoder_layers: int = 2, num_decoder_layers: int = 1,
                 use_attention: bool = True, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.encoder = GNNEncoder(
            vocab_size, embedding_dim, hidden_size,
            num_layers=num_encoder_layers,
            use_attention=use_attention,
            dropout=dropout
        )

        self.decoder = GRUDecoder(
            vocab_size, embedding_dim, hidden_size,
            num_layers=num_decoder_layers,
            dropout=dropout
        )

        # State capture for topological analysis (similar to GRU model)
        self.captured_encoder_states = []
        self.captured_decoder_states = []
        self.capture_enabled = False

    def enable_state_capture(self):
        """Enable capturing hidden states for topological analysis."""
        self.capture_enabled = True
        self.captured_encoder_states = []
        self.captured_decoder_states = []

    def disable_state_capture(self):
        """Disable state capture."""
        self.capture_enabled = False

    def get_captured_states(self) -> Dict:
        """Get captured states for topological analysis."""
        return {
            'encoder_states': self.captured_encoder_states,
            'decoder_states': self.captured_decoder_states
        }

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                target_ids: Optional[torch.Tensor] = None,
                teacher_forcing_ratio: float = 1.0) -> Dict:
        """
        Forward pass through the complete model.

        Args:
            input_ids: [batch_size, input_seq_len] input token IDs
            attention_mask: [batch_size, input_seq_len] input attention mask
            target_ids: [batch_size, target_seq_len] target token IDs (for training)
            teacher_forcing_ratio: probability of using teacher forcing

        Returns:
            dict with logits and other outputs
        """
        # Encode input as graph
        encoder_outputs = self.encoder(input_ids, attention_mask)
        graph_embedding = encoder_outputs['graph_embedding']

        # Capture encoder states if enabled
        if self.capture_enabled:
            self.captured_encoder_states.append(graph_embedding.detach().cpu())

        if target_ids is not None:
            # Training mode: decode with targets
            decoder_outputs = self.decoder(
                target_ids, graph_embedding,
                teacher_forcing_ratio=teacher_forcing_ratio
            )

            # Capture decoder states if enabled
            if self.capture_enabled and decoder_outputs['hidden_states'] is not None:
                # Average over layers and get last timestep
                decoder_state = decoder_outputs['hidden_states'].mean(dim=0)  # [batch_size, hidden_size]
                self.captured_decoder_states.append(decoder_state.detach().cpu())

            return {
                'logits': decoder_outputs['logits'],
                'encoder_embedding': graph_embedding,
                'decoder_hidden_states': decoder_outputs['hidden_states']
            }
        else:
            # Inference mode: generate
            return self.generate(graph_embedding, max_length=50)

    def generate(self, encoder_embedding: torch.Tensor, max_length: int = 50,
                 sos_token_id: int = 1, eos_token_id: int = 2) -> Dict:
        """
        Generate sequences using the decoder.

        Args:
            encoder_embedding: [batch_size, hidden_size] graph embedding
            max_length: maximum generation length
            sos_token_id: start of sequence token ID
            eos_token_id: end of sequence token ID

        Returns:
            dict with generated_ids
        """
        self.eval()
        batch_size = encoder_embedding.size(0)
        device = encoder_embedding.device

        # Initialize with SOS token
        generated = torch.full((batch_size, 1), sos_token_id, device=device, dtype=torch.long)

        # Initialize hidden state
        hidden = encoder_embedding.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(max_length - 1):
            # Get current input
            current_input = generated[:, -1]

            # Embed and decode
            embedded = self.decoder.embedding(current_input).unsqueeze(1)
            output, hidden = self.decoder.gru(embedded, hidden)

            # Project to vocabulary
            logits = self.decoder.output_projection(output.squeeze(1))
            next_token = logits.argmax(dim=-1)

            # Update generated sequence
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

            # Check for EOS
            finished = finished | (next_token == eos_token_id)
            if finished.all():
                break

        return {'generated_ids': generated}


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the GNN model
    print("Testing Epistemic GNN Model...")

    vocab_size = 59
    batch_size = 4
    seq_len = 20
    target_len = 5

    # Create model
    model = EpistemicSeq2SeqGNN(
        vocab_size=vocab_size,
        embedding_dim=32,
        hidden_size=128,
        num_encoder_layers=2,
        num_decoder_layers=1,
        use_attention=True
    )

    print(f"Model created with {count_parameters(model):,} trainable parameters")

    # Test data
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    target_ids = torch.randint(1, vocab_size, (batch_size, target_len))

    # Test forward pass
    print(f"\nTesting forward pass...")
    print(f"Input shape: {input_ids.shape}")
    print(f"Target shape: {target_ids.shape}")

    model.eval()
    with torch.no_grad():
        # Test training mode
        outputs = model(input_ids, attention_mask, target_ids)
        print(f"Output logits shape: {outputs['logits'].shape}")

        # Test inference mode
        outputs = model(input_ids, attention_mask)
        print(f"Generated sequence shape: {outputs['generated_ids'].shape}")

        # Test state capture
        model.enable_state_capture()
        outputs = model(input_ids, attention_mask, target_ids)
        captured = model.get_captured_states()
        print(f"Captured encoder states: {len(captured['encoder_states'])} batches")
        print(f"Captured decoder states: {len(captured['decoder_states'])} batches")

    print("\nGNN model test completed successfully!")
