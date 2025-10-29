#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
models/pocket_encoder.py: Protein pocket encoder with state conditioning.

Architecture:
    1. Residue GNN: 4-layer GNN with distance-based edges
    2. State conditioning: Add learned state embeddings (agonist/antagonist)
    3. Readout: Attention pooling to fixed-dimensional pocket embedding

The encoder produces:
    - z_poc: Global pocket embedding (d_model,)
    - h_residues: Per-residue hidden states (N_res, d_model) for cross-attention

Optional FiLM (Feature-wise Linear Modulation) for state conditioning.

Author: Aaryesh Deshpande
Last Modified: 10/25/2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
from torch_geometric.nn import GATv2Conv


class RBFEdgeEncoder(nn.Module):
    """
    Radial basis function encoder for distance-based edges.

    Encodes continuous distances as smooth RBF features for GNN edge attributes.

    Args:
        d_rbf: Number of RBF centers (input dimension)
        d_out: Output edge embedding dimension
    """

    def __init__(self, d_in: int, d_out: int = 64):
        super().__init__()
        self.edge_net = nn.Sequential(
            nn.Linear(d_in, d_out),  # RBF in includes contact flags
            nn.ReLU(),
            nn.Linear(d_out, d_out),
        )

    def forward(self, edge_attr: Tensor) -> Tensor:
        """
        Encode edge features.

        Args:
            edge_attr: Raw edge features (E, d_rbf + 3)

        Returns:
            Encoded edge features (E, d_out)
        """
        return self.edge_net(edge_attr)


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation for state conditioning.

    Applies affine transformation to node features based on global state:
        out = gamma(state) * x + beta(state)

    Args:
        d_node: Node feature dimension
        d_state: State embedding dimension
    """

    def __init__(self, d_node: int, d_state: int):
        super().__init__()
        self.gamma_net = nn.Linear(d_state, d_node)
        self.beta_net = nn.Linear(d_state, d_node)

    def forward(self, x: Tensor, state_emb: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        """
        Apply FiLM conditioning.

        Args:
            x: Node features (N, d_node)
            state_emb: State embedding (1, d_state) or (batch_size, d_state)

        Returns:
            Conditioned features (N, d_node)
        """
        gamma = self.gamma_net(state_emb)  # (1 or B, d_node)
        beta = self.beta_net(state_emb)

        # Broadcast if needed
        if gamma.size(0) == 1:
            return gamma * x + beta
        else:
            # Batched (would need batch indices)
            if batch is None:
                raise ValueError("batch tensor required for batched FiLM conditioning")
            gamma = gamma[batch]  # (N, d_node)
            beta = beta[batch]    # (N, d_node)
            return gamma * x + beta


class ResidueGNN(nn.Module):
    """
    Graph neural network for residue-level pocket representation.

    Uses GATv2 (Graph Attention Network v2) with distance-encoded edges
    and residual connections for multi-layer processing.

    Args:
        d_node: Input node feature dimension
        d_edge: Edge feature dimension
        d_hidden: Hidden dimension
        n_layers: Number of GNN layers
        n_heads: Number of attention heads
        dropout: Dropout probability
        use_film: Whether to use FiLM for state conditioning
        d_state: State embedding dimension (if use_film)
    """

    def __init__(self, d_node: int, d_edge: int, d_hidden: int,
                 n_layers: int = 4, n_heads: int = 4, dropout: float = 0.1,
                 use_film: bool = True, d_state: int = 16):
        super().__init__()
        self.n_layers = n_layers
        self.use_film = use_film

        # Input projection
        self.node_embed = nn.Linear(d_node, d_hidden)
        self.edge_encoder = RBFEdgeEncoder(d_in=d_edge, d_out=d_hidden)

        # GAT layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            # Use concat=False for mean aggregation of heads
            self.convs.append(GATv2Conv(d_hidden, d_hidden // n_heads, heads=n_heads,
                                        edge_dim=d_hidden, concat=True, dropout=dropout))
            self.norms.append(nn.LayerNorm(d_hidden))

        # FiLM layers for state conditioning
        if use_film:
            self.film_layers = nn.ModuleList([
                FiLMLayer(d_hidden, d_state) for _ in range(n_layers)
            ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor,
                state_emb: Optional[Tensor] = None,
                batch: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through residue GNN.

        Args:
            x: Node features (N_res, d_node)
            edge_index: Edge connectivity (2, E)
            edge_attr: Edge features (E, d_edge)
            state_emb: State embedding (1, d_state) for FiLM conditioning
            batch: Batch assignment for multiple pockets

        Returns:
            Node embeddings (N_res, d_hidden)
        """
        # Embed inputs
        x = self.node_embed(x)
        edge_attr_enc = self.edge_encoder(edge_attr)

        # Message passing with residual connections
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_in = x

            # GAT message passing
            x = conv(x, edge_index, edge_attr_enc)
            x = norm(x)

            # FiLM conditioning
            if self.use_film and state_emb is not None:
                x = self.film_layers[i](x, state_emb, batch)

            x = F.relu(x)
            x = self.dropout(x)

            # Residual connection (after first layer)
            if i > 0:
                x = x + x_in

        return x


class AttentionReadout(nn.Module):
    """
    Attention-based graph pooling for global pocket representation.

    Uses a learnable query vector to attend over residue features.

    Args:
        d_hidden: Hidden dimension
    """

    def __init__(self, d_hidden: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, d_hidden))
        self.key_net = nn.Linear(d_hidden, d_hidden)
        self.value_net = nn.Linear(d_hidden, d_hidden)

    def forward(self, x: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        """
        Attention pooling over residues.

        Args:
            x: Residue features (N, d_hidden)
            batch: Batch assignment (N,) for multiple pockets

        Returns:
            Global features (batch_size, d_hidden) or (1, d_hidden) if no batch
        """
        if batch is None:
            # Single pocket
            k = self.key_net(x)  # (N, d)
            v = self.value_net(x)  # (N, d)

            # Attention scores
            scores = torch.matmul(self.query, k.T)  # (1, N)
            attn = F.softmax(scores, dim=-1)  # (1, N)

            # Weighted sum
            out = torch.matmul(attn, v)  # (1, d)
            return out
        else:
            # Batched pockets
            batch_size = int(batch.max().item()) + 1
            outputs = []

            for b in range(batch_size):
                mask = (batch == b)
                x_b = x[mask]  # (N_b, d)

                k = self.key_net(x_b)
                v = self.value_net(x_b)

                scores = torch.matmul(self.query, k.T)
                attn = F.softmax(scores, dim=-1)

                out = torch.matmul(attn, v)
                outputs.append(out)

            return torch.cat(outputs, dim=0)  # (batch_size, d)


class PocketEncoder(nn.Module):
    """
    Complete protein pocket encoder with state conditioning.

    Architecture:
        1. State embedding: Encode agonist/antagonist state
        2. ResidueGNN: 4-layer GNN with FiLM conditioning
        3. Readout: Attention pooling for global embedding

    Args:
        d_node: Input node feature dimension (from residue features)
        d_edge: Edge feature dimension
        d_model: Output hidden dimension
        n_layers: Number of GNN layers
        n_heads: Number of attention heads
        dropout: Dropout probability
        use_state_token: Whether to use state conditioning
        d_state: State embedding dimension
        n_states: Number of possible states (2 for agonist/antagonist)
    """

    def _init_weights(self):
        """Initialize weights with careful scaling to prevent NaN."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use Xavier initialization with smaller gain
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def __init__(self, d_node: int = 30, d_edge: int = 19, d_model: int = 256,
                 n_layers: int = 4, n_heads: int = 4, dropout: float = 0.1,
                 use_state_token: bool = True, d_state: int = 16, n_states: int = 2):
        super().__init__()

        self.d_model = d_model
        self.use_state_token = use_state_token

        # State embedding
        if use_state_token:
            self.state_embed = nn.Embedding(n_states, d_state)

        # Residue GNN
        self.residue_gnn = ResidueGNN(d_node, d_edge, d_model, n_layers,
                                      n_heads, dropout, use_film=use_state_token,
                                      d_state=d_state)

        # Readout
        self.readout = AttentionReadout(d_model)

        # Final projection
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            # nn.LayerNorm(d_model),    # Disabled LayerNorm - smoothing issues for now!
        )
        self._init_weights()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor,
                state_id: Optional[Tensor] = None,
                batch: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Encode protein pocket.

        Args:
            x: Residue features (N_res, d_node)
            edge_index: Edge connectivity (2, E)
            edge_attr: Edge features (E, d_edge)
            state_id: State identifier (0=agonist, 1=antagonist) as tensor
            batch: Batch assignment (N_res,) for multiple pockets

        Returns:
            z_poc: Global pocket embedding (batch_size, d_model)
            h_residues: Per-residue hidden states (N_res, d_model)
        """
        # Get state embedding
        state_emb = None
        if self.use_state_token and state_id is not None:
            if not isinstance(state_id, Tensor):
                state_id = torch.tensor([state_id], dtype=torch.long, device=x.device)
            state_emb = self.state_embed(state_id)  # (1, d_state)

        # Process through residue GNN
        h_residues = self.residue_gnn(x, edge_index, edge_attr, state_emb, batch)

        # Global readout
        z_poc = self.readout(h_residues, batch)  # (batch_size, d_model)
        z_poc = self.out_proj(z_poc)

        # If pretrain loss wants unit vectors: TODO: Check
        # z_poc_for_loss = F.normalize(z_poc, dim=-1)
        return z_poc, h_residues


def create_pocket_encoder(d_node: int = 30, d_edge: int = 19, d_model: int = 256,
                          n_layers: int = 4, n_heads: int = 4, dropout: float = 0.1,
                          use_state_token: bool = True, d_state: int = 16) -> PocketEncoder:
    """Factory function for pocket encoder with default parameters."""
    return PocketEncoder(d_node, d_edge, d_model, n_layers, n_heads, dropout,
                         use_state_token, d_state)
