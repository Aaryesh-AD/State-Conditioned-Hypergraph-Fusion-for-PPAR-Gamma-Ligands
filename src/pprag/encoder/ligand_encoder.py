#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
models/ligand_encoder.py: Ligand molecular encoder with fused graph + hypergraph streams.

Architecture:
    1. AtomGNN: 3-layer GNN over molecular graph (atoms + bonds)
    2. HyperStream: 2-layer hypergraph encoder over pharmacophore groups
    3. Fusion: Gated combination of atom-level and hypergraph embeddings
    4. Readout: Attention pooling to fixed-dimensional ligand embedding

The encoder produces:
    - z_lig: Global ligand embedding (d_model,)
    - h_atoms: Per-atom hidden states (N_atoms, d_model) for cross-attention

Author: Aaryesh Deshpande
Last Modified: 10/25/2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
from torch_geometric.nn import GINEConv
from layers_hypergraph import HypergraphEncoder


class AtomGNN(nn.Module):
    """
    Graph neural network for atom-level molecular representation.

    Uses GINE (Graph Isomorphism Network with Edge features) for message passing
    with residual connections, layer normalization, and coordinate-aware edges.

    Args:
        d_node: Input node feature dimension
        d_edge: Edge feature dimension
        d_hidden: Hidden dimension
        n_layers: Number of GNN layers
        dropout: Dropout probability
    """

    def __init__(self, d_node: int, d_edge: int, d_hidden: int,
                 n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.n_layers = n_layers

        # Input projection
        self.node_embed = nn.Linear(d_node, d_hidden)
        self.edge_embed = nn.Linear(d_edge, d_hidden)

        # GINE layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(n_layers):
            nn_module = nn.Sequential(
                nn.Linear(d_hidden, d_hidden * 2),
                nn.ReLU(),
                nn.Linear(d_hidden * 2, d_hidden),
            )
            self.convs.append(GINEConv(nn_module, edge_dim=d_hidden))
            self.batch_norms.append(nn.LayerNorm(d_hidden))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor,
                batch: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through atom GNN.

        Args:
            x: Node features (N_atoms, d_node)
            edge_index: Edge connectivity (2, E)
            edge_attr: Edge features (E, d_edge)
            batch: Batch assignment for multiple molecules

        Returns:
            Node embeddings (N_atoms, d_hidden)
        """
        # Embed inputs
        x = self.node_embed(x)
        edge_attr = self.edge_embed(edge_attr)

        # Message passing with residual connections
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x_in = x
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)

            # Residual connection (after first layer)
            if i > 0:
                x = x + x_in

        return x


class GatedFusion(nn.Module):
    """
    Gated fusion of atom-level graph and hypergraph representations.

    Uses a learned gate to combine features from different structural views:
        out = gate * x_atom + (1 - gate) * x_hyper

    Args:
        d_atom: Atom feature dimension
        d_hyper: Hypergraph feature dimension (should equal d_atom)
    """

    def __init__(self, d_atom: int, d_hyper: int):
        super().__init__()
        assert d_atom == d_hyper, "Fusion requires equal dimensions"

        self.gate_net = nn.Sequential(
            nn.Linear(d_atom * 2, d_atom),
            nn.ReLU(),
            nn.Linear(d_atom, d_atom),
            nn.Sigmoid()
        )

    def forward(self, x_atom: Tensor, x_hyper: Tensor) -> Tensor:
        """
        Fuse atom and hypergraph features.

        Args:
            x_atom: Atom features from graph stream (N, d)
            x_hyper: Atom features from hypergraph stream (N, d)

        Returns:
            Fused features (N, d)
        """
        # Compute gate
        gate = self.gate_net(torch.cat([x_atom, x_hyper], dim=-1))

        # Gated combination
        out = gate * x_atom + (1 - gate) * x_hyper

        return out


class AttentionReadout(nn.Module):
    """
    Attention-based graph pooling for global representation.

    Uses a learnable query vector to attend over node features.

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
        Attention pooling over nodes.

        Args:
            x: Node features (N, d_hidden)
            batch: Batch assignment (N,) for multiple molecules

        Returns:
            Global features (batch_size, d_hidden) or (1, d_hidden) if no batch
        """
        if batch is None:
            # Single molecule
            k = self.key_net(x)  # (N, d)
            v = self.value_net(x)  # (N, d)

            # Attention scores
            scores = torch.matmul(self.query, k.T)  # (1, N)
            attn = F.softmax(scores, dim=-1)  # (1, N)

            # Weighted sum
            out = torch.matmul(attn, v)  # (1, d)
            return out
        else:
            # Batched molecules
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


class LigandEncoder(nn.Module):
    """
    Complete ligand encoder with fused graph + hypergraph streams.

    Architecture:
        1. AtomGNN: Process molecular graph (3 layers)
        2. HyperStream: Process pharmacophore hypergraph (2 layers)
        3. Fusion: Combine atom and hypergraph representations
        4. Readout: Attention pooling for global embedding

    Args:
        d_node: Input node feature dimension (from atom features)
        d_edge: Edge feature dimension
        d_hyper_feat: Hyperedge feature dimension
        d_model: Output hidden dimension
        n_gnn_layers: Number of GNN layers
        n_hyper_layers: Number of hypergraph layers
        n_heads: Number of attention heads in hypergraph
        dropout: Dropout probability
    """

    def __init__(self, d_node: int = 44, d_edge: int = 6, d_hyper_feat: int = 15,
                 d_model: int = 256, n_gnn_layers: int = 3, n_hyper_layers: int = 2,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Atom graph stream
        self.atom_gnn = AtomGNN(d_node, d_edge, d_model, n_gnn_layers, dropout)

        # Hypergraph stream
        self.hyper_embed = nn.Linear(d_hyper_feat, d_model)
        self.hyper_encoder = HypergraphEncoder(d_model, d_model, n_hyper_layers,
                                               n_heads, dropout, use_type_gate=True)

        # Fusion
        self.fusion = GatedFusion(d_model, d_model)

        # Readout
        self.readout = AttentionReadout(d_model)

        # Final projection
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor,
                pos: Tensor, hyperedge_attr: Optional[Tensor] = None,
                hyperedge_members: Optional[list] = None,
                batch: Optional[Tensor] = None,
                return_attn: bool = False) -> Tuple[Tensor, Tensor, Optional[dict]]:
        """
        Encode ligand molecule.

        Args:
            x: Atom features (N_atoms, d_node)
            edge_index: Bond connectivity (2, E)
            edge_attr: Bond features (E, d_edge)
            pos: 3D coordinates (N_atoms, 3) - for future use
            hyperedge_attr: Hyperedge features (H, d_hyper_feat)
            hyperedge_members: List of atom index lists per hyperedge
            batch: Batch assignment (N_atoms,) for multiple molecules
            return_attn: Whether to return attention weights

        Returns:
            z_lig: Global ligand embedding (batch_size, d_model)
            h_atoms: Per-atom hidden states (N_atoms, d_model)
            attn_dict: Optional attention weights from hypergraph
        """
        # Process through atom GNN
        h_atom = self.atom_gnn(x, edge_index, edge_attr, batch)  # (N, d_model)

        # Process through hypergraph if available
        if hyperedge_attr is not None and hyperedge_members is not None and len(hyperedge_members) > 0:
            # Embed hyperedges
            h_hyper = self.hyper_embed(hyperedge_attr)  # (H, d_model)

            # Hypergraph message passing
            h_atom_hyper, h_hyper_out, attn_dict = self.hyper_encoder(
                h_atom, h_hyper, hyperedge_members, return_attn
            )

            # Fusion
            h_fused = self.fusion(h_atom, h_atom_hyper)
        else:
            # No hypergraph - use atom features only
            h_fused = h_atom
            attn_dict = None

        # Global readout
        z_lig = self.readout(h_fused, batch)  # (batch_size, d_model)
        z_lig = self.out_proj(z_lig)

        return z_lig, h_fused, attn_dict


def create_ligand_encoder(d_node: int = 44, d_edge: int = 6, d_hyper_feat: int = 15,
                          d_model: int = 256, n_gnn_layers: int = 3,
                          n_hyper_layers: int = 2, n_heads: int = 4,
                          dropout: float = 0.1) -> LigandEncoder:
    """Factory function for ligand encoder with default parameters."""
    return LigandEncoder(d_node, d_edge, d_hyper_feat, d_model, n_gnn_layers,
                         n_hyper_layers, n_heads, dropout)
