#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
models/layers_hypergraph.py: Bipartite hypergraph attention layers.

Implements bipartite message passing for hypergraphs using atom <-> hyperedge attention:
1. Atom -> Hyperedge aggregation (size-normalized attention)
2. Hyperedge -> Atom message passing (type-gated attention)

This approach is more interpretable than incidence matrix methods and naturally
handles variable-size hyperedges with explicit attention weights for visualization.

Inspired by AllSet/AllSetTransformer architecture but adapted for pharmacophores.

Author: Aaryesh Deshpande
Last Modified: 10/25/2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, List
import math


class BipartiteHypergraphAttention(nn.Module):
    """
    Bipartite attention layer for hypergraph message passing.

    Architecture:
        1. Atom -> Hyperedge: Aggregate atom features to hyperedge embeddings
           using attention over member atoms (size-normalized)
        2. Hyperedge -> Atom: Broadcast hyperedge features back to atoms
           using attention over incident hyperedges (type-gated)

    Args:
        d_atom: Atom feature dimension
        d_hyper: Hyperedge feature dimension
        n_heads: Number of attention heads
        dropout: Dropout probability
        use_type_gate: Apply pharmacophore-type gating to messages
    """

    def __init__(self, d_atom: int, d_hyper: int, n_heads: int = 4,
                 dropout: float = 0.1, use_type_gate: bool = True):
        super().__init__()
        self.d_atom = d_atom
        self.d_hyper = d_hyper
        self.n_heads = n_heads
        self.d_head = d_atom // n_heads
        self.use_type_gate = use_type_gate

        assert d_atom % n_heads == 0, "d_atom must be divisible by n_heads"

        # Atom → Hyperedge attention
        self.q_a2h = nn.Linear(d_hyper, d_atom)  # Query from hyperedge
        self.k_a2h = nn.Linear(d_atom, d_atom)   # Key from atom
        self.v_a2h = nn.Linear(d_atom, d_atom)   # Value from atom

        # Hyperedge → Atom attention
        self.q_h2a = nn.Linear(d_atom, d_atom)   # Query from atom
        self.k_h2a = nn.Linear(d_hyper, d_atom)  # Key from hyperedge
        self.v_h2a = nn.Linear(d_hyper, d_atom)  # Value from hyperedge

        # Optional type-based gating
        if use_type_gate:
            self.type_gate = nn.Sequential(
                nn.Linear(d_hyper, d_atom),
                nn.Sigmoid()
            )

        # Output projections
        self.out_atom = nn.Linear(d_atom, d_atom)
        self.out_hyper = nn.Linear(d_atom, d_hyper)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_head)

    def forward(self, x_atom: Tensor, x_hyper: Tensor,
                membership: List[List[int]],
                return_attn: bool = False) -> Tuple[Tensor, Tensor, Optional[dict]]:
        """
        Forward pass through bipartite hypergraph layer.

        Args:
            x_atom: Atom features (N_atoms, d_atom)
            x_hyper: Hyperedge features (N_hyper, d_hyper)
            membership: List of atom index lists for each hyperedge
            return_attn: Whether to return attention weights

        Returns:
            x_atom_out: Updated atom features (N_atoms, d_atom)
            x_hyper_out: Updated hyperedge features (N_hyper, d_hyper)
            attn_dict: Optional attention weights for visualization
        """
        n_atoms = x_atom.size(0)
        n_hyper = x_hyper.size(0)   # noqa

        # === STEP 1: Atom -> Hyperedge aggregation ===
        # For each hyperedge, aggregate its member atoms with attention
        hyper_messages = []
        a2h_attn_weights: Optional[List[Tensor]] = [] if return_attn else None

        for h_idx, members in enumerate(membership):
            if len(members) == 0:
                # Empty hyperedge (shouldn't happen, but handle gracefully)
                hyper_messages.append(torch.zeros(1, self.d_atom, device=x_atom.device))
                if return_attn and a2h_attn_weights is not None:
                    a2h_attn_weights.append(torch.zeros(0, device=x_atom.device))
                continue

            # ADDED: Validate indices before creating tensor
            valid_members = [m for m in members if 0 <= m < n_atoms]
            if len(valid_members) == 0:
                print(f"WARNING: Hyperedge {h_idx} has no valid members (original: {members}, n_atoms: {n_atoms})")
                hyper_messages.append(torch.zeros(1, self.d_atom, device=x_atom.device))
                if return_attn and a2h_attn_weights is not None:
                    a2h_attn_weights.append(torch.zeros(0, device=x_atom.device))
                continue

            if len(valid_members) != len(members):
                print(f"WARNING: Hyperedge {h_idx} had invalid indices. Original: {members}, Valid: {valid_members}, n_atoms: {n_atoms}")

            # Get member atom features
            member_indices = torch.tensor(valid_members, dtype=torch.long, device=x_atom.device)
            x_members = x_atom[member_indices]  # (n_members, d_atom)

            # Multi-head attention: hyperedge queries atoms
            q = self.q_a2h(x_hyper[h_idx:h_idx + 1])  # (1, d_atom)
            k = self.k_a2h(x_members)                # (n_members, d_atom)
            v = self.v_a2h(x_members)                # (n_members, d_atom)

            # Reshape for multi-head
            q = q.view(1, self.n_heads, self.d_head)  # (1, H, d_head)
            k = k.view(-1, self.n_heads, self.d_head)  # (n_members, H, d_head)
            v = v.view(-1, self.n_heads, self.d_head)  # (n_members, H, d_head)

            # Attention scores
            scores = torch.einsum('qhd,khd->qhk', q, k) * self.scale  # (1, H, n_members)

            # Size normalization (larger hyperedges get dampened)
            size_norm = 1.0 / math.sqrt(max(len(members), 4))  # Prevent division by small numbers
            scores = scores * size_norm

            # Clip scores to prevent extreme values in softmax
            scores = torch.clamp(scores, min=-5.0, max=5.0)

            attn = F.softmax(scores, dim=-1)  # (1, H, n_members)
            attn = self.dropout(attn)

            if return_attn and a2h_attn_weights is not None:
                a2h_attn_weights.append(attn.squeeze(0).mean(0))  # Average over heads

            # Aggregate
            out = torch.einsum('qhk,khd->qhd', attn, v)  # (1, H, d_head)
            out = out.reshape(1, self.d_atom)

            hyper_messages.append(out)

        # Stack hyperedge messages
        x_hyper_agg = torch.cat(hyper_messages, dim=0)  # (N_hyper, d_atom)

        # === STEP 2: Hyperedge → Atom message passing ===
        # For each atom, aggregate messages from incident hyperedges
        atom_messages = torch.zeros_like(x_atom)  # (N_atoms, d_atom)
        atom_msg_counts = torch.zeros(n_atoms, device=x_atom.device)
        h2a_attn_weights: Optional[List[List[Tuple[int, float]]]] = [[] for _ in range(n_atoms)] if return_attn else None

        for h_idx, members in enumerate(membership):
            if len(members) == 0:
                continue

            # ADDED: Validate indices before creating tensor (same as first loop)
            valid_members = [m for m in members if 0 <= m < n_atoms]
            if len(valid_members) == 0:
                print(f"WARNING: Hyperedge {h_idx} has no valid members in h2a pass (original: {members}, n_atoms: {n_atoms})")
                continue

            if len(valid_members) != len(members):
                print(f"WARNING: Hyperedge {h_idx} had invalid indices in h2a pass. Original: {members}, Valid: {valid_members}, n_atoms: {n_atoms}")

            member_indices = torch.tensor(valid_members, dtype=torch.long, device=x_atom.device)
            x_recipients = x_atom[member_indices]  # (n_members, d_atom)

            # Multi-head attention: atoms query hyperedge
            q = self.q_h2a(x_recipients)             # (n_members, d_atom)
            k = self.k_h2a(x_hyper[h_idx:h_idx + 1])  # (1, d_atom)
            v = self.v_h2a(x_hyper[h_idx:h_idx + 1])  # (1, d_atom)

            # Reshape for multi-head
            q = q.view(-1, self.n_heads, self.d_head)  # (n_members, H, d_head)
            k = k.view(1, self.n_heads, self.d_head)   # (1, H, d_head)
            v = v.view(1, self.n_heads, self.d_head)   # (1, H, d_head)

            # Attention scores
            scores = torch.einsum('qhd,khd->qhk', q, k) * self.scale  # (n_members, H, 1)
            attn = F.softmax(scores, dim=-1)  # (n_members, H, 1)
            attn = self.dropout(attn)

            # Aggregate
            out = torch.einsum('qhk,khd->qhd', attn, v)  # (n_members, H, d_head)
            out = out.reshape(-1, self.d_atom)  # (n_members, d_atom)

            # Optional type-based gating
            if self.use_type_gate:
                gate = self.type_gate(x_hyper[h_idx:h_idx + 1])  # (1, d_atom)
                out = out * gate

            # Accumulate messages to atoms
            atom_messages[member_indices] += out
            atom_msg_counts[member_indices] += 1

            if return_attn and h2a_attn_weights is not None:
                attn_mean = attn.squeeze(-1).mean(1)  # (n_members,)
                for i, member_idx in enumerate(valid_members):
                    h2a_attn_weights[member_idx].append((h_idx, attn_mean[i].item()))

        # Normalize by number of incident hyperedges
        atom_msg_counts = atom_msg_counts.clamp(min=1.0)  # Avoid division by zero
        atom_messages = atom_messages / atom_msg_counts.unsqueeze(-1)

        # === Output projections ===
        x_atom_out = self.out_atom(atom_messages)
        x_hyper_out = self.out_hyper(x_hyper_agg)

        # Return attention weights if requested
        attn_dict = None
        if return_attn:
            attn_dict = {
                'a2h': a2h_attn_weights,  # List[Tensor] per hyperedge
                'h2a': h2a_attn_weights,  # List[List[Tuple]] per atom
            }

        return x_atom_out, x_hyper_out, attn_dict


class HypergraphConvLayer(nn.Module):
    """
    Hypergraph convolution layer with residual connections and layer norm.

    Combines bipartite attention with residual connections and normalization
    for stable training in deep architectures.

    Args:
        d_atom: Atom feature dimension
        d_hyper: Hyperedge feature dimension
        n_heads: Number of attention heads
        dropout: Dropout probability
        use_type_gate: Apply pharmacophore-type gating
    """

    def __init__(self, d_atom: int, d_hyper: int, n_heads: int = 4,
                 dropout: float = 0.1, use_type_gate: bool = True):
        super().__init__()
        self.attn = BipartiteHypergraphAttention(d_atom, d_hyper, n_heads, dropout, use_type_gate)
        self.norm_atom = nn.LayerNorm(d_atom)
        self.norm_hyper = nn.LayerNorm(d_hyper)
        self.dropout = nn.Dropout(dropout)

        # Feedforward networks
        self.ffn_atom = nn.Sequential(
            nn.Linear(d_atom, d_atom * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_atom * 2, d_atom),
        )
        self.ffn_hyper = nn.Sequential(
            nn.Linear(d_hyper, d_hyper * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hyper * 2, d_hyper),
        )

    def forward(self, x_atom: Tensor, x_hyper: Tensor,
                membership: List[List[int]],
                return_attn: bool = False) -> Tuple[Tensor, Tensor, Optional[dict]]:
        """Forward pass with residual connections."""
        # Bipartite attention
        dx_atom, dx_hyper, attn_dict = self.attn(x_atom, x_hyper, membership, return_attn)

        # Residual + norm for atoms
        x_atom = self.norm_atom(x_atom + self.dropout(dx_atom))
        x_atom = x_atom + self.dropout(self.ffn_atom(x_atom))

        # Residual + norm for hyperedges
        x_hyper = self.norm_hyper(x_hyper + self.dropout(dx_hyper))
        x_hyper = x_hyper + self.dropout(self.ffn_hyper(x_hyper))

        return x_atom, x_hyper, attn_dict


class HypergraphEncoder(nn.Module):
    """
    Stack of hypergraph convolution layers.

    Args:
        d_atom: Atom feature dimension
        d_hyper: Hyperedge feature dimension
        n_layers: Number of hypergraph layers
        n_heads: Number of attention heads per layer
        dropout: Dropout probability
        use_type_gate: Apply pharmacophore-type gating
    """

    def __init__(self, d_atom: int, d_hyper: int, n_layers: int = 2,
                 n_heads: int = 4, dropout: float = 0.1, use_type_gate: bool = True):
        super().__init__()
        self.layers = nn.ModuleList([
            HypergraphConvLayer(d_atom, d_hyper, n_heads, dropout, use_type_gate)
            for _ in range(n_layers)
        ])

    def forward(self, x_atom: Tensor, x_hyper: Tensor,
                membership: List[List[int]],
                return_attn: bool = False) -> Tuple[Tensor, Tensor, Optional[List[dict]]]:
        """
        Forward pass through all layers.

        Returns:
            x_atom: Final atom features
            x_hyper: Final hyperedge features
            attn_dicts: Optional list of attention dicts per layer
        """
        attn_dicts: Optional[List[dict]] = [] if return_attn else None

        for layer in self.layers:
            x_atom, x_hyper, attn_dict = layer(x_atom, x_hyper, membership, return_attn)
            if return_attn and attn_dicts is not None:
                attn_dicts.append(attn_dict)

        return x_atom, x_hyper, attn_dicts


# Alternative: Simpler incidence matrix-based hypergraph conv (baseline)
class IncidenceHypergraphConv(nn.Module):
    """
    Simplified hypergraph convolution using incidence matrix normalization.

    This is a baseline implementation for comparison with bipartite attention.
    Uses fixed normalization based on hyperedge sizes.

    Args:
        d_in: Input feature dimension
        d_out: Output feature dimension
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.w_msg = nn.Linear(d_in, d_out)
        self.w_self = nn.Linear(d_in, d_out)

    def forward(self, x: Tensor, membership: List[List[int]]) -> Tensor:
        """
        Simple hypergraph convolution.

        Args:
            x: Node features (N, d_in)
            membership: List of node index lists per hyperedge

        Returns:
            Updated node features (N, d_out)
        """
        n_nodes = x.size(0)   # noqa
        device = x.device

        # Self-loop
        out = self.w_self(x)

        # Aggregate from hyperedges
        for members in membership:
            if len(members) == 0:
                continue

            # Normalize by size
            size_norm = 1.0 / len(members)
            member_indices = torch.tensor(members, dtype=torch.long, device=device)

            # Average pooling within hyperedge
            x_members = x[member_indices]  # (n_members, d_in)
            msg = x_members.mean(dim=0, keepdim=True) * size_norm  # (1, d_in)
            msg = self.w_msg(msg)  # (1, d_out)

            # Broadcast to all members
            out[member_indices] += msg

        return out
