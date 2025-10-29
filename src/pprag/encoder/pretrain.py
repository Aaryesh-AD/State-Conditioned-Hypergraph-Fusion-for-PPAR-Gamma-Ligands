#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
encoder/pretrain.py: Self-supervised pretraining for ligand and pocket encoders.

Implements two pretraining strategies:

1. GraphCL (Graph Contrastive Learning):
   - Create augmented views of molecules/pockets
   - Use InfoNCE loss to learn invariant representations
   - Augmentations: node dropping, edge perturbation, subgraph sampling

2. State-Aware Contrastive Learning:
   - Learn state-specific representations (agonist vs antagonist)
   - Contrastive pairs based on conformational state
   - Encourages encoder to capture functional differences

These methods improve representation quality before fine-tuning on
classification tasks.

Author: Aaryesh Deshpande
Last Modified: 10/28/2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Dict, Union
import random
from dataclasses import dataclass


@dataclass
class ContrastiveConfig:
    """Configuration for contrastive learning."""
    temperature: float = 0.1
    projection_dim: int = 128
    use_projection_head: bool = True
    normalize_embeddings: bool = True


# AUGMENTATION STRATEGIES
class GraphAugmentor:
    """
    Graph augmentation for contrastive learning.

    Implements common augmentation strategies that preserve semantic meaning:
    - Node dropping: Remove random nodes
    - Edge perturbation: Add/remove random edges
    - Feature masking: Mask random node features
    - Subgraph sampling: Sample connected subgraph
    """

    @staticmethod
    def node_dropping(x: Tensor, edge_index: Tensor, edge_attr: Tensor,
                      drop_ratio: float = 0.1) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Drop random nodes and their incident edges.

        Returns:
            x_aug: Augmented node features
            edge_index_aug: Augmented edge index
            edge_attr_aug: Augmented edge attributes
            keep_mask: Boolean mask of kept nodes
        """
        n_nodes = x.size(0)
        n_keep = max(int(n_nodes * (1 - drop_ratio)), 1)

        # Random node selection
        keep_nodes = torch.randperm(n_nodes)[:n_keep].sort()[0]
        keep_mask = torch.zeros(n_nodes, dtype=torch.bool, device=x.device)
        keep_mask[keep_nodes] = True

        # Filter nodes
        x_aug = x[keep_mask]

        # Filter edges
        edge_mask = keep_mask[edge_index[0]] & keep_mask[edge_index[1]]
        edge_index_aug = edge_index[:, edge_mask]
        edge_attr_aug = edge_attr[edge_mask]

        # Remap node indices
        node_map = torch.zeros(n_nodes, dtype=torch.long, device=x.device)
        node_map[keep_mask] = torch.arange(n_keep, device=x.device)
        edge_index_aug = node_map[edge_index_aug]

        return x_aug, edge_index_aug, edge_attr_aug, keep_mask

    @staticmethod
    def edge_perturbation(edge_index: Tensor, edge_attr: Tensor,
                          n_nodes: int, add_ratio: float = 0.1,
                          drop_ratio: float = 0.1) -> Tuple[Tensor, Tensor]:
        """
        Add and remove random edges.

        Returns:
            edge_index_aug: Augmented edge index
            edge_attr_aug: Augmented edge attributes (new edges get mean features)
        """
        n_edges = edge_index.size(1)
        device = edge_index.device

        # Drop edges
        n_keep = max(int(n_edges * (1 - drop_ratio)), 1)
        keep_edges = torch.randperm(n_edges, device=device)[:n_keep]
        edge_index_aug = edge_index[:, keep_edges]
        edge_attr_aug = edge_attr[keep_edges]

        # Add edges
        n_add = int(n_edges * add_ratio)
        if n_add > 0:
            # Random edge pairs (avoid self-loops)
            new_src = torch.randint(0, n_nodes, (n_add,), device=device)
            new_dst = torch.randint(0, n_nodes, (n_add,), device=device)

            # Filter self-loops
            valid = new_src != new_dst
            new_src = new_src[valid]
            new_dst = new_dst[valid]

            if len(new_src) > 0:
                new_edges = torch.stack([new_src, new_dst], dim=0)
                edge_index_aug = torch.cat([edge_index_aug, new_edges], dim=1)

                # New edges get mean edge features
                mean_edge_attr = edge_attr.mean(dim=0, keepdim=True).expand(len(new_src), -1)
                edge_attr_aug = torch.cat([edge_attr_aug, mean_edge_attr], dim=0)

        return edge_index_aug, edge_attr_aug

    @staticmethod
    def feature_masking(x: Tensor, mask_ratio: float = 0.15) -> Tensor:
        """
        Mask random node features (set to zero).

        Similar to BERT masking but for graph nodes.
        """
        mask = torch.rand(x.size(0), device=x.device) > mask_ratio
        x_aug = x * mask.unsqueeze(-1).float()
        return x_aug

    @staticmethod
    def combine_augmentations(x: Tensor, edge_index: Tensor, edge_attr: Tensor,
                              aug_type: str = 'random') -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Apply random combination of augmentations.

        Args:
            aug_type: 'random', 'node_drop', 'edge_pert', 'feat_mask', 'all'

        Returns:
            x_aug: Augmented node features
            ei_aug: Augmented edge index
            ea_aug: Augmented edge attributes
            keep_mask: Boolean mask of kept nodes (None if no node dropping)
        """
        if aug_type == 'random':
            aug_type = random.choice(['node_drop', 'edge_pert', 'feat_mask'])

        keep_mask = None
        if aug_type == 'node_drop':
            x_aug, ei_aug, ea_aug, keep_mask = GraphAugmentor.node_dropping(
                x, edge_index, edge_attr, drop_ratio=0.1
            )
        elif aug_type == 'edge_pert':
            x_aug = x
            ei_aug, ea_aug = GraphAugmentor.edge_perturbation(
                edge_index, edge_attr, x.size(0), add_ratio=0.1, drop_ratio=0.1
            )
            keep_mask = torch.ones(x.size(0), dtype=torch.bool, device=x.device)
        elif aug_type == 'feat_mask':
            x_aug = GraphAugmentor.feature_masking(x, mask_ratio=0.15)
            ei_aug = edge_index
            ea_aug = edge_attr
            keep_mask = torch.ones(x.size(0), dtype=torch.bool, device=x.device)
        elif aug_type == 'all':
            # Apply all augmentations sequentially
            x_aug, ei_aug, ea_aug, keep_mask = GraphAugmentor.node_dropping(
                x, edge_index, edge_attr, drop_ratio=0.05
            )
            ei_aug, ea_aug = GraphAugmentor.edge_perturbation(
                ei_aug, ea_aug, x_aug.size(0), add_ratio=0.05, drop_ratio=0.05
            )
            x_aug = GraphAugmentor.feature_masking(x_aug, mask_ratio=0.10)
        else:
            raise ValueError(f"Unknown augmentation type: {aug_type}")

        return x_aug, ei_aug, ea_aug, keep_mask


# PROJECTION HEADS
class ProjectionHead(nn.Module):
    """
    MLP projection head for contrastive learning.

    Maps encoder outputs to lower-dimensional space for contrastive loss.
    Common in SimCLR, MoCo, etc.
    """

    def __init__(self, d_in: int, d_hidden: int = 256, d_out: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# CONTRASTIVE LOSSES
def info_nce_loss(z_i: Tensor, z_j: Tensor, temperature: float = 0.1) -> Tensor:
    """
    InfoNCE (NT-Xent) loss for contrastive learning.

    Pulls positive pairs together and pushes negative pairs apart.

    Args:
        z_i: Embeddings of augmented view 1 (batch_size, d)
        z_j: Embeddings of augmented view 2 (batch_size, d)
        temperature: Temperature parameter for softmax

    Returns:
        loss: Scalar InfoNCE loss
    """
    batch_size = z_i.size(0)

    # Normalize embeddings
    z_i = F.normalize(z_i, dim=-1)
    z_j = F.normalize(z_j, dim=-1)

    # Concatenate to create [z_i; z_j] with 2*batch_size samples
    z = torch.cat([z_i, z_j], dim=0)  # (2B, d)

    # Compute similarity matrix
    sim_matrix = torch.mm(z, z.T) / temperature  # (2B, 2B)

    # Mask diagonal (self-similarity)
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))

    # Positive pairs: (i, i+B) and (i+B, i)
    pos_indices = torch.arange(batch_size, device=z.device)
    pos_sim_i = sim_matrix[pos_indices, pos_indices + batch_size]   # noqa
    pos_sim_j = sim_matrix[pos_indices + batch_size, pos_indices]   # noqa

    # Compute loss for both directions
    logits_i = sim_matrix[pos_indices]  # (B, 2B)
    logits_j = sim_matrix[pos_indices + batch_size]  # (B, 2B)

    # Targets: positive pair is at index (i + B) for first half, i for second half
    targets_i = pos_indices + batch_size
    targets_j = pos_indices

    loss_i = F.cross_entropy(logits_i, targets_i)
    loss_j = F.cross_entropy(logits_j, targets_j)

    loss = (loss_i + loss_j) / 2.0

    return loss


def state_aware_contrastive_loss(z: Tensor, states: Tensor,
                                 temperature: float = 0.1) -> Tensor:
    """
    State-aware contrastive loss for pocket representations.

    Encourages:
    - Same state pockets to have similar representations (positive pairs)
    - Different state pockets to have different representations (negative pairs)

    Args:
        z: Pocket embeddings (batch_size, d)
        states: State labels (batch_size,) - 0=agonist, 1=antagonist
        temperature: Temperature parameter

    Returns:
        loss: Scalar contrastive loss
    """
    batch_size = z.size(0)
    device = z.device

    # Normalize embeddings
    z = F.normalize(z, dim=-1)

    # Compute similarity matrix (scaled by temperature)
    sim_matrix = torch.mm(z, z.T) / temperature  # (B, B)

    # Create masks
    state_match = (states.unsqueeze(0) == states.unsqueeze(1)).float()
    diag_mask = torch.eye(batch_size, dtype=torch.float, device=device)

    # Positive pairs: same state, not self
    pos_mask = state_match * (1 - diag_mask)

    # Negative pairs: different state
    neg_mask = (1 - state_match) * (1 - diag_mask)  # noqa

    # Safety check: If no positive pairs exist, return zero loss
    if pos_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Compute InfoNCE loss using logsumexp (numerically stable)
    # For each anchor, loss = -log(sum(exp(pos)) / sum(exp(pos + neg)))
    #                        = -logsumexp(pos) + logsumexp(pos + neg)

    losses = []
    for i in range(batch_size):
        # Get positive and negative similarities for anchor i
        pos_sims = sim_matrix[i][pos_mask[i] > 0]

        if len(pos_sims) == 0:
            continue

        # Get all non-self similarities (both pos and neg)
        all_mask = 1 - diag_mask[i]
        all_sims = sim_matrix[i][all_mask > 0]

        # For numerical stability, use logsumexp
        # Loss = -log(exp(pos) / sum(exp(all)))
        #      = -pos + logsumexp(all)
        for pos_sim in pos_sims:
            loss_i = -pos_sim + torch.logsumexp(all_sims, dim=0)
            losses.append(loss_i)

    if len(losses) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    loss = torch.stack(losses).mean()

    # Safety check for NaN/Inf
    if torch.isnan(loss) or torch.isinf(loss):
        print("\n[ERROR] NaN/Inf detected in loss computation!")
        print(f"  Similarity matrix range: [{sim_matrix.min():.4f}, {sim_matrix.max():.4f}]")
        print(f"  Embedding norm: {z.norm(dim=-1).mean():.4f}")
        print(f"  Temperature: {temperature}")
        print("  Returning zero loss to continue training...\n")
        return torch.tensor(0.0, device=device, requires_grad=True)

    return loss


# PRETRAINING WRAPPER
class ContrastivePretrainer(nn.Module):
    """
    Wrapper for contrastive pretraining of encoders.

    Supports both GraphCL and state-aware contrastive learning.
    """

    from typing import Union

    def __init__(self, encoder: nn.Module, config: ContrastiveConfig):
        super().__init__()
        self.encoder = encoder
        self.config = config

        # Projection head (optional)
        self.projection: Union[ProjectionHead, nn.Identity]
        if config.use_projection_head:
            # Ensure d_model is an integer, not a tensor or module
            d_model = int(getattr(encoder, 'd_model', 0)) if isinstance(getattr(encoder, 'd_model', 0), (int, float)) else encoder.output_dim if hasattr(encoder, 'output_dim') else None
            if not isinstance(d_model, int) or d_model <= 0:
                raise ValueError("Encoder must have an integer 'd_model' or 'output_dim' attribute for projection head.")
            self.projection = ProjectionHead(d_model, d_hidden=256, d_out=config.projection_dim)
        else:
            self.projection = nn.Identity()

    def forward_pair(self, x1: Tensor, edge_index1: Tensor, edge_attr1: Tensor,
                     x2: Tensor, edge_index2: Tensor, edge_attr2: Tensor,
                     **kwargs) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for augmented pair.

        Returns:
            z1, z2: Projected embeddings for contrastive loss
        """
        # Encode both views
        z1, _ = self.encoder(x1, edge_index1, edge_attr1, **kwargs)[:2]
        z2, _ = self.encoder(x2, edge_index2, edge_attr2, **kwargs)[:2]

        # Project
        z1 = self.projection(z1)
        z2 = self.projection(z2)

        # Normalize if configured
        if self.config.normalize_embeddings:
            z1 = F.normalize(z1, dim=-1)
            z2 = F.normalize(z2, dim=-1)

        return z1, z2

    def graphcl_loss(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor,
                     aug_type1: str = 'random', aug_type2: str = 'random',
                     **kwargs) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute GraphCL loss with two augmented views.

        Returns:
            loss: InfoNCE loss
            metrics: Dictionary of logging metrics
        """
        # Create augmented views
        x1, ei1, ea1, keep_mask1 = GraphAugmentor.combine_augmentations(
            x, edge_index, edge_attr, aug_type=aug_type1
        )
        x2, ei2, ea2, keep_mask2 = GraphAugmentor.combine_augmentations(
            x, edge_index, edge_attr, aug_type=aug_type2
        )

        # Update batch tensor and hyperedge_members if nodes were dropped
        kwargs_view1 = kwargs.copy()
        kwargs_view2 = kwargs.copy()

        if 'batch' in kwargs:
            batch_orig = kwargs['batch']

            # Update batch tensor for view 1
            if keep_mask1 is not None:
                batch_view1 = batch_orig[keep_mask1]
                # Remap batch indices to be contiguous
                unique_batches = batch_view1.unique(sorted=True)
                batch_mapping = torch.zeros(batch_orig.max().item() + 1,
                                            dtype=torch.long, device=batch_orig.device)
                for new_idx, old_idx in enumerate(unique_batches):
                    batch_mapping[old_idx] = new_idx
                kwargs_view1['batch'] = batch_mapping[batch_view1]

                # Update hyperedge_members if present
                if 'hyperedge_members' in kwargs and kwargs['hyperedge_members'] is not None:
                    # Create node index mapping (old -> new)
                    n_nodes_orig = keep_mask1.size(0)
                    node_map = torch.zeros(n_nodes_orig, dtype=torch.long, device=x.device)
                    node_map[keep_mask1] = torch.arange(keep_mask1.sum().item(), device=x.device)

                    # Filter and remap hyperedge members
                    new_hyperedge_members = []
                    for members in kwargs['hyperedge_members']:
                        # Keep only members that weren't dropped
                        valid_members = [m for m in members if keep_mask1[m]]
                        if len(valid_members) > 0:
                            # Remap to new indices
                            new_members = [node_map[m].item() for m in valid_members]
                            new_hyperedge_members.append(new_members)

                    kwargs_view1['hyperedge_members'] = new_hyperedge_members if new_hyperedge_members else None

                    # Also need to filter hyperedge_attr if present
                    if 'hyperedge_attr' in kwargs and kwargs['hyperedge_attr'] is not None:
                        if len(new_hyperedge_members) > 0:
                            kwargs_view1['hyperedge_attr'] = kwargs['hyperedge_attr'][:len(new_hyperedge_members)]
                        else:
                            kwargs_view1['hyperedge_attr'] = None

            # Update batch tensor for view 2
            if keep_mask2 is not None:
                batch_view2 = batch_orig[keep_mask2]
                # Remap batch indices to be contiguous
                unique_batches = batch_view2.unique(sorted=True)
                batch_mapping = torch.zeros(batch_orig.max().item() + 1,
                                            dtype=torch.long, device=batch_orig.device)
                for new_idx, old_idx in enumerate(unique_batches):
                    batch_mapping[old_idx] = new_idx
                kwargs_view2['batch'] = batch_mapping[batch_view2]

                # Update hyperedge_members if present
                if 'hyperedge_members' in kwargs and kwargs['hyperedge_members'] is not None:
                    # Create node index mapping (old -> new)
                    n_nodes_orig = keep_mask2.size(0)
                    node_map = torch.zeros(n_nodes_orig, dtype=torch.long, device=x.device)
                    node_map[keep_mask2] = torch.arange(keep_mask2.sum().item(), device=x.device)

                    # Filter and remap hyperedge members
                    new_hyperedge_members = []
                    for members in kwargs['hyperedge_members']:
                        # Keep only members that weren't dropped
                        valid_members = [m for m in members if keep_mask2[m]]
                        if len(valid_members) > 0:
                            # Remap to new indices
                            new_members = [node_map[m].item() for m in valid_members]
                            new_hyperedge_members.append(new_members)

                    kwargs_view2['hyperedge_members'] = new_hyperedge_members if new_hyperedge_members else None

                    # Also need to filter hyperedge_attr if present
                    if 'hyperedge_attr' in kwargs and kwargs['hyperedge_attr'] is not None:
                        if len(new_hyperedge_members) > 0:
                            kwargs_view2['hyperedge_attr'] = kwargs['hyperedge_attr'][:len(new_hyperedge_members)]
                        else:
                            kwargs_view2['hyperedge_attr'] = None

        # Forward pass with updated batch tensors
        z1, _ = self.encoder(x1, ei1, ea1, **kwargs_view1)[:2]
        z2, _ = self.encoder(x2, ei2, ea2, **kwargs_view2)[:2]

        # Project
        z1 = self.projection(z1)
        z2 = self.projection(z2)

        # Normalize if configured
        if self.config.normalize_embeddings:
            z1 = F.normalize(z1, dim=-1)
            z2 = F.normalize(z2, dim=-1)

        # Compute InfoNCE loss
        loss = info_nce_loss(z1, z2, temperature=self.config.temperature)

        # Metrics
        with torch.no_grad():
            # Similarity between positive pairs
            pos_sim = F.cosine_similarity(z1, z2, dim=-1).mean()

        metrics = {
            'loss': loss.item(),
            'pos_similarity': pos_sim.item(),
        }

        return loss, metrics

    def state_contrastive_loss(self, x: Tensor, edge_index: Tensor,
                               edge_attr: Tensor, states: Tensor,
                               **kwargs) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute state-aware contrastive loss for pockets.

        Args:
            states: State labels (batch_size,) - 0=agonist, 1=antagonist

        Returns:
            loss: State-aware contrastive loss
            metrics: Dictionary of logging metrics
        """
        # Encode pockets with state conditioning
        z, _ = self.encoder(x, edge_index, edge_attr, state_id=states, **kwargs)[:2]

        # Project
        z = self.projection(z)

        # Normalize
        if self.config.normalize_embeddings:
            z = F.normalize(z, dim=-1)

        # Compute state-aware loss
        loss = state_aware_contrastive_loss(z, states, self.config.temperature)

        # Metrics
        with torch.no_grad():
            # Within-state similarity
            agonist_mask = states == 0
            antagonist_mask = states == 1

            if agonist_mask.sum() > 1:
                z_ago = z[agonist_mask]
                ago_sim = torch.mm(z_ago, z_ago.T).mean()
            else:
                ago_sim = torch.tensor(0.0)

            if antagonist_mask.sum() > 1:
                z_ant = z[antagonist_mask]
                ant_sim = torch.mm(z_ant, z_ant.T).mean()
            else:
                ant_sim = torch.tensor(0.0)

            # Cross-state similarity
            if agonist_mask.any() and antagonist_mask.any():
                cross_sim = torch.mm(z[agonist_mask], z[antagonist_mask].T).mean()
            else:
                cross_sim = torch.tensor(0.0)

        metrics = {
            'loss': loss.item(),
            'agonist_similarity': ago_sim.item(),
            'antagonist_similarity': ant_sim.item(),
            'cross_state_similarity': cross_sim.item(),
        }

        return loss, metrics


def create_ligand_pretrainer(encoder: nn.Module,
                             temperature: float = 0.1,
                             projection_dim: int = 128) -> ContrastivePretrainer:
    """Factory function for ligand contrastive pretrainer."""
    config = ContrastiveConfig(
        temperature=temperature,
        projection_dim=projection_dim,
        use_projection_head=True,
        normalize_embeddings=True
    )
    return ContrastivePretrainer(encoder, config)


def create_pocket_pretrainer(encoder: nn.Module,
                             temperature: float = 0.1,
                             projection_dim: int = 128) -> ContrastivePretrainer:
    """Factory function for pocket contrastive pretrainer."""
    config = ContrastiveConfig(
        temperature=temperature,
        projection_dim=projection_dim,
        use_projection_head=True,
        normalize_embeddings=True
    )
    return ContrastivePretrainer(encoder, config)
