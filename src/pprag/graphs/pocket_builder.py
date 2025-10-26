#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
graphs/pocket_builder.py: Build pocket residue graphs from selected binding sites.

This module constructs protein binding site representations as residue-level graphs:
- Nodes: Residues within binding pocket (from PocketSelect)
- Edges: Distance-based connectivity (CA-CA distance <= threshold)
- Features: Amino acid identity, SASA, hydropathy, charge, HBD/HBA

Edge construction:
    - Primary: CA-CA distance <= 10 Angstroms
    - Supplement: kNN (k=8) to ensure connectivity
    - Features: RBF distance encoding + contact flags

Author: Aaryesh Deshpande
Last Modified: 10/25/2025
"""

import numpy as np
from typing import List, Tuple
from pprag.dataio.schema import PocketSelect, PocketGraph, FeatureSpec


# Utility
def rbf_encode_distance(dist: float, d_min: float = 0.0, d_max: float = 12.0,
                        n_bins: int = 16) -> np.ndarray:
    """
    Radial basis function encoding of distance.

    Args:
        dist: Distance value in Angstroms
        d_min: Minimum distance for encoding
        d_max: Maximum distance for encoding
        n_bins: Number of RBF centers

    Returns:
        RBF-encoded distance vector (n_bins,)
    """
    centers = np.linspace(d_min, d_max, n_bins)
    gamma = 1.0 / ((d_max - d_min) / n_bins) ** 2
    rbf = np.exp(-gamma * (dist - centers) ** 2)
    return rbf.astype(np.float32)


def build_residue_features(ps: PocketSelect, fe: FeatureSpec) -> np.ndarray:
    """
    Build feature matrix for residues in pocket.

    Features per residue (around 64 dims):
        - Amino acid one-hot (20 dims)
        - SASA (normalized, 1 dim)
        - Hydropathy (normalized, 1 dim)
        - Charge class one-hot (-1, 0, +1 → 3 dims)
        - HBD count (normalized, 1 dim)
        - HBA count (normalized, 1 dim)
        - Secondary structure flags (helix, sheet, coil from sequence position, 3 dims)

    Args:
        ps: PocketSelect with residue features
        fe: FeatureSpec with feature dimensions

    Returns:
        Feature matrix (N_res, d_poc_node)
    """
    n_res = len(ps.feats)   # noqa
    feat_list = []

    for res_feat in ps.feats:
        # AA one-hot (20 dims)
        aa_onehot = np.zeros(20, dtype=np.float32)
        aa_idx = res_feat.aa_idx
        if 0 <= aa_idx < 20:
            aa_onehot[aa_idx] = 1.0

        # SASA (normalized by typical max around 300 angstroms²)
        sasa_norm = min(1.0, res_feat.sasa / 300.0)

        # Hydropathy (normalized from Kyte-Doolittle range [-4.5, 4.5] to [0, 1])
        hydropathy_norm = (res_feat.hydropathy + 4.5) / 9.0
        hydropathy_norm = max(0.0, min(1.0, hydropathy_norm))

        # Charge class one-hot (-1, 0, +1)
        charge_onehot = np.zeros(3, dtype=np.float32)
        charge_idx = res_feat.charge_class + 1  # Map -1,0,1 to 0,1,2
        if 0 <= charge_idx < 3:
            charge_onehot[charge_idx] = 1.0

        # HBD/HBA (normalized by typical max around 5)
        hbd_norm = min(1.0, res_feat.hbd / 5.0)
        hba_norm = min(1.0, res_feat.hba / 5.0)

        # Secondary structure placeholder (would need DSSP in production)
        # For now, I am using simple heuristics: hydrophobic + buried = likely helix/sheet
        is_buried = sasa_norm < 0.3
        is_hydrophobic = res_feat.hydropathy > 0
        ss_helix = float(is_buried and is_hydrophobic)
        ss_sheet = float(is_buried and not is_hydrophobic)
        ss_coil = float(not is_buried)

        # Concatenate: 20 + 1 + 1 + 3 + 1 + 1 + 3 = 30 dims base
        feat = np.concatenate([
            aa_onehot,
            [sasa_norm, hydropathy_norm],
            charge_onehot,
            [hbd_norm, hba_norm],
            [ss_helix, ss_sheet, ss_coil]
        ])

        feat_list.append(feat)

    return np.stack(feat_list, axis=0)


def build_edges_distance(pos_ca: np.ndarray, threshold: float = 10.0,
                         knn: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build edges based on CA-CA distance with optional kNN supplement.

    Args:
        pos_ca: CA coordinates (N_res, 3)
        threshold: Maximum distance for edge creation (Angstroms)
        knn: Number of nearest neighbors to ensure connectivity

    Returns:
        edge_index: (2, E) array of edge indices
        distances: (E,) array of edge distances
    """
    n_res = pos_ca.shape[0]

    # Compute pairwise distances
    dist_mat = np.linalg.norm(pos_ca[:, None, :] - pos_ca[None, :, :], axis=2)

    # Create edges for distance threshold
    i_dist, j_dist = np.where((dist_mat <= threshold) & (dist_mat > 0))
    dist_edges = dist_mat[i_dist, j_dist]   # noqa

    # Add kNN edges to ensure connectivity
    edge_set = set(zip(i_dist.tolist(), j_dist.tolist()))

    # For each node, find k nearest neighbors
    for i in range(n_res):
        dists_i = dist_mat[i]
        nearest = np.argsort(dists_i)[1:knn + 1]  # Skip self (distance 0)
        for j in nearest:
            if (i, j) not in edge_set:
                edge_set.add((i, j))
                # Add reverse direction too
                if (j, i) not in edge_set:
                    edge_set.add((j, i))

    # Convert to arrays
    if len(edge_set) > 0:
        edges = np.array(list(edge_set), dtype=np.int64).T  # (2, E)
        distances = dist_mat[edges[0], edges[1]]
    else:
        # Single residue or disconnected
        edges = np.zeros((2, 0), dtype=np.int64)
        distances = np.array([], dtype=np.float32)

    return edges, distances


def build_edge_features(distances: np.ndarray, edge_index: np.ndarray,
                        residue_ids: List[int], n_rbf: int = 16) -> np.ndarray:
    """
    Build edge features from distances and residue positions.

    Features per edge:
        - RBF distance encoding (16 dims)
        - Sequential contact flag (|delta seq| <= 1, 1 dim)
        - Medium range contact (2 <= |delta seq| <= 4, 1 dim)
        - Long range contact (|delta seq| > 4, 1 dim)
    Total: 16 + 3 = 19 dims (can truncate to fe.d_poc_edge if smaller)

    Args:
        distances: Edge distances (E,)
        edge_index: Edge indices (2, E)
        residue_ids: Original residue numbers
        n_rbf: Number of RBF bins

    Returns:
        Edge feature matrix (E, d_edge)
    """
    n_edges = distances.shape[0]
    feat_list = []

    for k in range(n_edges):
        i, j = edge_index[0, k], edge_index[1, k]
        dist = distances[k]

        # RBF encode distance
        rbf = rbf_encode_distance(dist, d_min=0.0, d_max=12.0, n_bins=n_rbf)

        # Sequential separation
        seq_sep = abs(residue_ids[i] - residue_ids[j])
        is_sequential = float(seq_sep <= 1)
        is_medium = float(2 <= seq_sep <= 4)
        is_long = float(seq_sep > 4)

        # Concatenate
        feat = np.concatenate([rbf, [is_sequential, is_medium, is_long]])
        feat_list.append(feat)

    return np.stack(feat_list, axis=0) if len(feat_list) > 0 else np.zeros((0, n_rbf + 3), dtype=np.float32)


def build_pocket_graph(ps: PocketSelect, state_id: int, fe: FeatureSpec) -> PocketGraph:
    """
    Build complete pocket graph from PocketSelect.

    Args:
        ps: PocketSelect with residue features and coordinates
        state_id: State identifier (0=agonist, 1=antagonist)
        fe: FeatureSpec with feature dimensions and flags

    Returns:
        PocketGraph with residue features, edges, and coordinates
    """
    # Build node features
    x = build_residue_features(ps, fe)  # (N_res, d_poc_node)
    n_res = x.shape[0]  # noqa

    # Extract coordinates
    pos_ca_list = []
    pos_sc_list = []

    for res_feat in ps.feats:
        pos_ca_list.append(res_feat.ca_xyz)
        if res_feat.sc_xyz is not None:
            pos_sc_list.append(res_feat.sc_xyz)
        else:
            # Use CA as fallback for glycine or missing sidechain
            pos_sc_list.append(res_feat.ca_xyz)

    pos_ca = np.stack(pos_ca_list, axis=0)  # (N_res, 3)
    pos_sc = np.stack(pos_sc_list, axis=0) if len(pos_sc_list) > 0 else None

    # Build edges
    edge_index, distances = build_edges_distance(pos_ca, threshold=10.0, knn=8)

    # Build edge features
    edge_attr = build_edge_features(distances, edge_index, ps.residues, n_rbf=16)

    # Truncate edge features to match FeatureSpec if needed
    if edge_attr.shape[1] > fe.d_poc_edge:
        edge_attr = edge_attr[:, :fe.d_poc_edge]
    elif edge_attr.shape[1] < fe.d_poc_edge:
        # Padding if too small
        pad_width = fe.d_poc_edge - edge_attr.shape[1]
        edge_attr = np.pad(edge_attr, ((0, 0), (0, pad_width)), mode='constant')

    # Convert residue IDs to array
    residue_ids = np.array(ps.residues, dtype=np.int32)

    return PocketGraph(
        target_id=ps.target_id,
        state_id=state_id,
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos_ca=pos_ca,
        pos_sc=pos_sc,
        residue_ids=residue_ids
    )
