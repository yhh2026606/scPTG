from __future__ import annotations

import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans


def multiscale_communities(
    features: np.ndarray, edges: np.ndarray, weights: np.ndarray,
    n_clusters: int, resolutions: list[float], seed: int,
) -> np.ndarray:
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    adjacency = sp.csr_matrix((np.concatenate([weights, weights]), (rows, cols)), shape=(len(features), len(features)))
    graph = nx.from_scipy_sparse_array(adjacency, edge_attribute="weight")
    partitions: list[list[set[int]]] = []
    signatures: list[np.ndarray] = []
    for resolution in sorted(float(r) for r in resolutions):
        groups = nx.community.louvain_communities(graph, weight="weight", resolution=resolution, seed=seed)
        partitions.append(groups)
        labels = _labels(groups, len(features))
        signatures.append(np.eye(len(groups), dtype=np.float32)[labels])
    signature = np.concatenate(signatures, axis=1)
    groups = _refine(partitions[0], signature, n_clusters, seed)
    if len(groups) < n_clusters:
        groups = _refine(groups, features, n_clusters, seed)
    if len(groups) > n_clusters:
        groups = _merge(groups, signature, n_clusters, seed)
    labels = _labels(groups, len(features))
    if np.unique(labels).size != n_clusters:
        raise RuntimeError("Community initialization did not produce exactly K clusters")
    return labels


def cluster_centers(features: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    return np.stack([features[labels == c].mean(axis=0) for c in range(n_clusters)]).astype(np.float32)


def _labels(groups: list[set[int]], n: int) -> np.ndarray:
    labels = np.full(n, -1, dtype=np.int64)
    for label, members in enumerate(groups):
        labels[np.fromiter(members, dtype=np.int64)] = label
    if np.any(labels < 0):
        raise RuntimeError("Community partition left cells unassigned")
    return labels


def _refine(groups: list[set[int]], features: np.ndarray, k: int, seed: int) -> list[set[int]]:
    groups = [set(g) for g in groups]
    while len(groups) < k:
        candidates = []
        for index, members in enumerate(groups):
            if len(members) < 4:
                continue
            ids = np.fromiter(members, dtype=np.int64)
            local = features[ids]
            total = float(np.square(local - local.mean(axis=0)).sum())
            if total <= 1e-8:
                continue
            split = KMeans(2, n_init=10, random_state=seed).fit(local)
            candidates.append((1.0 - float(split.inertia_) / total, index, ids, split.labels_))
        if not candidates:
            break
        _, index, ids, labels = max(candidates, key=lambda item: item[0])
        groups[index] = set(ids[labels == 0].tolist())
        groups.append(set(ids[labels == 1].tolist()))
    return groups


def _merge(groups: list[set[int]], features: np.ndarray, k: int, seed: int) -> list[set[int]]:
    centers = np.stack([features[np.fromiter(g, dtype=np.int64)].mean(axis=0) for g in groups])
    sizes = np.asarray([len(g) for g in groups], dtype=np.float64)
    assignment = KMeans(k, n_init=20, random_state=seed).fit(centers, sample_weight=sizes).labels_
    merged = [set() for _ in range(k)]
    for group, label in zip(groups, assignment):
        merged[int(label)].update(group)
    return merged

