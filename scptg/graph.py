from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.neighbors import NearestNeighbors


@dataclass
class Graph:
    propagation: torch.Tensor
    edges: np.ndarray
    weights: np.ndarray
    metadata: dict[str, float | int]

    def to(self, device: torch.device) -> "Graph":
        self.propagation = self.propagation.to(device)
        return self


def build_graph(pca: np.ndarray, k: int, rank_sigma: float, self_loop: float) -> Graph:
    n = len(pca)
    k = min(int(k), max(1, n - 1))
    indices = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=1).fit(pca).kneighbors(return_distance=False)
    ranks: list[dict[int, int]] = []
    for i, row in enumerate(indices):
        neighbors = [int(j) for j in row if int(j) != i][:k]
        ranks.append({j: rank for rank, j in enumerate(neighbors, 1)})

    pairs = sorted({(min(i, j), max(i, j)) for i, row in enumerate(ranks) for j in row})
    edges = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    weights = np.asarray([
        np.exp(-(ranks[i].get(j, k + 1) + ranks[j].get(i, k + 1)) / (2.0 * rank_sigma))
        for i, j in pairs
    ], dtype=np.float32)
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    adjacency = sp.csr_matrix((np.concatenate([weights, weights]), (rows, cols)), shape=(n, n), dtype=np.float32)
    base = adjacency + sp.eye(n, dtype=np.float32, format="csr") * float(self_loop)
    degree = np.asarray(base.sum(axis=1)).ravel()
    scale = sp.diags(1.0 / np.sqrt(np.maximum(degree, 1e-12)))
    propagation = (scale @ base @ scale).tocoo()
    tensor = torch.sparse_coo_tensor(
        torch.from_numpy(np.vstack([propagation.row, propagation.col]).astype(np.int64)),
        torch.from_numpy(propagation.data.astype(np.float32)), propagation.shape,
    ).coalesce()
    components = sp.csgraph.connected_components(adjacency, directed=False, return_labels=False)
    return Graph(tensor, edges, weights, {
        "k": k,
        "n_edges": int(len(edges)),
        "n_components": int(components),
        "mean_degree": float((adjacency > 0).sum(axis=1).mean()),
    })

