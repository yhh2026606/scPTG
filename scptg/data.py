from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder


@dataclass
class SingleCellData:
    x: np.ndarray
    pca: np.ndarray
    labels: np.ndarray
    obs_names: np.ndarray
    n_clusters: int
    metadata: dict[str, object]


def load_data(path: str | Path, config: dict) -> SingleCellData:
    path = Path(path)
    adata = ad.read_h5ad(path)
    if "label" not in adata.obs:
        raise KeyError(f"{path.name} must contain adata.obs['label']")

    status = str(adata.obs["data_status"].iloc[0]) if "data_status" in adata.obs else "unknown"
    use_counts = bool(config.get("prefer_counts", True)) and "counts" in adata.layers
    matrix = adata.layers["counts"] if use_counts else adata.X
    matrix = matrix.tocsr().astype(np.float32) if sp.issparse(matrix) else sp.csr_matrix(matrix, dtype=np.float32)

    if use_counts or status == "raw_counts":
        totals = np.asarray(matrix.sum(axis=1)).ravel()
        matrix = sp.diags((float(config["target_sum"]) / np.maximum(totals, 1.0)).astype(np.float32)) @ matrix
        matrix.data = np.log1p(matrix.data)

    n_hvg = min(int(config["n_hvg"]), matrix.shape[1])
    mean = np.asarray(matrix.mean(axis=0)).ravel()
    variance = np.maximum(np.asarray(matrix.power(2).mean(axis=0)).ravel() - mean * mean, 0.0)
    selected = np.argpartition(variance, -n_hvg)[-n_hvg:]
    selected = selected[np.argsort(variance[selected])[::-1]]
    dense = matrix[:, selected].toarray().astype(np.float32, copy=False)
    dense = (dense - dense.mean(axis=0, keepdims=True)) / np.maximum(dense.std(axis=0, keepdims=True), 1e-6)
    np.clip(dense, -float(config["clip"]), float(config["clip"]), out=dense)
    x = np.ascontiguousarray(dense, dtype=np.float32)

    pca_dim = min(int(config["pca_dim"]), x.shape[0] - 1, x.shape[1])
    pca = PCA(n_components=pca_dim, svd_solver="randomized", random_state=0).fit_transform(x).astype(np.float32)
    labels = LabelEncoder().fit_transform(adata.obs["label"].astype(str)).astype(np.int64)
    return SingleCellData(
        x=x,
        pca=pca,
        labels=labels,
        obs_names=np.asarray(adata.obs_names.astype(str)),
        n_clusters=int(np.unique(labels).size),
        metadata={
            "dataset": path.stem,
            "path": str(path.resolve()),
            "n_cells": int(x.shape[0]),
            "n_genes_original": int(adata.n_vars),
            "n_genes_selected": int(x.shape[1]),
            "n_clusters": int(np.unique(labels).size),
            "matrix_source": "layers/counts" if use_counts else "X",
        },
    )

