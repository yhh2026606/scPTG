from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true_values, true_ids = np.unique(y_true, return_inverse=True)
    pred_values, pred_ids = np.unique(y_pred, return_inverse=True)
    table = np.zeros((len(true_values), len(pred_values)), dtype=np.int64)
    np.add.at(table, (true_ids, pred_ids), 1)
    rows, cols = linear_sum_assignment(table, maximize=True)
    return {
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "acc": float(table[rows, cols].sum() / len(y_true)),
    }

