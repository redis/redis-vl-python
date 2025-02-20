from typing import Any, Dict, List

import numpy as np

from redisvl.extensions.threshold_optimizer.schema import TestData
from redisvl.redis.utils import buffer_to_array


def calc_cosine_distance(
    vector1: bytes | np.ndarray | list[float], vector2: bytes | np.ndarray | list[float]
) -> float:
    if isinstance(vector1, bytes):
        vector1 = buffer_to_array(vector1, dtype="float32")
    if isinstance(vector2, bytes):
        vector2 = buffer_to_array(vector2, dtype="float32")

    return 1 - (
        np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    )


def calc_f1_metrics_per_threshold(
    distances: np.ndarray,
    test_data: List[TestData],
    cached_records: List[Dict[str, Any]],
):
    thresholds = np.linspace(0.01, 0.8, 60)
    metrics = {}

    for threshold in thresholds:
        TP, TN, FP, FN = 0, 0, 0, 0

        for i, test in enumerate(test_data):
            # print(i, test)
            # distance_of_nearest = np.min(distances[i, :])
            index_of_nearest = np.argmin(distances[:, i])
            print("nearest index: ", index_of_nearest)
            # print(distances[:, i][index_of_nearest])
            if distances[:, i][index_of_nearest] < threshold:
                print(test.query_match == cached_records[index_of_nearest]["id"])
                print(test.query_match, cached_records[index_of_nearest]["id"], "\n\n")
                if test.query_match == cached_records[index_of_nearest]["id"]:
                    TP += 1
                else:
                    FP += 1
            else:
                if test.query_match:
                    FN += 1
                else:
                    TN += 1

        # print(TP, TN, FP, FN)

        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        F1 = (
            2 * (precision * recall) / (precision + recall)
            if precision + recall > 0
            else 0
        )
        accuracy = (TP + TN) / len(test_data)

        metrics[threshold] = {
            "precision": precision,
            "recall": recall,
            "F1": F1,
            "accuracy": accuracy,
        }

    return metrics


def get_best_threshold(metrics: dict) -> float:
    """
    Returns the threshold with the highest F1 score.
    If multiple thresholds have the same F1 score, returns the lowest threshold.
    """
    return max(metrics.items(), key=lambda x: (x[1]["F1"], -x[0]))[0]
