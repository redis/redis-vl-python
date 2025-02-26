from typing import Any, Dict, List

import numpy as np
from ranx import Qrels, Run, evaluate

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


def format_qrels(test_data: List[TestData]) -> Qrels:
    qrels_dict = {}

    for td in test_data:
        if td.query_match:
            qrels_dict[td.q_id] = {td.query_match: 1}

    return Qrels(qrels_dict)


def generate_run_router(test_data, router):
    run_dict = {}

    for td in test_data:
        run_dict[td.q_id] = {}
        route_match = router(td.query)
        print(f"{td.query=}\n\n{route_match=}\n\n")
        if route_match and route_match.name == td.query_match:
            run_dict[td.q_id][td.query_match] = 1
        else:
            run_dict[td.q_id]["no_match"] = 1

    return Run(run_dict)


def eval_router_f1(router, test_data, qrels):
    run = generate_run_router(test_data, router)
    f1 = evaluate(qrels, run, "f1", make_comparable=True)
    return f1


def generate_run(test_data: List[TestData], threshold: float) -> Run:
    run_dict = {}

    for td in test_data:
        run_dict[td.q_id] = {}
        for res in td.response:
            if float(res["vector_distance"]) < threshold:
                # value of 1 is irrelevant checks only on match for f1
                run_dict[td.q_id][res["id"]] = 1

        if not run_dict[td.q_id]:
            # ranx is a little odd in that if there are no matches it errors
            # if however there are no keys that match you get the correct score
            run_dict[td.q_id]["no_results"] = 1

    return Run(run_dict)


def calc_f1_metrics_per_threshold(
    qrels: Qrels,
    test_data: List[TestData],
):
    thresholds = np.linspace(0.01, 0.8, 60)
    metrics = {}

    for threshold in thresholds:
        run = generate_run(test_data, threshold)
        f1 = evaluate(qrels, run, "f1", make_comparable=True)
        metrics[threshold] = {"F1": f1}

    return metrics


def get_best_threshold(metrics: dict) -> float:
    """
    Returns the threshold with the highest F1 score.
    If multiple thresholds have the same F1 score, returns the lowest threshold.
    """
    return max(metrics.items(), key=lambda x: (x[1]["F1"], -x[0]))[0]
