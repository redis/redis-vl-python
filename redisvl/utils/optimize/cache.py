from typing import Any, Callable, Dict, List

import numpy as np
from ranx import Qrels, Run, evaluate

from redisvl.extensions.llmcache.semantic import SemanticCache
from redisvl.query import RangeQuery
from redisvl.utils.optimize.base import BaseThresholdOptimizer, EvalMetric
from redisvl.utils.optimize.schema import TestData
from redisvl.utils.optimize.utils import NULL_RESPONSE_KEY, _format_qrels


def _generate_run_cache(test_data: List[TestData], threshold: float) -> Run:
    """Format observed data for evaluation with ranx"""
    run_dict: Dict[str, Dict[str, int]] = {}

    for td in test_data:
        run_dict[td.id] = {}
        for res in td.response:
            if float(res["vector_distance"]) < threshold:
                # value of 1 is irrelevant checks only on match for f1
                run_dict[td.id][res["id"]] = 1

        if not run_dict[td.id]:
            # ranx is a little odd in that if there are no matches it errors
            # if however there are no keys that match you get the correct score
            run_dict[td.id][NULL_RESPONSE_KEY] = 1

    return Run(run_dict)


def _eval_cache(
    test_data: List[TestData], threshold: float, qrels: Qrels, metric: str
) -> float:
    """Formats run data and evaluates supported metric"""
    run = _generate_run_cache(test_data, threshold)
    return evaluate(qrels, run, metric, make_comparable=True)


def _get_best_threshold(metrics: dict) -> float:
    """
    Returns the threshold with the highest F1 score.
    If multiple thresholds have the same F1 score, returns the lowest threshold.
    """
    return max(metrics.items(), key=lambda x: (x[1]["score"], -x[0]))[0]


def _grid_search_opt_cache(
    cache: SemanticCache, test_data: List[TestData], eval_metric: EvalMetric
):
    """Evaluates all thresholds in linspace for cache to determine optimal"""
    thresholds = np.linspace(0.01, 0.8, 60)
    metrics = {}

    for td in test_data:
        vec = cache._vectorizer.embed(td.query)
        query = RangeQuery(
            vec, vector_field_name="prompt_vector", distance_threshold=1.0
        )
        res = cache.index.query(query)
        td.response = res

    qrels = _format_qrels(test_data)

    for threshold in thresholds:
        score = _eval_cache(test_data, threshold, qrels, eval_metric.value)
        metrics[threshold] = {"score": score}

    best_threshold = _get_best_threshold(metrics)
    cache.set_threshold(best_threshold)


class CacheThresholdOptimizer(BaseThresholdOptimizer):
    def __init__(
        self,
        cache: SemanticCache,
        test_dict: List[Dict],
        opt_fn: Callable = _grid_search_opt_cache,
        eval_metric: str = "f1",
    ):
        super().__init__(cache, test_dict, opt_fn, eval_metric)

    def optimize(self, **kwargs: Any):
        """Optimize thresholds using the provided optimization function for cache case."""
        self.opt_fn(self.optimizable, self.test_data, self.eval_metric, **kwargs)
