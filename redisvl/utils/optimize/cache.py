from typing import Any, Callable, Dict, List

import numpy as np
from ranx import Qrels, Run, evaluate

from redisvl.extensions.llmcache.semantic import SemanticCache
from redisvl.query import RangeQuery
from redisvl.utils.optimize.base import BaseThresholdOptimizer, EvalMetric
from redisvl.utils.optimize.schema import LabeledData
from redisvl.utils.optimize.utils import NULL_RESPONSE_KEY, _format_qrels


def _generate_run_cache(test_data: List[LabeledData], threshold: float) -> Run:
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
    test_data: List[LabeledData], threshold: float, qrels: Qrels, metric: str
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
    cache: SemanticCache, test_data: List[LabeledData], eval_metric: EvalMetric
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
    """
    Class for optimizing thresholds for a SemanticCache.

    .. code-block:: python

        from redisvl.extensions.llmcache import SemanticCache
        from redisvl.utils.optimize import CacheThresholdOptimizer

        sem_cache = SemanticCache(
            name="sem_cache",                    # underlying search index name
            redis_url="redis://localhost:6379",  # redis connection url string
            distance_threshold=0.5               # semantic cache distance threshold
        )

        paris_key = sem_cache.store(prompt="what is the capital of france?", response="paris")
        rabat_key = sem_cache.store(prompt="what is the capital of morocco?", response="rabat")

        test_data = [
            {
                "query": "What's the capital of Britain?",
                "query_match": ""
            },
            {
                "query": "What's the capital of France??",
                "query_match": paris_key
            },
            {
                "query": "What's the capital city of Morocco?",
                "query_match": rabat_key
            },
        ]

        optimizer = CacheThresholdOptimizer(sem_cache, test_data)
        optimizer.optimize()
    """

    def __init__(
        self,
        cache: SemanticCache,
        test_dict: List[Dict[str, Any]],
        opt_fn: Callable = _grid_search_opt_cache,
        eval_metric: str = "f1",
    ):
        """Initialize the cache optimizer.

        Args:
            cache (SemanticCache): The RedisVL SemanticCache instance to optimize.
            test_dict (List[Dict[str, Any]]): List of test cases.
            opt_fn (Callable): Function to perform optimization. Defaults to
                grid search.
            eval_metric (str): Evaluation metric for threshold optimization.
                Defaults to "f1" score.

        Raises:
            ValueError: If the test_dict not in LabeledData format.
        """
        super().__init__(cache, test_dict, opt_fn, eval_metric)

    def optimize(self, **kwargs: Any):
        """Optimize thresholds using the provided optimization function for cache case."""
        self.opt_fn(self.optimizable, self.test_data, self.eval_metric, **kwargs)
