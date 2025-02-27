import random
from typing import Any, Callable, Dict, List

import numpy as np
from ranx import Qrels, Run, evaluate

from redisvl.extensions.router.semantic import SemanticRouter
from redisvl.extensions.threshold_optimizer.base import (
    BaseThresholdOptimizer,
    EvalMetric,
)
from redisvl.extensions.threshold_optimizer.schema import TestData
from redisvl.extensions.threshold_optimizer.utils import NULL_RESPONSE_KEY, format_qrels


def _generate_run_router(test_data: List[TestData], router: SemanticRouter) -> Run:
    """Format router results into format for ranx Run"""
    run_dict: Dict[Any, Any] = {}

    for td in test_data:
        run_dict[td.q_id] = {}
        route_match = router(td.query)
        if route_match and route_match.name == td.query_match:
            run_dict[td.q_id][td.query_match] = 1
        else:
            run_dict[td.q_id][NULL_RESPONSE_KEY] = 1

    return Run(run_dict)


def _eval_router(
    router: SemanticRouter, test_data: List[TestData], qrels: Qrels, eval_metric: str
) -> float:
    """Evaluate acceptable metric given run and qrels data"""
    run = _generate_run_router(test_data, router)
    return evaluate(qrels, run, eval_metric, make_comparable=True)


def _router_random_search(
    route_names: List[str], route_thresholds: dict, search_step=0.10
):
    """Performances random search for many threshold to many route context"""
    score_threshold_values = []
    for route in route_names:
        score_threshold_values.append(
            np.linspace(
                start=max(route_thresholds[route] - search_step, 0),
                stop=route_thresholds[route] + search_step,
                num=100,
            )
        )

    return {
        route: float(random.choice(score_threshold_values[i]))
        for i, route in enumerate(route_names)
    }


def _random_search_opt_router(
    router: SemanticRouter,
    test_data: List[TestData],
    qrels: Qrels,
    eval_metric: EvalMetric,
    **kwargs: Any,
):
    """Performs complete optimization for router cases provide acceptable metric"""
    best_score = _eval_router(router, test_data, qrels, eval_metric.value)
    best_thresholds = router.route_thresholds

    max_iterations = kwargs.get("max_iterations", 20)

    print(f"Starting score {best_score}, starting thresholds {router.route_thresholds}")
    for _ in range(max_iterations):
        route_names = router.route_names
        route_thresholds = router.route_thresholds
        thresholds = _router_random_search(
            route_names=route_names, route_thresholds=route_thresholds
        )
        router.update_route_thresholds(thresholds)
        score = _eval_router(router, test_data, qrels, eval_metric.value)
        if score > best_score:
            best_score = score
            best_thresholds = thresholds

    print(f"Ending score {best_score}, ending thresholds {router.route_thresholds}")
    router.update_route_thresholds(best_thresholds)


class RouterThresholdOptimizer(BaseThresholdOptimizer):
    def __init__(
        self,
        router: SemanticRouter,
        test_dict: List[Dict],
        opt_fn: Callable = _random_search_opt_router,
        eval_metric: str = "f1",
    ):
        super().__init__(router, test_dict, opt_fn, eval_metric)

    def optimize(self, **kwargs: Any):
        """Optimize thresholds using the provided optimization function for router case."""
        qrels = format_qrels(self.test_data)
        self.opt_fn(self.optimizable, self.test_data, qrels, self.eval_metric, **kwargs)
