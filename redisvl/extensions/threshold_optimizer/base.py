from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, TypeVar

from redisvl.extensions.threshold_optimizer.utils import validate_test_dict


class EvalMetric(Enum):
    """Evaluation metrics for threshold optimization."""

    F1 = "f1"
    PRECISION = "precision"
    RECALL = "recall"
    ACCURACY = "accuracy"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_string(cls, metric: str) -> "EvalMetric":
        """Convert string to EvalMetric enum."""
        try:
            return cls(metric.lower())
        except ValueError:
            raise ValueError(
                f"Invalid metric: {metric}. Valid options are: {', '.join(m.value for m in cls)}"
            )


T = TypeVar("T")  # Type variable for the optimizable object (Cache or Router)


class BaseThresholdOptimizer(ABC):
    """Abstract base class for threshold optimizers."""

    def __init__(
        self,
        optimizable: T,
        test_dict: List[Dict],
        opt_fn: Callable,
        eval_metric: str = "f1",
    ):
        """Initialize the optimizer.

        Args:
            optimizable: The object to optimize (Cache or Router)
            test_dict: List of test cases
            eval_fn: Function to evaluate performance
            opt_fn: Function to perform optimization
        """
        self.test_data = validate_test_dict(test_dict)
        self.optimizable = optimizable
        self.eval_metric = EvalMetric(eval_metric)
        self.opt_fn = opt_fn

    @abstractmethod
    def optimize(self, **kwargs: Any):
        """Optimize thresholds using the provided optimization function."""
        pass
