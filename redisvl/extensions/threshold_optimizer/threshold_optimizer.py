# from enum import Enum
# from typing import List

# from redisvl.extensions.router.semantic import SemanticRouter
# from redisvl.extensions.threshold_optimizer.schema import TestData


# class OptimizerType(Enum):
#     RANDOM = "random"


# class ThresholdOptimizer:
#     def optimize_router_thresholds(
#         self,
#         router: SemanticRouter,
#         optimizer: OptimizerType,
#         test_data: List[dict],
#         max_iterations: int = 300,
#     ):
#         test_data = [TestData(**data) for data in test_data]  # validate with pydantic

#         # starting condition for search
#         best_acc = self._eval_accuracy(test_data)
#         best_thresholds = self.route_thresholds

#         for _ in range(max_iterations):
#             route_names = self.route_names
#             route_thresholds = self.route_thresholds
#             thresholds = self._random_search(
#                 route_names=route_names, route_thresholds=route_thresholds
#             )
#             self.update_route_thresholds(thresholds)
#             acc = self._eval_accuracy(test_data)
#             print(f"Accuracy: {acc}, Best: {best_acc}, Thresholds: {thresholds}")
#             if acc > best_acc:
#                 best_acc = acc
#                 best_thresholds = thresholds
#                 print("Updated best accuracy")

#         self.router.update_route_thresholds(best_thresholds)
