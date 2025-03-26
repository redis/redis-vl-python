from typing import List

from ranx import Qrels

from redisvl.utils.optimize.schema import LabeledData

NULL_RESPONSE_KEY = "no_match"


def _format_qrels(test_data: List[LabeledData]) -> Qrels:
    """Utility function for creating qrels for evaluation with ranx"""
    qrels_dict = {}

    for td in test_data:
        if td.query_match:
            qrels_dict[td.id] = {td.query_match: 1}
        else:
            # This is for capturing true negatives from test set
            qrels_dict[td.id] = {NULL_RESPONSE_KEY: 1}

    return Qrels(qrels_dict)


def _validate_test_dict(test_dict: List[dict]) -> List[LabeledData]:
    """Convert/validate test_dict for use in optimizer"""
    return [LabeledData(**d) for d in test_dict]
