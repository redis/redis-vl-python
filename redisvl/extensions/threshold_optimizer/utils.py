from typing import List

from ranx import Qrels

from redisvl.extensions.threshold_optimizer.schema import TestData

NULL_RESPONSE_KEY = "no_match"


def format_qrels(test_data: List[TestData]) -> Qrels:
    qrels_dict = {}

    for td in test_data:
        if td.query_match:
            qrels_dict[td.q_id] = {td.query_match: 1}
        else:
            # This is for capturing true negatives from test set
            qrels_dict[td.q_id] = {NULL_RESPONSE_KEY: 1}

    return Qrels(qrels_dict)


def validate_test_dict(test_dict: List[dict]) -> List[TestData]:
    return [TestData(**d) for d in test_dict]
