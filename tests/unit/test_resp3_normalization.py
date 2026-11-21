import pytest
from redis.commands.search.aggregation import AggregateResult, Cursor
from redis.exceptions import ResponseError

try:
    from redis.commands.search.hybrid_result import HybridResult
except ImportError:
    HybridResult = None  # type: ignore[misc, assignment]

from redisvl.redis.utils import normalize_resp3_result


def test_passthrough():
    # Pass-through for AggregateResult
    agg = AggregateResult([], None, None)
    assert normalize_resp3_result(agg, target_type=AggregateResult) is agg

    # Pass-through for HybridResult
    if HybridResult is not None:
        hyb = HybridResult(total_results=0, results=[], warnings=[], execution_time=0.0)
        assert normalize_resp3_result(hyb, target_type=HybridResult) is hyb

    # Pass-through for non-dict/non-list
    assert (
        normalize_resp3_result("some_string", target_type=AggregateResult)
        == "some_string"
    )
    assert normalize_resp3_result(12345, target_type=AggregateResult) == 12345


def test_aggregate_resp3_decode_responses_true():
    raw_wire = {
        "total_results": 2,
        "results": [
            {"extra_attributes": {"tag": "category_0", "txt": "hello 0"}, "values": []},
            {"extra_attributes": {"tag": "category_1", "txt": "hello 1"}, "values": []},
        ],
        "warning": ["some warning"],
        "format": "STRING",
        "attributes": [],
    }
    res = normalize_resp3_result(raw_wire, target_type=AggregateResult)
    assert isinstance(res, AggregateResult)
    assert res.cursor is None
    assert len(res.rows) == 2
    assert res.rows[0] == ["tag", "category_0", "txt", "hello 0"]
    assert res.rows[1] == ["tag", "category_1", "txt", "hello 1"]
    assert getattr(res, "total", None) == 2
    assert getattr(res, "warnings", None) == ["some warning"]


def test_aggregate_resp3_decode_responses_false():
    raw_wire = {
        b"total_results": 1,
        b"results": [
            {b"extra_attributes": {b"tag": b"news", b"txt": b"world"}, b"values": []},
        ],
        b"warning": [b"raw byte warning"],
        b"format": b"STRING",
        b"attributes": [],
    }
    res = normalize_resp3_result(raw_wire, target_type=AggregateResult)
    assert isinstance(res, AggregateResult)
    assert res.cursor is None
    assert len(res.rows) == 1
    assert res.rows[0] == [b"tag", b"news", b"txt", b"world"]
    assert getattr(res, "total", None) == 1
    assert getattr(res, "warnings", None) == ["raw byte warning"]


def test_aggregate_resp3_cursor_batch_and_final():
    # Initial / Intermediate cursor read
    raw_cursor_1 = [
        {
            "total_results": 2,
            "results": [
                {"extra_attributes": {"__key": "doc:0"}, "values": []},
                {"extra_attributes": {"__key": "doc:1"}, "values": []},
            ],
            "warning": [],
        },
        987654321,
    ]
    res1 = normalize_resp3_result(raw_cursor_1, target_type=AggregateResult)
    assert isinstance(res1, AggregateResult)
    assert res1.cursor is not None
    assert res1.cursor.cid == 987654321
    assert len(res1.rows) == 2
    assert res1.rows[0] == ["__key", "doc:0"]

    # Final cursor read (exhausted)
    raw_cursor_final = [
        {
            "total_results": 1,
            "results": [
                {"extra_attributes": {"__key": "doc:2"}, "values": []},
            ],
            "warning": [],
        },
        0,
    ]
    res_final = normalize_resp3_result(raw_cursor_final, target_type=AggregateResult)
    assert isinstance(res_final, AggregateResult)
    assert res_final.cursor is not None
    assert res_final.cursor.cid == 0
    assert len(res_final.rows) == 1
    assert res_final.rows[0] == ["__key", "doc:2"]


def test_aggregate_empty_results():
    raw_empty = {
        "total_results": 0,
        "results": [],
        "warning": [],
        "format": "STRING",
        "attributes": [],
    }
    res = normalize_resp3_result(raw_empty, target_type=AggregateResult)
    assert isinstance(res, AggregateResult)
    assert res.rows == []
    assert getattr(res, "total", None) == 0


def test_aggregate_rows_empty_or_missing_extra_attributes():
    # When queries do not project attributes (e.g. FT.AGGREGATE * without LOAD),
    # Redis sends {"extra_attributes": {}, "values": []} per matched row.
    # Rows with empty or missing extra_attributes must produce an empty list row []
    # to maintain row indexing alignment and 1:1 parity with total_results.
    raw_rows = {
        "total_results": 3,
        "results": [
            {"extra_attributes": {}, "values": []},
            {"values": []},  # missing extra_attributes key
            {"extra_attributes": {"title": "item 3"}},
        ],
        "warning": [],
    }
    res = normalize_resp3_result(raw_rows, target_type=AggregateResult)
    assert isinstance(res, AggregateResult)
    assert len(res.rows) == 3
    assert res.rows[0] == []
    assert res.rows[1] == []
    assert res.rows[2] == ["title", "item 3"]


def test_error_handling():
    # Non-empty error in aggregate map raises ResponseError
    raw_with_error = {
        "error": "Syntax error in aggregation",
        "results": [],
        "total_results": 0,
    }
    with pytest.raises(ResponseError, match="Syntax error in aggregation"):
        normalize_resp3_result(raw_with_error, target_type=AggregateResult)

    # Byte error in aggregate map raises ResponseError
    raw_with_byte_error = {
        b"error": b"Unknown function",
        b"results": [],
    }
    with pytest.raises(ResponseError, match="Unknown function"):
        normalize_resp3_result(raw_with_byte_error, target_type=AggregateResult)

    # Empty error does NOT raise
    raw_with_empty_error = {
        "error": "",
        "results": [],
        "total_results": 0,
    }
    res = normalize_resp3_result(raw_with_empty_error, target_type=AggregateResult)
    assert isinstance(res, AggregateResult)


@pytest.mark.skipif(HybridResult is None, reason="Requires redis-py >= 7.1.0")
def test_hybrid_search_resp3():
    raw_hybrid = {
        "total_results": 2,
        "results": [
            {"__key": "doc:0", "__score": "0.95", "txt": "hello"},
            {"__key": "doc:1", "__score": "0.80", "txt": "world"},
        ],
        "warnings": ["warning 1"],
        "execution_time": 0.123,
    }
    res = normalize_resp3_result(raw_hybrid, target_type=HybridResult)
    assert isinstance(res, HybridResult)
    assert res.total_results == 2
    assert len(res.results) == 2
    assert res.results[0]["__key"] == "doc:0"
    assert res.warnings == ["warning 1"]
    assert res.execution_time == 0.123

    # Error in hybrid search
    raw_hybrid_err = {
        "error": "Failed hybrid search sub-query",
    }
    with pytest.raises(ResponseError, match="Failed hybrid search sub-query"):
        normalize_resp3_result(raw_hybrid_err, target_type=HybridResult)
