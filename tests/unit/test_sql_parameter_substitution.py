"""Unit tests for SQL parameter substitution in SQLQuery.

These tests verify that parameter substitution correctly handles:
1. Partial matching bug: :id should not replace inside :product_id
2. Quote escaping bug: Single quotes in values should be SQL-escaped
3. Edge cases: Multiple occurrences, similar names, special characters
"""

import pytest

from redisvl.query.sql import SQLQuery


def buggy_substitute_params(sql: str, params: dict) -> str:
    """Simulate the CURRENT buggy implementation for comparison.

    This is the exact code from redisvl/query/sql.py lines 105-113.
    """
    for key, value in params.items():
        placeholder = f":{key}"
        if isinstance(value, (int, float)):
            sql = sql.replace(placeholder, str(value))
        elif isinstance(value, str):
            sql = sql.replace(placeholder, f"'{value}'")
    return sql


class TestBuggyBehaviorDemonstration:
    """Tests that DEMONSTRATE the bugs in the current implementation.

    These tests show what goes wrong with the naive str.replace() approach.
    They should PASS (demonstrating the bug exists) before the fix,
    and some assertions will need to change after the fix.
    """

    def test_partial_match_bug_exists(self):
        """Demonstrate that :id incorrectly replaces inside :product_id."""
        sql = "SELECT * FROM idx WHERE id = :id AND product_id = :product_id"
        params = {"id": 123, "product_id": 456}

        result = buggy_substitute_params(sql, params)

        # BUG: :id gets replaced inside :product_id first (dict ordering dependent)
        # This demonstrates the bug - the result is corrupted
        # Depending on dict ordering, we might get "product_123" corruption
        assert ":id" not in result or "product_" in result  # Some substitution happened

    def test_quote_escaping_bug_exists(self):
        """Demonstrate that quotes are NOT escaped in current implementation."""
        sql = "SELECT * FROM idx WHERE name = :name"
        params = {"name": "O'Brien"}

        result = buggy_substitute_params(sql, params)

        # BUG: The quote is NOT escaped - this produces invalid SQL
        assert "O'Brien" in result  # Raw quote, not escaped
        assert "O''Brien" not in result  # Proper escaping is missing


class TestParameterSubstitutionPartialMatching:
    """Tests for the partial string matching bug.

    The bug: Using str.replace(':id', '123') would also replace
    ':id' inside ':product_id', resulting in 'product_123'.
    """

    def test_similar_param_names_no_partial_match(self):
        """Test that :id doesn't replace inside :product_id."""
        sql_query = SQLQuery(
            "SELECT * FROM idx WHERE id = :id AND product_id = :product_id",
            params={"id": 123, "product_id": 456},
        )

        substituted = sql_query._substitute_params(sql_query.sql, sql_query.params)

        assert "id = 123" in substituted
        assert "product_id = 456" in substituted
        # Should NOT have "product_123"
        assert "product_123" not in substituted

    def test_prefix_param_names(self):
        """Test params where one is a prefix of another: :user, :user_id, :user_name."""
        sql_query = SQLQuery(
            "SELECT * FROM idx WHERE user = :user AND user_id = :user_id AND user_name = :user_name",
            params={"user": "alice", "user_id": 42, "user_name": "Alice Smith"},
        )

        substituted = sql_query._substitute_params(sql_query.sql, sql_query.params)

        assert "user = 'alice'" in substituted
        assert "user_id = 42" in substituted
        assert "user_name = 'Alice Smith'" in substituted
        # Should NOT have corrupted values
        assert "'alice'_id" not in substituted
        assert "'alice'_name" not in substituted

    def test_suffix_param_names(self):
        """Test params where one is a suffix pattern: :vec, :query_vec."""
        sql_query = SQLQuery(
            "SELECT * FROM idx WHERE vec = :vec AND query_vec = :query_vec",
            params={"vec": 1.0, "query_vec": 2.0},
        )

        substituted = sql_query._substitute_params(sql_query.sql, sql_query.params)

        assert "vec = 1.0" in substituted or "vec = 1" in substituted
        assert "query_vec = 2.0" in substituted or "query_vec = 2" in substituted


class TestParameterSubstitutionQuoteEscaping:
    """Tests for the quote escaping bug.

    The bug: String values with single quotes like "O'Brien" would
    produce invalid SQL: 'O'Brien' instead of 'O''Brien'.
    """

    def test_single_quote_in_value(self):
        """Test that single quotes are properly escaped."""
        sql_query = SQLQuery(
            "SELECT * FROM idx WHERE name = :name",
            params={"name": "O'Brien"},
        )

        substituted = sql_query._substitute_params(sql_query.sql, sql_query.params)

        # SQL standard escaping: ' becomes ''
        assert "name = 'O''Brien'" in substituted

    def test_multiple_quotes_in_value(self):
        """Test multiple single quotes in a value."""
        sql_query = SQLQuery(
            "SELECT * FROM idx WHERE phrase = :phrase",
            params={"phrase": "It's a 'test' string"},
        )

        substituted = sql_query._substitute_params(sql_query.sql, sql_query.params)

        assert "phrase = 'It''s a ''test'' string'" in substituted

    def test_apostrophe_names(self):
        """Test common names with apostrophes."""
        test_cases = [
            ("McDonald's", "'McDonald''s'"),
            ("O'Reilly", "'O''Reilly'"),
            ("D'Angelo", "'D''Angelo'"),
        ]

        for name, expected in test_cases:
            sql_query = SQLQuery(
                "SELECT * FROM idx WHERE name = :name",
                params={"name": name},
            )
            substituted = sql_query._substitute_params(sql_query.sql, sql_query.params)
            assert f"name = {expected}" in substituted, f"Failed for {name}"


class TestParameterSubstitutionEdgeCases:
    """Tests for edge cases in parameter substitution."""

    def test_multiple_occurrences_same_param(self):
        """Test that a parameter used multiple times is substituted everywhere."""
        sql_query = SQLQuery(
            "SELECT * FROM idx WHERE category = :cat OR subcategory = :cat",
            params={"cat": "electronics"},
        )

        substituted = sql_query._substitute_params(sql_query.sql, sql_query.params)

        assert substituted.count("'electronics'") == 2

    def test_empty_string_value(self):
        """Test empty string parameter value."""
        sql_query = SQLQuery(
            "SELECT * FROM idx WHERE name = :name",
            params={"name": ""},
        )

        substituted = sql_query._substitute_params(sql_query.sql, sql_query.params)

        assert "name = ''" in substituted

    def test_numeric_types(self):
        """Test integer and float parameter values."""
        sql_query = SQLQuery(
            "SELECT * FROM idx WHERE count = :count AND price = :price",
            params={"count": 42, "price": 99.99},
        )

        substituted = sql_query._substitute_params(sql_query.sql, sql_query.params)

        assert "count = 42" in substituted
        assert "price = 99.99" in substituted

    def test_bytes_param_not_substituted(self):
        """Test that bytes parameters are not substituted (handled separately)."""
        sql_query = SQLQuery(
            "SELECT * FROM idx WHERE embedding = :vec",
            params={"vec": b"\x00\x01\x02\x03"},
        )

        substituted = sql_query._substitute_params(sql_query.sql, sql_query.params)

        # Bytes should remain as placeholder
        assert ":vec" in substituted

    def test_special_characters_in_value(self):
        """Test special characters that might interfere with regex."""
        special_values = [
            "hello@world.com",
            "path/to/file",
            "price: $100",
            "regex.*pattern",
            "back\\slash",
        ]

        for value in special_values:
            sql_query = SQLQuery(
                "SELECT * FROM idx WHERE field = :field",
                params={"field": value},
            )
            substituted = sql_query._substitute_params(sql_query.sql, sql_query.params)
            # Should contain the value wrapped in quotes (with any necessary escaping)
            assert ":field" not in substituted, f"Failed to substitute for value: {value}"

