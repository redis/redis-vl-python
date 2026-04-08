from typing import Any, Dict, Iterable, List, Optional, Union

from redisvl.mcp.errors import MCPErrorCode, RedisVLMCPError
from redisvl.query.filter import FilterExpression, Num, Tag, Text
from redisvl.schema import IndexSchema


def parse_filter(
    value: Optional[Union[str, Dict[str, Any]]], schema: IndexSchema
) -> Optional[Union[str, FilterExpression]]:
    """Parse an MCP filter value into a RedisVL filter representation."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        raise RedisVLMCPError(
            "filter must be a string or object",
            code=MCPErrorCode.INVALID_FILTER,
            retryable=False,
        )
    return _parse_expression(value, schema)


def _parse_expression(value: Dict[str, Any], schema: IndexSchema) -> FilterExpression:
    logical_keys = [key for key in ("and", "or", "not") if key in value]
    if logical_keys:
        if len(logical_keys) != 1 or len(value) != 1:
            raise RedisVLMCPError(
                "logical filter objects must contain exactly one operator",
                code=MCPErrorCode.INVALID_FILTER,
                retryable=False,
            )

        logical_key = logical_keys[0]
        if logical_key == "not":
            child = value["not"]
            if not isinstance(child, dict):
                raise RedisVLMCPError(
                    "not filter must wrap a single object expression",
                    code=MCPErrorCode.INVALID_FILTER,
                    retryable=False,
                )
            return FilterExpression(f"(-({str(_parse_expression(child, schema))}))")

        children = value[logical_key]
        if not isinstance(children, list) or not children:
            raise RedisVLMCPError(
                f"{logical_key} filter must contain a non-empty array",
                code=MCPErrorCode.INVALID_FILTER,
                retryable=False,
            )

        expressions: List[FilterExpression] = []
        for child in children:
            if not isinstance(child, dict):
                raise RedisVLMCPError(
                    "logical filter children must be objects",
                    code=MCPErrorCode.INVALID_FILTER,
                    retryable=False,
                )
            expressions.append(_parse_expression(child, schema))

        combined = expressions[0]
        for child in expressions[1:]:
            combined = combined & child if logical_key == "and" else combined | child
        return combined

    field_name = value.get("field")
    op = value.get("op")
    if not isinstance(field_name, str) or not field_name.strip():
        raise RedisVLMCPError(
            "filter.field must be a non-empty string",
            code=MCPErrorCode.INVALID_FILTER,
            retryable=False,
        )
    if not isinstance(op, str) or not op.strip():
        raise RedisVLMCPError(
            "filter.op must be a non-empty string",
            code=MCPErrorCode.INVALID_FILTER,
            retryable=False,
        )

    field = schema.fields.get(field_name)
    if field is None:
        raise RedisVLMCPError(
            f"Unknown filter field: {field_name}",
            code=MCPErrorCode.INVALID_FILTER,
            retryable=False,
        )

    normalized_op = op.lower()
    if normalized_op == "exists":
        return FilterExpression(f"(-ismissing(@{field_name}))")

    if "value" not in value:
        raise RedisVLMCPError(
            "filter.value is required for this operator",
            code=MCPErrorCode.INVALID_FILTER,
            retryable=False,
        )

    operand = value["value"]
    if field.type == "tag":
        return _parse_tag_expression(field_name, normalized_op, operand)
    if field.type == "text":
        return _parse_text_expression(field_name, normalized_op, operand)
    if field.type == "numeric":
        return _parse_numeric_expression(field_name, normalized_op, operand)

    raise RedisVLMCPError(
        f"Unsupported filter field type for {field_name}: {field.type}",
        code=MCPErrorCode.INVALID_FILTER,
        retryable=False,
    )


def _parse_tag_expression(field_name: str, op: str, operand: Any) -> FilterExpression:
    field = Tag(field_name)
    if op == "eq":
        return field == _require_string(operand, field_name, op)
    if op == "ne":
        return field != _require_string(operand, field_name, op)
    if op == "in":
        return field == _require_string_list(operand, field_name, op)
    if op == "like":
        return field % _require_string(operand, field_name, op)
    raise RedisVLMCPError(
        f"Unsupported operator '{op}' for tag field '{field_name}'",
        code=MCPErrorCode.INVALID_FILTER,
        retryable=False,
    )


def _parse_text_expression(field_name: str, op: str, operand: Any) -> FilterExpression:
    field = Text(field_name)
    if op == "eq":
        return field == _require_string(operand, field_name, op)
    if op == "ne":
        return field != _require_string(operand, field_name, op)
    if op == "like":
        return field % _require_string(operand, field_name, op)
    if op == "in":
        return _combine_or(
            [field == item for item in _require_string_list(operand, field_name, op)]
        )
    raise RedisVLMCPError(
        f"Unsupported operator '{op}' for text field '{field_name}'",
        code=MCPErrorCode.INVALID_FILTER,
        retryable=False,
    )


def _parse_numeric_expression(
    field_name: str, op: str, operand: Any
) -> FilterExpression:
    field = Num(field_name)
    if op == "eq":
        return field == _require_number(operand, field_name, op)
    if op == "ne":
        return field != _require_number(operand, field_name, op)
    if op == "gt":
        return field > _require_number(operand, field_name, op)
    if op == "gte":
        return field >= _require_number(operand, field_name, op)
    if op == "lt":
        return field < _require_number(operand, field_name, op)
    if op == "lte":
        return field <= _require_number(operand, field_name, op)
    if op == "in":
        return _combine_or(
            [field == item for item in _require_number_list(operand, field_name, op)]
        )
    raise RedisVLMCPError(
        f"Unsupported operator '{op}' for numeric field '{field_name}'",
        code=MCPErrorCode.INVALID_FILTER,
        retryable=False,
    )


def _combine_or(expressions: Iterable[FilterExpression]) -> FilterExpression:
    expression_list = list(expressions)
    if not expression_list:
        raise RedisVLMCPError(
            "in operator requires a non-empty array",
            code=MCPErrorCode.INVALID_FILTER,
            retryable=False,
        )

    combined = expression_list[0]
    for expression in expression_list[1:]:
        combined = combined | expression
    return combined


def _require_string(value: Any, field_name: str, op: str) -> str:
    if not isinstance(value, str) or not value:
        raise RedisVLMCPError(
            f"filter value for field '{field_name}' and operator '{op}' must be a non-empty string",
            code=MCPErrorCode.INVALID_FILTER,
            retryable=False,
        )
    return value


def _require_string_list(value: Any, field_name: str, op: str) -> List[str]:
    if not isinstance(value, list) or not value:
        raise RedisVLMCPError(
            f"filter value for field '{field_name}' and operator '{op}' must be a non-empty array",
            code=MCPErrorCode.INVALID_FILTER,
            retryable=False,
        )
    strings = [_require_string(item, field_name, op) for item in value]
    return strings


def _require_number(value: Any, field_name: str, op: str) -> Union[int, float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RedisVLMCPError(
            f"filter value for field '{field_name}' and operator '{op}' must be numeric",
            code=MCPErrorCode.INVALID_FILTER,
            retryable=False,
        )
    return value


def _require_number_list(
    value: Any, field_name: str, op: str
) -> List[Union[int, float]]:
    if not isinstance(value, list) or not value:
        raise RedisVLMCPError(
            f"filter value for field '{field_name}' and operator '{op}' must be a non-empty array",
            code=MCPErrorCode.INVALID_FILTER,
            retryable=False,
        )
    return [_require_number(item, field_name, op) for item in value]
