from enum import Enum
from typing import Any, List, Optional, Union

from redisvl.utils.utils import TokenEscaper


class FilterOperator(Enum):
    EQ = 1
    NE = 2
    LT = 3
    GT = 4
    LE = 5
    GE = 6
    OR = 7
    AND = 8
    LIKE = 9
    IN = 10


class FilterField:
    escaper = TokenEscaper()
    OPERATORS = []

    def __init__(self, field: str):
        self._field = field
        self._value = None
        self._operator = None

    def _set_value(self, val: Any, val_type: type, operator: FilterOperator):
        # check that the operator is supported by this class
        if operator not in self.OPERATORS:
            raise ValueError(
                f"Operator {operator} not supported by {self.__class__.__name__}. "
                + f"Supported operators are {self.OPERATORS.values()}"
            )

        if not isinstance(val, val_type):
            raise TypeError(
                f"Right side argument passed to operator {self.OPERATORS[operator]} with left side "
                f"argument {self.__class__.__name__} must be of type {val_type}"
            )
        self._value = val
        self._operator = operator


class Tag(FilterField):
    OPERATORS = {
        FilterOperator.EQ: "==",
        FilterOperator.NE: "!=",
        FilterOperator.IN: "==",
    }

    OPERATOR_MAP = {
        FilterOperator.EQ: "@%s:{%s}",
        FilterOperator.NE: "(-@%s:{%s})",
        FilterOperator.IN: "@%s:{%s}",
    }

    def __init__(self, field: str):
        super().__init__(field)

    def _set_tag_value(self, other: Union[List[str], str], operator: FilterOperator):
        if isinstance(other, list):
            if not all(isinstance(tag, str) for tag in other):
                raise ValueError("All tags must be strings")
        else:
            other = [other]
        self._set_value(other, list, operator)

    def __eq__(self, other) -> "FilterExpression":
        self._set_tag_value(other, FilterOperator.EQ)
        return FilterExpression(str(self))

    def __ne__(self, other) -> "FilterExpression":
        self._set_tag_value(other, FilterOperator.NE)
        return FilterExpression(str(self))

    @property
    def _formatted_tag_value(self) -> str:
        return "|".join([self.escaper.escape(tag) for tag in self._value])

    def __str__(self) -> str:
        return self.OPERATOR_MAP[self._operator] % (
            self._field,
            self._formatted_tag_value,
        )


class Geo(FilterField):
    OPERATORS = {
        FilterOperator.EQ: "==",
        FilterOperator.NE: "!=",
    }
    OPERATOR_MAP = {
        FilterOperator.EQ: "@%s:[%f %f %i %s]",
        FilterOperator.NE: "(-@%s:[%f %f %i %s])",
    }

    def __eq__(self, other) -> "FilterExpression":
        # TODO raise typeError
        self._set_value(other, GeoSpec, FilterOperator.EQ)
        return FilterExpression(str(self))

    def __ne__(self, other) -> "FilterExpression":
        self._set_value(other, GeoSpec, FilterOperator.NE)
        return FilterExpression(str(self))

    def __str__(self) -> "FilterExpression":
        return self.OPERATOR_MAP[self._operator] % (
            self._field,
            *self._value.get_args(),
        )


class GeoSpec:
    GEO_UNITS = ["m", "km", "mi", "ft"]

    # class for the operand for FilterExpressions with Geo
    def __init__(self, longitude: float, latitude: float, unit: str = "km"):
        if unit.lower() not in self.GEO_UNITS:
            raise ValueError(f"Unit must be one of {self.GEO_UNITS}")
        self._longitude = longitude
        self._latitude = latitude
        self._unit = unit.lower()


class GeoRadius(GeoSpec):
    # class for the operand for FilterExpressions with Geo
    def __init__(
        self, longitude: float, latitude: float, radius: int = 1, unit: str = "km"
    ):
        super().__init__(longitude, latitude, unit)
        self._radius = radius

    def get_args(self) -> List[Union[float, int, str]]:
        return [self._longitude, self._latitude, self._radius, self._unit]


class Num(FilterField):
    OPERATORS = {
        FilterOperator.EQ: "==",
        FilterOperator.NE: "!=",
        FilterOperator.LT: "<",
        FilterOperator.GT: ">",
        FilterOperator.LE: "<=",
        FilterOperator.GE: ">=",
    }
    OPERATOR_MAP = {
        FilterOperator.EQ: "@%s:[%i %i]",
        FilterOperator.NE: "(-@%s:[%i %i])",
        FilterOperator.GT: "@%s:[(%i +inf]",
        FilterOperator.LT: "@%s:[-inf (%i]",
        FilterOperator.GE: "@%s:[%i +inf]",
        FilterOperator.LE: "@%s:[-inf %i]",
    }

    def __str__(self) -> str:
        if self._operator == FilterOperator.EQ or self._operator == FilterOperator.NE:
            return self.OPERATOR_MAP[self._operator] % (
                self._field,
                self._value,
                self._value,
            )
        else:
            return self.OPERATOR_MAP[self._operator] % (self._field, self._value)

    def __eq__(self, other: str) -> "FilterExpression":
        self._set_value(other, int, FilterOperator.EQ)
        return FilterExpression(str(self))

    def __ne__(self, other: str) -> "FilterExpression":
        self._set_value(other, int, FilterOperator.NE)
        return FilterExpression(str(self))

    def __gt__(self, other: str) -> "FilterExpression":
        self._set_value(other, int, FilterOperator.GT)
        return FilterExpression(str(self))

    def __lt__(self, other: str) -> "FilterExpression":
        self._set_value(other, int, FilterOperator.LT)
        return FilterExpression(str(self))

    def __ge__(self, other: str) -> "FilterExpression":
        self._set_value(other, int, FilterOperator.GE)
        return FilterExpression(str(self))

    def __le__(self, other: str) -> "FilterExpression":
        self._set_value(other, int, FilterOperator.LE)
        return FilterExpression(str(self))


class Text(FilterField):
    OPERATORS = {
        FilterOperator.EQ: "==",
        FilterOperator.NE: "!=",
        FilterOperator.LIKE: "%",
    }
    OPERATOR_MAP = {
        FilterOperator.EQ: '@%s:"%s"',
        FilterOperator.NE: '(-@%s:"%s")',
        FilterOperator.LIKE: "@%s:%s",
    }

    def __eq__(self, other: str) -> "FilterExpression":
        self._set_value(other, str, FilterOperator.EQ)
        return FilterExpression(str(self))

    def __ne__(self, other: str) -> "FilterExpression":
        self._set_value(other, str, FilterOperator.NE)
        return FilterExpression(str(self))

    def __mod__(self, other: str) -> "FilterExpression":
        self._set_value(other, str, FilterOperator.LIKE)
        return FilterExpression(str(self))

    def __str__(self) -> str:
        try:
            return self.OPERATOR_MAP[self._operator] % (self._field, self._value)
        except KeyError:
            raise Exception("Invalid operator")


class FilterExpression:
    def __init__(
        self,
        _filter: str = None,
        operator: FilterOperator = None,
        left: Optional["FilterExpression"] = None,
        right: Optional["FilterExpression"] = None,
    ):
        self._filter = _filter
        self._operator = operator
        self._left = left
        self._right = right

    def __and__(self, other):
        return FilterExpression(operator=FilterOperator.AND, left=self, right=other)

    def __or__(self, other):
        return FilterExpression(operator=FilterOperator.OR, left=self, right=other)

    def __str__(self):
        if self._operator:
            operator_str = " | " if self._operator == FilterOperator.OR else " "
            return f"({str(self._left)}{operator_str}{str(self._right)})"
        return self._filter
