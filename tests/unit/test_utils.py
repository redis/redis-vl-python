import importlib
import io
import logging
import re
import sys
from functools import wraps

import numpy as np
import pytest

from redisvl.redis.utils import (
    array_to_buffer,
    buffer_to_array,
    convert_bytes,
    make_dict,
)
from redisvl.utils.utils import (
    assert_no_warnings,
    deprecated_argument,
    deprecated_function,
    norm_cosine_distance,
)


def test_norm_cosine_distance():
    input = 2
    expected = 0
    assert norm_cosine_distance(input) == expected


def test_even_number_of_elements():
    """Test with an even number of elements"""
    values = ["key1", "value1", "key2", "value2"]
    expected = {"key1": "value1", "key2": "value2"}
    assert make_dict(values) == expected


def test_odd_number_of_elements():
    """Test with an odd number of elements - expecting the last element to be ignored"""
    values = ["key1", "value1", "key2"]
    expected = {"key1": "value1"}  # 'key2' has no pair, so it's ignored
    assert make_dict(values) == expected


def test_different_data_types():
    """Test with different data types as keys and values"""
    values = [1, "one", 2.0, "two"]
    expected = {1: "one", 2.0: "two"}
    assert make_dict(values) == expected


def test_empty_list():
    """Test with an empty list"""
    values = []
    expected = {}
    assert make_dict(values) == expected


def test_with_complex_objects():
    """Test with complex objects like lists and dicts as values"""
    key = "a list"
    value = [1, 2, 3]
    values = [key, value]
    expected = {key: value}
    assert make_dict(values) == expected


def test_simple_byte_buffer_to_floats():
    """Test conversion of a simple byte buffer into floats"""
    buffer = np.array([1.0, 2.0, 3.0], dtype=np.float32).tobytes()
    expected = [1.0, 2.0, 3.0]
    assert buffer_to_array(buffer, dtype="float32") == expected


def test_converting_different_data_types():
    """Test conversion with different data types"""
    # Float64 test
    buffer = np.array([1.0, 2.0, 3.0], dtype=np.float64).tobytes()
    expected = [1.0, 2.0, 3.0]
    assert buffer_to_array(buffer, dtype="float64") == expected


def test_empty_byte_buffer():
    """Test conversion of an empty byte buffer"""
    buffer = b""
    expected = []
    assert buffer_to_array(buffer, dtype="float32") == expected


def test_plain_bytes_to_string():
    """Test conversion of plain bytes to string"""
    data = b"hello world"
    expected = "hello world"
    assert convert_bytes(data) == expected


def test_bytes_in_dict():
    """Test conversion of bytes in a dictionary, including nested dictionaries"""
    data = {"key": b"value", "nested": {"nkey": b"nvalue"}}
    expected = {"key": "value", "nested": {"nkey": "nvalue"}}
    assert convert_bytes(data) == expected


def test_bytes_in_list():
    """Test conversion of bytes in a list, including nested lists"""
    data = [b"item1", b"item2", ["nested", b"nested item"]]
    expected = ["item1", "item2", ["nested", "nested item"]]
    assert convert_bytes(data) == expected


def test_bytes_in_tuple():
    """Test conversion of bytes in a tuple, including nested tuples"""
    data = (b"item1", b"item2", ("nested", b"nested item"))
    expected = ("item1", "item2", ("nested", "nested item"))
    assert convert_bytes(data) == expected


def test_non_bytes_data():
    """Test handling of non-bytes data types"""
    data = "already a string"
    expected = "already a string"
    assert convert_bytes(data) == expected


def test_bytes_with_invalid_utf8():
    """Test handling bytes that cannot be decoded with UTF-8"""
    data = b"\xff\xff"  # Invalid in UTF-8
    expected = data
    assert convert_bytes(data) == expected


def test_simple_list_to_bytes_default_dtype():
    """Test conversion of a simple list of floats to bytes using the default dtype"""
    array = [1.0, 2.0, 3.0]
    expected = np.array(array, dtype=np.float32).tobytes()
    assert array_to_buffer(array, "float32") == expected


def test_list_to_bytes_non_default_dtype():
    """Test conversion with a non-default dtype"""
    array = [1.0, 2.0, 3.0]
    dtype = np.float64
    expected = np.array(array, dtype=dtype).tobytes()
    assert array_to_buffer(array, dtype="float64") == expected


def test_empty_list_to_bytes():
    """Test conversion of an empty list"""
    array = []
    expected = np.array(array, dtype=np.float32).tobytes()
    assert array_to_buffer(array, dtype="float32") == expected


@pytest.mark.parametrize("dtype", ["float64", "float32", "float16", "bfloat16"])
def test_conversion_with_various_dtypes(dtype):
    """Test conversion of a list of floats to bytes with various dtypes"""
    array = [1.0, -2.0, 3.5]
    expected = np.array(array, dtype=dtype).tobytes()
    assert array_to_buffer(array, dtype=dtype) == expected


@pytest.mark.parametrize("dtype", ["int8", "uint8"])
def test_conversion_with_integer_dtypes(dtype):
    """Test conversion of a list of floats to bytes with various dtypes"""
    array = [0.0, 1.0, 2.2, 3.5]
    expected = np.array(array, dtype=dtype).tobytes()
    assert array_to_buffer(array, dtype=dtype) == expected


def test_conversion_with_invalid_floats():
    """Test conversion with invalid float values (numpy should handle them)"""
    array = [float("inf"), float("-inf"), float("nan")]
    result = array_to_buffer(array, "float16")
    assert len(result) > 0  # Simple check to ensure it returns anything


def decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("boop")
        return func(*args, **kwargs)

    return wrapper


class TestDeprecatedArgument:
    def test_deprecation_warning_text_with_replacement(self):
        @deprecated_argument("old_arg", "new_arg")
        def test_func(old_arg=None, new_arg=None, neutral_arg=None):
            pass

        # Test that passing the deprecated argument as a keyword triggers the warning.
        with pytest.warns(DeprecationWarning) as record:
            test_func(old_arg="float32")

        assert len(record) == 1
        assert str(record[0].message) == (
            "Argument old_arg is deprecated and will be removed in the next major release. Use new_arg instead."
        )

        # Test that passing the deprecated argument as a positional argument also triggers the warning.
        with pytest.warns(DeprecationWarning) as record:
            test_func("float32", neutral_arg="test_vector")

        assert len(record) == 1
        assert str(record[0].message) == (
            "Argument old_arg is deprecated and will be removed in the next major release. Use new_arg instead."
        )

        with assert_no_warnings():
            test_func(new_arg="float32")
            test_func(new_arg="float32", neutral_arg="test_vector")

    def test_deprecation_warning_text_without_replacement(self):
        @deprecated_argument("old_arg")
        def test_func(old_arg=None, neutral_arg=None):
            pass

        # As a kwarg
        with pytest.warns(DeprecationWarning) as record:
            test_func(old_arg="float32")

        assert len(record) == 1
        assert str(record[0].message) == (
            "Argument old_arg is deprecated and will be removed"
            " in the next major release."
        )

        # As a positional arg
        with pytest.warns(DeprecationWarning):
            test_func("float32", neutral_arg="test_vector")

        assert len(record) == 1
        assert str(record[0].message) == (
            "Argument old_arg is deprecated and will be removed"
            " in the next major release."
        )

        with assert_no_warnings():
            test_func(neutral_arg="test_vector")

    def test_function_positional_argument_required(self):
        """
        NOTE: In this situation, it's not possible to avoid a deprecation
        warning because the argument is currently required.
        """

        @deprecated_argument("old_arg")
        def test_func(old_arg, neutral_arg):
            pass

        with pytest.warns(DeprecationWarning):
            test_func("float32", "bob")

    def test_function_positional_argument_optional(self):
        @deprecated_argument("old_arg")
        def test_func(neutral_arg, old_arg=None):
            pass

        with pytest.warns(DeprecationWarning):
            test_func("bob", "float32")

        with assert_no_warnings():
            test_func("bob")

    def test_function_keyword_argument(self):
        @deprecated_argument("old_arg", "new_arg")
        def test_func(old_arg=None, new_arg=None):
            pass

        with pytest.warns(DeprecationWarning):
            test_func(old_arg="float32")

        with assert_no_warnings():
            test_func(new_arg="float32")

    def test_function_keyword_argument_multiple_decorators(self):
        @deprecated_argument("old_arg", "new_arg")
        @decorator
        def test_func(old_arg=None, new_arg=None):
            pass

        with pytest.warns(DeprecationWarning):
            test_func(old_arg="float32")

        with assert_no_warnings():
            test_func(new_arg="float32")

    def test_method_positional_argument_optional(self):
        class TestClass:
            @deprecated_argument("old_arg", "new_arg")
            def test_method(self, new_arg=None, old_arg=None):
                pass

        with pytest.warns(DeprecationWarning):
            TestClass().test_method("float32", "bob")

        with assert_no_warnings():
            TestClass().test_method("float32")

    def test_method_positional_argument_required(self):
        """
        NOTE: In this situation, it's not possible to avoid a deprecation
        warning because the argument is currently required.
        """

        class TestClass:
            @deprecated_argument("old_arg", "new_arg")
            def test_method(self, old_arg, new_arg):
                pass

        with pytest.warns(DeprecationWarning):
            TestClass().test_method("float32", new_arg="bob")

    def test_method_keyword_argument(self):
        class TestClass:
            @deprecated_argument("old_arg", "new_arg")
            def test_method(self, old_arg=None, new_arg=None):
                pass

        with pytest.warns(DeprecationWarning):
            TestClass().test_method(old_arg="float32")

        with assert_no_warnings():
            TestClass().test_method(new_arg="float32")

    def test_classmethod_positional_argument_required(self):
        """
        NOTE: In this situation, it's impossible to avoid a deprecation
        warning because the argument is currently required.
        """

        class TestClass:
            @deprecated_argument("old_arg", "new_arg")
            @classmethod
            def test_method(cls, old_arg, new_arg):
                pass

        with pytest.warns(DeprecationWarning):
            TestClass.test_method("float32", new_arg="bob")

    def test_classmethod_positional_argument_optional(self):
        class TestClass:
            @deprecated_argument("old_arg", "new_arg")
            @classmethod
            def test_method(cls, new_arg=None, old_arg=None):
                pass

        with pytest.warns(DeprecationWarning):
            TestClass.test_method("float32", "bob")

        with assert_no_warnings():
            TestClass.test_method("float32")

    def test_classmethod_keyword_argument(self):
        class TestClass:
            @deprecated_argument("old_arg", "new_arg")
            @classmethod
            def test_method(cls, old_arg=None, new_arg=None):
                pass

        with pytest.warns(DeprecationWarning):
            TestClass.test_method(old_arg="float32")

        with assert_no_warnings():
            TestClass.test_method(new_arg="float32")

    def test_classmethod_keyword_argument_multiple_decorators(self):
        """
        NOTE: The @deprecated_argument decorator should come between @classmethod
        and the method definition.
        """

        class TestClass:
            @classmethod
            @deprecated_argument("old_arg", "new_arg")
            @decorator
            def test_method(cls, old_arg=None, new_arg=None):
                pass

        with pytest.warns(DeprecationWarning):
            TestClass.test_method(old_arg="float32")

        with assert_no_warnings():
            TestClass.test_method(new_arg="float32")

    def test_class_init_argument(self):
        class TestClass:
            @deprecated_argument("old_arg", "new_arg")
            def __init__(self, old_arg=None, new_arg=None):
                pass

        with pytest.warns(DeprecationWarning):
            TestClass(old_arg="float32")

    def test_class_init_keyword_argument(self):
        class TestClass:
            @deprecated_argument("old_arg", "new_arg")
            def __init__(self, old_arg=None, new_arg=None):
                pass

        with pytest.warns(DeprecationWarning):
            TestClass(old_arg="float32")

        with assert_no_warnings():
            TestClass(new_arg="float32")

    def test_class_init_keyword_argument_kwargs(self):
        class TestClass:
            @deprecated_argument("old_arg", "new_arg")
            def __init__(self, new_arg=None, **kwargs):
                pass

        with pytest.warns(DeprecationWarning):
            TestClass(old_arg="float32")

        with assert_no_warnings():
            TestClass(new_arg="float32")

    async def test_async_function_argument(self):
        @deprecated_argument("old_arg", "new_arg")
        async def test_func(old_arg=None, new_arg=None):
            return 1

        with pytest.warns(DeprecationWarning):
            result = await test_func(old_arg="float32")
        assert result == 1

    async def test_ignores_local_variable(self):
        @deprecated_argument("old_arg", "new_arg")
        async def test_func(old_arg=None, new_arg=None):
            # The presence of this variable should not trigger a warning
            old_arg = "float32"
            return 1

        with assert_no_warnings():
            await test_func()


class TestDeprecatedFunction:
    def test_deprecated_function_warning(self):
        @deprecated_function("new_func", "Use new_func2")
        def old_func():
            pass

        with pytest.warns(DeprecationWarning):
            old_func()

    def test_deprecated_function_warning_with_name(self):
        @deprecated_function("new_func", "Use new_func2")
        def old_func():
            pass

        with pytest.warns(DeprecationWarning):
            old_func()

    def test_logging_configuration_not_overridden(self):
        """Test that RedisVL imports don't override existing logging configurations."""
        import os
        import subprocess

        # Get the path to the helper script relative to this test file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        helper_script = os.path.join(current_dir, "logger_interference_checker.py")

        # Run the helper script in a separate process
        result = subprocess.run(
            [sys.executable, helper_script], capture_output=True, text=True
        )

        # Extract the log lines
        output_lines = result.stdout.strip().split("\n")
        pre_import_line = ""
        post_import_line = ""

        for line in output_lines:
            if "PRE_IMPORT_FORMAT" in line:
                pre_import_line = line
            elif "POST_IMPORT_FORMAT" in line:
                post_import_line = line

        # Check if we found both lines
        assert pre_import_line, "No pre-import log message found"
        assert post_import_line, "No post-import log message found"

        # Print for debugging
        print(f"Pre-import format: {pre_import_line}")
        print(f"Post-import format: {post_import_line}")

        # Check format preservation
        # Look for bracketed logger name format
        has_bracketed_logger_pre = "[app]" in pre_import_line
        has_bracketed_logger_post = "[app]" in post_import_line
        assert has_bracketed_logger_pre == has_bracketed_logger_post, (
            f"Logger format changed from {'bracketed' if has_bracketed_logger_pre else 'unbracketed'} "
            f"to {'bracketed' if has_bracketed_logger_post else 'unbracketed'}"
        )

        # Look for file/line information
        has_file_line_pre = bool(re.search(r"\[\w+\.py:\d+\]", pre_import_line))
        has_file_line_post = bool(re.search(r"\[\w+\.py:\d+\]", post_import_line))
        assert (
            has_file_line_pre == has_file_line_post
        ), f"File/line format changed: was present before: {has_file_line_pre}, present after: {has_file_line_post}"

        # Check date format (RedisVL strips the date portion)
        has_date_pre = bool(re.match(r"\d{4}-\d{2}-\d{2}", pre_import_line))
        has_date_post = bool(re.match(r"\d{4}-\d{2}-\d{2}", post_import_line))
        assert (
            has_date_pre == has_date_post
        ), f"Date format changed: was present before: {has_date_pre}, present after: {has_date_post}"
