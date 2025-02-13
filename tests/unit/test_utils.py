import warnings
import numpy as np
import pytest

from redisvl.redis.utils import (
    array_to_buffer,
    buffer_to_array,
    convert_bytes,
    make_dict,
)
from redisvl.utils.utils import deprecated_argument


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


def test_conversion_with_invalid_floats():
    """Test conversion with invalid float values (numpy should handle them)"""
    array = [float("inf"), float("-inf"), float("nan")]
    result = array_to_buffer(array, "float16")
    assert len(result) > 0  # Simple check to ensure it returns anything


class TestDeprecatedArgument:
    def test_deprecation_warning_text_with_replacement(self):
        @deprecated_argument("dtype", "vectorizer")
        def test_func(dtype=None, vectorizer=None):
            pass

        with pytest.warns(DeprecationWarning) as record:
            test_func(dtype="float32")

        assert len(record) == 1
        assert str(record[0].message) == (
            "Argument dtype is deprecated and will be removed"
            " in the next major release. Use vectorizer instead."
        )

    def test_deprecation_warning_text_without_replacement(self):
        @deprecated_argument("dtype")
        def test_func(dtype=None):
            pass

        with pytest.warns(DeprecationWarning) as record:
            test_func(dtype="float32")

        assert len(record) == 1
        assert str(record[0].message) == (
            "Argument dtype is deprecated and will be removed"
            " in the next major release."
        )

    def test_function_argument(self):
        @deprecated_argument("dtype", "vectorizer")
        def test_func(dtype=None, vectorizer=None):
            pass

        with pytest.warns(DeprecationWarning):
            test_func(dtype="float32")

    def test_function_keyword_argument(self):
        @deprecated_argument("dtype", "vectorizer")
        def test_func(dtype=None, vectorizer=None):
            pass

        with pytest.warns(DeprecationWarning):
            test_func(vectorizer="float32")

    def test_class_method_argument(self):
        class TestClass:
            @deprecated_argument("dtype", "vectorizer")
            def test_method(self, dtype=None, vectorizer=None):
                pass

        with pytest.warns(DeprecationWarning):
            TestClass().test_method(dtype="float32")

    def test_class_method_keyword_argument(self):
        class TestClass:
            @deprecated_argument("dtype", "vectorizer")
            def test_method(self, dtype=None, vectorizer=None):
                pass

        with pytest.warns(DeprecationWarning):
            TestClass().test_method(vectorizer="float32")

    def test_class_init_argument(self):
        class TestClass:
            @deprecated_argument("dtype", "vectorizer")
            def __init__(self, dtype=None, vectorizer=None):
                pass

        with pytest.warns(DeprecationWarning):
            TestClass(dtype="float32")

    def test_class_init_keyword_argument(self):
        class TestClass:
            @deprecated_argument("dtype", "vectorizer")
            def __init__(self, dtype=None, vectorizer=None):
                pass

        with pytest.warns(DeprecationWarning):
            TestClass(dtype="float32")

    async def test_async_function_argument(self):
        @deprecated_argument("dtype", "vectorizer")
        async def test_func(dtype=None, vectorizer=None):
            return 1

        with pytest.warns(DeprecationWarning):
            result = await test_func(dtype="float32")
        assert result == 1
        
    async def test_ignores_local_variable(self):
        @deprecated_argument("dtype")
        async def test_func(vectorizer=None):
            # The presence of this variable should not trigger a warning
            dtype = "float32"
            return 1

        # This will raise an error if any warning is emitted
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            await test_func()
