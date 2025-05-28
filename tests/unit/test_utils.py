import re
import sys
from functools import wraps

import numpy as np
import pytest

from redisvl.redis.utils import (
    _keys_share_hash_tag,
    array_to_buffer,
    buffer_to_array,
    convert_bytes,
    make_dict,
)
from redisvl.utils.utils import (
    assert_no_warnings,
    denorm_cosine_distance,
    deprecated_argument,
    deprecated_function,
    lazy_import,
    norm_cosine_distance,
)


def test_norm_cosine_distance():
    input = 2
    expected = 0
    assert norm_cosine_distance(input) == expected


def test_denorm_cosine_distance():
    input = 0
    expected = 2
    assert denorm_cosine_distance(input) == expected


def test_norm_denorm_cosine():
    input = 0.6
    assert input == round(denorm_cosine_distance(norm_cosine_distance(input)), 6)


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

    # Special handling for bfloat16 which requires explicit import from ml_dtypes
    if dtype == "bfloat16":
        from ml_dtypes import bfloat16 as bf16

        expected = np.array(array, dtype=bf16).tobytes()
    else:
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


class TestLazyImport:
    def test_import_standard_library(self):
        """Test lazy importing of a standard library module"""
        # Remove the module from sys.modules if it's already imported
        if "json" in sys.modules:
            del sys.modules["json"]

        # Lazy import the module
        json = lazy_import("json")

        # Verify the module is not imported yet
        assert "json" not in sys.modules

        # Use the module, which should trigger the import
        result = json.dumps({"key": "value"})

        # Verify the module is now imported
        assert "json" in sys.modules
        assert result == '{"key": "value"}'

    def test_cached_module_import(self):
        """Test that _import_module returns the cached module if it exists"""
        # Remove the module from sys.modules if it's already imported
        if "json" in sys.modules:
            del sys.modules["json"]

        # Lazy import the module
        json = lazy_import("json")

        # Access an attribute to trigger the import
        json.dumps

        # The module should now be cached
        # We need to access the private _import_module method directly
        # to test the cached path
        module = json._import_module()

        # Verify that the cached module was returned
        assert module is json._module

    def test_import_already_imported_module(self):
        """Test lazy importing of an already imported module"""
        # Make sure the module is imported
        import math

        assert "math" in sys.modules

        # Lazy import the module
        math_lazy = lazy_import("math")

        # Since the module is already imported, it should be returned directly
        assert math_lazy is sys.modules["math"]

        # Use the module
        assert math_lazy.sqrt(4) == 2.0

    def test_import_submodule(self):
        """Test lazy importing of a submodule"""
        # Remove the module from sys.modules if it's already imported
        if "os.path" in sys.modules:
            del sys.modules["os.path"]
        if "os" in sys.modules:
            del sys.modules["os"]

        # Lazy import the submodule
        path = lazy_import("os.path")

        # Verify the module is not imported yet
        assert "os" not in sys.modules

        # Use the submodule, which should trigger the import
        result = path.join("dir", "file.txt")

        # Verify the module is now imported
        assert "os" in sys.modules
        assert (
            result == "dir/file.txt" or result == "dir\\file.txt"
        )  # Handle Windows paths

    def test_import_function(self):
        """Test lazy importing of a function"""
        # Remove the module from sys.modules if it's already imported
        if "math" in sys.modules:
            del sys.modules["math"]

        # Lazy import the function
        sqrt = lazy_import("math.sqrt")

        # Verify the module is not imported yet
        assert "math" not in sys.modules

        # Use the function, which should trigger the import
        result = sqrt(4)

        # Verify the module is now imported
        assert "math" in sys.modules
        assert result == 2.0

    def test_import_nonexistent_module(self):
        """Test lazy importing of a nonexistent module"""
        # Lazy import a nonexistent module
        nonexistent = lazy_import("nonexistent_module_xyz")

        # Accessing an attribute should raise ImportError
        with pytest.raises(ImportError) as excinfo:
            nonexistent.some_attribute

        assert "Failed to lazily import nonexistent_module_xyz" in str(excinfo.value)

    def test_call_nonexistent_module(self):
        """Test calling a nonexistent module"""
        # Lazy import a nonexistent module
        nonexistent = lazy_import("nonexistent_module_xyz")

        # Calling the nonexistent module should raise ImportError
        with pytest.raises(ImportError) as excinfo:
            nonexistent()

        assert "Failed to lazily import nonexistent_module_xyz" in str(excinfo.value)

    def test_import_nonexistent_attribute(self):
        """Test lazy importing of a nonexistent attribute"""
        # Lazy import a nonexistent attribute
        nonexistent_attr = lazy_import("math.nonexistent_attribute")

        # Accessing the attribute should raise ImportError
        with pytest.raises(ImportError) as excinfo:
            nonexistent_attr()

        assert "module 'math' has no attribute 'nonexistent_attribute'" in str(
            excinfo.value
        )

    def test_getattr_on_nonexistent_attribute_path(self):
        """Test accessing an attribute on a nonexistent attribute path"""
        # Lazy import a nonexistent attribute path
        nonexistent_attr = lazy_import("math.nonexistent_attribute")

        # Accessing an attribute on the nonexistent attribute should raise AttributeError
        with pytest.raises(AttributeError) as excinfo:
            nonexistent_attr.some_attribute

        assert "module 'math' has no attribute 'nonexistent_attribute'" in str(
            excinfo.value
        )

    def test_import_noncallable(self):
        """Test calling a non-callable lazy imported object"""
        # Lazy import a non-callable attribute
        pi = lazy_import("math.pi")

        # Calling it should raise TypeError
        with pytest.raises(TypeError) as excinfo:
            pi()

        assert "math.pi is not callable" in str(excinfo.value)

    def test_attribute_error(self):
        """Test accessing a nonexistent attribute on a lazy imported module"""
        # Lazy import a module
        math = lazy_import("math")

        # Accessing a nonexistent attribute should raise AttributeError
        with pytest.raises(AttributeError) as excinfo:
            math.nonexistent_attribute

        assert "module 'math' has no attribute 'nonexistent_attribute'" in str(
            excinfo.value
        )

    def test_attribute_error_after_import(self):
        """Test accessing a nonexistent attribute on a module after it's been imported"""
        # Create a simple module with a known attribute
        import types

        test_module = types.ModuleType("test_module")
        test_module.existing_attr = "exists"

        # Add it to sys.modules so lazy_import can find it
        sys.modules["test_module"] = test_module

        try:
            # Lazy import the module
            lazy_mod = lazy_import("test_module")

            # Access the existing attribute to trigger the import
            assert lazy_mod.existing_attr == "exists"

            # Now access a nonexistent attribute
            with pytest.raises(AttributeError) as excinfo:
                lazy_mod.nonexistent_attribute

            assert (
                "module 'test_module' has no attribute 'nonexistent_attribute'"
                in str(excinfo.value)
            )
        finally:
            # Clean up
            if "test_module" in sys.modules:
                del sys.modules["test_module"]

    def test_attribute_error_with_direct_module_access(self):
        """Test accessing a nonexistent attribute by directly setting the _module attribute"""
        # Get the lazy_import function
        from redisvl.utils.utils import lazy_import

        # Create a lazy import for a module that doesn't exist yet
        lazy_mod = lazy_import("test_direct_module")

        # Create a simple object with no __getattr__ method
        class SimpleObject:
            def __init__(self, value):
                self.value = value

        obj = SimpleObject(42)

        # Directly set the _module attribute to our simple object
        # This bypasses the normal import mechanism
        lazy_mod._module = obj

        # Now access a nonexistent attribute
        # This should go through our LazyModule.__getattr__ and hit line 332
        with pytest.raises(AttributeError) as excinfo:
            lazy_mod.nonexistent_attribute

        assert (
            "module 'test_direct_module' has no attribute 'nonexistent_attribute'"
            in str(excinfo.value)
        )


# Hash tag validation tests for Redis Cluster compatibility
def test_keys_share_hash_tag_same_tags():
    """Test that keys with the same hash tag are considered compatible."""
    keys = ["prefix:{tag1}:key1", "prefix:{tag1}:key2", "prefix:{tag1}:key3"]
    assert _keys_share_hash_tag(keys) is True


def test_keys_share_hash_tag_different_tags():
    """Test that keys with different hash tags are considered incompatible."""
    keys = ["prefix:{tag1}:key1", "prefix:{tag2}:key2"]
    assert _keys_share_hash_tag(keys) is False


def test_keys_share_hash_tag_no_tags():
    """Test that keys without hash tags are considered compatible."""
    keys = ["prefix:key1", "prefix:key2", "prefix:key3"]
    assert _keys_share_hash_tag(keys) is True


def test_keys_share_hash_tag_mixed_tags_and_no_tags():
    """Test that mixing keys with and without hash tags is incompatible."""
    keys = ["prefix:{tag1}:key1", "prefix:key2"]
    assert _keys_share_hash_tag(keys) is False


def test_keys_share_hash_tag_empty_list():
    """Test that an empty list of keys is considered compatible."""
    assert _keys_share_hash_tag([]) is True


def test_keys_share_hash_tag_single_key():
    """Test that a single key is always compatible."""
    assert _keys_share_hash_tag(["prefix:{tag1}:key1"]) is True
    assert _keys_share_hash_tag(["prefix:key1"]) is True


def test_keys_share_hash_tag_complex_tags():
    """Test with complex hash tag patterns."""
    keys_same = [
        "user:{user123}:profile",
        "user:{user123}:settings",
        "user:{user123}:history",
    ]
    assert _keys_share_hash_tag(keys_same) is True

    keys_different = ["user:{user123}:profile", "user:{user456}:profile"]
    assert _keys_share_hash_tag(keys_different) is False


def test_keys_share_hash_tag_malformed_tags():
    """Test with malformed hash tags (missing closing brace)."""
    keys = [
        "prefix:{tag1:key1",  # Missing closing brace
        "prefix:{tag1:key2",  # Missing closing brace
    ]
    # These should be treated as no hash tags (empty string)
    assert _keys_share_hash_tag(keys) is True


def test_keys_share_hash_tag_nested_braces():
    """Test with nested braces in hash tags."""
    keys_same = ["prefix:{{nested}tag}:key1", "prefix:{{nested}tag}:key2"]
    assert _keys_share_hash_tag(keys_same) is True

    keys_different = ["prefix:{{nested}tag}:key1", "prefix:{{other}tag}:key2"]
    assert _keys_share_hash_tag(keys_different) is False


def test_keys_share_hash_tag_multiple_braces():
    """Test with multiple sets of braces in a key."""
    keys = ["prefix:{tag1}:middle:{tag2}:key1", "prefix:{tag1}:middle:{tag2}:key2"]
    # Should use the first hash tag found
    assert _keys_share_hash_tag(keys) is True
