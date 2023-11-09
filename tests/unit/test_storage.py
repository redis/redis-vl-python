import pytest

from redisvl.storage import BaseStorage, HashStorage, JsonStorage


@pytest.fixture(params=[JsonStorage, HashStorage])
def storage_instance(request):
    StorageClass = request.param
    instance = StorageClass(prefix="test", key_separator=":")
    return instance


def test_key_formatting(storage_instance):
    key = "1234"
    generated_key = storage_instance._key(key, "", "")
    assert generated_key == key, "The generated key does not match the expected format."
    generated_key = storage_instance._key(key, "", ":")
    assert generated_key == key, "The generated key does not match the expected format."
    generated_key = storage_instance._key(key, "test", ":")
    assert (
        generated_key == f"test:{key}"
    ), "The generated key does not match the expected format."


def test_create_key(storage_instance):
    key_field = "id"
    obj = {key_field: "1234"}
    expected_key = (
        f"{storage_instance._prefix}{storage_instance._key_separator}{obj[key_field]}"
    )
    generated_key = storage_instance._create_key(obj, key_field)
    assert (
        generated_key == expected_key
    ), "The generated key does not match the expected format."


def test_validate_success(storage_instance):
    data = {"foo": "bar"}
    try:
        storage_instance._validate(data)
    except Exception as e:
        pytest.fail(f"_validate should not raise an exception here, but raised {e}")


def test_validate_failure(storage_instance):
    data = "Some invalid data type"
    with pytest.raises(TypeError):
        storage_instance._validate(data)
    data = 12345
    with pytest.raises(TypeError):
        storage_instance._validate(data)


def test_preprocess(storage_instance):
    data = {"key": "value"}
    preprocessed_data = storage_instance._preprocess(preprocess=None, obj=data)
    assert preprocessed_data == data

    def fn(d):
        d["foo"] = "bar"
        return d

    preprocessed_data = storage_instance._preprocess(fn, data)
    assert "foo" in preprocessed_data
    assert preprocessed_data["foo"] == "bar"


@pytest.mark.asyncio
async def test_preprocess(storage_instance):
    data = {"key": "value"}
    preprocessed_data = await storage_instance._apreprocess(preprocess=None, obj=data)
    assert preprocessed_data == data

    async def fn(d):
        d["foo"] = "bar"
        return d

    preprocessed_data = await storage_instance._apreprocess(fn, data)
    assert "foo" in preprocessed_data
    assert preprocessed_data["foo"] == "bar"
