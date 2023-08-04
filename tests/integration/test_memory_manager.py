import pytest

from time import sleep
from redisvl.memory.manager import MemoryManager
from redisvl.vectorize.text import HFTextVectorizer
from redisvl.memory.interaction import Interaction

@pytest.fixture
def vectorizer():
    return HFTextVectorizer("sentence-transformers/all-mpnet-base-v2")

@pytest.fixture
def memory():
    return MemoryManager()

@pytest.fixture
def interaction():
    return Interaction(
        session_id="test",
        prompt="foo",
        response="response"
    )

def test_memory_params(memory):
    # Check that we can store and retrieve a response
    with pytest.raises(ValueError):
        memory.set_threshold(-1)
    assert memory._session_key("test") == f"{memory._index._name}:session:test"

def test_add_memory(memory, interaction):
    for _ in range(5):
        assert memory.add(interaction)

def test_memory_len(memory):
    # check length
    assert memory.len("test") == 5

# def test_seek_memory(memory, interaction):
#     with pytest.raises(ValueError):
#         memory.seek("test", -3)
#     assert memory.seek("test", 1) == interaction
#     assert memory.seek("test", 2) == [interaction, interaction]
#     assert memory.seek_range("test", 0, 4) == [interaction for _ in range(4)]
#     assert memory.seek_range("test", 5, 6) == []
#     # test fake session id
#     assert memory.seek("fake", 1) == []

# def test_seek_relevant(memory, interaction):
#     results = memory.seek_relevant("test", context="foo", n=3)
#     assert len(results) == 3
#     assert results[0] == interaction
#     results = memory.seek_relevant("test", context="george washington", n=3)
#     assert len(results) == 0

# def test_clear_memory(memory):
#     memory.clear("test")
#     assert memory.len("test") == 0

