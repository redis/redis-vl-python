from collections import namedtuple
from time import sleep

import pytest

from redisvl.extensions.session_manager import (
    SemanticSessionManager,
    StandardSessionManager,
)
from redisvl.index.index import SearchIndex
from redisvl.utils.vectorise import HFTextVectorizer


@pytest.fixture
def vectorizer():
    return HFTextVectorizer("sentence_transformers/all-mpnet-base-v2")


def test_pass():
    assert True
