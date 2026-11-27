import sys
from unittest.mock import MagicMock, patch

import pytest

from redisvl.utils.rerank import HFCrossEncoderReranker


@pytest.fixture
def reranker():
    # Replace sentence_transformers so no model is downloaded.
    with patch.dict(sys.modules, {"sentence_transformers": MagicMock()}):
        reranker = HFCrossEncoderReranker()
    # Score each (query, text) pair by the length of the text.
    reranker._client.predict.side_effect = lambda pairs: [len(t) for _, t in pairs]
    return reranker


def test_dict_docs_without_content_are_ranked(reranker):
    docs = [{"text": "a"}, {"text": "a longer one"}]

    reranked, scores = reranker.rank("query", docs)

    assert reranked == [docs[1], docs[0]]
    assert scores == [float(len(str(docs[1]))), float(len(str(docs[0])))]


def test_mixed_dict_docs_keep_every_doc(reranker):
    docs = [{"content": "a"}, {"text": "a longer one"}]

    reranked, scores = reranker.rank("query", docs)

    assert reranked == [docs[1], docs[0]]
    # A doc with "content" is still scored on that field only.
    assert scores == [float(len(str(docs[1]))), 1.0]
