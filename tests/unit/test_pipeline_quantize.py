"""Tests for pipelined read/write quantization.

TDD: tests written BEFORE refactoring _quantize_vectors.

Tests the new quantize flow:
  1. Pipeline-read original vectors (dump phase)
  2. Convert dtype in memory
  3. Pipeline-write converted vectors (quantize phase)
"""

import struct
from unittest.mock import MagicMock


def _make_float32_vector(dims: int = 4, seed: float = 0.0) -> bytes:
    """Create a fake float32 vector."""
    return struct.pack(f"<{dims}f", *[seed + i for i in range(dims)])


class TestPipelineReadBatch:
    """Test that vector reads are pipelined, not individual HGET calls."""

    def test_pipeline_read_batches_hgets(self):
        """A batch of N keys with F fields should produce N*F pipelined HGET
        calls and exactly 1 pipe.execute() — not N*F individual client.hget()."""
        mock_client = MagicMock()
        mock_pipe = MagicMock()
        mock_client.pipeline.return_value = mock_pipe

        dims = 4
        keys = [f"doc:{i}" for i in range(5)]
        vec = _make_float32_vector(dims)
        # Pipeline execute returns one result per hget call
        mock_pipe.execute.return_value = [vec] * 5

        datatype_changes = {
            "embedding": {"source": "float32", "target": "float16", "dims": dims}
        }

        from redisvl.migration.quantize import pipeline_read_vectors

        result = pipeline_read_vectors(mock_client, keys, datatype_changes)

        # Should call pipeline(), not client.hget()
        mock_client.pipeline.assert_called_once_with(transaction=False)
        assert mock_pipe.hget.call_count == 5
        # Exactly 1 execute call (not 5)
        mock_pipe.execute.assert_called_once()
        # Should NOT call client.hget directly
        mock_client.hget.assert_not_called()
        # Returns dict of {key: {field: bytes}}
        assert len(result) == 5
        assert result["doc:0"]["embedding"] == vec

    def test_pipeline_read_multiple_fields(self):
        """Keys with multiple vector fields produce N*F pipelined HGETs."""
        mock_client = MagicMock()
        mock_pipe = MagicMock()
        mock_client.pipeline.return_value = mock_pipe

        dims = 4
        keys = ["doc:0", "doc:1"]
        vec = _make_float32_vector(dims)
        # 2 keys × 2 fields = 4 results
        mock_pipe.execute.return_value = [vec, vec, vec, vec]

        datatype_changes = {
            "embedding": {"source": "float32", "target": "float16", "dims": dims},
            "title_vec": {"source": "float32", "target": "float16", "dims": dims},
        }

        from redisvl.migration.quantize import pipeline_read_vectors

        result = pipeline_read_vectors(mock_client, keys, datatype_changes)

        assert mock_pipe.hget.call_count == 4
        mock_pipe.execute.assert_called_once()
        assert "embedding" in result["doc:0"]
        assert "title_vec" in result["doc:0"]

    def test_pipeline_read_handles_missing_keys(self):
        """Missing keys (hget returns None) should be excluded from results."""
        mock_client = MagicMock()
        mock_pipe = MagicMock()
        mock_client.pipeline.return_value = mock_pipe

        keys = ["doc:0", "doc:1"]
        vec = _make_float32_vector()
        # doc:0 has data, doc:1 is missing
        mock_pipe.execute.return_value = [vec, None]

        datatype_changes = {
            "embedding": {"source": "float32", "target": "float16", "dims": 4}
        }

        from redisvl.migration.quantize import pipeline_read_vectors

        result = pipeline_read_vectors(mock_client, keys, datatype_changes)

        assert "embedding" in result["doc:0"]
        # doc:1 should have empty field dict or be excluded
        assert result.get("doc:1", {}).get("embedding") is None


class TestPipelineWriteBatch:
    """Test that converted vectors are written via pipeline."""

    def test_pipeline_write_batches_hsets(self):
        """Writing N keys should produce N pipelined HSET calls and 1 execute."""
        mock_client = MagicMock()
        mock_pipe = MagicMock()
        mock_client.pipeline.return_value = mock_pipe

        converted = {
            "doc:0": {"embedding": b"\x00\x01\x02\x03"},
            "doc:1": {"embedding": b"\x04\x05\x06\x07"},
        }

        from redisvl.migration.quantize import pipeline_write_vectors

        pipeline_write_vectors(mock_client, converted)

        mock_client.pipeline.assert_called_once_with(transaction=False)
        assert mock_pipe.hset.call_count == 2
        mock_pipe.execute.assert_called_once()

    def test_pipeline_write_skips_empty(self):
        """If no keys to write, don't create a pipeline at all."""
        mock_client = MagicMock()

        from redisvl.migration.quantize import pipeline_write_vectors

        pipeline_write_vectors(mock_client, {})

        mock_client.pipeline.assert_not_called()


class TestConvertVectors:
    """Test dtype conversion logic."""

    def test_convert_float32_to_float16(self):
        import numpy as np

        from redisvl.migration.quantize import convert_vectors

        dims = 4
        vec = _make_float32_vector(dims, seed=1.0)
        originals = {"doc:0": {"embedding": vec}}
        changes = {
            "embedding": {"source": "float32", "target": "float16", "dims": dims}
        }

        converted = convert_vectors(originals, changes)

        # Result should be float16 bytes (2 bytes per dim)
        assert len(converted["doc:0"]["embedding"]) == dims * 2
        # Verify values round-trip through float16
        arr = np.frombuffer(converted["doc:0"]["embedding"], dtype=np.float16)
        np.testing.assert_allclose(arr, [1.0, 2.0, 3.0, 4.0], rtol=1e-3)

    def test_convert_float32_to_int8_scales_correctly(self):
        """Float-to-int8 should scale values to [-128, 127], not truncate."""
        import numpy as np

        from redisvl.migration.quantize import convert_vectors

        # Typical embedding values in [-1, 1] — would all become 0 without scaling.
        dims = 4
        source = np.array([-1.0, -0.5, 0.0, 1.0], dtype=np.float32)
        originals = {"doc:0": {"embedding": source.tobytes()}}
        changes = {"embedding": {"source": "float32", "target": "int8", "dims": dims}}

        converted = convert_vectors(originals, changes)
        result = np.frombuffer(converted["doc:0"]["embedding"], dtype=np.int8)

        # 1 byte per dim
        assert len(converted["doc:0"]["embedding"]) == dims * 1
        # Min should map to -128, max to 127
        assert result[0] == -128  # min value
        assert result[3] == 127  # max value
        # Values should span the full int8 range, NOT be all zeros
        assert result.min() == -128
        assert result.max() == 127
        # Middle values should be proportionally scaled
        # -0.5 → (-0.5 - (-1)) / 2 * 255 + (-128) = 63.75 - 128 = -64.25 → -64
        assert result[1] == -64
        # 0.0 → (0 - (-1)) / 2 * 255 + (-128) = 127.5 - 128 = -0.5 → 0
        assert result[2] == 0

    def test_convert_float16_to_int8_scales_correctly(self):
        """Float16-to-int8 should also scale properly (the benchmark bug path)."""
        import numpy as np

        from redisvl.migration.quantize import convert_vectors

        # Simulate what the benchmark did: random [0, 1] float16 vectors
        source = np.array([0.1, 0.3, 0.7, 0.9], dtype=np.float16)
        originals = {"doc:0": {"embedding": source.tobytes()}}
        changes = {"embedding": {"source": "float16", "target": "int8", "dims": 4}}

        converted = convert_vectors(originals, changes)
        result = np.frombuffer(converted["doc:0"]["embedding"], dtype=np.int8)

        # Should NOT be all zeros (the original bug)
        assert not np.all(
            result == 0
        ), "INT8 conversion produced all zeros — scaling is not being applied"
        # Should use the full range
        assert result.min() == -128
        assert result.max() == 127

    def test_convert_float32_to_uint8_scales_correctly(self):
        """Float-to-uint8 should scale values to [0, 255]."""
        import numpy as np

        from redisvl.migration.quantize import convert_vectors

        source = np.array([-1.0, 0.0, 0.5, 1.0], dtype=np.float32)
        originals = {"doc:0": {"embedding": source.tobytes()}}
        changes = {"embedding": {"source": "float32", "target": "uint8", "dims": 4}}

        converted = convert_vectors(originals, changes)
        result = np.frombuffer(converted["doc:0"]["embedding"], dtype=np.uint8)

        assert len(converted["doc:0"]["embedding"]) == 4 * 1
        assert result[0] == 0  # min maps to 0
        assert result[3] == 255  # max maps to 255
        assert result.min() == 0
        assert result.max() == 255

    def test_convert_constant_vector_to_int8(self):
        """A constant vector (all same value) should not divide by zero."""
        import numpy as np

        from redisvl.migration.quantize import convert_vectors

        source = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        originals = {"doc:0": {"embedding": source.tobytes()}}
        changes = {"embedding": {"source": "float32", "target": "int8", "dims": 4}}

        converted = convert_vectors(originals, changes)
        result = np.frombuffer(converted["doc:0"]["embedding"], dtype=np.int8)

        # Should not raise and should produce a valid int8 vector
        assert len(result) == 4
        # All values should be identical (mapped to midpoint)
        assert np.all(result == result[0])

    def test_convert_skips_unknown_fields(self):
        """Fields not in datatype_changes should be skipped."""
        from redisvl.migration.quantize import convert_vectors

        originals = {"doc:0": {"other_field": b"\x00\x01"}}
        changes = {"embedding": {"source": "float32", "target": "float16", "dims": 4}}

        converted = convert_vectors(originals, changes)
        assert converted["doc:0"] == {}
