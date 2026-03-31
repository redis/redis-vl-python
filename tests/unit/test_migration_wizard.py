import pytest

from redisvl.migration.wizard import MigrationWizard


def _make_vector_source_schema(algorithm="hnsw", datatype="float32"):
    """Helper to create a source schema with a vector field."""
    return {
        "index": {
            "name": "test_index",
            "prefix": "test:",
            "storage_type": "hash",
        },
        "fields": [
            {"name": "title", "type": "text"},
            {
                "name": "embedding",
                "type": "vector",
                "attrs": {
                    "algorithm": algorithm,
                    "dims": 384,
                    "distance_metric": "cosine",
                    "datatype": datatype,
                    "m": 16,
                    "ef_construction": 200,
                },
            },
        ],
    }


def test_wizard_builds_patch_from_interactive_inputs(monkeypatch):
    source_schema = {
        "index": {
            "name": "docs",
            "prefix": "docs",
            "storage_type": "json",
        },
        "fields": [
            {"name": "title", "type": "text", "path": "$.title"},
            {"name": "category", "type": "tag", "path": "$.category"},
            {
                "name": "embedding",
                "type": "vector",
                "path": "$.embedding",
                "attrs": {
                    "algorithm": "flat",
                    "dims": 3,
                    "distance_metric": "cosine",
                    "datatype": "float32",
                },
            },
        ],
    }

    answers = iter(
        [
            # Add field
            "1",
            "status",  # field name
            "tag",  # field type
            "$.status",  # JSON path
            "y",  # sortable
            "n",  # index_missing
            "n",  # index_empty
            "|",  # separator (tag-specific)
            "n",  # case_sensitive (tag-specific)
            "n",  # no_index (prompted since sortable=y)
            # Update field
            "2",
            "title",  # select field
            "y",  # sortable
            "n",  # index_missing
            "n",  # index_empty
            "n",  # no_stem (text-specific)
            "",  # weight (blank to skip, text-specific)
            "",  # phonetic_matcher (blank to skip)
            "n",  # unf (prompted since sortable=y)
            "n",  # no_index (prompted since sortable=y)
            # Remove field
            "3",
            "category",
            # Finish
            "8",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    wizard = MigrationWizard()
    patch = wizard._build_patch(source_schema)  # noqa: SLF001

    assert patch.changes.add_fields == [
        {
            "name": "status",
            "type": "tag",
            "path": "$.status",
            "attrs": {
                "sortable": True,
                "index_missing": False,
                "index_empty": False,
                "separator": "|",
                "case_sensitive": False,
                "no_index": False,
            },
        }
    ]
    assert patch.changes.remove_fields == ["category"]
    assert len(patch.changes.update_fields) == 1
    assert patch.changes.update_fields[0].name == "title"
    assert patch.changes.update_fields[0].attrs["sortable"] is True
    assert patch.changes.update_fields[0].attrs["no_stem"] is False


# =============================================================================
# Vector Algorithm Tests
# =============================================================================


class TestVectorAlgorithmChanges:
    """Test wizard handling of vector algorithm changes."""

    def test_hnsw_to_flat(self, monkeypatch):
        """Test changing from HNSW to FLAT algorithm."""
        source_schema = _make_vector_source_schema(algorithm="hnsw")

        answers = iter(
            [
                "2",  # Update field
                "embedding",  # Select vector field
                "FLAT",  # Change to FLAT
                "",  # datatype (keep current)
                "",  # distance_metric (keep current)
                # No HNSW params prompted for FLAT
                "8",  # Finish
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        assert len(patch.changes.update_fields) == 1
        update = patch.changes.update_fields[0]
        assert update.name == "embedding"
        assert update.attrs["algorithm"] == "FLAT"

    def test_flat_to_hnsw_with_params(self, monkeypatch):
        """Test changing from FLAT to HNSW with custom M and EF_CONSTRUCTION."""
        source_schema = _make_vector_source_schema(algorithm="flat")

        answers = iter(
            [
                "2",  # Update field
                "embedding",  # Select vector field
                "HNSW",  # Change to HNSW
                "",  # datatype (keep current)
                "",  # distance_metric (keep current)
                "32",  # M
                "400",  # EF_CONSTRUCTION
                "",  # EF_RUNTIME (keep current)
                "",  # EPSILON (keep current)
                "8",  # Finish
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        update = patch.changes.update_fields[0]
        assert update.attrs["algorithm"] == "HNSW"
        assert update.attrs["m"] == 32
        assert update.attrs["ef_construction"] == 400

    def test_hnsw_to_svs_vamana_with_underscore(self, monkeypatch):
        """Test changing to SVS_VAMANA (underscore format) is normalized."""
        source_schema = _make_vector_source_schema(algorithm="hnsw")

        answers = iter(
            [
                "2",  # Update field
                "embedding",  # Select vector field
                "SVS_VAMANA",  # Underscore format (should be normalized)
                "float16",  # SVS only supports float16/float32
                "",  # distance_metric (keep current)
                "64",  # GRAPH_MAX_DEGREE
                "LVQ8",  # COMPRESSION
                "8",  # Finish
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        update = patch.changes.update_fields[0]
        assert update.attrs["algorithm"] == "SVS-VAMANA"  # Normalized to hyphen
        assert update.attrs["datatype"] == "float16"
        assert update.attrs["graph_max_degree"] == 64
        assert update.attrs["compression"] == "LVQ8"

    def test_hnsw_to_svs_vamana_with_hyphen(self, monkeypatch):
        """Test changing to SVS-VAMANA (hyphen format) works directly."""
        source_schema = _make_vector_source_schema(algorithm="hnsw")

        answers = iter(
            [
                "2",  # Update field
                "embedding",  # Select vector field
                "SVS-VAMANA",  # Hyphen format
                "",  # datatype (keep current)
                "",  # distance_metric (keep current)
                "",  # GRAPH_MAX_DEGREE (keep default)
                "",  # COMPRESSION (none)
                "8",  # Finish
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        update = patch.changes.update_fields[0]
        assert update.attrs["algorithm"] == "SVS-VAMANA"

    def test_svs_vamana_with_leanvec_compression(self, monkeypatch):
        """Test SVS-VAMANA with LeanVec compression type."""
        source_schema = _make_vector_source_schema(algorithm="hnsw")

        answers = iter(
            [
                "2",  # Update field
                "embedding",  # Select vector field
                "SVS-VAMANA",
                "float16",
                "",  # distance_metric
                "48",  # GRAPH_MAX_DEGREE
                "LEANVEC8X8",  # COMPRESSION
                "192",  # REDUCE (dims/2)
                "8",  # Finish
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        update = patch.changes.update_fields[0]
        assert update.attrs["algorithm"] == "SVS-VAMANA"
        assert update.attrs["compression"] == "LeanVec8x8"
        assert update.attrs["reduce"] == 192


# =============================================================================
# Vector Datatype (Quantization) Tests
# =============================================================================


class TestVectorDatatypeChanges:
    """Test wizard handling of vector datatype/quantization changes."""

    def test_float32_to_float16(self, monkeypatch):
        """Test quantization from float32 to float16."""
        source_schema = _make_vector_source_schema(datatype="float32")

        answers = iter(
            [
                "2",  # Update field
                "embedding",
                "",  # algorithm (keep current)
                "float16",  # datatype
                "",  # distance_metric
                "",  # M (keep current)
                "",  # EF_CONSTRUCTION (keep current)
                "",  # EF_RUNTIME (keep current)
                "",  # EPSILON (keep current)
                "8",  # Finish
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        update = patch.changes.update_fields[0]
        assert update.attrs["datatype"] == "float16"

    def test_float16_to_float32(self, monkeypatch):
        """Test changing from float16 back to float32."""
        source_schema = _make_vector_source_schema(datatype="float16")

        answers = iter(
            [
                "2",  # Update field
                "embedding",
                "",  # algorithm
                "float32",  # datatype
                "",  # distance_metric
                "",  # M
                "",  # EF_CONSTRUCTION
                "",  # EF_RUNTIME (keep current)
                "",  # EPSILON (keep current)
                "8",  # Finish
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        update = patch.changes.update_fields[0]
        assert update.attrs["datatype"] == "float32"

    def test_int8_accepted_for_hnsw(self, monkeypatch):
        """Test that int8 is accepted for HNSW/FLAT (but not SVS-VAMANA)."""
        source_schema = _make_vector_source_schema(datatype="float32")

        answers = iter(
            [
                "2",  # Update field
                "embedding",
                "",  # algorithm (keep HNSW)
                "int8",  # Valid for HNSW/FLAT
                "",  # distance_metric
                "",  # M
                "",  # EF_CONSTRUCTION
                "",  # EF_RUNTIME (keep current)
                "",  # EPSILON (keep current)
                "8",  # Finish
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        # int8 is now valid for HNSW/FLAT
        update = patch.changes.update_fields[0]
        assert update.attrs["datatype"] == "int8"


# =============================================================================
# Distance Metric Tests
# =============================================================================


class TestDistanceMetricChanges:
    """Test wizard handling of distance metric changes."""

    def test_cosine_to_l2(self, monkeypatch):
        """Test changing distance metric from cosine to L2."""
        source_schema = _make_vector_source_schema()

        answers = iter(
            [
                "2",  # Update field
                "embedding",
                "",  # algorithm
                "",  # datatype
                "l2",  # distance_metric
                "",  # M
                "",  # EF_CONSTRUCTION
                "",  # EF_RUNTIME (keep current)
                "",  # EPSILON (keep current)
                "8",  # Finish
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        update = patch.changes.update_fields[0]
        assert update.attrs["distance_metric"] == "l2"

    def test_cosine_to_ip(self, monkeypatch):
        """Test changing distance metric from cosine to inner product."""
        source_schema = _make_vector_source_schema()

        answers = iter(
            [
                "2",  # Update field
                "embedding",
                "",  # algorithm
                "",  # datatype
                "ip",  # distance_metric (inner product)
                "",  # M
                "",  # EF_CONSTRUCTION
                "",  # EF_RUNTIME (keep current)
                "",  # EPSILON (keep current)
                "8",  # Finish
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        update = patch.changes.update_fields[0]
        assert update.attrs["distance_metric"] == "ip"


# =============================================================================
# Combined Changes Tests
# =============================================================================


class TestCombinedVectorChanges:
    """Test wizard handling of multiple vector attribute changes."""

    def test_algorithm_datatype_and_metric_change(self, monkeypatch):
        """Test changing algorithm, datatype, and distance metric together."""
        source_schema = _make_vector_source_schema(algorithm="flat", datatype="float32")

        answers = iter(
            [
                "2",  # Update field
                "embedding",
                "HNSW",  # algorithm
                "float16",  # datatype
                "l2",  # distance_metric
                "24",  # M
                "300",  # EF_CONSTRUCTION
                "",  # EF_RUNTIME (keep current)
                "",  # EPSILON (keep current)
                "8",  # Finish
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        update = patch.changes.update_fields[0]
        assert update.attrs["algorithm"] == "HNSW"
        assert update.attrs["datatype"] == "float16"
        assert update.attrs["distance_metric"] == "l2"
        assert update.attrs["m"] == 24
        assert update.attrs["ef_construction"] == 300

    def test_svs_vamana_full_config(self, monkeypatch):
        """Test SVS-VAMANA with all parameters configured."""
        source_schema = _make_vector_source_schema(algorithm="hnsw", datatype="float32")

        answers = iter(
            [
                "2",  # Update field
                "embedding",
                "SVS-VAMANA",  # algorithm
                "float16",  # datatype (required for SVS)
                "ip",  # distance_metric
                "50",  # GRAPH_MAX_DEGREE
                "LVQ4X8",  # COMPRESSION
                "8",  # Finish
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        update = patch.changes.update_fields[0]
        assert update.attrs["algorithm"] == "SVS-VAMANA"
        assert update.attrs["datatype"] == "float16"
        assert update.attrs["distance_metric"] == "ip"
        assert update.attrs["graph_max_degree"] == 50
        assert update.attrs["compression"] == "LVQ4x8"

    def test_no_changes_when_all_blank(self, monkeypatch):
        """Test that blank inputs result in no changes."""
        source_schema = _make_vector_source_schema()

        answers = iter(
            [
                "2",  # Update field
                "embedding",
                "",  # algorithm (keep current)
                "",  # datatype (keep current)
                "",  # distance_metric (keep current)
                "",  # M (keep current)
                "",  # EF_CONSTRUCTION (keep current)
                "",  # EF_RUNTIME (keep current)
                "",  # EPSILON (keep current)
                "8",  # Finish
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        # No changes collected means no update_fields
        assert len(patch.changes.update_fields) == 0


# =============================================================================
# Adversarial / Edge Case Tests
# =============================================================================


class TestWizardAdversarialInputs:
    """Test wizard robustness against malformed, malicious, or edge case inputs."""

    # -------------------------------------------------------------------------
    # Invalid Algorithm Inputs
    # -------------------------------------------------------------------------

    def test_typo_in_algorithm_ignored(self, monkeypatch):
        """Test that typos in algorithm name are ignored."""
        source_schema = _make_vector_source_schema(algorithm="hnsw")

        answers = iter(
            [
                "2",
                "embedding",
                "HNSW_TYPO",  # Invalid algorithm
                "",  # datatype
                "",  # distance_metric
                "",  # M
                "",  # EF_CONSTRUCTION
                "",  # EF_RUNTIME
                "",  # EPSILON
                "8",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        # Invalid algorithm should be ignored, no changes
        assert len(patch.changes.update_fields) == 0

    def test_partial_algorithm_name_ignored(self, monkeypatch):
        """Test that partial algorithm names are ignored."""
        source_schema = _make_vector_source_schema(algorithm="hnsw")

        answers = iter(
            [
                "2",
                "embedding",
                "HNS",  # Partial name
                "",  # datatype
                "",  # distance_metric
                "",  # M
                "",  # EF_CONSTRUCTION
                "",  # EF_RUNTIME
                "",  # EPSILON
                "8",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        assert len(patch.changes.update_fields) == 0

    def test_algorithm_with_special_chars_ignored(self, monkeypatch):
        """Test that algorithm with special characters is ignored."""
        source_schema = _make_vector_source_schema(algorithm="hnsw")

        answers = iter(
            [
                "2",
                "embedding",
                "HNSW; DROP TABLE users;--",  # SQL injection attempt
                "",  # datatype
                "",  # distance_metric
                "",  # M
                "",  # EF_CONSTRUCTION
                "",  # EF_RUNTIME
                "",  # EPSILON
                "8",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        assert len(patch.changes.update_fields) == 0

    def test_algorithm_lowercase_works(self, monkeypatch):
        """Test that lowercase algorithm names work (case insensitive)."""
        source_schema = _make_vector_source_schema(algorithm="hnsw")

        answers = iter(
            [
                "2",
                "embedding",
                "flat",  # lowercase
                "",
                "",
                "8",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        update = patch.changes.update_fields[0]
        assert update.attrs["algorithm"] == "FLAT"

    def test_algorithm_mixed_case_works(self, monkeypatch):
        """Test that mixed case algorithm names work."""
        source_schema = _make_vector_source_schema(algorithm="hnsw")

        answers = iter(
            [
                "2",
                "embedding",
                "SvS_VaMaNa",  # Mixed case with underscore
                "",
                "",
                "",
                "",
                "8",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        update = patch.changes.update_fields[0]
        assert update.attrs["algorithm"] == "SVS-VAMANA"

    # -------------------------------------------------------------------------
    # Invalid Numeric Inputs
    # -------------------------------------------------------------------------

    def test_negative_m_ignored(self, monkeypatch):
        """Test that negative M value is ignored."""
        source_schema = _make_vector_source_schema(algorithm="hnsw")

        answers = iter(
            [
                "2",
                "embedding",
                "HNSW",
                "",  # datatype
                "",  # distance_metric
                "-16",  # Negative M
                "",  # EF_CONSTRUCTION
                "",  # EF_RUNTIME
                "",  # EPSILON
                "8",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        update = patch.changes.update_fields[0]
        assert "m" not in update.attrs  # Negative should be ignored

    def test_float_m_ignored(self, monkeypatch):
        """Test that float M value is ignored."""
        source_schema = _make_vector_source_schema(algorithm="hnsw")

        answers = iter(
            [
                "2",
                "embedding",
                "HNSW",
                "",  # datatype
                "",  # distance_metric
                "16.5",  # Float M
                "",  # EF_CONSTRUCTION
                "",  # EF_RUNTIME
                "",  # EPSILON
                "8",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        update = patch.changes.update_fields[0]
        assert "m" not in update.attrs

    def test_string_m_ignored(self, monkeypatch):
        """Test that string M value is ignored."""
        source_schema = _make_vector_source_schema(algorithm="hnsw")

        answers = iter(
            [
                "2",
                "embedding",
                "HNSW",
                "",  # datatype
                "",  # distance_metric
                "sixteen",  # String M
                "",  # EF_CONSTRUCTION
                "",  # EF_RUNTIME
                "",  # EPSILON
                "8",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        update = patch.changes.update_fields[0]
        assert "m" not in update.attrs

    def test_zero_m_accepted(self, monkeypatch):
        """Test that zero M is accepted (validation happens at schema level)."""
        source_schema = _make_vector_source_schema(algorithm="hnsw")

        answers = iter(
            [
                "2",
                "embedding",
                "HNSW",
                "",  # datatype
                "",  # distance_metric
                "0",  # Zero M
                "",  # EF_CONSTRUCTION
                "",  # EF_RUNTIME
                "",  # EPSILON
                "8",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        # Zero is a valid digit, wizard accepts it (validation at apply time)
        # isdigit() returns False for "0" in some edge cases, let's check
        update = patch.changes.update_fields[0]
        # "0".isdigit() returns True, so it should be accepted
        assert update.attrs.get("m") == 0

    def test_very_large_ef_construction_accepted(self, monkeypatch):
        """Test that very large EF_CONSTRUCTION is accepted by wizard."""
        source_schema = _make_vector_source_schema(algorithm="hnsw")

        answers = iter(
            [
                "2",
                "embedding",
                "HNSW",
                "",
                "",
                "",
                "999999999",  # Very large EF_CONSTRUCTION
                "",  # EF_RUNTIME (keep current)
                "",  # EPSILON (keep current)
                "8",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        update = patch.changes.update_fields[0]
        assert update.attrs["ef_construction"] == 999999999

    # -------------------------------------------------------------------------
    # Invalid Datatype Inputs
    # -------------------------------------------------------------------------

    def test_bfloat16_accepted_for_hnsw(self, monkeypatch):
        """Test that bfloat16 is accepted for HNSW/FLAT."""
        source_schema = _make_vector_source_schema(datatype="float32")

        answers = iter(
            [
                "2",
                "embedding",
                "",  # algorithm
                "bfloat16",  # Valid for HNSW/FLAT
                "",  # distance_metric
                "",  # M
                "",  # EF_CONSTRUCTION
                "",  # EF_RUNTIME
                "",  # EPSILON
                "8",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        update = patch.changes.update_fields[0]
        assert update.attrs["datatype"] == "bfloat16"

    def test_uint8_accepted_for_hnsw(self, monkeypatch):
        """Test that uint8 is accepted for HNSW/FLAT."""
        source_schema = _make_vector_source_schema(datatype="float32")

        answers = iter(
            [
                "2",
                "embedding",
                "",  # algorithm
                "uint8",  # Valid for HNSW/FLAT
                "",  # distance_metric
                "",  # M
                "",  # EF_CONSTRUCTION
                "",  # EF_RUNTIME
                "",  # EPSILON
                "8",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        update = patch.changes.update_fields[0]
        assert update.attrs["datatype"] == "uint8"

    def test_int8_rejected_for_svs_vamana(self, monkeypatch):
        """Test that int8 is rejected for SVS-VAMANA (only float16/float32 allowed)."""
        source_schema = _make_vector_source_schema(datatype="float32", algorithm="hnsw")

        answers = iter(
            [
                "2",
                "embedding",
                "SVS-VAMANA",  # Switch to SVS-VAMANA
                "int8",  # Invalid for SVS-VAMANA
                "",
                "",
                "",  # graph_max_degree
                "",  # compression
                "8",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        # Should have algorithm change but NOT datatype
        update = patch.changes.update_fields[0]
        assert update.attrs["algorithm"] == "SVS-VAMANA"
        assert "datatype" not in update.attrs  # int8 rejected

    # -------------------------------------------------------------------------
    # Invalid Distance Metric Inputs
    # -------------------------------------------------------------------------

    def test_invalid_distance_metric_ignored(self, monkeypatch):
        """Test that invalid distance metric is ignored."""
        source_schema = _make_vector_source_schema()

        answers = iter(
            [
                "2",
                "embedding",
                "",  # algorithm
                "",  # datatype
                "euclidean",  # Invalid (should be 'l2')
                "",  # M
                "",  # EF_CONSTRUCTION
                "",  # EF_RUNTIME
                "",  # EPSILON
                "8",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        assert len(patch.changes.update_fields) == 0

    def test_distance_metric_uppercase_works(self, monkeypatch):
        """Test that uppercase distance metric works."""
        source_schema = _make_vector_source_schema()

        answers = iter(
            [
                "2",
                "embedding",
                "",  # algorithm
                "",  # datatype
                "L2",  # Uppercase
                "",  # M
                "",  # EF_CONSTRUCTION
                "",  # EF_RUNTIME
                "",  # EPSILON
                "8",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        update = patch.changes.update_fields[0]
        assert update.attrs["distance_metric"] == "l2"

    # -------------------------------------------------------------------------
    # Invalid Compression Inputs
    # -------------------------------------------------------------------------

    def test_invalid_compression_ignored(self, monkeypatch):
        """Test that invalid compression type is ignored."""
        source_schema = _make_vector_source_schema(algorithm="hnsw")

        answers = iter(
            [
                "2",
                "embedding",
                "SVS-VAMANA",
                "",
                "",
                "",
                "INVALID_COMPRESSION",  # Invalid
                "8",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        update = patch.changes.update_fields[0]
        assert "compression" not in update.attrs

    def test_compression_lowercase_works(self, monkeypatch):
        """Test that lowercase compression works."""
        source_schema = _make_vector_source_schema(algorithm="hnsw")

        answers = iter(
            [
                "2",
                "embedding",
                "SVS-VAMANA",
                "",
                "",
                "",
                "lvq8",  # lowercase
                "8",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        update = patch.changes.update_fields[0]
        assert update.attrs["compression"] == "LVQ8"

    # -------------------------------------------------------------------------
    # Whitespace and Special Character Inputs
    # -------------------------------------------------------------------------

    def test_whitespace_only_treated_as_blank(self, monkeypatch):
        """Test that whitespace-only input is treated as blank."""
        source_schema = _make_vector_source_schema()

        answers = iter(
            [
                "2",
                "embedding",
                "   ",  # Whitespace only (algorithm)
                "   ",  # datatype
                "   ",  # distance_metric
                "   ",  # M
                "   ",  # EF_CONSTRUCTION
                "   ",  # EF_RUNTIME
                "   ",  # EPSILON
                "8",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        assert len(patch.changes.update_fields) == 0

    def test_algorithm_with_leading_trailing_whitespace(self, monkeypatch):
        """Test that algorithm with whitespace is trimmed and works."""
        source_schema = _make_vector_source_schema(algorithm="hnsw")

        answers = iter(
            [
                "2",
                "embedding",
                "  FLAT  ",  # Whitespace around (FLAT has no extra params)
                "",  # datatype
                "",  # distance_metric
                "8",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        update = patch.changes.update_fields[0]
        assert update.attrs["algorithm"] == "FLAT"

    def test_unicode_input_ignored(self, monkeypatch):
        """Test that unicode/emoji inputs are ignored."""
        source_schema = _make_vector_source_schema()

        answers = iter(
            [
                "2",
                "embedding",
                "HNSW\U0001f680",  # Unicode emoji
                "",  # datatype
                "",  # distance_metric
                "",  # M
                "",  # EF_CONSTRUCTION
                "",  # EF_RUNTIME
                "",  # EPSILON
                "8",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        assert len(patch.changes.update_fields) == 0

    def test_very_long_input_ignored(self, monkeypatch):
        """Test that very long inputs are ignored."""
        source_schema = _make_vector_source_schema()

        answers = iter(
            [
                "2",
                "embedding",
                "A" * 10000,  # Very long string
                "",  # datatype
                "",  # distance_metric
                "",  # M
                "",  # EF_CONSTRUCTION
                "",  # EF_RUNTIME
                "",  # EPSILON
                "8",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        assert len(patch.changes.update_fields) == 0

    # -------------------------------------------------------------------------
    # Field Selection Edge Cases
    # -------------------------------------------------------------------------

    def test_nonexistent_field_selection(self, monkeypatch):
        """Test selecting a nonexistent field."""
        source_schema = _make_vector_source_schema()

        answers = iter(
            [
                "2",
                "nonexistent_field",  # Doesn't exist
                "8",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        # Should print "Invalid field selection" and continue
        assert len(patch.changes.update_fields) == 0

    def test_field_selection_by_number_out_of_range(self, monkeypatch):
        """Test selecting a field by out-of-range number."""
        source_schema = _make_vector_source_schema()

        answers = iter(
            [
                "2",
                "99",  # Out of range
                "8",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        assert len(patch.changes.update_fields) == 0

    def test_field_selection_negative_number(self, monkeypatch):
        """Test selecting a field with negative number."""
        source_schema = _make_vector_source_schema()

        answers = iter(
            [
                "2",
                "-1",  # Negative
                "8",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        assert len(patch.changes.update_fields) == 0

    # -------------------------------------------------------------------------
    # Menu Action Edge Cases
    # -------------------------------------------------------------------------

    def test_invalid_menu_action(self, monkeypatch):
        """Test invalid menu action selection."""
        source_schema = _make_vector_source_schema()

        answers = iter(
            [
                "99",  # Invalid action
                "abc",  # Invalid action
                "",  # Empty
                "8",  # Finally finish
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        # Should handle invalid actions gracefully and eventually finish
        assert patch is not None

    # -------------------------------------------------------------------------
    # SVS-VAMANA Specific Edge Cases
    # -------------------------------------------------------------------------

    def test_svs_vamana_negative_graph_max_degree_ignored(self, monkeypatch):
        """Test that negative GRAPH_MAX_DEGREE is ignored."""
        source_schema = _make_vector_source_schema(algorithm="hnsw")

        answers = iter(
            [
                "2",
                "embedding",
                "SVS-VAMANA",
                "",
                "",
                "-40",  # Negative
                "",
                "8",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        update = patch.changes.update_fields[0]
        assert "graph_max_degree" not in update.attrs

    def test_svs_vamana_string_graph_max_degree_ignored(self, monkeypatch):
        """Test that string GRAPH_MAX_DEGREE is ignored."""
        source_schema = _make_vector_source_schema(algorithm="hnsw")

        answers = iter(
            [
                "2",
                "embedding",
                "SVS-VAMANA",
                "",
                "",
                "forty",  # String
                "",
                "8",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        update = patch.changes.update_fields[0]
        assert "graph_max_degree" not in update.attrs
