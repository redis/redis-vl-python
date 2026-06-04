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
# TDD: Wizard rename/remove interaction bug fixes
# =============================================================================


class TestWizardRenameRemoveInteractions:
    """Tests for rename/remove interaction edge cases in the wizard."""

    def test_rename_then_remove_target_cleans_rename(self, monkeypatch):
        """Rename a→b, then remove b should cancel the rename and update."""
        source_schema = {
            "index": {"name": "idx", "prefix": "t:", "storage_type": "hash"},
            "fields": [
                {"name": "a", "type": "text"},
                {"name": "c", "type": "text"},
            ],
        }

        answers = iter(
            [
                # Rename a→b
                "4",
                "a",
                "b",
                # Remove b (which is renamed-from a)
                "3",
                "b",
                # Finish
                "8",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        # The rename a→b should be cancelled
        assert len(patch.changes.rename_fields) == 0
        # b should be in remove_fields (it's the working-name after rename)
        assert "b" in patch.changes.remove_fields

    def test_chained_rename_collapsed(self, monkeypatch):
        """Rename a→b then b→c should collapse into a single a→c."""
        source_schema = {
            "index": {"name": "idx", "prefix": "t:", "storage_type": "hash"},
            "fields": [
                {"name": "a", "type": "text"},
                {"name": "d", "type": "text"},
            ],
        }

        answers = iter(
            [
                # Rename a→b
                "4",
                "a",
                "b",
                # Rename b→c (chained)
                "4",
                "b",
                "c",
                # Finish
                "8",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        assert len(patch.changes.rename_fields) == 1
        assert patch.changes.rename_fields[0].old_name == "a"
        assert patch.changes.rename_fields[0].new_name == "c"

    def test_rename_to_staged_removal_blocked(self, monkeypatch):
        """Renaming field to a name that is staged for removal should be blocked."""
        source_schema = {
            "index": {"name": "idx", "prefix": "t:", "storage_type": "hash"},
            "fields": [
                {"name": "a", "type": "text"},
                {"name": "b", "type": "text"},
            ],
        }

        answers = iter(
            [
                # Remove b
                "3",
                "b",
                # Try to rename a→b (should be blocked)
                "4",
                "a",
                "b",
                # Finish
                "8",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        # The rename should NOT have been accepted
        assert len(patch.changes.rename_fields) == 0
        # b should still be in remove_fields
        assert "b" in patch.changes.remove_fields

    def test_update_then_rename_then_remove_cleans_update(self, monkeypatch):
        """Update a, rename a→b, remove b should clean both rename and update."""
        source_schema = {
            "index": {"name": "idx", "prefix": "t:", "storage_type": "hash"},
            "fields": [
                {"name": "a", "type": "text"},
                {"name": "c", "type": "text"},
            ],
        }

        answers = iter(
            [
                # Update a: set sortable=y, then defaults
                "2",
                "a",
                "y",  # sortable
                "n",  # index_missing
                "n",  # index_empty
                "n",  # no_stem
                "",  # weight
                "",  # phonetic
                "n",  # unf
                "n",  # no_index
                # Rename a→b
                "4",
                "a",
                "b",
                # Remove b
                "3",
                "b",
                # Finish
                "8",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

        wizard = MigrationWizard()
        patch = wizard._build_patch(source_schema)

        # Rename cancelled, update for 'a' cleaned
        assert len(patch.changes.rename_fields) == 0
        assert len(patch.changes.update_fields) == 0
        assert "b" in patch.changes.remove_fields
