import os
import subprocess
import sys
import tempfile
import yaml
import pytest


@pytest.fixture
def cli_schema_file(redis_test_name):
    """Creates a temporary schema YAML file for integration testing CLI commands."""
    index_name = redis_test_name("cli_int_index")
    prefix = redis_test_name("cli_int_doc")
    schema_data = {
        "index": {
            "name": index_name,
            "prefix": prefix,
            "storage_type": "hash",
        },
        "fields": [
            {"name": "title", "type": "text"},
            {"name": "tag", "type": "tag"},
        ],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(schema_data, f)
        temp_path = f.name

    yield temp_path, index_name

    if os.path.exists(temp_path):
        os.remove(temp_path)


def run_cli(*args, redis_url=None):
    """Helper to execute rvl CLI commands via subprocess runner."""
    cmd = [sys.executable, "-m", "redisvl.cli.runner"] + list(args)
    env = os.environ.copy()
    if redis_url:
        env["REDIS_URL"] = redis_url
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    return result


def test_cli_version():
    res = run_cli("version")
    assert res.returncode == 0
    assert "redisvl" in res.stdout.lower() or "version" in res.stdout.lower() or "." in res.stdout


def test_cli_index_lifecycle(redis_url, cli_schema_file):
    schema_path, index_name = cli_schema_file

    # 1. Create index
    create_res = run_cli("index", "create", "-s", schema_path, "--url", redis_url)
    assert create_res.returncode == 0, f"create failed: {create_res.stderr}"

    # 2. List all indexes
    list_res = run_cli("index", "listall", "--url", redis_url)
    assert list_res.returncode == 0, f"listall failed: {list_res.stderr}"
    assert index_name in list_res.stdout

    # 3. Get index info
    info_res = run_cli("index", "info", "-i", index_name, "--url", redis_url)
    assert info_res.returncode == 0, f"info failed: {info_res.stderr}"
    assert index_name in info_res.stdout

    # 4. Get index stats
    stats_res = run_cli("stats", "-i", index_name, "--url", redis_url)
    assert stats_res.returncode == 0, f"stats failed: {stats_res.stderr}"
    assert index_name in stats_res.stdout or "num_docs" in stats_res.stdout or "STAT" in stats_res.stdout

    # 5. Delete index
    delete_res = run_cli("index", "delete", "-i", index_name, "--url", redis_url)
    assert delete_res.returncode == 0, f"delete failed: {delete_res.stderr}"

    # 6. Re-create index to test destroy
    create_res2 = run_cli("index", "create", "-s", schema_path, "--url", redis_url)
    assert create_res2.returncode == 0, f"second create failed: {create_res2.stderr}"

    # 7. Destroy index (with --drop / clear keys)
    destroy_res = run_cli("index", "destroy", "-i", index_name, "--url", redis_url)
    assert destroy_res.returncode == 0, f"destroy failed: {destroy_res.stderr}"


def test_cli_error_paths(redis_url):
    # Non-existent schema file
    schema_res = run_cli("index", "create", "-s", "non_existent_file_xyz.yaml", "--url", redis_url)
    assert schema_res.returncode != 0

    # Info for non-existent index
    info_res = run_cli("index", "info", "-i", "non_existent_index_xyz_9999", "--url", redis_url)
    assert info_res.returncode != 0

    # Stats for non-existent index
    stats_res = run_cli("stats", "-i", "non_existent_index_xyz_9999", "--url", redis_url)
    assert stats_res.returncode != 0

    # Invalid CLI subcommand
    invalid_res = run_cli("invalid_command_name")
    assert invalid_res.returncode == 2
