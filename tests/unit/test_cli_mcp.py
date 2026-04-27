import builtins
import importlib
import sys
import types
from collections import namedtuple

import pytest

from redisvl.cli.main import RedisVlCLI, _usage


def _import_cli_mcp():
    sys.modules.pop("redisvl.cli.mcp", None)
    return importlib.import_module("redisvl.cli.mcp")


def _make_version_info(major, minor, micro=0):
    version_info = namedtuple(
        "VersionInfo", ["major", "minor", "micro", "releaselevel", "serial"]
    )
    return version_info(major, minor, micro, "final", 0)


def _install_fake_redisvl_mcp(monkeypatch, settings_factory, server_factory):
    fake_module = types.ModuleType("redisvl.mcp")
    fake_module.MCPSettings = settings_factory
    fake_module.RedisVLMCPServer = server_factory
    monkeypatch.setitem(sys.modules, "redisvl.mcp", fake_module)
    return fake_module


def test_usage_includes_mcp():
    assert "mcp" in _usage()


def test_cli_dispatches_mcp_command_lazily(monkeypatch):
    calls = []
    fake_module = types.ModuleType("redisvl.cli.mcp")

    class FakeMCP(object):
        def __init__(self):
            calls.append(list(sys.argv))

    fake_module.MCP = FakeMCP
    monkeypatch.setitem(sys.modules, "redisvl.cli.mcp", fake_module)
    monkeypatch.setattr(sys, "argv", ["rvl", "mcp", "--config", "/tmp/mcp.yaml"])

    cli = RedisVlCLI.__new__(RedisVlCLI)

    with pytest.raises(SystemExit) as exc_info:
        RedisVlCLI.mcp(cli)

    assert exc_info.value.code == 0
    assert calls == [["rvl", "mcp", "--config", "/tmp/mcp.yaml"]]


def test_mcp_command_reports_missing_optional_dependencies(monkeypatch, capsys):
    monkeypatch.delitem(sys.modules, "redisvl.mcp", raising=False)
    monkeypatch.delitem(sys.modules, "redisvl.cli.mcp", raising=False)
    monkeypatch.setattr(sys, "version_info", _make_version_info(3, 11, 0))

    original_import = builtins.__import__

    def missing_mcp_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "redisvl.mcp" or name.startswith("redisvl.mcp."):
            raise ModuleNotFoundError(name)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", missing_mcp_import)

    module = _import_cli_mcp()
    monkeypatch.setattr(sys, "argv", ["rvl", "mcp", "--config", "/tmp/mcp.yaml"])

    with pytest.raises(SystemExit) as exc_info:
        module.MCP()

    out = capsys.readouterr()

    assert exc_info.value.code == 1
    assert "redisvl[mcp]" in out.err or "redisvl[mcp]" in out.out


def test_mcp_help_includes_description_and_example(monkeypatch, capsys):
    monkeypatch.delitem(sys.modules, "redisvl.cli.mcp", raising=False)
    monkeypatch.setattr(sys, "argv", ["rvl", "mcp", "--help"])

    module = _import_cli_mcp()

    with pytest.raises(SystemExit) as exc_info:
        module.MCP()

    out = capsys.readouterr()

    assert exc_info.value.code == 0
    assert "Expose a configured Redis index to MCP clients" in out.out
    assert "Use this command when wiring RedisVL into an MCP client" in out.out
    assert (
        "uvx --from redisvl[mcp] rvl mcp --config /path/to/mcp_config.yaml" in out.out
    )
    assert "--transport" in out.out
    assert "streamable-http" in out.out
    assert "--host" in out.out
    assert "--port" in out.out


def test_mcp_command_preserves_env_read_only_when_flag_is_omitted(monkeypatch):
    monkeypatch.delitem(sys.modules, "redisvl.cli.mcp", raising=False)
    monkeypatch.delitem(sys.modules, "redisvl.mcp", raising=False)
    monkeypatch.setattr(sys, "version_info", _make_version_info(3, 11, 0))
    monkeypatch.setattr(sys, "argv", ["rvl", "mcp", "--config", "/tmp/mcp.yaml"])

    calls = []

    class FakeSettings(object):
        @classmethod
        def from_env(cls, config=None, read_only=None):
            calls.append(("settings", config, read_only))
            return cls()

    class FakeServer(object):
        def __init__(self, settings):
            self.settings = settings

        async def startup(self):
            calls.append(("startup",))

        async def run(self, transport="stdio"):
            calls.append(("run", transport))

        async def shutdown(self):
            calls.append(("shutdown",))

    _install_fake_redisvl_mcp(monkeypatch, FakeSettings, FakeServer)
    module = _import_cli_mcp()

    with pytest.raises(SystemExit) as exc_info:
        module.MCP()

    assert exc_info.value.code == 0
    assert calls == [
        ("settings", "/tmp/mcp.yaml", None),
        ("startup",),
        ("run", "stdio"),
        ("shutdown",),
    ]


def test_mcp_command_runs_startup_then_stdio_then_shutdown(monkeypatch):
    monkeypatch.delitem(sys.modules, "redisvl.cli.mcp", raising=False)
    monkeypatch.delitem(sys.modules, "redisvl.mcp", raising=False)
    monkeypatch.setattr(sys, "version_info", _make_version_info(3, 11, 0))
    monkeypatch.setattr(
        sys, "argv", ["rvl", "mcp", "--config", "/tmp/mcp.yaml", "--read-only"]
    )

    calls = []

    class FakeSettings(object):
        def __init__(self, config, read_only=False):
            self.config = config
            self.read_only = read_only

        @classmethod
        def from_env(cls, config=None, read_only=None):
            calls.append(("settings", config, read_only))
            return cls(config=config, read_only=read_only)

    class FakeServer(object):
        def __init__(self, settings):
            self.settings = settings

        async def startup(self):
            calls.append(("startup", self.settings.config, self.settings.read_only))

        async def run(self, transport="stdio"):
            calls.append(("run", transport))

        async def shutdown(self):
            calls.append(("shutdown",))

    _install_fake_redisvl_mcp(monkeypatch, FakeSettings, FakeServer)
    module = _import_cli_mcp()

    with pytest.raises(SystemExit) as exc_info:
        module.MCP()

    assert exc_info.value.code == 0
    assert calls == [
        ("settings", "/tmp/mcp.yaml", True),
        ("startup", "/tmp/mcp.yaml", True),
        ("run", "stdio"),
        ("shutdown",),
    ]


def test_mcp_command_prefers_run_async_without_manual_lifecycle(monkeypatch):
    monkeypatch.delitem(sys.modules, "redisvl.cli.mcp", raising=False)
    monkeypatch.delitem(sys.modules, "redisvl.mcp", raising=False)
    monkeypatch.setattr(sys, "version_info", _make_version_info(3, 11, 0))
    monkeypatch.setattr(sys, "argv", ["rvl", "mcp", "--config", "/tmp/mcp.yaml"])

    calls = []

    class FakeSettings(object):
        @classmethod
        def from_env(cls, config=None, read_only=None):
            calls.append(("settings", config, read_only))
            return cls()

    class FakeServer(object):
        def __init__(self, settings):
            self.settings = settings

        async def startup(self):
            calls.append(("startup",))

        async def run_async(self, transport="stdio"):
            calls.append(("run_async", transport))

        async def shutdown(self):
            calls.append(("shutdown",))

    _install_fake_redisvl_mcp(monkeypatch, FakeSettings, FakeServer)
    module = _import_cli_mcp()

    with pytest.raises(SystemExit) as exc_info:
        module.MCP()

    assert exc_info.value.code == 0
    assert calls == [
        ("settings", "/tmp/mcp.yaml", None),
        ("run_async", "stdio"),
    ]


def test_mcp_command_reports_startup_failures(monkeypatch, capsys):
    monkeypatch.delitem(sys.modules, "redisvl.cli.mcp", raising=False)
    monkeypatch.delitem(sys.modules, "redisvl.mcp", raising=False)
    monkeypatch.setattr(sys, "version_info", _make_version_info(3, 11, 0))
    monkeypatch.setattr(sys, "argv", ["rvl", "mcp", "--config", "/tmp/mcp.yaml"])

    calls = []

    class FakeSettings(object):
        @classmethod
        def from_env(cls, config=None, read_only=None):
            calls.append(("settings", config, read_only))
            return cls()

    class FakeServer(object):
        def __init__(self, settings):
            self.settings = settings

        async def startup(self):
            calls.append(("startup",))
            raise RuntimeError("boom")

        async def run(self, transport="stdio"):
            calls.append(("run", transport))

        async def shutdown(self):
            calls.append(("shutdown",))

    _install_fake_redisvl_mcp(monkeypatch, FakeSettings, FakeServer)
    module = _import_cli_mcp()

    with pytest.raises(SystemExit) as exc_info:
        module.MCP()

    out = capsys.readouterr()

    assert exc_info.value.code == 1
    assert calls == [("settings", "/tmp/mcp.yaml", None), ("startup",)]
    assert "boom" in out.err or "boom" in out.out


def test_mcp_command_reports_run_async_failures_without_manual_shutdown(
    monkeypatch, capsys
):
    monkeypatch.delitem(sys.modules, "redisvl.cli.mcp", raising=False)
    monkeypatch.delitem(sys.modules, "redisvl.mcp", raising=False)
    monkeypatch.setattr(sys, "version_info", _make_version_info(3, 11, 0))
    monkeypatch.setattr(sys, "argv", ["rvl", "mcp", "--config", "/tmp/mcp.yaml"])

    calls = []

    class FakeSettings(object):
        @classmethod
        def from_env(cls, config=None, read_only=None):
            calls.append(("settings", config, read_only))
            return cls()

    class FakeServer(object):
        def __init__(self, settings):
            self.settings = settings

        async def startup(self):
            calls.append(("startup",))

        async def run_async(self, transport="stdio"):
            calls.append(("run_async", transport))
            raise RuntimeError("run_async failed")

        async def shutdown(self):
            calls.append(("shutdown",))

    _install_fake_redisvl_mcp(monkeypatch, FakeSettings, FakeServer)
    module = _import_cli_mcp()

    with pytest.raises(SystemExit) as exc_info:
        module.MCP()

    out = capsys.readouterr()

    assert exc_info.value.code == 1
    assert calls == [
        ("settings", "/tmp/mcp.yaml", None),
        ("run_async", "stdio"),
    ]
    assert "run_async failed" in out.err or "run_async failed" in out.out


def test_mcp_command_shuts_down_when_run_fails(monkeypatch, capsys):
    monkeypatch.delitem(sys.modules, "redisvl.cli.mcp", raising=False)
    monkeypatch.delitem(sys.modules, "redisvl.mcp", raising=False)
    monkeypatch.setattr(sys, "version_info", _make_version_info(3, 11, 0))
    monkeypatch.setattr(sys, "argv", ["rvl", "mcp", "--config", "/tmp/mcp.yaml"])

    calls = []

    class FakeSettings(object):
        @classmethod
        def from_env(cls, config=None, read_only=None):
            calls.append(("settings", config, read_only))
            return cls()

    class FakeServer(object):
        def __init__(self, settings):
            self.settings = settings

        async def startup(self):
            calls.append(("startup",))

        async def run(self, transport="stdio"):
            calls.append(("run", transport))
            raise RuntimeError("run failed")

        async def shutdown(self):
            calls.append(("shutdown",))

    _install_fake_redisvl_mcp(monkeypatch, FakeSettings, FakeServer)
    module = _import_cli_mcp()

    with pytest.raises(SystemExit) as exc_info:
        module.MCP()

    out = capsys.readouterr()

    assert exc_info.value.code == 1
    assert calls == [
        ("settings", "/tmp/mcp.yaml", None),
        ("startup",),
        ("run", "stdio"),
        ("shutdown",),
    ]
    assert "run failed" in out.err or "run failed" in out.out


def test_mcp_command_passes_streamable_http_transport(monkeypatch):
    monkeypatch.delitem(sys.modules, "redisvl.cli.mcp", raising=False)
    monkeypatch.delitem(sys.modules, "redisvl.mcp", raising=False)
    monkeypatch.setattr(sys, "version_info", _make_version_info(3, 11, 0))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rvl",
            "mcp",
            "--config",
            "/tmp/mcp.yaml",
            "--transport",
            "streamable-http",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
        ],
    )

    calls = []

    class FakeSettings(object):
        @classmethod
        def from_env(cls, config=None, read_only=None):
            calls.append(("settings", config, read_only))
            return cls()

    class FakeServer(object):
        def __init__(self, settings):
            self.settings = settings

        async def run_async(self, transport="stdio", **kwargs):
            calls.append(("run_async", transport, kwargs))

    _install_fake_redisvl_mcp(monkeypatch, FakeSettings, FakeServer)
    module = _import_cli_mcp()

    with pytest.raises(SystemExit) as exc_info:
        module.MCP()

    assert exc_info.value.code == 0
    assert calls == [
        ("settings", "/tmp/mcp.yaml", None),
        ("run_async", "streamable-http", {"host": "0.0.0.0", "port": 9000}),
    ]


def test_mcp_command_stdio_does_not_pass_host_port(monkeypatch):
    monkeypatch.delitem(sys.modules, "redisvl.cli.mcp", raising=False)
    monkeypatch.delitem(sys.modules, "redisvl.mcp", raising=False)
    monkeypatch.setattr(sys, "version_info", _make_version_info(3, 11, 0))
    monkeypatch.setattr(sys, "argv", ["rvl", "mcp", "--config", "/tmp/mcp.yaml"])

    calls = []

    class FakeSettings(object):
        @classmethod
        def from_env(cls, config=None, read_only=None):
            calls.append(("settings", config, read_only))
            return cls()

    class FakeServer(object):
        def __init__(self, settings):
            self.settings = settings

        async def run_async(self, transport="stdio", **kwargs):
            calls.append(("run_async", transport, kwargs))

    _install_fake_redisvl_mcp(monkeypatch, FakeSettings, FakeServer)
    module = _import_cli_mcp()

    with pytest.raises(SystemExit) as exc_info:
        module.MCP()

    assert exc_info.value.code == 0
    assert calls == [
        ("settings", "/tmp/mcp.yaml", None),
        ("run_async", "stdio", {}),
    ]


def test_mcp_command_rejects_invalid_transport(monkeypatch, capsys):
    """TDD: argparse should reject transports not in the choices list."""
    monkeypatch.delitem(sys.modules, "redisvl.cli.mcp", raising=False)
    monkeypatch.setattr(sys, "version_info", _make_version_info(3, 11, 0))
    monkeypatch.setattr(
        sys,
        "argv",
        ["rvl", "mcp", "--config", "/tmp/mcp.yaml", "--transport", "websocket"],
    )

    module = _import_cli_mcp()

    with pytest.raises(SystemExit) as exc_info:
        module.MCP()

    assert exc_info.value.code == 1
    out = capsys.readouterr()
    assert "invalid choice" in out.err or "invalid choice" in out.out


def test_mcp_command_fallback_run_path_passes_http_transport_kwargs(monkeypatch):
    """TDD: When run_async is absent, the fallback run() path must also
    forward transport and host/port kwargs for HTTP transports."""
    monkeypatch.delitem(sys.modules, "redisvl.cli.mcp", raising=False)
    monkeypatch.delitem(sys.modules, "redisvl.mcp", raising=False)
    monkeypatch.setattr(sys, "version_info", _make_version_info(3, 11, 0))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rvl",
            "mcp",
            "--config",
            "/tmp/mcp.yaml",
            "--transport",
            "streamable-http",
            "--host",
            "0.0.0.0",
            "--port",
            "7777",
        ],
    )

    calls = []

    class FakeSettings(object):
        @classmethod
        def from_env(cls, config=None, read_only=None):
            calls.append(("settings", config, read_only))
            return cls()

    class FakeServer(object):
        """No run_async -- forces the fallback startup/run/shutdown path."""

        def __init__(self, settings):
            self.settings = settings

        async def startup(self):
            calls.append(("startup",))

        async def run(self, transport="stdio", **kwargs):
            calls.append(("run", transport, kwargs))

        async def shutdown(self):
            calls.append(("shutdown",))

    _install_fake_redisvl_mcp(monkeypatch, FakeSettings, FakeServer)
    module = _import_cli_mcp()

    with pytest.raises(SystemExit) as exc_info:
        module.MCP()

    assert exc_info.value.code == 0
    assert calls == [
        ("settings", "/tmp/mcp.yaml", None),
        ("startup",),
        ("run", "streamable-http", {"host": "0.0.0.0", "port": 7777}),
        ("shutdown",),
    ]
