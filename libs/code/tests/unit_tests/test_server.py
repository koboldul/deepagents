"""Tests for server lifecycle helpers."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Self
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepagents_code._posix_shell import _SyncPosixOwnerGuard
from deepagents_code.client.launch import server as server_module
from deepagents_code.client.launch.server import (
    ServerProcess,
    _find_free_port,
    _port_in_use,
    _server_process_group,
    _spawn_posix_server_process,
    _terminate_server_process,
    _wait_for_process_group_exit,
    emit_preserved_log_notices,
    wait_for_server_healthy,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import IO

_SIGKILL = getattr(signal, "SIGKILL", 9)


@pytest.fixture
def _posix_process_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide POSIX-only process symbols for tests that simulate Linux."""
    if not hasattr(os, "getpgid"):
        monkeypatch.setattr(os, "getpgid", None, raising=False)
    if not hasattr(os, "killpg"):
        monkeypatch.setattr(os, "killpg", None, raising=False)
    if not hasattr(signal, "SIGKILL"):
        monkeypatch.setattr(signal, "SIGKILL", 9, raising=False)


@pytest.fixture
def _real_posix_server_lifecycle() -> None:
    """Opt a test into the real gated POSIX watchdog launch."""


@pytest.fixture(autouse=True)
def _stub_posix_server_lifecycle(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep ordinary subprocess mocks independent of POSIX watchdog processes."""
    if "_real_posix_server_lifecycle" in request.fixturenames:
        return

    def spawn(
        cmd: list[str],
        *,
        cwd: str,
        env: dict[str, str],
        stdout: int | IO[str] | None,
    ) -> tuple[subprocess.Popen[Any], None]:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            creationflags=0,
        )
        return process, None

    monkeypatch.setattr(
        "deepagents_code.client.launch.server._spawn_posix_server_process",
        spawn,
    )


@pytest.fixture
def _real_windows_server_lifecycle() -> None:
    """Opt a test into real Windows Job assignment instead of the mock."""


@pytest.fixture(autouse=True)
def _stub_windows_server_job_assignment(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> list[MagicMock]:
    """Keep ordinary subprocess mocks independent of native Windows handles."""
    jobs: list[MagicMock] = []
    if "_real_windows_server_lifecycle" in request.fixturenames:
        return jobs

    def assign(_process: subprocess.Popen[bytes]) -> MagicMock:
        job = MagicMock()
        job.terminate.return_value = True
        jobs.append(job)
        return job

    monkeypatch.setattr(
        "deepagents_code.client.launch.server._assign_windows_server_job_and_resume",
        assign,
    )
    return jobs


class _FakeSocket:
    """Small socket stand-in for unit tests running with `--disable-socket`."""

    def __init__(
        self,
        *,
        bind_error: OSError | None = None,
        sockname: tuple[str, int] = ("127.0.0.1", 0),
    ) -> None:
        self._bind_error = bind_error
        self._sockname = sockname
        self.bound_addr: tuple[str, int] | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def bind(self, addr: tuple[str, int]) -> None:
        """Record the bind call or raise the configured error."""
        if self._bind_error is not None:
            raise self._bind_error
        self.bound_addr = addr

    def getsockname(self) -> tuple[str, int]:
        """Return the configured socket name tuple."""
        return self._sockname


class _FakeAsyncClient:
    """Minimal async `httpx.AsyncClient` stand-in for readiness tests."""

    def __init__(self, response: object) -> None:
        self.response = response
        self.urls: list[str] = []

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def get(
        self,
        url: str,
        *,
        timeout: float,  # noqa: ARG002, ASYNC109  # mirrors httpx.AsyncClient.get
    ) -> object:
        self.urls.append(url)
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


class TestPortInUse:
    def test_free_port(self) -> None:
        fake_socket = _FakeSocket()

        with patch("socket.socket", return_value=fake_socket) as socket_cls:
            assert not _port_in_use("127.0.0.1", 2024)

        socket_cls.assert_called_once_with(socket.AF_INET, socket.SOCK_STREAM)
        assert fake_socket.bound_addr == ("127.0.0.1", 2024)

    def test_occupied_port(self) -> None:
        fake_socket = _FakeSocket(bind_error=OSError("port already in use"))

        with patch("socket.socket", return_value=fake_socket) as socket_cls:
            assert _port_in_use("127.0.0.1", 2024)

        socket_cls.assert_called_once_with(socket.AF_INET, socket.SOCK_STREAM)
        assert fake_socket.bound_addr is None


class TestFindFreePort:
    def test_returns_valid_port(self) -> None:
        fake_socket = _FakeSocket(sockname=("127.0.0.1", 43210))

        with patch("socket.socket", return_value=fake_socket) as socket_cls:
            port = _find_free_port("127.0.0.1")

        socket_cls.assert_called_once_with(socket.AF_INET, socket.SOCK_STREAM)
        assert fake_socket.bound_addr == ("127.0.0.1", 0)
        assert 1 <= port <= 65535

    def test_returns_port_reported_by_socket(self) -> None:
        fake_socket = _FakeSocket(sockname=("127.0.0.1", 53123))

        with patch("socket.socket", return_value=fake_socket):
            port = _find_free_port("127.0.0.1")

        assert port == 53123


class TestServerPortSelection:
    """Port resolution in `ServerProcess.start()`."""

    @staticmethod
    def _make_server(tmp_path: Path, port: int = 0) -> ServerProcess:
        config_dir = tmp_path / "runtime"
        config_dir.mkdir()
        (config_dir / "langgraph.json").write_text("{}")
        return ServerProcess(config_dir=config_dir, port=port)

    @staticmethod
    def _mock_log_file(tmp_path: Path) -> MagicMock:
        log_file = MagicMock()
        log_file.name = str(tmp_path / "server.log")
        return log_file

    async def test_default_uses_ephemeral_port(self, tmp_path: Path) -> None:
        """Default port (0) resolves via `_find_free_port`, never squats 2024."""
        server = self._make_server(tmp_path)
        assert server.port == 0

        process = MagicMock(pid=1234)
        process.poll.return_value = None
        with (
            patch(
                "deepagents_code.client.launch.server.tempfile.NamedTemporaryFile",
                return_value=self._mock_log_file(tmp_path),
            ),
            patch(
                "deepagents_code.client.launch.server.subprocess.Popen",
                return_value=process,
            ),
            patch(
                "deepagents_code.client.launch.server.wait_for_server_healthy",
                new=AsyncMock(),
            ),
            patch(
                "deepagents_code.client.launch.server._find_free_port",
                return_value=43210,
            ) as find_free,
            patch("deepagents_code.client.launch.server._port_in_use") as in_use,
        ):
            await server.start()

        find_free.assert_called_once_with("127.0.0.1")
        in_use.assert_not_called()
        assert server.port == 43210

    async def test_explicit_free_port_is_kept(self, tmp_path: Path) -> None:
        """An explicit, free port is honored without searching for another."""
        server = self._make_server(tmp_path, port=2024)

        process = MagicMock(pid=1234)
        process.poll.return_value = None
        with (
            patch(
                "deepagents_code.client.launch.server.tempfile.NamedTemporaryFile",
                return_value=self._mock_log_file(tmp_path),
            ),
            patch(
                "deepagents_code.client.launch.server.subprocess.Popen",
                return_value=process,
            ),
            patch(
                "deepagents_code.client.launch.server.wait_for_server_healthy",
                new=AsyncMock(),
            ),
            patch(
                "deepagents_code.client.launch.server._port_in_use", return_value=False
            ) as in_use,
            patch("deepagents_code.client.launch.server._find_free_port") as find_free,
        ):
            await server.start()

        in_use.assert_called_once_with("127.0.0.1", 2024)
        find_free.assert_not_called()
        assert server.port == 2024

    async def test_explicit_busy_port_falls_back(self, tmp_path: Path) -> None:
        """An explicit but busy port falls back to a free port."""
        server = self._make_server(tmp_path, port=2024)

        process = MagicMock(pid=1234)
        process.poll.return_value = None
        with (
            patch(
                "deepagents_code.client.launch.server.tempfile.NamedTemporaryFile",
                return_value=self._mock_log_file(tmp_path),
            ),
            patch(
                "deepagents_code.client.launch.server.subprocess.Popen",
                return_value=process,
            ),
            patch(
                "deepagents_code.client.launch.server.wait_for_server_healthy",
                new=AsyncMock(),
            ),
            patch(
                "deepagents_code.client.launch.server._port_in_use", return_value=True
            ) as in_use,
            patch(
                "deepagents_code.client.launch.server._find_free_port",
                return_value=43210,
            ) as find_free,
        ):
            await server.start()

        in_use.assert_called_once_with("127.0.0.1", 2024)
        find_free.assert_called_once_with("127.0.0.1")
        assert server.port == 43210


class TestWaitForServerHealthy:
    """Tests for the health-check polling loop."""

    async def test_returns_on_200(self) -> None:
        """Happy path: server responds 200 immediately."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            await wait_for_server_healthy("http://localhost:2024", timeout=5)

        mock_client.get.assert_awaited_once()

    async def test_raises_on_early_process_exit(self) -> None:
        """Process dies before health check succeeds -> fail fast."""
        process = MagicMock()
        process.poll.return_value = 1
        process.returncode = 1

        with pytest.raises(RuntimeError, match="exited with code 1"):
            await wait_for_server_healthy(
                "http://localhost:2024",
                timeout=5,
                process=process,
            )

    async def test_early_exit_includes_log_output(self) -> None:
        """read_log output is included in the error message."""
        process = MagicMock()
        process.poll.return_value = 1
        process.returncode = 1

        with pytest.raises(RuntimeError, match="some log output"):
            await wait_for_server_healthy(
                "http://localhost:2024",
                timeout=5,
                process=process,
                read_log=lambda: "some log output",
            )

    async def test_early_exit_promotes_marked_startup_error(self) -> None:
        """Marked server startup errors should survive app error trimming."""
        process = MagicMock()
        process.poll.return_value = 1
        process.returncode = 3

        log = (
            "Traceback (most recent call last):\n"
            "ValueError: No Runloop API key found\n"
            "Sandbox creation failed for 'runloop': No Runloop API key found. "
            "Set RUNLOOP_API_KEY or DEEPAGENTS_CODE_RUNLOOP_API_KEY.\n"
            "DEEPAGENTS_STARTUP_ERROR:Sandbox creation failed for 'runloop': "
            "No Runloop API key found. Set RUNLOOP_API_KEY or "
            "DEEPAGENTS_CODE_RUNLOOP_API_KEY.\n"
            "2026-05-11T03:37:44.911664Z [error    ] "
            "Application startup failed. Exiting. [uvicorn.error]"
        )

        with pytest.raises(RuntimeError) as exc_info:
            await wait_for_server_healthy(
                "http://localhost:2024",
                timeout=5,
                process=process,
                read_log=lambda: log,
            )

        first_line = str(exc_info.value).splitlines()[0]
        assert first_line == (
            "Server process exited with code 3: Sandbox creation failed for "
            "'runloop': No Runloop API key found. Set RUNLOOP_API_KEY or "
            "DEEPAGENTS_CODE_RUNLOOP_API_KEY."
        )

    async def test_raises_on_timeout(self) -> None:
        """Timeout exhaustion raises RuntimeError."""
        import httpx

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch(
                "deepagents_code.client.launch.server._HEALTH_POLL_INTERVAL_LOCAL", 0
            ),
            patch(
                "deepagents_code.client.launch.server._HEALTH_POLL_INTERVAL_REMOTE", 0
            ),
            pytest.raises(RuntimeError, match="did not become healthy"),
        ):
            await wait_for_server_healthy("http://localhost:2024", timeout=0.01)

    async def test_timeout_reports_last_status(self) -> None:
        """Timeout error includes the last HTTP status code."""
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch(
                "deepagents_code.client.launch.server._HEALTH_POLL_INTERVAL_LOCAL", 0
            ),
            patch(
                "deepagents_code.client.launch.server._HEALTH_POLL_INTERVAL_REMOTE", 0
            ),
            pytest.raises(RuntimeError, match="last status: 503"),
        ):
            await wait_for_server_healthy("http://localhost:2024", timeout=0.01)


class TestServerProcess:
    @pytest.fixture(autouse=True)
    def _no_real_process_group(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Force the process-only shutdown fallback for placeholder-pid mocks.

        `_stop_process` now tears down the server's whole process group via
        `os.getpgid`/`os.killpg`. These tests drive `MagicMock` subprocesses
        with fake pids, so patch `_server_process_group` to return `None`. This
        keeps teardown deterministic and guarantees no unrelated real pid is
        ever signaled. Group-teardown behavior is covered explicitly in
        `TestTerminateServerProcess`.
        """
        monkeypatch.setattr(
            "deepagents_code.client.launch.server._server_process_group",
            lambda _pid: None,
        )
        monkeypatch.setattr(
            "deepagents_code.client.launch.server.sys.platform",
            "linux",
        )

    async def test_start_is_noop_when_already_running(self) -> None:
        """`start()` on an already-running server spawns no second process.

        The old `if self.running: return` guard now lives inside
        `_spawn_process` (under `_state_lock`); a regression would double-spawn
        the subprocess and leak the port.
        """
        server = ServerProcess(host="127.0.0.1", port=2024)
        running_proc = MagicMock()
        running_proc.poll.return_value = None  # still alive
        server._process = running_proc

        with patch("deepagents_code.client.launch.server.subprocess.Popen") as popen:
            await server.start()

        popen.assert_not_called()
        assert server._process is running_proc

    async def test_wait_for_graph_ready_resolves_graph_endpoint(self) -> None:
        """Graph readiness should force LangGraph to resolve graph factories."""
        client = _FakeAsyncClient(SimpleNamespace(status_code=200))
        process = MagicMock()
        process.poll.return_value = None
        server = ServerProcess(host="127.0.0.1", port=2024)
        server._process = process

        with patch("httpx.AsyncClient", return_value=client):
            await server.wait_for_graph_ready("agent")

        assert client.urls == ["http://127.0.0.1:2024/assistants/agent/graph"]

    async def test_wait_for_graph_ready_surfaces_startup_marker(
        self, tmp_path: Path
    ) -> None:
        """Readiness failures should preserve marked subprocess startup errors."""
        log_path = tmp_path / "server.log"
        log_path.write_text(
            "booting\n"
            "DEEPAGENTS_STARTUP_ERROR:Sandbox creation failed for 'modal': boom\n"
        )

        log_file = MagicMock()
        log_file.name = str(log_path)

        client = _FakeAsyncClient(SimpleNamespace(status_code=500))
        process = MagicMock()
        process.poll.return_value = None
        server = ServerProcess(host="127.0.0.1", port=2024)
        server._process = process
        server._log_file = log_file

        with (
            patch("httpx.AsyncClient", return_value=client),
            pytest.raises(RuntimeError, match="Sandbox creation failed"),
        ):
            await server.wait_for_graph_ready("agent")

    async def test_wait_for_graph_ready_checks_logs_after_transport_error(
        self, tmp_path: Path
    ) -> None:
        """Dropped graph requests should still surface startup markers."""
        log_path = tmp_path / "server.log"
        log_path.write_text(
            "booting\nDEEPAGENTS_STARTUP_ERROR:ModelConfigError: missing API key\n"
        )

        log_file = MagicMock()
        log_file.name = str(log_path)

        client = _FakeAsyncClient(OSError("connection closed"))
        process = MagicMock()
        process.poll.return_value = 1
        process.returncode = 3
        server = ServerProcess(host="127.0.0.1", port=2024)
        server._process = process
        server._log_file = log_file

        with (
            patch("httpx.AsyncClient", return_value=client),
            pytest.raises(RuntimeError, match="ModelConfigError: missing API key"),
        ):
            await server.wait_for_graph_ready("agent")

    async def test_start_cleans_up_partial_state_on_health_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Failed startup should stop the process and remove owned resources."""
        # `_stop_process` preserves the log file when debug mode is on, which
        # would defeat the `not log_path.exists()` assertion below; pin it off
        # so an ambient `DEEPAGENTS_CODE_DEBUG` in the environment can't flake.
        monkeypatch.delenv("DEEPAGENTS_CODE_DEBUG", raising=False)

        config_dir = tmp_path / "runtime"
        config_dir.mkdir()
        (config_dir / "langgraph.json").write_text("{}")

        log_path = tmp_path / "server.log"
        log_path.write_text("booting")

        process = MagicMock()
        process.pid = 1234
        process.poll.return_value = None

        log_file = MagicMock()
        log_file.name = str(log_path)

        server = ServerProcess(config_dir=config_dir, owns_config_dir=True)

        with (
            patch(
                "deepagents_code.client.launch.server._find_free_port",
                return_value=12345,
            ),
            patch(
                "deepagents_code.client.launch.server.tempfile.NamedTemporaryFile",
                return_value=log_file,
            ),
            patch(
                "deepagents_code.client.launch.server.subprocess.Popen",
                return_value=process,
            ),
            patch(
                "deepagents_code.client.launch.server.wait_for_server_healthy",
                new=AsyncMock(side_effect=RuntimeError("boom")),
            ),
            pytest.raises(RuntimeError, match="boom"),
        ):
            await server.start()

        process.send_signal.assert_called_once_with(signal.SIGTERM)
        process.wait.assert_called_once()
        log_file.close.assert_called_once()
        assert server._process is None
        assert server._log_file is None
        assert not config_dir.exists()
        assert not log_path.exists()

    async def test_start_cleans_up_partial_state_on_cancellation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cancelled startup must reap the subprocess it already spawned.

        `asyncio.CancelledError` is a `BaseException`, not an `Exception`, so
        `start()` must clean up in a `finally` — otherwise the `langgraph dev`
        subprocess spawned before the health check is orphaned when the caller
        is cancelled mid-startup (e.g. Ctrl+D). Regression: PR #4629.
        """
        # See sibling health-failure test: pin debug off so the log file is
        # unlinked and the `not log_path.exists()` assertion can't flake.
        monkeypatch.delenv("DEEPAGENTS_CODE_DEBUG", raising=False)

        config_dir = tmp_path / "runtime"
        config_dir.mkdir()
        (config_dir / "langgraph.json").write_text("{}")

        log_path = tmp_path / "server.log"
        log_path.write_text("booting")

        process = MagicMock()
        process.pid = 1234
        process.poll.return_value = None
        loop_thread_id = threading.get_ident()
        cleanup_thread_id: int | None = None

        def record_cleanup_thread(*, timeout: float) -> None:  # noqa: ARG001
            nonlocal cleanup_thread_id
            cleanup_thread_id = threading.get_ident()

        process.wait.side_effect = record_cleanup_thread

        log_file = MagicMock()
        log_file.name = str(log_path)

        server = ServerProcess(config_dir=config_dir, owns_config_dir=True)

        with (
            patch(
                "deepagents_code.client.launch.server._find_free_port",
                return_value=12345,
            ),
            patch(
                "deepagents_code.client.launch.server.tempfile.NamedTemporaryFile",
                return_value=log_file,
            ),
            patch(
                "deepagents_code.client.launch.server.subprocess.Popen",
                return_value=process,
            ),
            patch(
                "deepagents_code.client.launch.server.wait_for_server_healthy",
                new=AsyncMock(side_effect=asyncio.CancelledError),
            ),
            pytest.raises(asyncio.CancelledError),
        ):
            await server.start()

        process.send_signal.assert_called_once_with(signal.SIGTERM)
        process.wait.assert_called_once()
        assert cleanup_thread_id is not None
        assert cleanup_thread_id != loop_thread_id
        log_file.close.assert_called_once()
        assert server._process is None
        assert server._log_file is None
        assert not config_dir.exists()
        assert not log_path.exists()

    async def test_start_cleanup_error_does_not_mask_startup_error(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A failing `stop()` during cleanup must not mask the startup error.

        `start()`'s `finally` guards `stop()` so that if reaping the subprocess
        itself raises, the in-flight startup exception still propagates
        unchanged (rather than being replaced by the cleanup error) and the
        failure is logged at `error` level.
        """
        config_dir = tmp_path / "runtime"
        config_dir.mkdir()
        (config_dir / "langgraph.json").write_text("{}")

        log_path = tmp_path / "server.log"
        log_path.write_text("booting")

        process = MagicMock()
        process.pid = 1234
        process.poll.return_value = None

        log_file = MagicMock()
        log_file.name = str(log_path)

        server = ServerProcess(config_dir=config_dir, owns_config_dir=True)

        with (
            patch(
                "deepagents_code.client.launch.server._find_free_port",
                return_value=12345,
            ),
            patch(
                "deepagents_code.client.launch.server.tempfile.NamedTemporaryFile",
                return_value=log_file,
            ),
            patch(
                "deepagents_code.client.launch.server.subprocess.Popen",
                return_value=process,
            ),
            patch(
                "deepagents_code.client.launch.server.wait_for_server_healthy",
                new=AsyncMock(side_effect=RuntimeError("startup boom")),
            ),
            patch.object(
                server, "stop", side_effect=RuntimeError("cleanup boom")
            ) as mock_stop,
            caplog.at_level(logging.ERROR),
            # The original startup error propagates, not the cleanup error.
            pytest.raises(RuntimeError, match="startup boom"),
        ):
            await server.start()

        mock_stop.assert_called_once()
        assert "Error stopping server during startup cleanup" in caplog.text

    async def test_start_rescaffolds_when_config_missing(self, tmp_path: Path) -> None:
        """A missing langgraph.json should be rebuilt via the scaffold hook."""
        config_dir = tmp_path / "runtime"
        config_dir.mkdir()

        def scaffold(work_dir: Path) -> None:
            (work_dir / "langgraph.json").write_text("{}")

        scaffold_mock = MagicMock(side_effect=scaffold)

        process = MagicMock()
        process.pid = 1234
        process.poll.return_value = None

        log_file = MagicMock()
        log_file.name = str(tmp_path / "server.log")

        server = ServerProcess(config_dir=config_dir, scaffold=scaffold_mock)

        with (
            patch(
                "deepagents_code.client.launch.server._find_free_port",
                return_value=12345,
            ),
            patch(
                "deepagents_code.client.launch.server.tempfile.NamedTemporaryFile",
                return_value=log_file,
            ),
            patch(
                "deepagents_code.client.launch.server.subprocess.Popen",
                return_value=process,
            ),
            patch(
                "deepagents_code.client.launch.server.wait_for_server_healthy",
                new=AsyncMock(),
            ),
        ):
            await server.start()

        scaffold_mock.assert_called_once_with(config_dir)
        assert (config_dir / "langgraph.json").exists()

    async def test_start_raises_when_scaffold_does_not_restore_config(
        self, tmp_path: Path
    ) -> None:
        """A scaffold hook that runs but produces no config still raises.

        The error must report the failed rescaffold (not the misleading
        "call generate_langgraph_json() first") and the hook must have run.
        """
        config_dir = tmp_path / "runtime"
        config_dir.mkdir()

        scaffold_mock = MagicMock()
        server = ServerProcess(config_dir=config_dir, scaffold=scaffold_mock)

        with pytest.raises(RuntimeError, match=r"did not produce langgraph\.json"):
            await server.start()

        scaffold_mock.assert_called_once_with(config_dir)

    async def test_start_raises_without_scaffold_when_config_missing(
        self, tmp_path: Path
    ) -> None:
        """With no scaffold hook, a missing config raises the original error."""
        config_dir = tmp_path / "runtime"
        config_dir.mkdir()

        server = ServerProcess(config_dir=config_dir)

        with pytest.raises(RuntimeError, match=r"langgraph\.json not found"):
            await server.start()

    async def test_start_wraps_scaffold_oserror(self, tmp_path: Path) -> None:
        """An OSError raised mid-scaffold surfaces as RuntimeError, cause kept."""
        config_dir = tmp_path / "runtime"
        config_dir.mkdir()

        boom = OSError("No space left on device")
        server = ServerProcess(
            config_dir=config_dir, scaffold=MagicMock(side_effect=boom)
        )

        with pytest.raises(RuntimeError, match=r"Failed to rescaffold") as exc_info:
            await server.start()

        assert exc_info.value.__cause__ is boom

    async def test_start_creates_work_dir_when_purged(self, tmp_path: Path) -> None:
        """A fully purged work dir is recreated before the scaffold runs.

        Exercises the `mkdir(parents=True)` recovery: the directory itself —
        not just `langgraph.json` — is gone (the OS tmp reaper removing the
        whole temp dir), so the scaffold would fail without the mkdir.
        """
        # Deliberately not created: the directory is missing entirely.
        config_dir = tmp_path / "runtime"

        def scaffold(work_dir: Path) -> None:
            (work_dir / "langgraph.json").write_text("{}")

        scaffold_mock = MagicMock(side_effect=scaffold)

        process = MagicMock()
        process.pid = 1234
        process.poll.return_value = None

        log_file = MagicMock()
        log_file.name = str(tmp_path / "server.log")

        server = ServerProcess(config_dir=config_dir, scaffold=scaffold_mock)

        with (
            patch(
                "deepagents_code.client.launch.server._find_free_port",
                return_value=12345,
            ),
            patch(
                "deepagents_code.client.launch.server.tempfile.NamedTemporaryFile",
                return_value=log_file,
            ),
            patch(
                "deepagents_code.client.launch.server.subprocess.Popen",
                return_value=process,
            ),
            patch(
                "deepagents_code.client.launch.server.wait_for_server_healthy",
                new=AsyncMock(),
            ),
        ):
            await server.start()

        scaffold_mock.assert_called_once_with(config_dir)
        assert config_dir.is_dir()
        assert (config_dir / "langgraph.json").exists()

    async def test_start_does_not_rescaffold_when_config_present(
        self, tmp_path: Path
    ) -> None:
        """The scaffold hook is a recovery path only; skip it when config exists."""
        config_dir = tmp_path / "runtime"
        config_dir.mkdir()
        (config_dir / "langgraph.json").write_text("{}")

        scaffold_mock = MagicMock()

        process = MagicMock()
        process.pid = 1234
        process.poll.return_value = None

        log_file = MagicMock()
        log_file.name = str(tmp_path / "server.log")

        server = ServerProcess(config_dir=config_dir, scaffold=scaffold_mock)

        with (
            patch(
                "deepagents_code.client.launch.server._find_free_port",
                return_value=12345,
            ),
            patch(
                "deepagents_code.client.launch.server.tempfile.NamedTemporaryFile",
                return_value=log_file,
            ),
            patch(
                "deepagents_code.client.launch.server.subprocess.Popen",
                return_value=process,
            ),
            patch(
                "deepagents_code.client.launch.server.wait_for_server_healthy",
                new=AsyncMock(),
            ),
        ):
            await server.start()

        scaffold_mock.assert_not_called()

    async def test_restart_rescaffolds_after_config_purged(
        self, tmp_path: Path
    ) -> None:
        """restart() rebuilds a config purged between boot and the restart.

        This is the motivating scenario: the server boots with config present,
        the work dir is purged externally, and a later `/restart` recovers via
        the scaffold hook rather than failing.
        """
        config_dir = tmp_path / "runtime"
        config_dir.mkdir()
        config_path = config_dir / "langgraph.json"
        config_path.write_text("{}")

        def scaffold(work_dir: Path) -> None:
            (work_dir / "langgraph.json").write_text("{}")

        scaffold_mock = MagicMock(side_effect=scaffold)

        process = MagicMock()
        process.pid = 1234
        process.poll.return_value = None

        log_file = MagicMock()
        log_file.name = str(tmp_path / "server.log")

        server = ServerProcess(config_dir=config_dir, scaffold=scaffold_mock)

        with (
            patch(
                "deepagents_code.client.launch.server._find_free_port",
                return_value=12345,
            ),
            patch(
                "deepagents_code.client.launch.server._port_in_use", return_value=False
            ),
            patch(
                "deepagents_code.client.launch.server.tempfile.NamedTemporaryFile",
                return_value=log_file,
            ),
            patch(
                "deepagents_code.client.launch.server.subprocess.Popen",
                return_value=process,
            ),
            patch(
                "deepagents_code.client.launch.server.wait_for_server_healthy",
                new=AsyncMock(),
            ),
        ):
            await server.start()
            # Config present on boot: the scaffold hook must not have fired yet.
            scaffold_mock.assert_not_called()

            # Simulate the OS tmp reaper purging the work dir.
            config_path.unlink()

            await server.restart()

        scaffold_mock.assert_called_once_with(config_dir)
        assert config_path.exists()

    async def test_update_env_and_restart(self, tmp_path: Path) -> None:
        """update_env stages overrides that restart() applies."""
        config_dir = tmp_path / "runtime"
        config_dir.mkdir()
        (config_dir / "langgraph.json").write_text("{}")

        log_path = tmp_path / "server.log"
        log_path.write_text("")

        process = MagicMock()
        process.pid = 1234
        process.poll.return_value = None

        log_file = MagicMock()
        log_file.name = str(log_path)

        server = ServerProcess(config_dir=config_dir, owns_config_dir=False)

        with (
            patch(
                "deepagents_code.client.launch.server._find_free_port",
                return_value=12345,
            ),
            patch(
                "deepagents_code.client.launch.server._port_in_use", return_value=False
            ),
            patch(
                "deepagents_code.client.launch.server.tempfile.NamedTemporaryFile",
                return_value=log_file,
            ),
            patch(
                "deepagents_code.client.launch.server.subprocess.Popen",
                return_value=process,
            ),
            patch(
                "deepagents_code.client.launch.server.wait_for_server_healthy",
                new=AsyncMock(),
            ),
        ):
            await server.start()
            assert server.running

            server.update_env(DEEPAGENTS_CODE_SERVER_MODEL="anthropic:claude-opus-4-6")

            # Restart: should stop the old process and start a new one
            await server.restart()

        # Env override was applied
        env_key = "DEEPAGENTS_CODE_SERVER_MODEL"
        assert os.environ.get(env_key) == "anthropic:claude-opus-4-6"
        # Overrides cleared after successful restart
        assert server._env_overrides == {}

    async def test_restart_runs_blocking_stop_off_event_loop(
        self, tmp_path: Path
    ) -> None:
        """restart() must run the blocking subprocess stop off the event loop.

        `_stop_process` blocks up to `_SHUTDOWN_TIMEOUT` (plus a SIGKILL grace
        wait) on `process.wait`; running it directly on the loop freezes the
        Textual reactor so `/restart` wedges the TUI input. `restart()` must
        offload it to a worker thread, so the stop executes on a thread other
        than the one running the event loop.
        """
        config_dir = tmp_path / "runtime"
        config_dir.mkdir()
        (config_dir / "langgraph.json").write_text("{}")

        server = ServerProcess(config_dir=config_dir, owns_config_dir=False)
        server._process = MagicMock()

        loop_thread_id = threading.get_ident()
        stop_thread_id: int | None = None

        def recording_stop() -> None:
            nonlocal stop_thread_id
            stop_thread_id = threading.get_ident()
            server._process = None

        # Patch only `_start` (avoid spawning a real server) and `_stop_process`
        # (record its executing thread). The real `restart()` and real
        # `asyncio.to_thread` run, so a regression to a direct call would run
        # `_stop_process` on the loop thread and fail the off-loop assertion.
        with (
            patch.object(server, "_start", new=AsyncMock()),
            patch.object(server, "_stop_process", new=recording_stop),
        ):
            await server.restart()

        assert stop_thread_id is not None
        assert stop_thread_id != loop_thread_id

    async def test_restart_cancellation_awaits_stop_cleanup(
        self, tmp_path: Path
    ) -> None:
        """Cancelling restart() lets the offloaded stop finish before re-raising.

        The shield around the `_stop_process` thread must keep it running to
        completion even when the restart task is cancelled mid-cleanup, so the
        subprocess is never left half-torn-down; `_start` must not run, so no
        replacement is spawned.
        """
        config_dir = tmp_path / "runtime"
        config_dir.mkdir()
        (config_dir / "langgraph.json").write_text("{}")

        server = ServerProcess(config_dir=config_dir, owns_config_dir=False)
        stop_entered = threading.Event()
        release_stop = threading.Event()
        stop_completed = threading.Event()

        def controlled_stop_process() -> None:
            stop_entered.set()
            release_stop.wait(timeout=2.0)
            stop_completed.set()

        start_mock = AsyncMock()
        with (
            patch.object(server, "_stop_process", new=controlled_stop_process),
            patch.object(server, "_start", new=start_mock),
        ):
            restart = asyncio.create_task(server.restart())
            assert await asyncio.to_thread(stop_entered.wait, 2.0)

            # Cancel while the shielded stop thread is mid-flight. A bare await
            # (no shield) or dropping the `await stop_task` on cancel would let
            # cleanup be abandoned here — this asserts it still completes.
            restart.cancel()
            await asyncio.sleep(0)
            assert not stop_completed.is_set()
            release_stop.set()

            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(restart, timeout=2.0)

        assert stop_completed.is_set()
        start_mock.assert_not_awaited()

    async def test_restart_lifecycle_is_serialized_with_stop(
        self, tmp_path: Path
    ) -> None:
        """Terminal stop during restart cleanup prevents a replacement spawn."""
        config_dir = tmp_path / "runtime"
        config_dir.mkdir()
        (config_dir / "langgraph.json").write_text("{}")

        server = ServerProcess(config_dir=config_dir, owns_config_dir=False)
        restart_stop_entered = threading.Event()
        release_restart_stop = threading.Event()

        def controlled_stop_process() -> None:
            restart_stop_entered.set()
            release_restart_stop.wait(timeout=2.0)

        with patch.object(server, "_stop_process", new=controlled_stop_process):
            restart = asyncio.create_task(server.restart())
            assert await asyncio.to_thread(restart_stop_entered.wait, 2.0)

            # This synchronous fallback runs on another thread while restart is
            # suspended. It must win permanently rather than allowing restart
            # to re-arm `_stopped` and spawn after shutdown.
            await asyncio.to_thread(server.stop)
            release_restart_stop.set()

            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(restart, timeout=2.0)

        assert server.running is False

    async def test_concurrent_restarts_are_serialized_by_task(
        self, tmp_path: Path
    ) -> None:
        """A second asyncio task cannot enter while the first restart awaits."""
        config_dir = tmp_path / "runtime"
        config_dir.mkdir()
        (config_dir / "langgraph.json").write_text("{}")

        server = ServerProcess(config_dir=config_dir, owns_config_dir=False)
        first_start_entered = asyncio.Event()
        release_first_start = asyncio.Event()
        stop_calls = 0
        start_calls = 0

        def controlled_stop_process() -> None:
            nonlocal stop_calls
            stop_calls += 1

        async def controlled_start(**_: object) -> None:
            nonlocal start_calls
            start_calls += 1
            if start_calls == 1:
                first_start_entered.set()
                await release_first_start.wait()

        with (
            patch.object(server, "_stop_process", new=controlled_stop_process),
            patch.object(server, "_start", new=controlled_start),
        ):
            first = asyncio.create_task(server.restart())
            await first_start_entered.wait()
            second = asyncio.create_task(server.restart())
            await asyncio.sleep(0)

            assert stop_calls == 1
            assert start_calls == 1
            assert not second.done()

            release_first_start.set()
            await asyncio.gather(first, second)

        assert stop_calls == 2
        assert start_calls == 2

    async def test_persistent_env_applies_to_later_restarts(
        self, tmp_path: Path
    ) -> None:
        """Persistent env overrides should apply without restaging."""
        config_dir = tmp_path / "runtime"
        config_dir.mkdir()
        (config_dir / "langgraph.json").write_text("{}")

        first_process = MagicMock()
        first_process.pid = 1234
        first_process.poll.return_value = None
        second_process = MagicMock()
        second_process.pid = 5678
        second_process.poll.return_value = None

        first_log_file = MagicMock()
        first_log_file.name = str(tmp_path / "server-1.log")
        second_log_file = MagicMock()
        second_log_file.name = str(tmp_path / "server-2.log")

        env_key = "DEEPAGENTS_CODE_SERVER_RUBRIC_MAX_ITERATIONS"
        server = ServerProcess(config_dir=config_dir, owns_config_dir=False)
        server.persist_env(**{env_key: "12"})

        with (
            patch(
                "deepagents_code.client.launch.server._find_free_port",
                return_value=12345,
            ),
            patch(
                "deepagents_code.client.launch.server._port_in_use", return_value=False
            ),
            patch(
                "deepagents_code.client.launch.server.tempfile.NamedTemporaryFile",
                side_effect=[first_log_file, second_log_file],
            ),
            patch(
                "deepagents_code.client.launch.server.subprocess.Popen",
                side_effect=[first_process, second_process],
            ) as popen,
            patch(
                "deepagents_code.client.launch.server.wait_for_server_healthy",
                new=AsyncMock(),
            ),
        ):
            await server.start()
            assert server._env_overrides == {}

            await server.restart()

        first_env = popen.call_args_list[0].kwargs["env"]
        second_env = popen.call_args_list[1].kwargs["env"]
        assert first_env[env_key] == "12"
        assert second_env[env_key] == "12"
        assert server._env_overrides == {}

    async def test_one_shot_override_wins_over_persisted(self, tmp_path: Path) -> None:
        """A one-shot `update_env` must override a persisted default on restart.

        Regression: persisting a value (via `persist_env`) and then staging a
        different value (via `update_env`) before a restart must launch the
        subprocess with the freshly staged value, not the stale persisted one.
        """
        config_dir = tmp_path / "runtime"
        config_dir.mkdir()
        (config_dir / "langgraph.json").write_text("{}")

        first_process = MagicMock()
        first_process.pid = 1234
        first_process.poll.return_value = None
        second_process = MagicMock()
        second_process.pid = 5678
        second_process.poll.return_value = None
        first_log_file = MagicMock()
        first_log_file.name = str(tmp_path / "server-1.log")
        second_log_file = MagicMock()
        second_log_file.name = str(tmp_path / "server-2.log")

        env_key = "DEEPAGENTS_CODE_SERVER_RUBRIC_MAX_ITERATIONS"
        server = ServerProcess(config_dir=config_dir, owns_config_dir=False)
        server.persist_env(**{env_key: "10"})

        with (
            patch(
                "deepagents_code.client.launch.server._find_free_port",
                return_value=12345,
            ),
            patch(
                "deepagents_code.client.launch.server._port_in_use", return_value=False
            ),
            patch(
                "deepagents_code.client.launch.server.tempfile.NamedTemporaryFile",
                side_effect=[first_log_file, second_log_file],
            ),
            patch(
                "deepagents_code.client.launch.server.subprocess.Popen",
                side_effect=[first_process, second_process],
            ) as popen,
            patch(
                "deepagents_code.client.launch.server.wait_for_server_healthy",
                new=AsyncMock(),
            ),
        ):
            await server.start()
            # Stage a different value for the next restart only.
            server.update_env(**{env_key: "12"})
            await server.restart()

        # The restart that applies the staged override must use it, not the
        # persisted default it temporarily supersedes.
        second_env = popen.call_args_list[1].kwargs["env"]
        assert second_env[env_key] == "12"

    async def test_restart_rollback_on_failure(self, tmp_path: Path) -> None:
        """Env overrides are rolled back when restart fails."""
        config_dir = tmp_path / "runtime"
        config_dir.mkdir()
        (config_dir / "langgraph.json").write_text("{}")

        process = MagicMock()
        process.pid = 1234
        process.poll.return_value = None

        server = ServerProcess(config_dir=config_dir, owns_config_dir=False)
        server._process = process  # simulate already started

        old_value = os.environ.get("DEEPAGENTS_CODE_SERVER_MODEL")

        async def failing_start(**_: object) -> None:
            await asyncio.sleep(0)
            msg = "restart failed"
            raise RuntimeError(msg)

        server._start = failing_start  # ty: ignore
        server.update_env(DEEPAGENTS_CODE_SERVER_MODEL="should-be-rolled-back")

        with pytest.raises(RuntimeError, match="restart failed"):
            await server.restart()

        # Env should be rolled back
        assert os.environ.get("DEEPAGENTS_CODE_SERVER_MODEL") == old_value
        # Overrides NOT cleared (available for retry)
        assert "DEEPAGENTS_CODE_SERVER_MODEL" in server._env_overrides


class TestServerProcessStopIdempotency:
    """`stop()` must be idempotent and safe against concurrent callers.

    The interactive shutdown path may invoke `stop()` from two places — the
    coordinated teardown task (offloaded to a worker thread) and the outer
    `finally` fallback in `run_textual_app` — so process teardown and resource
    cleanup must run exactly once and never interleave.
    """

    def test_stop_is_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Repeated `stop()` calls tear the process down only once."""
        server = ServerProcess(host="127.0.0.1", port=2024)
        calls: list[int] = []
        monkeypatch.setattr(server, "_stop_process_locked", lambda: calls.append(1))

        server.stop()
        server.stop()
        server.stop()

        assert calls == [1]

    def test_concurrent_stop_runs_teardown_once(self) -> None:
        """Two threads calling `stop()` at once run teardown exactly once.

        The first caller holds the lock across `_stop_process`; the second
        blocks on it and then short-circuits on the `_stopped` flag instead of
        re-running teardown, so there is no double cleanup and no interleave.
        """
        server = ServerProcess(host="127.0.0.1", port=2024)
        entered = threading.Event()
        release = threading.Event()
        calls: list[int] = []

        def slow_stop_process() -> None:
            calls.append(1)
            entered.set()
            release.wait(timeout=2.0)

        server._stop_process_locked = slow_stop_process  # ty: ignore[invalid-assignment]

        first = threading.Thread(target=server.stop)
        second = threading.Thread(target=server.stop)
        first.start()
        # Wait until the first caller is inside the locked teardown, then start
        # the second so it is guaranteed to contend for the lock.
        assert entered.wait(timeout=2.0)
        second.start()
        release.set()
        first.join(timeout=2.0)
        second.join(timeout=2.0)

        assert calls == [1]
        assert server._stopped is True

    async def test_start_rearms_stop_guard(self) -> None:
        """A fresh `start()` clears the idempotency guard set by `stop()`."""
        server = ServerProcess(host="127.0.0.1", port=2024)
        server.stop()
        assert server._stopped is True

        # `start()` re-arms the guard before doing any real work; the missing
        # langgraph.json then aborts startup, which is fine for this assertion.
        with contextlib.suppress(RuntimeError):
            await server.start()

        assert server._stopped is False


class TestPreservedLogNotice:
    """The `Server log preserved at:` notice must survive TUI teardown.

    `stop()` can run while Textual still owns the alternate screen (the
    coordinated teardown stops the server before `super().exit()` restores the
    terminal), so a stderr print inside teardown is discarded. `stop()` queues
    the preserved path onto a process-global list and
    `emit_preserved_log_notices()` prints every queued path later, once the
    terminal is restored. Regression: PR #4831.
    """

    @pytest.fixture(autouse=True)
    def _clear_pending_logs(self) -> Iterator[None]:
        """Isolate the process-global queue from other tests in both directions."""
        server_module._PENDING_PRESERVED_LOGS.clear()
        yield
        server_module._PENDING_PRESERVED_LOGS.clear()

    @staticmethod
    def _make_stopped_server(log_path: Path) -> ServerProcess:
        """Build a server wired to an already-exited process and real log file."""
        log_path.write_text("booting", encoding="utf-8")
        log_file = MagicMock()
        log_file.name = str(log_path)

        process = MagicMock()
        process.poll.return_value = 0  # already exited: skip termination

        server = ServerProcess(host="127.0.0.1", port=2024)
        server._process = process
        server._log_file = log_file
        return server

    def test_stop_queues_path_without_printing_when_debug_on(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """With debug on, `stop()` preserves the log and defers the notice."""
        monkeypatch.setenv("DEEPAGENTS_CODE_DEBUG", "1")
        log_path = tmp_path / "server.log"
        server = self._make_stopped_server(log_path)

        server.stop()

        assert list(server_module._PENDING_PRESERVED_LOGS) == [log_path]
        assert log_path.exists()
        assert server._log_file is None
        assert "Server log preserved at:" not in capsys.readouterr().err

    def test_emit_prints_queued_path_once(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """`emit_preserved_log_notices()` prints the path once, then drains it."""
        monkeypatch.setenv("DEEPAGENTS_CODE_DEBUG", "1")
        log_path = tmp_path / "server.log"
        server = self._make_stopped_server(log_path)
        server.stop()

        emit_preserved_log_notices()

        first = capsys.readouterr().err
        assert f"Server log preserved at: {log_path}" in first
        assert server_module._PENDING_PRESERVED_LOGS == []

        # A second call is a no-op: nothing is printed and no error is raised.
        emit_preserved_log_notices()
        assert "Server log preserved at:" not in capsys.readouterr().err

    def test_emit_prints_every_queued_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Restarts queue multiple paths; the drain announces each of them.

        A single restart reuses one `ServerProcess`, so successive `stop()`s
        must not clobber earlier preserved paths (PR #4999 review): every log
        stays announceable until the post-terminal drain.
        """
        monkeypatch.setenv("DEEPAGENTS_CODE_DEBUG", "1")
        first_log = tmp_path / "server-1.log"
        second_log = tmp_path / "server-2.log"
        server = self._make_stopped_server(first_log)

        server.stop()
        # Simulate the restart wiring a fresh process + log onto the same server.
        second_log.write_text("booting", encoding="utf-8")
        log_file = MagicMock()
        log_file.name = str(second_log)
        process = MagicMock()
        process.poll.return_value = 0
        server._process = process
        server._log_file = log_file
        server._stopped = False
        server.stop()

        emit_preserved_log_notices()

        err = capsys.readouterr().err
        assert f"Server log preserved at: {first_log}" in err
        assert f"Server log preserved at: {second_log}" in err

    def test_no_notice_when_debug_off(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """With debug off, the log is unlinked and no path is queued."""
        monkeypatch.delenv("DEEPAGENTS_CODE_DEBUG", raising=False)
        log_path = tmp_path / "server.log"
        server = self._make_stopped_server(log_path)

        server.stop()

        assert server_module._PENDING_PRESERVED_LOGS == []
        assert not log_path.exists()

        emit_preserved_log_notices()
        assert "Server log preserved at:" not in capsys.readouterr().err

    def test_stop_then_emit_surfaces_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The terminal stop-then-emit sequence surfaces the preserved path.

        Guards the #4831 ordering: stopping the server (during teardown, when a
        print would be swallowed) then emitting after the terminal is restored
        yields exactly one visible line.
        """
        monkeypatch.setenv("DEEPAGENTS_CODE_DEBUG", "1")
        log_path = tmp_path / "server.log"
        server = self._make_stopped_server(log_path)

        server.stop()
        assert "Server log preserved at:" not in capsys.readouterr().err
        emit_preserved_log_notices()

        assert f"Server log preserved at: {log_path}" in capsys.readouterr().err


class TestServerSessionIsolation:
    """Tests that `start()` detaches the server from dcode's terminal."""

    @staticmethod
    def _make_server(tmp_path: Path) -> ServerProcess:
        config_dir = tmp_path / "runtime"
        config_dir.mkdir()
        (config_dir / "langgraph.json").write_text("{}")
        return ServerProcess(config_dir=config_dir)

    @staticmethod
    def _mock_log_file(tmp_path: Path) -> MagicMock:
        log_file = MagicMock()
        log_file.name = str(tmp_path / "server.log")
        return log_file

    async def _spawn_and_capture(
        self, tmp_path: Path, platform: str, monkeypatch: pytest.MonkeyPatch
    ) -> MagicMock:
        """Run `start()` on a patched platform and return the `Popen` mock."""
        monkeypatch.setattr(
            "deepagents_code.client.launch.server.sys.platform", platform
        )
        server = self._make_server(tmp_path)
        process = MagicMock(pid=4321)
        process.poll.return_value = None
        popen = MagicMock(return_value=process)
        with (
            patch(
                "deepagents_code.client.launch.server.tempfile.NamedTemporaryFile",
                return_value=self._mock_log_file(tmp_path),
            ),
            patch("deepagents_code.client.launch.server.subprocess.Popen", popen),
            patch(
                "deepagents_code.client.launch.server.wait_for_server_healthy",
                new=AsyncMock(),
            ),
            patch(
                "deepagents_code.client.launch.server._find_free_port",
                return_value=43210,
            ),
            patch("deepagents_code.client.launch.server._port_in_use"),
        ):
            await server.start()
        return popen

    async def test_posix_spawn_starts_new_session(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On POSIX the server is spawned in its own session/process group."""
        popen = await self._spawn_and_capture(tmp_path, "linux", monkeypatch)
        assert popen.call_args.kwargs["start_new_session"] is True

    async def test_windows_spawn_suspended_without_new_session(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Windows suspends the root before Job assignment and skips `setsid()`."""
        from deepagents_code.update_check import _WINDOWS_CREATE_SUSPENDED

        popen = await self._spawn_and_capture(tmp_path, "win32", monkeypatch)
        assert popen.call_args.kwargs["start_new_session"] is False
        assert popen.call_args.kwargs["creationflags"] == _WINDOWS_CREATE_SUSPENDED


@pytest.mark.usefixtures("_posix_process_api")
class TestPosixServerOwnership:
    """Tests for durable POSIX process-group ownership."""

    def test_spawn_arms_owner_before_opening_exec_gate(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The server command cannot execute before its watchdog is ready."""
        events: list[str] = []
        process = MagicMock(pid=4321)
        owner = MagicMock()
        popen = MagicMock(
            side_effect=lambda *_args, **_kwargs: events.append("spawn") or process
        )

        def start_owner(process_group_id: int, *, term_timeout: float) -> MagicMock:
            assert process_group_id == 4321
            assert term_timeout > 0
            events.append("owner")
            return owner

        def open_gate(file_descriptor: int, payload: bytes) -> int:
            assert file_descriptor == 42
            assert payload == b"G"
            events.append("gate")
            return 1

        monkeypatch.setattr(
            "deepagents_code.client.launch.server.subprocess.Popen",
            popen,
        )
        monkeypatch.setattr(
            "deepagents_code._posix_shell._start_sync_posix_owner_guard",
            start_owner,
        )
        monkeypatch.setattr(
            "deepagents_code.client.launch.server.os.pipe",
            lambda: (41, 42),
        )
        monkeypatch.setattr(
            "deepagents_code.client.launch.server.os.write",
            open_gate,
        )
        monkeypatch.setattr(
            "deepagents_code.client.launch.server._close_posix_file_descriptor",
            lambda _file_descriptor: None,
        )

        spawned, retained_owner = _spawn_posix_server_process(
            ["python", "-m", "langgraph_cli"],
            cwd="runtime",
            env={"PATH": "test"},
            stdout=subprocess.DEVNULL,
        )

        assert events == ["spawn", "owner", "gate"]
        assert spawned is process
        assert retained_owner is owner
        assert popen.call_args.kwargs["start_new_session"] is True
        assert popen.call_args.kwargs["pass_fds"] == (41,)
        assert popen.call_args.args[0][-3:] == [
            "python",
            "-m",
            "langgraph_cli",
        ]

    def test_spawn_interruption_after_gate_terminates_owner(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An interruption after exec is allowed still tears down the group."""
        process = MagicMock(pid=4321)
        owner = MagicMock()
        owner.terminate.return_value = True
        monkeypatch.setattr(
            "deepagents_code.client.launch.server.subprocess.Popen",
            MagicMock(return_value=process),
        )
        monkeypatch.setattr(
            "deepagents_code._posix_shell._start_sync_posix_owner_guard",
            MagicMock(return_value=owner),
        )
        monkeypatch.setattr(
            "deepagents_code.client.launch.server.os.pipe",
            lambda: (41, 42),
        )
        monkeypatch.setattr(
            "deepagents_code.client.launch.server._close_posix_file_descriptor",
            lambda _file_descriptor: None,
        )
        monkeypatch.setattr(
            "deepagents_code.client.launch.server._open_posix_server_launch_gate",
            MagicMock(side_effect=KeyboardInterrupt),
        )

        with pytest.raises(KeyboardInterrupt):
            _spawn_posix_server_process(
                ["python", "-m", "langgraph_cli"],
                cwd="runtime",
                env={"PATH": "test"},
                stdout=subprocess.DEVNULL,
            )

        owner.terminate.assert_called_once_with(reap=process.poll)
        owner.close.assert_not_called()
        process.wait.assert_called_once()

    def test_dead_owner_never_signals_reused_process_group(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A completed watchdog prevents fallback to the exited root's PGID."""
        monkeypatch.setattr(
            "deepagents_code.client.launch.server.sys.platform",
            "linux",
        )
        process = MagicMock(pid=4321)
        process.poll.return_value = 7
        watchdog = MagicMock()
        watchdog.poll.return_value = 0
        watchdog.wait.return_value = 0
        owner = _SyncPosixOwnerGuard(93, watchdog, wait_timeout=0)

        with (
            patch(
                "deepagents_code._posix_shell._close_file_descriptor",
            ) as close_control,
            patch(
                "deepagents_code._posix_shell.os.write",
            ) as write_control,
            patch(
                "deepagents_code.client.launch.server._server_process_group",
            ) as resolve_group,
            patch("os.killpg") as killpg,
        ):
            _terminate_server_process(process, posix_owner=owner)

        close_control.assert_called_once_with(93)
        write_control.assert_not_called()
        resolve_group.assert_not_called()
        killpg.assert_not_called()
        process.send_signal.assert_not_called()
        process.kill.assert_not_called()

    def test_stop_uses_owner_after_root_exit(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An exited root does not suppress owned descendant teardown."""
        monkeypatch.setattr(
            "deepagents_code.client.launch.server.sys.platform",
            "linux",
        )
        server = ServerProcess()
        process = MagicMock(pid=4321)
        process.poll.return_value = 7
        owner = MagicMock()
        owner.terminate.return_value = True
        server._process = process
        server._posix_owner = owner

        server._stop_process()

        owner.terminate.assert_called_once_with(reap=process.poll)
        process.send_signal.assert_not_called()
        process.kill.assert_not_called()
        assert server._process is None
        assert server._posix_owner is None

    def test_repeated_stop_terminates_owner_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Repeated terminal shutdown consumes the ownership token once."""
        monkeypatch.setattr(
            "deepagents_code.client.launch.server.sys.platform",
            "linux",
        )
        server = ServerProcess()
        process = MagicMock(pid=4321)
        process.poll.return_value = 7
        owner = MagicMock()
        owner.terminate.return_value = True
        server._process = process
        server._posix_owner = owner

        server.stop()
        server.stop()

        owner.terminate.assert_called_once_with(reap=process.poll)

    @pytest.mark.parametrize(
        ("failure", "error_type"),
        [
            pytest.param(
                RuntimeError("Server did not become healthy within 0.01s"),
                RuntimeError,
                id="health-timeout",
            ),
            pytest.param(
                asyncio.CancelledError(),
                asyncio.CancelledError,
                id="cancellation",
            ),
        ],
    )
    async def test_failed_start_terminates_owner(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        failure: BaseException,
        error_type: type[BaseException],
    ) -> None:
        """Health timeout and cancellation both consume POSIX ownership."""
        monkeypatch.setattr(
            "deepagents_code.client.launch.server.sys.platform",
            "linux",
        )
        config_dir = tmp_path / "runtime"
        config_dir.mkdir()
        (config_dir / "langgraph.json").write_text("{}")
        process = MagicMock(pid=4321)
        process.poll.return_value = None
        owner = MagicMock()

        def terminate(*, reap: object) -> bool:
            assert reap == process.poll
            process.poll.return_value = 7
            return True

        owner.terminate.side_effect = terminate
        log_file = MagicMock()
        log_file.name = str(tmp_path / "server.log")
        server = ServerProcess(config_dir=config_dir)

        monkeypatch.setattr(
            "deepagents_code.client.launch.server._spawn_posix_server_process",
            lambda *_args, **_kwargs: (process, owner),
        )
        with (
            patch(
                "deepagents_code.client.launch.server._find_free_port",
                return_value=12345,
            ),
            patch(
                "deepagents_code.client.launch.server.tempfile.NamedTemporaryFile",
                return_value=log_file,
            ),
            patch(
                "deepagents_code.client.launch.server.wait_for_server_healthy",
                new=AsyncMock(side_effect=failure),
            ),
            pytest.raises(error_type),
        ):
            await server.start()

        owner.terminate.assert_called_once_with(reap=process.poll)
        assert server._process is None
        assert server._posix_owner is None


@pytest.mark.usefixtures("_posix_process_api")
class TestServerProcessGroup:
    """Tests for `_server_process_group` targeting logic."""

    def test_returns_none_on_windows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Windows has no POSIX process groups, so signaling targets the root."""
        monkeypatch.setattr(
            "deepagents_code.client.launch.server.sys.platform", "win32"
        )
        assert _server_process_group(4321) is None

    def test_returns_none_when_process_gone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A vanished process yields no group to signal."""
        monkeypatch.setattr(
            "deepagents_code.client.launch.server.sys.platform", "linux"
        )

        def _raise(_pid: int) -> int:
            raise ProcessLookupError

        monkeypatch.setattr("deepagents_code.client.launch.server.os.getpgid", _raise)
        assert _server_process_group(4321) is None

    def test_returns_none_when_not_group_leader(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A process that does not lead its own group is not group-signaled."""
        monkeypatch.setattr(
            "deepagents_code.client.launch.server.sys.platform", "linux"
        )
        # pgid (5000) != pid (4321): the server shares another group.
        monkeypatch.setattr(
            "deepagents_code.client.launch.server.os.getpgid",
            lambda pid: 5000 if pid == 4321 else 999,
        )
        assert _server_process_group(4321) is None

    def test_returns_none_when_group_is_dcodes_own(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The server's group is never signaled when it equals dcode's own."""
        monkeypatch.setattr(
            "deepagents_code.client.launch.server.sys.platform", "linux"
        )
        # Both the server and dcode (pid 0) resolve to the same group.
        monkeypatch.setattr(
            "deepagents_code.client.launch.server.os.getpgid",
            lambda _pid: 4321,
        )
        assert _server_process_group(4321) is None

    def test_returns_dedicated_group(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A dedicated leader group distinct from dcode's is returned."""
        monkeypatch.setattr(
            "deepagents_code.client.launch.server.sys.platform", "linux"
        )
        monkeypatch.setattr(
            "deepagents_code.client.launch.server.os.getpgid",
            lambda pid: 4321 if pid == 4321 else 999,
        )
        assert _server_process_group(4321) == 4321

    def test_unexpected_getpgid_error_warns_and_falls_back(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A non-`ProcessLookupError` failure falls back to root but is logged."""
        monkeypatch.setattr(
            "deepagents_code.client.launch.server.sys.platform", "linux"
        )

        def _raise(_pid: int) -> int:
            raise PermissionError

        monkeypatch.setattr("deepagents_code.client.launch.server.os.getpgid", _raise)

        with caplog.at_level(logging.WARNING):
            assert _server_process_group(4321) is None

        assert "falling back to root-only signaling" in caplog.text


@pytest.mark.usefixtures("_posix_process_api")
class TestTerminateServerProcess:
    """Tests for detached process-group teardown."""

    @staticmethod
    def _own_group_process() -> MagicMock:
        process = MagicMock()
        process.pid = 4321
        process.poll.return_value = None
        return process

    @staticmethod
    def _patch_own_group(monkeypatch: pytest.MonkeyPatch) -> None:
        """Make pid 4321 a dedicated group leader distinct from dcode's."""
        monkeypatch.setattr(
            "deepagents_code.client.launch.server.sys.platform", "linux"
        )
        monkeypatch.setattr(
            "deepagents_code.client.launch.server.os.getpgid",
            lambda pid: 4321 if pid == 4321 else 999,
        )

    @staticmethod
    def _gone_after_signal(_pgid: int, sig: int) -> None:
        """Model a group that disappears after receiving its real signal."""
        if sig == 0:
            raise ProcessLookupError

    def test_graceful_group_shutdown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SIGTERM is sent to the whole group and the process exits gracefully."""
        self._patch_own_group(monkeypatch)
        killpg = MagicMock(side_effect=self._gone_after_signal)
        monkeypatch.setattr("deepagents_code.client.launch.server.os.killpg", killpg)
        process = self._own_group_process()

        _terminate_server_process(process)

        assert killpg.call_args_list == [
            ((4321, signal.SIGTERM),),
            ((4321, 0),),
        ]
        process.wait.assert_called_once()
        process.send_signal.assert_not_called()
        process.kill.assert_not_called()

    def test_timeout_escalates_to_group_sigkill(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A hung group is escalated to a group SIGKILL after the timeout."""
        self._patch_own_group(monkeypatch)
        group_killed = False

        def _killpg(_pgid: int, sig: int) -> None:
            nonlocal group_killed
            if sig == _SIGKILL:
                group_killed = True
            elif sig == 0 and group_killed:
                raise ProcessLookupError

        killpg = MagicMock(side_effect=_killpg)
        monkeypatch.setattr("deepagents_code.client.launch.server.os.killpg", killpg)
        monkeypatch.setattr("deepagents_code.client.launch.server._SHUTDOWN_TIMEOUT", 0)
        process = self._own_group_process()

        _terminate_server_process(process)

        assert killpg.call_args_list == [
            ((4321, signal.SIGTERM),),
            ((4321, 0),),
            ((4321, _SIGKILL),),
            ((4321, 0),),
        ]
        process.wait.assert_called_once_with()
        process.kill.assert_not_called()

    def test_group_wait_ignores_exited_leader(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A group remains live while a descendant survives its leader."""
        probes = iter([None, ProcessLookupError()])

        def _probe(_pgid: int, sig: int) -> None:
            assert sig == 0
            result = next(probes)
            if isinstance(result, BaseException):
                raise result

        monkeypatch.setattr("deepagents_code.client.launch.server.os.killpg", _probe)
        monkeypatch.setattr(
            "deepagents_code.client.launch.server.time.sleep", lambda _: None
        )
        process = self._own_group_process()
        process.poll.return_value = 0

        assert _wait_for_process_group_exit(process, 4321, 1) is True

        # Reaping the leader does not end the group-level wait while a
        # descendant keeps the first probe alive.
        assert process.poll.call_count == 2
        process.wait.assert_called_once_with()

    def test_group_wait_reaps_leader_before_first_probe(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A zombie leader cannot make an otherwise empty group look alive."""
        leader_reaped = False

        def _poll() -> int:
            nonlocal leader_reaped
            leader_reaped = True
            return 0

        def _probe(_pgid: int, sig: int) -> None:
            assert sig == 0
            if leader_reaped:
                raise ProcessLookupError

        monkeypatch.setattr("deepagents_code.client.launch.server.os.killpg", _probe)
        process = self._own_group_process()
        process.poll.side_effect = _poll

        assert _wait_for_process_group_exit(process, 4321, 0) is True

        process.poll.assert_called_once_with()
        process.wait.assert_called_once_with()

    def test_group_wait_permission_error_probe_keeps_waiting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An EPERM probe means "still alive", so the wait must not stop early."""
        probes = iter([PermissionError(), ProcessLookupError()])

        def _probe(_pgid: int, sig: int) -> None:
            assert sig == 0
            raise next(probes)

        monkeypatch.setattr("deepagents_code.client.launch.server.os.killpg", _probe)
        monkeypatch.setattr(
            "deepagents_code.client.launch.server.time.sleep", lambda _: None
        )
        process = self._own_group_process()

        # First probe raises EPERM (unsignalable but live); only the second,
        # which proves the group is gone, ends the wait successfully.
        assert _wait_for_process_group_exit(process, 4321, 1) is True
        process.wait.assert_called_once_with()

    def test_never_signals_dcodes_process_group(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the server shares dcode's group, only the root proc is signaled."""
        monkeypatch.setattr(
            "deepagents_code.client.launch.server.sys.platform", "linux"
        )
        # Server pid and dcode (pid 0) resolve to the same group id.
        monkeypatch.setattr(
            "deepagents_code.client.launch.server.os.getpgid",
            lambda _pid: 4321,
        )
        killpg = MagicMock()
        monkeypatch.setattr("deepagents_code.client.launch.server.os.killpg", killpg)
        process = self._own_group_process()

        _terminate_server_process(process)

        killpg.assert_not_called()
        process.send_signal.assert_called_once_with(signal.SIGTERM)

    def test_windows_signals_root_process_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On Windows only the root process is signaled (no `killpg`)."""
        monkeypatch.setattr(
            "deepagents_code.client.launch.server.sys.platform", "win32"
        )
        process = self._own_group_process()
        process.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="langgraph", timeout=3),
            None,
        ]

        _terminate_server_process(process)

        process.send_signal.assert_called_once_with(signal.SIGTERM)
        process.kill.assert_called_once_with()

    def test_initial_sigterm_process_lookup_is_benign(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A group that vanishes before SIGTERM is not escalated to SIGKILL."""
        self._patch_own_group(monkeypatch)

        def _killpg(_pgid: int, _sig: int) -> None:
            raise ProcessLookupError

        killpg = MagicMock(side_effect=_killpg)
        monkeypatch.setattr("deepagents_code.client.launch.server.os.killpg", killpg)
        process = self._own_group_process()

        _terminate_server_process(process)

        killpg.assert_called_once_with(4321, signal.SIGTERM)
        process.kill.assert_not_called()

    def test_initial_sigterm_oserror_reports_orphan(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An undeliverable SIGTERM leaves the group running and is reported."""
        self._patch_own_group(monkeypatch)

        def _killpg(_pgid: int, _sig: int) -> None:
            raise PermissionError

        killpg = MagicMock(side_effect=_killpg)
        monkeypatch.setattr("deepagents_code.client.launch.server.os.killpg", killpg)
        process = self._own_group_process()

        with caplog.at_level(logging.ERROR):
            _terminate_server_process(process)

        killpg.assert_called_once_with(4321, signal.SIGTERM)
        process.kill.assert_not_called()
        assert "may be orphaned" in caplog.text

    def test_sigkill_process_lookup_is_benign(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A group that exits just before the escalation SIGKILL is benign."""
        self._patch_own_group(monkeypatch)

        def _killpg(_pgid: int, sig: int) -> None:
            if sig == _SIGKILL:
                raise ProcessLookupError

        killpg = MagicMock(side_effect=_killpg)
        monkeypatch.setattr("deepagents_code.client.launch.server.os.killpg", killpg)
        monkeypatch.setattr("deepagents_code.client.launch.server._SHUTDOWN_TIMEOUT", 0)
        process = self._own_group_process()

        _terminate_server_process(process)

        assert killpg.call_args_list == [
            ((4321, signal.SIGTERM),),
            ((4321, 0),),
            ((4321, _SIGKILL),),
        ]

    def test_sigkill_oserror_reports_orphan(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A failed escalation SIGKILL surfaces the orphan risk."""
        self._patch_own_group(monkeypatch)

        def _killpg(_pgid: int, sig: int) -> None:
            if sig == _SIGKILL:
                raise PermissionError

        killpg = MagicMock(side_effect=_killpg)
        monkeypatch.setattr("deepagents_code.client.launch.server.os.killpg", killpg)
        monkeypatch.setattr("deepagents_code.client.launch.server._SHUTDOWN_TIMEOUT", 0)
        process = self._own_group_process()

        with caplog.at_level(logging.ERROR):
            _terminate_server_process(process)

        assert "may be orphaned" in caplog.text

    def test_sigkill_group_probe_timeout_warns(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A group that outlives SIGKILL is reported, not silently dropped."""
        self._patch_own_group(monkeypatch)
        # The group never disappears: every liveness probe reports it alive.
        killpg = MagicMock(return_value=None)
        monkeypatch.setattr("deepagents_code.client.launch.server.os.killpg", killpg)
        monkeypatch.setattr("deepagents_code.client.launch.server._SHUTDOWN_TIMEOUT", 0)
        monkeypatch.setattr("deepagents_code.client.launch.server._SIGKILL_TIMEOUT", 0)
        process = self._own_group_process()

        with caplog.at_level(logging.WARNING):
            _terminate_server_process(process)

        assert killpg.call_args_list == [
            ((4321, signal.SIGTERM),),
            ((4321, 0),),
            ((4321, _SIGKILL),),
            ((4321, 0),),
        ]
        assert "did not exit after SIGKILL" in caplog.text

    def test_windows_sigkill_timeout_warns(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """On Windows a process that ignores SIGKILL is reported after the wait."""
        monkeypatch.setattr(
            "deepagents_code.client.launch.server.sys.platform", "win32"
        )
        process = self._own_group_process()
        process.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="langgraph", timeout=3),
            subprocess.TimeoutExpired(cmd="langgraph", timeout=2),
        ]

        with caplog.at_level(logging.WARNING):
            _terminate_server_process(process)

        process.kill.assert_called_once_with()
        assert "did not exit after SIGKILL" in caplog.text

    async def test_startup_failure_terminates_group(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed startup tears down the detached server group, not just root."""
        monkeypatch.delenv("DEEPAGENTS_CODE_DEBUG", raising=False)
        self._patch_own_group(monkeypatch)
        killpg = MagicMock(side_effect=self._gone_after_signal)
        monkeypatch.setattr("deepagents_code.client.launch.server.os.killpg", killpg)

        config_dir = tmp_path / "runtime"
        config_dir.mkdir()
        (config_dir / "langgraph.json").write_text("{}")
        log_file = MagicMock()
        log_file.name = str(tmp_path / "server.log")

        process = self._own_group_process()
        server = ServerProcess(config_dir=config_dir, owns_config_dir=True)

        with (
            patch(
                "deepagents_code.client.launch.server._find_free_port",
                return_value=12345,
            ),
            patch(
                "deepagents_code.client.launch.server.tempfile.NamedTemporaryFile",
                return_value=log_file,
            ),
            patch(
                "deepagents_code.client.launch.server.subprocess.Popen",
                return_value=process,
            ),
            patch(
                "deepagents_code.client.launch.server.wait_for_server_healthy",
                new=AsyncMock(side_effect=RuntimeError("boom")),
            ),
            pytest.raises(RuntimeError, match="boom"),
        ):
            await server.start()

        assert killpg.call_args_list == [
            ((4321, signal.SIGTERM),),
            ((4321, 0),),
        ]
        assert server._process is None

    async def test_restart_terminates_group_and_restarts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Restart tears down the old server group before starting a new one."""
        self._patch_own_group(monkeypatch)
        killpg = MagicMock(side_effect=self._gone_after_signal)
        monkeypatch.setattr("deepagents_code.client.launch.server.os.killpg", killpg)

        config_dir = tmp_path / "runtime"
        config_dir.mkdir()
        (config_dir / "langgraph.json").write_text("{}")

        server = ServerProcess(config_dir=config_dir, owns_config_dir=False)
        server._process = self._own_group_process()

        start_mock = AsyncMock()
        with patch.object(server, "_start", start_mock):
            await server.restart()

        assert killpg.call_args_list == [
            ((4321, signal.SIGTERM),),
            ((4321, 0),),
        ]
        start_mock.assert_awaited_once()
        assert server._process is None

    def test_root_sigterm_process_lookup_is_benign(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On the root-only path a vanished process is not escalated to SIGKILL."""
        monkeypatch.setattr(
            "deepagents_code.client.launch.server.sys.platform", "win32"
        )
        process = self._own_group_process()
        process.send_signal.side_effect = ProcessLookupError

        _terminate_server_process(process)

        process.send_signal.assert_called_once_with(signal.SIGTERM)
        process.kill.assert_not_called()

    def test_root_sigterm_oserror_reports_orphan(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """On the root-only path an undeliverable SIGTERM reports the orphan."""
        monkeypatch.setattr(
            "deepagents_code.client.launch.server.sys.platform", "win32"
        )
        process = self._own_group_process()
        process.send_signal.side_effect = PermissionError

        with caplog.at_level(logging.ERROR):
            _terminate_server_process(process)

        process.send_signal.assert_called_once_with(signal.SIGTERM)
        process.kill.assert_not_called()
        assert "may be orphaned" in caplog.text

    def test_root_sigkill_oserror_reports_orphan(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """On the root-only path a failed escalation SIGKILL reports the orphan."""
        monkeypatch.setattr(
            "deepagents_code.client.launch.server.sys.platform", "win32"
        )
        process = self._own_group_process()
        process.wait.side_effect = subprocess.TimeoutExpired(cmd="langgraph", timeout=3)
        process.kill.side_effect = PermissionError

        with caplog.at_level(logging.ERROR):
            _terminate_server_process(process)

        process.kill.assert_called_once_with()
        assert "may be orphaned" in caplog.text

    def test_stop_process_warns_when_teardown_leaves_process_alive(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A process still alive after teardown surfaces the orphan risk."""
        server = ServerProcess()
        process = MagicMock()
        process.pid = 4321
        # `poll()` never reports exit, modeling teardown that could not kill it.
        process.poll.return_value = None
        server._process = process
        monkeypatch.setattr(
            "deepagents_code.client.launch.server._terminate_server_process",
            MagicMock(),
        )

        with caplog.at_level(logging.WARNING):
            server._stop_process()

        assert "Dropping handle to server pid=4321 that is still running" in caplog.text
        assert server._process is None


_WINDOWS_PROCESS_TEST_TIMEOUT = 10.0
_POSIX_PROCESS_TEST_TIMEOUT = 10.0


def _wait_for_pid_file(path: Path) -> int:
    """Wait for a subprocess to publish a child pid."""
    deadline = time.monotonic() + _WINDOWS_PROCESS_TEST_TIMEOUT
    while time.monotonic() < deadline:
        if path.exists():
            try:
                return int(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        time.sleep(0.02)
    msg = f"Timed out waiting for pid file: {path}"
    raise AssertionError(msg)


def _posix_root_exits_with_child_script(
    pid_file: Path,
    terminated_file: Path,
) -> str:
    """Build a POSIX root that exits after starting a long-lived child."""
    child_code = (
        "import os, signal, time\n"
        "from pathlib import Path\n"
        f"pid_file = Path({str(pid_file)!r})\n"
        f"terminated_file = Path({str(terminated_file)!r})\n"
        "def stop(_signal, _frame):\n"
        "    terminated_file.write_text('terminated', encoding='utf-8')\n"
        "    raise SystemExit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "pid_file.write_text(str(os.getpid()), encoding='utf-8')\n"
        "while True:\n"
        "    time.sleep(1)\n"
    )
    return (
        "import subprocess, sys, time\n"
        "from pathlib import Path\n"
        f"pid_file = Path({str(pid_file)!r})\n"
        "child = subprocess.Popen(\n"
        f"    [sys.executable, '-c', {child_code!r}],\n"
        "    stdin=subprocess.DEVNULL,\n"
        "    stdout=subprocess.DEVNULL,\n"
        "    stderr=subprocess.DEVNULL,\n"
        ")\n"
        f"deadline = time.monotonic() + {_POSIX_PROCESS_TEST_TIMEOUT!r}\n"
        "while not pid_file.exists() and time.monotonic() < deadline:\n"
        "    time.sleep(0.02)\n"
        "if not pid_file.exists():\n"
        "    child.kill()\n"
        "    child.wait()\n"
        "    raise SystemExit(8)\n"
        "raise SystemExit(7)\n"
    )


def _posix_process_is_running(pid: int) -> bool:
    """Return whether a POSIX test process is live rather than a zombie."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

    stat_path = Path("/proc") / str(pid) / "stat"
    try:
        return stat_path.read_text(encoding="utf-8").split()[2] != "Z"
    except (IndexError, OSError):
        return True


def _wait_for_posix_process_exit(pid: int) -> bool:
    """Wait for a real POSIX process to exit or become a zombie."""
    deadline = time.monotonic() + _POSIX_PROCESS_TEST_TIMEOUT
    while time.monotonic() < deadline:
        if not _posix_process_is_running(pid):
            return True
        time.sleep(0.02)
    return False


def _force_stop_posix_test_process(pid: int | None) -> None:
    """Best-effort cleanup for a POSIX child if an assertion aborts a test."""
    if pid is None or not _posix_process_is_running(pid):
        return
    with contextlib.suppress(OSError, ProcessLookupError):
        os.kill(pid, _SIGKILL)
    _wait_for_posix_process_exit(pid)


@pytest.mark.skipif(sys.platform == "win32", reason="requires POSIX process groups")
@pytest.mark.usefixtures("_real_posix_server_lifecycle")
class TestPosixServerProcessTree:
    """Real POSIX coverage for server roots that exit before descendants."""

    @staticmethod
    def _make_server(tmp_path: Path) -> ServerProcess:
        config_dir = tmp_path / "runtime"
        config_dir.mkdir()
        (config_dir / "langgraph.json").write_text("{}", encoding="utf-8")
        return ServerProcess(config_dir=config_dir, port=43210)

    @staticmethod
    def _patch_server_command(
        monkeypatch: pytest.MonkeyPatch,
        script: str,
    ) -> None:
        monkeypatch.setattr(
            "deepagents_code.client.launch.server._build_server_cmd",
            lambda *_args, **_kwargs: [sys.executable, "-c", script],
        )
        monkeypatch.setattr(
            "deepagents_code.client.launch.server._port_in_use",
            lambda *_args: False,
        )

    async def test_stop_cleans_child_after_root_exit(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Explicit stop terminates descendants after the root is reaped."""
        pid_file = tmp_path / "posix-child.pid"
        terminated_file = tmp_path / "posix-child.terminated"
        server = self._make_server(tmp_path)
        self._patch_server_command(
            monkeypatch,
            _posix_root_exits_with_child_script(pid_file, terminated_file),
        )
        child_pid: int | None = None

        async def observe_root_exit(
            *_args: object,
            **kwargs: object,
        ) -> None:
            nonlocal child_pid
            process = kwargs["process"]
            assert isinstance(process, subprocess.Popen)
            child_pid = await asyncio.to_thread(_wait_for_pid_file, pid_file)
            await asyncio.to_thread(
                process.wait,
                timeout=_POSIX_PROCESS_TEST_TIMEOUT,
            )
            assert process.returncode == 7
            assert _posix_process_is_running(child_pid)

        monkeypatch.setattr(
            "deepagents_code.client.launch.server.wait_for_server_healthy",
            observe_root_exit,
        )

        try:
            await server.start()
            assert child_pid is not None
            assert server._posix_owner is not None

            server.stop()

            assert _wait_for_posix_process_exit(child_pid)
            assert terminated_file.read_text(encoding="utf-8") == "terminated"
            assert server._process is None
            assert server._posix_owner is None
        finally:
            server.stop()
            _force_stop_posix_test_process(child_pid)

    async def test_repeated_stop_consumes_real_owner_once(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Repeated stop is idempotent after a real root exits early."""
        pid_file = tmp_path / "repeated-posix-child.pid"
        terminated_file = tmp_path / "repeated-posix-child.terminated"
        server = self._make_server(tmp_path)
        self._patch_server_command(
            monkeypatch,
            _posix_root_exits_with_child_script(pid_file, terminated_file),
        )
        child_pid: int | None = None

        async def observe_root_exit(
            *_args: object,
            **kwargs: object,
        ) -> None:
            nonlocal child_pid
            process = kwargs["process"]
            assert isinstance(process, subprocess.Popen)
            child_pid = await asyncio.to_thread(_wait_for_pid_file, pid_file)
            await asyncio.to_thread(
                process.wait,
                timeout=_POSIX_PROCESS_TEST_TIMEOUT,
            )

        monkeypatch.setattr(
            "deepagents_code.client.launch.server.wait_for_server_healthy",
            observe_root_exit,
        )

        try:
            await server.start()
            assert child_pid is not None
            owner = server._posix_owner
            assert owner is not None

            with patch.object(
                owner,
                "terminate",
                wraps=owner.terminate,
            ) as terminate:
                server.stop()
                server.stop()

            terminate.assert_called_once()
            assert _wait_for_posix_process_exit(child_pid)
            assert terminated_file.read_text(encoding="utf-8") == "terminated"
        finally:
            server.stop()
            _force_stop_posix_test_process(child_pid)

    async def test_startup_failure_cleans_child_after_root_exit(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Startup failure tears down descendants after the root exits."""
        pid_file = tmp_path / "failed-posix-child.pid"
        terminated_file = tmp_path / "failed-posix-child.terminated"
        server = self._make_server(tmp_path)
        self._patch_server_command(
            monkeypatch,
            _posix_root_exits_with_child_script(pid_file, terminated_file),
        )
        child_pid: int | None = None

        async def fail_after_root_exit(
            *_args: object,
            **kwargs: object,
        ) -> None:
            nonlocal child_pid
            process = kwargs["process"]
            assert isinstance(process, subprocess.Popen)
            child_pid = await asyncio.to_thread(_wait_for_pid_file, pid_file)
            await asyncio.to_thread(
                process.wait,
                timeout=_POSIX_PROCESS_TEST_TIMEOUT,
            )
            msg = "root exited before readiness"
            raise RuntimeError(msg)

        monkeypatch.setattr(
            "deepagents_code.client.launch.server.wait_for_server_healthy",
            fail_after_root_exit,
        )

        try:
            with pytest.raises(RuntimeError, match="root exited before readiness"):
                await server.start()

            assert child_pid is not None
            assert _wait_for_posix_process_exit(child_pid)
            assert terminated_file.read_text(encoding="utf-8") == "terminated"
            assert server._process is None
            assert server._posix_owner is None
        finally:
            server.stop()
            _force_stop_posix_test_process(child_pid)


def _open_windows_process_handle(pid: int) -> int:
    """Open a synchronization handle for a real Windows process."""
    import ctypes
    from ctypes import wintypes

    synchronize = 0x00100000
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    )
    kernel32.OpenProcess.restype = wintypes.HANDLE
    handle = kernel32.OpenProcess(synchronize, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        msg = f"OpenProcess failed for pid={pid}"
        raise OSError(error, msg)
    return int(handle)


def _windows_process_handle_is_running(handle: int) -> bool:
    """Return whether a process handle is still unsignaled."""
    import ctypes
    from ctypes import wintypes

    wait_timeout = 0x00000102
    wait_failed = 0xFFFFFFFF
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    result = int(kernel32.WaitForSingleObject(handle, 0))
    if result == wait_failed:
        error = ctypes.get_last_error()
        msg = "WaitForSingleObject failed"
        raise OSError(error, msg)
    return result == wait_timeout


def _wait_for_windows_process_handle_exit(handle: int) -> bool:
    """Wait for a real Windows process handle to become signaled."""
    import ctypes
    from ctypes import wintypes

    wait_object_0 = 0
    wait_failed = 0xFFFFFFFF
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    result = int(
        kernel32.WaitForSingleObject(
            handle,
            int(_WINDOWS_PROCESS_TEST_TIMEOUT * 1000),
        )
    )
    if result == wait_failed:
        error = ctypes.get_last_error()
        msg = "WaitForSingleObject failed"
        raise OSError(error, msg)
    return result == wait_object_0


def _close_windows_process_handle(handle: int) -> None:
    """Close a native process handle opened by a Windows lifecycle test."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    assert kernel32.CloseHandle(handle)


def _child_process_script(pid_file: Path, *, exit_root: bool = False) -> str:
    """Build a Python script that publishes and then outlives a child process."""
    child_code = "import time; time.sleep(300)"
    root_tail = "raise SystemExit(7)" if exit_root else "time.sleep(300)"
    return (
        "import subprocess, sys, time\n"
        "from pathlib import Path\n"
        "child = subprocess.Popen(\n"
        f"    [sys.executable, '-c', {child_code!r}],\n"
        "    stdin=subprocess.DEVNULL,\n"
        "    stdout=subprocess.DEVNULL,\n"
        "    stderr=subprocess.DEVNULL,\n"
        ")\n"
        f"Path({str(pid_file)!r}).write_text(str(child.pid), encoding='utf-8')\n"
        f"{root_tail}\n"
    )


def _root_process_script(ready_file: Path) -> str:
    """Build a Python script that reports readiness and remains alive."""
    return (
        "import os, time\n"
        "from pathlib import Path\n"
        f"Path({str(ready_file)!r}).write_text(str(os.getpid()), encoding='utf-8')\n"
        "time.sleep(300)\n"
    )


@pytest.mark.skipif(sys.platform != "win32", reason="requires Windows Job Objects")
@pytest.mark.usefixtures("_real_windows_server_lifecycle")
class TestWindowsServerProcessTree:
    """Real Windows coverage for the owned `langgraph dev` process tree."""

    @staticmethod
    def _make_server(tmp_path: Path) -> ServerProcess:
        config_dir = tmp_path / "runtime"
        config_dir.mkdir()
        (config_dir / "langgraph.json").write_text("{}", encoding="utf-8")
        return ServerProcess(config_dir=config_dir, port=43210)

    @staticmethod
    def _patch_server_command(
        monkeypatch: pytest.MonkeyPatch,
        script: str,
    ) -> None:
        monkeypatch.setattr(
            "deepagents_code.client.launch.server._build_server_cmd",
            lambda *_args, **_kwargs: [sys.executable, "-c", script],
        )
        monkeypatch.setattr(
            "deepagents_code.client.launch.server._port_in_use",
            lambda *_args: False,
        )

    @staticmethod
    def _assert_no_cleanup_warning(caplog: pytest.LogCaptureFixture) -> None:
        warnings = [
            record.getMessage()
            for record in caplog.records
            if record.levelno >= logging.WARNING
            and (
                "server process" in record.getMessage().lower()
                or "server job" in record.getMessage().lower()
                or "orphan" in record.getMessage().lower()
            )
        ]
        assert warnings == []

    async def test_child_descendant_cleanup(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Stopping the server terminates a real child descendant."""
        child_pid_file = tmp_path / "child.pid"
        server = self._make_server(tmp_path)
        self._patch_server_command(
            monkeypatch,
            _child_process_script(child_pid_file),
        )

        async def wait_until_child_starts(
            *_args: object,
            **_kwargs: object,
        ) -> None:
            await asyncio.to_thread(_wait_for_pid_file, child_pid_file)

        monkeypatch.setattr(
            "deepagents_code.client.launch.server.wait_for_server_healthy",
            wait_until_child_starts,
        )

        child_handle: int | None = None
        try:
            await server.start()
            process = server._process
            assert process is not None
            child_pid = _wait_for_pid_file(child_pid_file)
            child_handle = _open_windows_process_handle(child_pid)
            assert _windows_process_handle_is_running(child_handle)

            with caplog.at_level(logging.WARNING):
                server.stop()

            assert process.poll() is not None
            assert _wait_for_windows_process_handle_exit(child_handle)
            assert server._windows_job is None
            self._assert_no_cleanup_warning(caplog)
        finally:
            server.stop()
            if child_handle is not None:
                _close_windows_process_handle(child_handle)

    async def test_root_exits_early_but_child_is_cleaned_up(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Startup cleanup terminates the Job after its root already exited."""
        child_pid_file = tmp_path / "early-child.pid"
        server = self._make_server(tmp_path)
        self._patch_server_command(
            monkeypatch,
            _child_process_script(child_pid_file, exit_root=True),
        )
        child_handles: list[int] = []

        async def observe_root_exit(
            *_args: object,
            **kwargs: object,
        ) -> None:
            process = kwargs["process"]
            assert isinstance(process, subprocess.Popen)
            child_pid = await asyncio.to_thread(_wait_for_pid_file, child_pid_file)
            child_handle = _open_windows_process_handle(child_pid)
            child_handles.append(child_handle)
            await asyncio.to_thread(
                process.wait,
                timeout=_WINDOWS_PROCESS_TEST_TIMEOUT,
            )
            assert process.returncode == 7
            assert _windows_process_handle_is_running(child_handle)
            msg = "root exited before readiness"
            raise RuntimeError(msg)

        monkeypatch.setattr(
            "deepagents_code.client.launch.server.wait_for_server_healthy",
            observe_root_exit,
        )

        try:
            with (
                caplog.at_level(logging.WARNING),
                pytest.raises(RuntimeError, match="root exited before readiness"),
            ):
                await server.start()

            assert len(child_handles) == 1
            assert _wait_for_windows_process_handle_exit(child_handles[0])
            assert server._process is None
            assert server._windows_job is None
            self._assert_no_cleanup_warning(caplog)
        finally:
            server.stop()
            for handle in child_handles:
                _close_windows_process_handle(handle)

    async def test_job_assignment_failure_reaps_suspended_root(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Assignment failure kills the real suspended root before it can run."""
        from deepagents_code.update_check import _WindowsJobObject

        sentinel = tmp_path / "unowned-process-ran"
        server = self._make_server(tmp_path)
        script = (
            "import time\n"
            "from pathlib import Path\n"
            f"Path({str(sentinel)!r}).write_text('ran', encoding='utf-8')\n"
            "time.sleep(300)\n"
        )
        self._patch_server_command(monkeypatch, script)

        real_popen = subprocess.Popen
        processes: list[subprocess.Popen[Any]] = []

        def capture_process(*args: Any, **kwargs: Any) -> subprocess.Popen[Any]:
            process = real_popen(*args, **kwargs)
            processes.append(process)
            return process

        monkeypatch.setattr(
            "deepagents_code.client.launch.server.subprocess.Popen",
            capture_process,
        )
        healthy = AsyncMock()
        monkeypatch.setattr(
            "deepagents_code.client.launch.server.wait_for_server_healthy",
            healthy,
        )

        def fail_assignment(_process_handle: int) -> None:
            time.sleep(0.2)
            assert not sentinel.exists()
            msg = "Job assignment unavailable"
            raise OSError(msg)

        try:
            with (
                patch.object(
                    _WindowsJobObject,
                    "create_for_process_handle",
                    side_effect=fail_assignment,
                ),
                caplog.at_level(logging.WARNING),
                pytest.raises(OSError, match="Job assignment unavailable"),
            ):
                await server.start()

            assert len(processes) == 1
            assert processes[0].poll() is not None
            assert not sentinel.exists()
            healthy.assert_not_awaited()
            assert server._process is None
            assert server._windows_job is None
            self._assert_no_cleanup_warning(caplog)
        finally:
            server.stop()
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=_WINDOWS_PROCESS_TEST_TIMEOUT)

    async def test_normal_shutdown(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A normally running real server root is terminated and reaped."""
        ready_file = tmp_path / "ready"
        server = self._make_server(tmp_path)
        self._patch_server_command(monkeypatch, _root_process_script(ready_file))

        async def wait_until_ready(
            *_args: object,
            **_kwargs: object,
        ) -> None:
            await asyncio.to_thread(_wait_for_pid_file, ready_file)

        monkeypatch.setattr(
            "deepagents_code.client.launch.server.wait_for_server_healthy",
            wait_until_ready,
        )

        try:
            await server.start()
            process = server._process
            assert process is not None

            with caplog.at_level(logging.WARNING):
                server.stop()

            assert process.poll() is not None
            assert server._process is None
            assert server._windows_job is None
            self._assert_no_cleanup_warning(caplog)
        finally:
            server.stop()

    async def test_repeated_stop_is_idempotent(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Repeated stop closes the real Job once without cleanup warnings."""
        ready_file = tmp_path / "repeated-ready"
        server = self._make_server(tmp_path)
        self._patch_server_command(monkeypatch, _root_process_script(ready_file))

        async def wait_until_ready(
            *_args: object,
            **_kwargs: object,
        ) -> None:
            await asyncio.to_thread(_wait_for_pid_file, ready_file)

        monkeypatch.setattr(
            "deepagents_code.client.launch.server.wait_for_server_healthy",
            wait_until_ready,
        )

        try:
            await server.start()
            process = server._process
            job = server._windows_job
            assert process is not None
            assert job is not None

            with (
                patch.object(job, "terminate", wraps=job.terminate) as terminate,
                caplog.at_level(logging.WARNING),
            ):
                server.stop()
                server.stop()

            terminate.assert_called_once_with()
            assert process.poll() is not None
            assert server._process is None
            assert server._windows_job is None
            self._assert_no_cleanup_warning(caplog)
        finally:
            server.stop()
