"""Unit tests for LocalShellBackend."""

import ctypes
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import uuid
import warnings
from contextlib import suppress
from ctypes import wintypes
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deepagents._api.deprecation import LangChainDeprecationWarning
from deepagents.backends import local_shell as local_shell_module
from deepagents.backends.local_shell import LocalShellBackend
from deepagents.backends.protocol import ExecuteResponse


def _python_command(code: str) -> str:
    """Build a shell-safe command using the active Python interpreter."""
    argv = [sys.executable, "-c", code]
    if sys.platform == "win32":
        return subprocess.list2cmdline(argv)
    return shlex.join(argv)


def _windows_pid_is_running(pid: int) -> bool:
    """Return whether a Windows process is still active."""
    process_query_limited_information = 0x1000
    still_active = 259
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    )
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.DWORD),
    )
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.OpenProcess(
        process_query_limited_information,
        wintypes.BOOL(0),
        pid,
    )
    if not handle:
        return False
    try:
        exit_code = wintypes.DWORD()
        return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))) and exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def _wait_for_windows_pid_exit(pid: int, *, timeout: float = 2.0) -> bool:
    """Wait for a Windows process to stop running."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _windows_pid_is_running(pid):
            return True
        time.sleep(0.02)
    return not _windows_pid_is_running(pid)


def _cleanup_windows_process_tree(pid: int) -> None:
    """Best-effort cleanup for a failed Windows process-tree assertion."""
    windows_dir = os.environ.get("SYSTEMROOT") or os.environ.get("WINDIR")
    executable = str(Path(windows_dir) / "System32" / "taskkill.exe") if windows_dir else "taskkill.exe"
    with suppress(OSError, subprocess.TimeoutExpired):
        subprocess.run(  # noqa: S603
            [executable, "/PID", str(pid), "/T", "/F"],
            check=False,
            capture_output=True,
            timeout=5,
        )


def _windows_admin_unc_path(path: Path, *, require_exists: bool = True) -> Path:
    """Convert a local drive path to its localhost administrative-share path."""
    resolved = path.resolve()
    if len(resolved.drive) != 2 or resolved.drive[1] != ":":
        pytest.skip(f"No local drive is available for a UNC regression: {resolved}")
    relative = resolved.relative_to(Path(resolved.anchor))
    unc_path = Path(f"\\\\localhost\\{resolved.drive[0]}$\\{relative}")
    if require_exists and not unc_path.exists():
        pytest.skip(f"Local administrative share is unavailable: {unc_path}")
    return unc_path


def _posix_pid_is_running(pid: int) -> bool:
    """Return whether a POSIX process still exists."""
    stat_path = Path("/proc") / str(pid) / "stat"
    try:
        if stat_path.read_text(encoding="utf-8").split()[2] == "Z":
            return False
    except (IndexError, OSError):
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_posix_pid_exit(pid: int, *, timeout: float = 2.0) -> bool:
    """Wait for a POSIX process to stop running."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _posix_pid_is_running(pid):
            return True
        time.sleep(0.02)
    return not _posix_pid_is_running(pid)


def _posix_process_state(pid: int) -> str | None:
    """Return one Linux process state without reaping the process."""
    try:
        return (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8").split()[2]
    except (IndexError, OSError):
        return None


def _posix_file_descriptor_count() -> int:
    """Return the current process's Linux file-descriptor count."""
    return len(tuple((Path("/proc") / "self" / "fd").iterdir()))


def _pid_is_running(pid: int) -> bool:
    """Return whether a process is active on the current platform."""
    if sys.platform == "win32":
        return _windows_pid_is_running(pid)
    return _posix_pid_is_running(pid)


def _wait_for_pid_exit(pid: int, *, timeout: float = 2.0) -> bool:
    """Wait for a process to exit on the current platform."""
    if sys.platform == "win32":
        return _wait_for_windows_pid_exit(pid, timeout=timeout)
    return _wait_for_posix_pid_exit(pid, timeout=timeout)


def _cleanup_process_tree(pid: int) -> None:
    """Best-effort cleanup for a failed cross-platform ownership assertion."""
    if sys.platform == "win32":
        _cleanup_windows_process_tree(pid)
        return
    with suppress(OSError):
        os.kill(pid, signal.SIGKILL)


def test_local_shell_backend_initialization() -> None:
    """Test that LocalShellBackend initializes correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(root_dir=tmpdir)

        assert backend.cwd == Path(tmpdir).resolve()
        assert backend.id.startswith("local-")
        assert len(backend.id) == 14  # "local-" + 8 hex chars


def test_local_shell_backend_execute_simple_command() -> None:
    """Test executing a simple shell command."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(root_dir=tmpdir, inherit_env=True)

        result = backend.execute(_python_command("print('Hello World')"))

        assert isinstance(result, ExecuteResponse)
        assert result.exit_code == 0
        assert "Hello World" in result.output
        assert result.truncated is False


def test_local_shell_backend_execute_with_error() -> None:
    """Test executing a command that fails."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(root_dir=tmpdir, inherit_env=True)

        result = backend.execute(_python_command("import sys; sys.stderr.write('missing file\\n'); raise SystemExit(2)"))

        assert result.exit_code != 0
        assert "[stderr]" in result.output
        assert "Exit code:" in result.output


def test_local_shell_backend_execute_in_working_directory() -> None:
    """Test that commands execute in the specified working directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test file
        test_file = Path(tmpdir) / "test.txt"
        test_file.write_text("test content")

        backend = LocalShellBackend(root_dir=tmpdir, inherit_env=True)

        # Execute command that relies on working directory
        result = backend.execute(_python_command("from pathlib import Path; print(Path('test.txt').read_text())"))

        assert result.exit_code == 0
        assert "test content" in result.output


def test_local_shell_backend_execute_empty_command() -> None:
    """Test executing an empty command returns an error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(root_dir=tmpdir)

        result = backend.execute("")

        assert result.exit_code == 1
        assert "must be a non-empty string" in result.output


def test_local_shell_backend_execute_timeout() -> None:
    """Test that long-running commands timeout correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(root_dir=tmpdir, timeout=1, inherit_env=True)

        # Sleep for longer than timeout
        result = backend.execute(_python_command("import time; time.sleep(5)"))

        assert result.exit_code == 124  # Standard timeout exit code
        assert "timed out" in result.output


def test_local_shell_backend_close_kills_successful_command_descendant(
    tmp_path: Path,
) -> None:
    """A successful root's background child remains owned until backend close."""
    pid_file = tmp_path / "successful-descendant.pid"
    parent_code = (
        "import subprocess, sys; from pathlib import Path; "
        "child = subprocess.Popen("
        "[sys.executable, '-c', 'import time; time.sleep(60)'], "
        "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, "
        "stderr=subprocess.DEVNULL); "
        f"Path({str(pid_file)!r}).write_text(str(child.pid), encoding='utf-8')"
    )
    backend = LocalShellBackend(root_dir=tmp_path, inherit_env=True)
    descendant_pid: int | None = None

    try:
        result = backend.execute(_python_command(parent_code))
        assert result.exit_code == 0, result.output
        descendant_pid = int(pid_file.read_text(encoding="utf-8"))
        assert _pid_is_running(descendant_pid)

        backend.close()

        exited = _wait_for_pid_exit(descendant_pid)
    finally:
        backend.close()
        if descendant_pid is not None and _pid_is_running(descendant_pid):
            _cleanup_process_tree(descendant_pid)

    assert exited


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="Windows Job lifecycle regression",
)
def test_local_shell_backend_releases_empty_successful_job(tmp_path: Path) -> None:
    """A foreground-only successful command does not retain an empty Job."""
    backend = LocalShellBackend(root_dir=tmp_path, inherit_env=True)

    try:
        result = backend.execute(_python_command("print('done')"))

        assert result.exit_code == 0, result.output
        assert backend._process_registry._processes == []
    finally:
        backend.close()


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="Windows process-tree regression",
)
def test_local_shell_backend_windows_timeout_kills_descendant_promptly() -> None:
    """A suspended root cannot spawn before delayed Job assignment."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pid_file = Path(tmpdir) / "descendant.pid"
        parent_code = (
            "import subprocess, sys, time; from pathlib import Path; "
            "child = subprocess.Popen("
            "[sys.executable, '-c', 'import time; time.sleep(30)']); "
            f"Path({str(pid_file)!r}).write_text("
            "str(child.pid), encoding='utf-8'); "
            "time.sleep(30)"
        )
        backend = LocalShellBackend(
            root_dir=tmpdir,
            timeout=1,
            inherit_env=True,
        )
        original_create = local_shell_module._WindowsJobObject.create_for_process_handle
        spawned_before_assignment = False
        created_jobs: list[local_shell_module._WindowsJobObject] = []

        def delayed_create(
            process_handle: int,
        ) -> local_shell_module._WindowsJobObject:
            nonlocal spawned_before_assignment
            time.sleep(0.25)
            spawned_before_assignment = pid_file.exists()
            job = original_create(process_handle)
            created_jobs.append(job)
            return job

        with patch.object(
            local_shell_module._WindowsJobObject,
            "create_for_process_handle",
            side_effect=delayed_create,
        ):
            started = time.monotonic()
            result = backend.execute(_python_command(parent_code))
            elapsed = time.monotonic() - started

        assert result.exit_code == 124
        assert not spawned_before_assignment
        assert created_jobs
        assert pid_file.exists()
        descendant_pid = int(pid_file.read_text(encoding="utf-8"))
        exited = _wait_for_windows_pid_exit(descendant_pid)
        if not exited:
            _cleanup_windows_process_tree(descendant_pid)
        assert elapsed < 2.5, f"Job-first timeout cleanup took {elapsed:.3f}s"
        assert exited


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="Windows suspended-launch regression",
)
def test_local_shell_backend_windows_assignment_failure_never_runs_command(
    tmp_path: Path,
) -> None:
    """Unavailable Job assignment reaps the root before user code can run."""
    pid_file = tmp_path / "descendant.pid"
    parent_code = (
        "import subprocess, sys; from pathlib import Path; "
        "child = subprocess.Popen("
        "[sys.executable, '-c', 'import time; time.sleep(30)']); "
        f"Path({str(pid_file)!r}).write_text("
        "str(child.pid), encoding='utf-8')"
    )

    def unavailable_job(_process_handle: int) -> local_shell_module._WindowsJobObject:
        time.sleep(0.25)
        assert not pid_file.exists()
        msg = "Job assignment unavailable"
        raise OSError(msg)

    started = time.monotonic()
    try:
        with (
            patch.object(
                local_shell_module._WindowsJobObject,
                "create_for_process_handle",
                side_effect=unavailable_job,
            ),
            pytest.raises(OSError, match="Job assignment unavailable"),
        ):
            local_shell_module._run_shell_command(
                _python_command(parent_code),
                cwd=tmp_path,
                env=dict(os.environ),
                timeout=10,
            )
    finally:
        if pid_file.exists():
            _cleanup_windows_process_tree(int(pid_file.read_text(encoding="utf-8")))

    assert time.monotonic() - started < 4.0
    assert not pid_file.exists()


def test_assign_windows_job_resumes_only_after_assignment() -> None:
    """The suspended root resumes only after its Job assignment succeeds."""
    process = MagicMock()
    process._handle = 5678
    job = MagicMock()
    order: list[str] = []

    def record_assignment(
        _process_handle: int,
    ) -> local_shell_module._WindowsJobObject:
        order.append("assign")
        return job

    with (
        patch.object(
            local_shell_module._WindowsJobObject,
            "create_for_process_handle",
            side_effect=record_assignment,
        ) as create,
        patch.object(
            local_shell_module,
            "_resume_windows_process",
            side_effect=lambda _handle: order.append("resume"),
        ) as resume,
    ):
        result = local_shell_module._assign_windows_job_and_resume(process)

    create.assert_called_once_with(5678)
    resume.assert_called_once_with(5678)
    assert result is job
    assert order == ["assign", "resume"]


def test_assign_windows_job_resume_failure_fails_closed() -> None:
    """A resume failure terminates and reaps the still-suspended root."""
    process = MagicMock()
    process._handle = 5678
    job = MagicMock()

    with (
        patch.object(
            local_shell_module._WindowsJobObject,
            "create_for_process_handle",
            return_value=job,
        ),
        patch.object(
            local_shell_module,
            "_resume_windows_process",
            side_effect=OSError("resume failed"),
        ),
        patch.object(
            local_shell_module,
            "_terminate_windows_process_tree",
        ) as terminate,
        pytest.raises(OSError, match="resume failed"),
    ):
        local_shell_module._assign_windows_job_and_resume(process)

    terminate.assert_called_once_with(process, job)


def test_terminate_windows_process_tree_closes_job_before_postcheck() -> None:
    """Successful Job termination avoids the slower `taskkill` fallback."""
    process = MagicMock()
    process.pid = 1234
    process.poll.return_value = None
    job = MagicMock()
    teardown_order: list[str] = []

    def terminate_job() -> bool:
        teardown_order.append("job")
        return True

    def record_wait(_process: object, *, timeout: float) -> bool:
        assert timeout == local_shell_module._WINDOWS_TERMINATE_TIMEOUT
        teardown_order.append("wait")
        return True

    def reap_root(_process: object, *, timeout: float) -> None:
        assert timeout == local_shell_module._WINDOWS_TERMINATE_TIMEOUT
        teardown_order.append("reap")

    job.terminate.side_effect = terminate_job
    with (
        patch.object(
            local_shell_module,
            "_wait_for_windows_process_root_exit",
            side_effect=record_wait,
        ) as wait_for_root,
        patch.object(
            local_shell_module,
            "_taskkill_windows_process_tree",
            side_effect=lambda _pid: teardown_order.append("taskkill"),
        ) as taskkill,
        patch.object(
            local_shell_module,
            "_reap_process_root",
            side_effect=reap_root,
        ) as reap,
    ):
        local_shell_module._terminate_windows_process_tree(process, job)

    wait_for_root.assert_called_once_with(
        process,
        timeout=local_shell_module._WINDOWS_TERMINATE_TIMEOUT,
    )
    taskkill.assert_not_called()
    process.kill.assert_not_called()
    reap.assert_called_once_with(
        process,
        timeout=local_shell_module._WINDOWS_TERMINATE_TIMEOUT,
    )
    assert teardown_order == ["job", "wait", "reap"]


def test_terminate_windows_process_tree_uses_taskkill_when_job_close_fails() -> None:
    """A failed Job close falls back to bounded PID-scoped `taskkill`."""
    process = MagicMock()
    process.pid = 1234
    process.poll.return_value = 0
    job = MagicMock()
    job.terminate.return_value = False

    with (
        patch.object(
            local_shell_module,
            "_taskkill_windows_process_tree",
        ) as taskkill,
        patch.object(
            local_shell_module,
            "_wait_for_windows_process_root_exit",
            return_value=True,
        ) as wait_for_root,
        patch.object(local_shell_module, "_reap_process_root"),
    ):
        local_shell_module._terminate_windows_process_tree(process, job)

    job.terminate.assert_called_once_with()
    taskkill.assert_called_once_with(process.pid)
    wait_for_root.assert_called_once_with(
        process,
        timeout=local_shell_module._WINDOWS_TERMINATE_TIMEOUT,
    )
    process.kill.assert_not_called()


def test_terminate_windows_process_tree_taskkills_surviving_job_root() -> None:
    """A root surviving Job close triggers the bounded post-check fallback."""
    process = MagicMock()
    process.pid = 1234
    process.poll.return_value = None
    job = MagicMock()
    job.terminate.return_value = True

    with (
        patch.object(
            local_shell_module,
            "_wait_for_windows_process_root_exit",
            side_effect=[False, True],
        ) as wait_for_root,
        patch.object(
            local_shell_module,
            "_taskkill_windows_process_tree",
        ) as taskkill,
        patch.object(local_shell_module, "_reap_process_root"),
    ):
        local_shell_module._terminate_windows_process_tree(process, job)

    assert wait_for_root.call_count == 2
    taskkill.assert_called_once_with(process.pid)
    process.kill.assert_not_called()


def test_terminate_windows_process_tree_taskkills_without_owned_job() -> None:
    """An unassigned suspended root uses the bounded PID fallback."""
    process = MagicMock()
    process.pid = 1234
    process.poll.return_value = None

    with (
        patch.object(
            local_shell_module,
            "_taskkill_windows_process_tree",
        ) as taskkill,
        patch.object(
            local_shell_module,
            "_wait_for_windows_process_root_exit",
            return_value=True,
        ) as wait_for_root,
        patch.object(local_shell_module, "_reap_process_root"),
    ):
        local_shell_module._terminate_windows_process_tree(process, None)

    taskkill.assert_called_once_with(process.pid)
    wait_for_root.assert_called_once_with(
        process,
        timeout=local_shell_module._WINDOWS_TERMINATE_TIMEOUT,
    )
    process.kill.assert_not_called()


def test_terminate_windows_process_tree_reaps_idempotently() -> None:
    """Repeated cleanup of an exited root stays bounded and skips `taskkill`."""
    process = MagicMock()
    process.pid = 1234
    process.poll.return_value = 0
    process.wait.return_value = 0

    with patch.object(
        local_shell_module,
        "_taskkill_windows_process_tree",
    ) as taskkill:
        local_shell_module._terminate_windows_process_tree(process, None)
        local_shell_module._terminate_windows_process_tree(process, None)

    taskkill.assert_not_called()
    process.kill.assert_not_called()
    assert process.wait.call_count == 2


def test_run_shell_command_windows_error_uses_tree_cleanup(
    tmp_path: Path,
) -> None:
    """A Windows stream failure tears down the still-running process tree."""
    process = MagicMock()
    process.pid = 1234
    process.communicate.side_effect = OSError("pipe read failed")
    job = MagicMock()

    with (
        patch.object(local_shell_module.os, "name", "nt"),
        patch.object(
            local_shell_module.subprocess,
            "Popen",
            return_value=process,
        ) as popen,
        patch.object(
            local_shell_module,
            "_assign_windows_job_and_resume",
            return_value=job,
        ) as assign,
        patch.object(
            local_shell_module,
            "_terminate_windows_process_tree",
        ) as terminate,
        pytest.raises(OSError, match="pipe read failed"),
    ):
        local_shell_module._run_shell_command(
            "cmd",
            cwd=tmp_path,
            env={},
            timeout=1,
        )

    assert popen.call_args.kwargs["creationflags"] == local_shell_module._WINDOWS_CREATE_SUSPENDED
    assign.assert_called_once_with(process)
    terminate.assert_called_once_with(process, job)
    job.close.assert_not_called()


def test_run_shell_command_windows_keyboard_interrupt_uses_job_cleanup(
    tmp_path: Path,
) -> None:
    """KeyboardInterrupt terminates the Job before propagating."""
    process = MagicMock()
    process.pid = 1234
    process.returncode = 0
    process.communicate.side_effect = KeyboardInterrupt
    job = MagicMock()

    with (
        patch.object(local_shell_module.os, "name", "nt"),
        patch.object(local_shell_module.subprocess, "Popen", return_value=process),
        patch.object(
            local_shell_module,
            "_assign_windows_job_and_resume",
            return_value=job,
        ),
        patch.object(
            local_shell_module,
            "_terminate_windows_process_tree",
        ) as terminate,
        pytest.raises(KeyboardInterrupt),
    ):
        local_shell_module._run_shell_command(
            "cmd",
            cwd=tmp_path,
            env={},
            timeout=1,
        )

    terminate.assert_called_once_with(process, job)
    job.close.assert_not_called()


@pytest.mark.skipif(
    os.name != "posix",
    reason="POSIX environment fidelity regression",
)
def test_local_shell_backend_posix_execs_with_exact_environment(
    tmp_path: Path,
) -> None:
    """Python locale coercion must not leak into the final `/bin/sh` environment."""
    backend = LocalShellBackend(
        root_dir=tmp_path,
        env={
            "LANG": "C",
            "PATH": "/usr/bin:/bin",
        },
    )

    try:
        result = backend.execute("if env | grep -q '^LC_CTYPE='; then echo mutated; else echo exact; fi")
    finally:
        backend.close()

    assert result.exit_code == 0, result.output
    assert result.output.strip() == "exact"


@pytest.mark.skipif(
    os.name != "posix",
    reason="POSIX signal-disposition regression",
)
def test_local_shell_backend_posix_restores_sigpipe_before_exec(
    tmp_path: Path,
) -> None:
    """Pipeline writers inherit direct-shell SIGPIPE behavior, not Python's ignore."""
    backend = LocalShellBackend(root_dir=tmp_path, inherit_env=True)

    try:
        result = backend.execute("yes | head -n 1")
    finally:
        backend.close()

    assert result.exit_code == 0, result.output
    assert result.output.strip() == "y"


@pytest.mark.skipif(
    os.name != "posix",
    reason="POSIX high-file-descriptor regression",
)
def test_local_shell_backend_posix_watchdog_supports_high_file_descriptors(
    tmp_path: Path,
) -> None:
    """Watchdog readiness and control work when inherited FDs exceed 1023."""
    open_file_descriptors: list[int] = []
    backend: LocalShellBackend | None = None
    try:
        while not open_file_descriptors or open_file_descriptors[-1] < 1050:
            try:
                read_fd, write_fd = os.pipe()
            except OSError:
                pytest.skip("Process file-descriptor limit is below the regression range")
            open_file_descriptors.extend((read_fd, write_fd))

        backend = LocalShellBackend(root_dir=tmp_path, inherit_env=True)
        result = backend.execute("printf high-fd-ready")
    finally:
        if backend is not None:
            backend.close()
        for file_descriptor in reversed(open_file_descriptors):
            with suppress(OSError):
                os.close(file_descriptor)

    assert result.exit_code == 0, result.output
    assert result.output == "high-fd-ready"


@pytest.mark.skipif(
    os.name != "posix" or not Path("/proc/self/fd").is_dir(),
    reason="Linux foreground-command resource regression",
)
def test_local_shell_backend_posix_foreground_commands_do_not_leak_watchdogs(
    tmp_path: Path,
) -> None:
    """Fifty foreground commands leave no retained guards, FDs, or zombies."""
    command_count = 50
    watchdog_pids: list[int] = []
    original_init = local_shell_module._PosixOwnerGuard.__init__

    def capture_watchdog(
        owner_guard: local_shell_module._PosixOwnerGuard,
        control_fd: int,
        watchdog: subprocess.Popen[str],
        process_group_id: int,
    ) -> None:
        watchdog_pids.append(watchdog.pid)
        original_init(
            owner_guard,
            control_fd,
            watchdog,
            process_group_id,
        )

    backend = LocalShellBackend(
        root_dir=tmp_path,
        inherit_env=True,
        virtual_mode=True,
    )
    baseline_file_descriptors = _posix_file_descriptor_count()
    maximum_registry_size = 0

    try:
        with patch.object(
            local_shell_module._PosixOwnerGuard,
            "__init__",
            new=capture_watchdog,
        ):
            for _ in range(command_count):
                result = backend.execute(":")
                assert result.exit_code == 0, result.output
                maximum_registry_size = max(
                    maximum_registry_size,
                    len(backend._process_registry._processes),
                )

        time.sleep(local_shell_module._OWNER_WATCHDOG_POLL_INTERVAL * 2)
        final_file_descriptors = _posix_file_descriptor_count()
        zombie_watchdogs = [pid for pid in watchdog_pids if _posix_process_state(pid) == "Z"]

        assert len(watchdog_pids) == command_count
        assert maximum_registry_size == 0
        assert backend._process_registry._processes == []
        assert final_file_descriptors <= baseline_file_descriptors + 2
        assert zombie_watchdogs == []
    finally:
        backend.close()


@pytest.mark.skipif(
    os.name != "posix",
    reason="POSIX process-group regression",
)
def test_local_shell_backend_posix_timeout_kills_pipe_holding_descendant() -> None:
    """A pipe-holding descendant cannot delay timeout cleanup."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pid_file = Path(tmpdir) / "descendant.pid"
        parent_code = (
            "import subprocess, sys; from pathlib import Path; "
            "child = subprocess.Popen("
            "[sys.executable, '-c', 'import time; time.sleep(30)']); "
            f"Path({str(pid_file)!r}).write_text("
            "str(child.pid), encoding='utf-8')"
        )
        backend = LocalShellBackend(
            root_dir=tmpdir,
            timeout=1,
            inherit_env=True,
        )

        descendant_pid: int | None = None
        try:
            started = time.monotonic()
            result = backend.execute(_python_command(parent_code))
            elapsed = time.monotonic() - started

            assert result.exit_code == 124
            assert pid_file.exists()
            descendant_pid = int(pid_file.read_text(encoding="utf-8"))
            exited = _wait_for_posix_pid_exit(descendant_pid)
        finally:
            if descendant_pid is None and pid_file.exists():
                descendant_pid = int(pid_file.read_text(encoding="utf-8"))
            if descendant_pid is not None and _posix_pid_is_running(descendant_pid):
                sigkill = getattr(signal, "SIGKILL", None)
                if isinstance(sigkill, int):
                    with suppress(OSError):
                        os.kill(descendant_pid, sigkill)

        assert 0.8 <= elapsed < 3.0
        assert exited


@pytest.mark.skipif(
    os.name != "posix",
    reason="POSIX owner-death regression",
)
def test_local_shell_backend_posix_owner_death_kills_shell_and_descendant(
    tmp_path: Path,
) -> None:
    """Abrupt owner death triggers bounded watchdog cleanup of the whole group."""
    shell_pid_file = tmp_path / "shell.pid"
    descendant_pid_file = tmp_path / "descendant.pid"
    descendant_code = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    command_code = (
        "import os, signal, subprocess, sys, time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(shell_pid_file)!r}).write_text("
        "str(os.getpgrp()), encoding='utf-8'); "
        f"child = subprocess.Popen([sys.executable, '-c', {descendant_code!r}]); "
        f"Path({str(descendant_pid_file)!r}).write_text("
        "str(child.pid), encoding='utf-8'); "
        "time.sleep(60)"
    )
    shell_command = _python_command(command_code)
    owner_code = (
        "from deepagents.backends.local_shell import LocalShellBackend; "
        f"backend = LocalShellBackend(root_dir={str(tmp_path)!r}, "
        "timeout=30, inherit_env=True); "
        f"result = backend.execute({shell_command!r}); "
        "raise SystemExit(result.exit_code)"
    )
    owner = subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", owner_code],
        cwd=tmp_path,
        env=dict(os.environ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    shell_pid: int | None = None
    descendant_pid: int | None = None
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if shell_pid_file.exists() and descendant_pid_file.exists():
                break
            if owner.poll() is not None:
                stdout, stderr = owner.communicate()
                pytest.fail(f"owner exited before launching descendants: {stdout}{stderr}")
            time.sleep(0.02)
        else:
            pytest.fail("owner did not launch the shell descendant before the deadline")

        shell_pid = int(shell_pid_file.read_text(encoding="utf-8"))
        descendant_pid = int(descendant_pid_file.read_text(encoding="utf-8"))
        assert _posix_pid_is_running(shell_pid)
        assert _posix_pid_is_running(descendant_pid)

        owner.kill()
        owner.wait(timeout=5)

        assert _wait_for_posix_pid_exit(shell_pid, timeout=8)
        assert _wait_for_posix_pid_exit(descendant_pid, timeout=8)
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=5)
        getpgrp = getattr(os, "getpgrp", None)
        killpg = getattr(os, "killpg", None)
        sigkill = getattr(signal, "SIGKILL", None)
        if shell_pid is not None and callable(getpgrp) and callable(killpg) and isinstance(sigkill, int) and shell_pid != getpgrp():
            with suppress(OSError):
                killpg(shell_pid, sigkill)
        if descendant_pid is not None and _posix_pid_is_running(descendant_pid):
            _cleanup_process_tree(descendant_pid)


@pytest.mark.skipif(
    os.name != "posix",
    reason="POSIX owner-death regression",
)
def test_local_shell_backend_posix_owner_death_after_success_kills_descendant(
    tmp_path: Path,
) -> None:
    """A retained watchdog kills a 60s child after its successful root is gone."""
    descendant_pid_file = tmp_path / "successful-descendant.pid"
    owner_ready_file = tmp_path / "owner-ready"
    descendant_code = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    command_code = (
        "import subprocess, sys; from pathlib import Path; "
        f"child = subprocess.Popen([sys.executable, '-c', {descendant_code!r}], "
        "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, "
        "stderr=subprocess.DEVNULL); "
        f"Path({str(descendant_pid_file)!r}).write_text("
        "str(child.pid), encoding='utf-8')"
    )
    shell_command = _python_command(command_code)
    owner_code = (
        "import time; from pathlib import Path; "
        "from deepagents.backends.local_shell import LocalShellBackend; "
        f"backend = LocalShellBackend(root_dir={str(tmp_path)!r}, "
        "timeout=30, inherit_env=True); "
        f"result = backend.execute({shell_command!r}); "
        "assert result.exit_code == 0, result.output; "
        f"Path({str(owner_ready_file)!r}).touch(); "
        "time.sleep(60)"
    )
    owner = subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", owner_code],
        cwd=tmp_path,
        env=dict(os.environ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    descendant_pid: int | None = None
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if descendant_pid_file.exists() and owner_ready_file.exists():
                break
            if owner.poll() is not None:
                stdout, stderr = owner.communicate()
                pytest.fail(f"owner exited before retaining descendant: {stdout}{stderr}")
            time.sleep(0.02)
        else:
            pytest.fail("owner did not retain the successful descendant before the deadline")

        descendant_pid = int(descendant_pid_file.read_text(encoding="utf-8"))
        assert _posix_pid_is_running(descendant_pid)

        owner.kill()
        owner.wait(timeout=5)

        assert _wait_for_posix_pid_exit(descendant_pid, timeout=8)
    finally:
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=5)
        if descendant_pid is not None and _posix_pid_is_running(descendant_pid):
            _cleanup_process_tree(descendant_pid)


def test_start_posix_shell_process_gates_exec_until_watchdog_ready(
    tmp_path: Path,
) -> None:
    """The shell gate opens only after an independent watchdog is active."""
    process = MagicMock()
    process.pid = 1234
    watchdog = MagicMock()
    watchdog.poll.return_value = None
    events: list[tuple[str, object]] = []

    def record_ready(ready_fd: int) -> None:
        events.append(("ready", ready_fd))

    def record_write(file_descriptor: int, value: bytes) -> int:
        events.append(("write", (file_descriptor, bytes(value))))
        return len(value)

    with (
        patch.object(
            local_shell_module.os,
            "pipe",
            side_effect=[(10, 11), (12, 13), (14, 15), (16, 17)],
        ),
        patch.object(local_shell_module.os, "close"),
        patch.object(local_shell_module.os, "write", side_effect=record_write),
        patch.object(local_shell_module.os, "getpid", return_value=4321),
        patch.object(
            local_shell_module.subprocess,
            "Popen",
            side_effect=[process, watchdog],
        ) as popen,
        patch.object(
            local_shell_module,
            "_wait_for_posix_watchdog_ready",
            side_effect=record_ready,
        ),
        patch.object(local_shell_module, "_reap_process_root") as reap,
    ):
        started_process, owner_guard = local_shell_module._start_posix_shell_process(
            "printf ready",
            cwd=tmp_path,
            env={"PATH": "/bin"},
        )
        owner_guard.close()

    assert started_process is process
    shell_call, watchdog_call = popen.call_args_list
    assert shell_call.args[0][-1] == "printf ready"
    assert shell_call.kwargs["shell"] is False
    assert shell_call.kwargs["start_new_session"] is True
    assert shell_call.kwargs["pass_fds"] == (10, 12)
    assert watchdog_call.args[0][6:8] == ["1234", "4321"]
    assert watchdog_call.kwargs["shell"] is False
    assert watchdog_call.kwargs["start_new_session"] is True
    assert watchdog_call.kwargs["pass_fds"] == (14, 17)
    assert events == [
        ("ready", 16),
        ("write", (13, b'[["PATH","/bin"]]')),
        ("write", (11, b"G")),
        ("write", (15, b"C")),
    ]
    reap.assert_called_once_with(
        watchdog,
        timeout=local_shell_module._POSIX_WATCHDOG_EXIT_TIMEOUT,
    )


def test_posix_watchdog_acknowledges_only_after_selector_registration() -> None:
    """The shell gate cannot open before the watchdog control FD is armed."""
    watchdog_source = local_shell_module._POSIX_OWNER_WATCHDOG

    assert watchdog_source.index("selector.register") < watchdog_source.index('os.write(ready_fd, b"R")')


def test_posix_owner_guard_terminates_through_live_watchdog() -> None:
    """Retained cleanup sends its action to the specific live watchdog."""
    watchdog = MagicMock()
    watchdog.poll.return_value = None
    owner_guard = local_shell_module._PosixOwnerGuard(15, watchdog, 1234)
    killpg = MagicMock()

    with (
        patch.object(local_shell_module.os, "write") as write,
        patch.object(local_shell_module.os, "close") as close,
        patch.object(local_shell_module, "_reap_process_root") as reap,
        patch.object(
            local_shell_module,
            "_posix_process_group_api",
            return_value=(killpg, signal.SIGTERM, 9),
        ),
    ):
        owner_guard.terminate()
        owner_guard.terminate()

    killpg.assert_called_once_with(1234, 0)
    write.assert_called_once_with(15, b"T")
    close.assert_called_once_with(15)
    reap.assert_called_once_with(
        watchdog,
        timeout=local_shell_module._POSIX_WATCHDOG_EXIT_TIMEOUT,
    )


def test_retained_posix_cleanup_never_signals_reused_pgid_after_watchdog_loss() -> None:
    """A dead watchdog prevents delayed signaling of its root's reused PGID."""
    process = MagicMock()
    process.pid = 1234
    watchdog = MagicMock()
    watchdog.poll.return_value = 0
    owner_guard = local_shell_module._PosixOwnerGuard(15, watchdog, process.pid)
    owned_process = local_shell_module._OwnedShellProcess(
        process,
        None,
        owner_guard,
    )

    with (
        patch.object(local_shell_module.os, "write") as write,
        patch.object(local_shell_module.os, "close"),
        patch.object(local_shell_module, "_reap_process_root") as reap,
        patch.object(
            local_shell_module,
            "_terminate_posix_process_group",
        ) as terminate_group,
    ):
        owned_process.terminate()
        owned_process.terminate()

    write.assert_not_called()
    terminate_group.assert_not_called()
    reap.assert_any_call(
        watchdog,
        timeout=local_shell_module._POSIX_WATCHDOG_EXIT_TIMEOUT,
    )
    reap.assert_any_call(
        process,
        timeout=local_shell_module._POSIX_TERMINATE_TIMEOUT,
    )
    assert reap.call_count == 2


def test_run_shell_command_posix_timeout_uses_new_session_and_group_cleanup(
    tmp_path: Path,
) -> None:
    """The POSIX path isolates the shell and delegates timeout tree cleanup."""
    process = MagicMock()
    process.communicate.side_effect = subprocess.TimeoutExpired("cmd", 1)
    owner_guard = MagicMock()

    with (
        patch.object(local_shell_module.os, "name", "posix"),
        patch.object(
            local_shell_module,
            "_start_shell_process",
            return_value=(process, None, owner_guard),
        ) as start,
        patch.object(local_shell_module, "_terminate_posix_process_group") as terminate,
        pytest.raises(subprocess.TimeoutExpired),
    ):
        local_shell_module._run_shell_command(
            "cmd",
            cwd=tmp_path,
            env={},
            timeout=1,
        )

    start.assert_called_once_with(
        "cmd",
        cwd=tmp_path,
        env={},
    )
    terminate.assert_called_once_with(process)
    owner_guard.close.assert_called_once_with()


def test_run_shell_command_posix_error_cleans_group_after_root_exit(
    tmp_path: Path,
) -> None:
    """A stream error still cleans descendants after the shell root exits."""
    process = MagicMock()
    process.poll.return_value = 0
    msg = "pipe read failed"
    process.communicate.side_effect = OSError(msg)
    owner_guard = MagicMock()

    with (
        patch.object(local_shell_module.os, "name", "posix"),
        patch.object(
            local_shell_module,
            "_start_shell_process",
            return_value=(process, None, owner_guard),
        ),
        patch.object(local_shell_module, "_terminate_posix_process_group") as terminate,
        pytest.raises(OSError, match="pipe read failed"),
    ):
        local_shell_module._run_shell_command(
            "cmd",
            cwd=tmp_path,
            env={},
            timeout=1,
        )

    terminate.assert_called_once_with(process)
    owner_guard.close.assert_called_once_with()


def test_run_shell_command_posix_keyboard_interrupt_cleans_group(
    tmp_path: Path,
) -> None:
    """KeyboardInterrupt terminates the known POSIX group before propagating."""
    process = MagicMock()
    process.pid = 1234
    process.returncode = 0
    process.communicate.side_effect = KeyboardInterrupt
    owner_guard = MagicMock()

    with (
        patch.object(local_shell_module.os, "name", "posix"),
        patch.object(
            local_shell_module,
            "_start_shell_process",
            return_value=(process, None, owner_guard),
        ),
        patch.object(local_shell_module, "_terminate_posix_process_group") as terminate,
        pytest.raises(KeyboardInterrupt),
    ):
        local_shell_module._run_shell_command(
            "cmd",
            cwd=tmp_path,
            env={},
            timeout=1,
        )

    terminate.assert_called_once_with(process)
    owner_guard.close.assert_called_once_with()


def test_terminate_posix_process_group_bounds_cleanup_waits() -> None:
    """POSIX cleanup bounds SIGTERM grace, SIGKILL, and root reaping."""
    process = MagicMock()
    process.pid = 1234
    process.poll.return_value = None
    process.wait.side_effect = subprocess.TimeoutExpired("cmd", 1)
    killpg = MagicMock()

    with (
        patch.object(
            local_shell_module,
            "_posix_process_group_api",
            return_value=(killpg, signal.SIGTERM, 9),
        ),
        patch.object(local_shell_module, "_POSIX_TERMINATE_TIMEOUT", 0.01),
    ):
        started = time.monotonic()
        local_shell_module._terminate_posix_process_group(process)
        elapsed = time.monotonic() - started

    killpg.assert_any_call(process.pid, signal.SIGTERM)
    killpg.assert_any_call(process.pid, 0)
    killpg.assert_any_call(process.pid, 9)
    assert process.wait.call_count == 2
    process.stdout.close.assert_called_once_with()
    process.stderr.close.assert_called_once_with()
    assert elapsed < 0.5


def test_local_shell_backend_execute_output_truncation() -> None:
    """Test that large output gets truncated."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(
            root_dir=tmpdir,
            max_output_bytes=100,
            inherit_env=True,
        )

        # Generate lots of output
        result = backend.execute(_python_command("print('\\n'.join(str(i) for i in range(1000)))"))

        assert result.truncated is True
        assert "Output truncated" in result.output
        assert len(result.output) <= 150  # Some buffer for truncation message


def test_local_shell_backend_filesystem_operations() -> None:
    """Test that filesystem operations work (inherited from FilesystemBackend)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(root_dir=tmpdir, virtual_mode=True)

        # Write a file
        write_result = backend.write("/test.txt", "Hello\nWorld\n")
        assert write_result.error is None
        assert write_result.path == "/test.txt"

        # Read the file
        content = backend.read("/test.txt")
        assert content.file_data is not None
        assert "Hello" in content.file_data["content"]
        assert "World" in content.file_data["content"]

        # Edit the file
        edit_result = backend.edit("/test.txt", "World", "Universe")
        assert edit_result.error is None
        assert edit_result.occurrences == 1

        # Verify edit
        content = backend.read("/test.txt")
        assert content.file_data is not None
        assert "Universe" in content.file_data["content"]
        assert "World" not in content.file_data["content"]


def test_local_shell_backend_integration_shell_and_filesystem() -> None:
    """Test that shell commands and filesystem operations work together."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(
            root_dir=tmpdir,
            virtual_mode=True,
            inherit_env=True,
        )

        # Create file via filesystem and read it via command execution.
        backend.write("/script.txt", "Script output")
        result = backend.execute(_python_command("from pathlib import Path; print(Path('script.txt').read_text())"))

        assert result.exit_code == 0
        assert "Script output" in result.output

        # Create file via shell
        backend.execute(_python_command("from pathlib import Path; Path('shell_file.txt').write_text('Shell created')"))

        # Read via filesystem
        content = backend.read("/shell_file.txt")
        assert content.file_data is not None
        assert "Shell created" in content.file_data["content"]


def test_local_shell_backend_ls_info() -> None:
    """Test listing directory contents."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(root_dir=tmpdir, virtual_mode=True)

        # Create some files
        backend.write("/file1.txt", "content1")
        backend.write("/file2.txt", "content2")

        # List files
        files = backend.ls("/").entries

        assert files is not None
        assert len(files) == 2
        paths = [f["path"] for f in files]
        assert "/file1.txt" in paths
        assert "/file2.txt" in paths


def test_local_shell_backend_grep() -> None:
    """Test grep functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(root_dir=tmpdir, virtual_mode=True)

        # Create files with searchable content
        backend.write("/file1.txt", "TODO: implement this")
        backend.write("/file2.txt", "DONE: completed")

        # Search for TODO
        matches = backend.grep("TODO").matches

        assert matches is not None
        assert len(matches) == 1
        assert matches[0]["text"] == "TODO: implement this"


def test_local_shell_backend_glob() -> None:
    """Test glob functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(root_dir=tmpdir, virtual_mode=True)

        # Create files with different extensions
        backend.write("/file1.txt", "content")
        backend.write("/file2.py", "content")
        backend.write("/file3.txt", "content")

        # Find all .txt files
        txt_files = backend.glob("*.txt").matches

        assert txt_files is not None
        assert len(txt_files) == 2
        paths = [f["path"] for f in txt_files]
        assert "/file1.txt" in paths
        assert "/file3.txt" in paths
        assert "/file2.py" not in paths


def test_local_shell_backend_virtual_mode_restrictions() -> None:
    """Test that virtual_mode restricts filesystem paths but not shell commands."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        root = base / "root"
        root.mkdir()
        outside = base / "outside.txt"
        outside.write_text("outside shell access", encoding="utf-8")
        backend = LocalShellBackend(root_dir=root, virtual_mode=True)

        # Filesystem operations should be restricted
        with pytest.raises(ValueError, match="Path traversal not allowed"):
            backend.read("/../outside.txt")

        # But shell commands are NOT restricted (by design)
        code = f"from pathlib import Path; print(Path({str(outside)!r}).read_text())"
        result = backend.execute(_python_command(code))
        assert result.exit_code == 0
        assert "outside shell access" in result.output


def test_local_shell_backend_environment_variables() -> None:
    """Test that custom environment variables are passed to commands."""
    with tempfile.TemporaryDirectory() as tmpdir:
        custom_env = {"CUSTOM_VAR": "custom_value"}
        backend = LocalShellBackend(root_dir=tmpdir, env=custom_env)

        result = backend.execute(_python_command("import os; print(os.environ.get('CUSTOM_VAR', ''))"))

        assert result.exit_code == 0
        assert "custom_value" in result.output


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="Windows shell semantics test",
)
def test_local_shell_backend_windows_unc_cwd_and_relative_access(
    tmp_path: Path,
) -> None:
    """A UNC root is entered through `pushd`, never cmd's fallback directory."""
    target = tmp_path / "unc cwd"
    target.mkdir()
    marker = target / "relative.txt"
    marker.write_text("intended-share", encoding="utf-8")
    unc_target = _windows_admin_unc_path(target)
    code = (
        "import sys; from pathlib import Path; "
        "cwd = Path.cwd(); "
        "print(f'drive={cwd.drive}'); "
        "print(f'same={cwd.samefile(Path(sys.argv[1]))}'); "
        "print(Path('relative.txt').read_text(encoding='utf-8'))"
    )
    command = subprocess.list2cmdline([sys.executable, "-c", code, str(unc_target)])
    backend = LocalShellBackend(root_dir=unc_target, inherit_env=True)

    result = backend.execute(command)

    assert result.exit_code == 0, result.output
    assert "same=True" in result.output
    assert "intended-share" in result.output
    drive_line = next(line for line in result.output.splitlines() if line.startswith("drive="))
    mapped_drive = drive_line.removeprefix("drive=")
    assert mapped_drive
    assert not Path(f"{mapped_drive}\\").exists()


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="Windows shell semantics test",
)
def test_local_shell_backend_windows_unc_percent_cd_expands_after_pushd(
    tmp_path: Path,
) -> None:
    """The nested command processor expands `%CD%` in the mapped UNC directory."""
    target = tmp_path / "percent cd cwd"
    target.mkdir()
    unc_target = _windows_admin_unc_path(target)
    code = "import sys; from pathlib import Path; print(Path(sys.argv[1]).samefile(Path(sys.argv[2]))); print(sys.argv[1])"
    command = f'{subprocess.list2cmdline([sys.executable, "-c", code])} "%CD%" {subprocess.list2cmdline([str(unc_target)])}'
    backend = LocalShellBackend(root_dir=unc_target, inherit_env=True)

    try:
        result = backend.execute(command)
    finally:
        backend.close()

    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[0] == "True"
    assert str(Path(os.environ["SYSTEMROOT"]) / "System32") not in result.output


def test_windows_unc_launch_defers_user_command_to_nested_cmd() -> None:
    """The outer `cmd.exe` receives no user text to expand before `pushd`."""
    command = 'echo %CD% && echo "quoted & shell"'
    cwd = Path(r"\\server\share\project")
    with (
        patch.object(
            local_shell_module,
            "_windows_cmd_launch_paths",
            return_value=(
                r"C:\Windows\System32\cmd.exe",
                r"C:\Windows\System32",
                r"C:\Windows",
            ),
        ),
        patch.object(local_shell_module.uuid, "uuid4") as uuid4,
    ):
        uuid4.return_value.hex = "fixed"
        command_line, executable, bootstrap_cwd, child_env = local_shell_module._windows_unc_shell_launch(
            command,
            cwd=cwd,
            env={},
        )

    assert executable == r"C:\Windows\System32\cmd.exe"
    assert bootstrap_cwd == r"C:\Windows\System32"
    assert command not in command_line
    assert "@pushd" in command_line
    assert command_line.index("@pushd") < command_line.index(sys.executable)
    assert child_env["DEEPAGENTS_UNC_COMMAND_fixed"] == command
    assert "DEEPAGENTS_UNC_COMMAND_fixed" in command_line


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="Windows shell semantics test",
)
def test_local_shell_backend_windows_unc_quotes_metacharacters_and_exit_status(
    tmp_path: Path,
) -> None:
    """UNC quoting preserves literal paths, shell operators, and command status."""
    target = tmp_path / "cwd %UNC_LITERAL%! & (meta)^"
    target.mkdir()
    relative_name = "relative & (quoted)^!.txt"
    (target / relative_name).write_text("literal-metacharacters", encoding="utf-8")
    unc_target = _windows_admin_unc_path(target)
    code = (
        "import sys; from pathlib import Path; cwd = Path.cwd(); print(f'cwd_name={cwd.name}'); print(Path(sys.argv[1]).read_text(encoding='utf-8'))"
    )
    python_command = subprocess.list2cmdline([sys.executable, "-c", code, relative_name])
    command = f'{python_command} && echo "quoted & shell" && exit /b 37'
    backend = LocalShellBackend(
        root_dir=unc_target,
        env={"UNC_LITERAL": "must-not-expand"},
    )

    result = backend.execute(command)

    assert result.exit_code == 37
    assert "cwd_name=cwd %UNC_LITERAL%! & (meta)^" in result.output
    assert "literal-metacharacters" in result.output
    assert '"quoted & shell"' in result.output


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="Windows shell semantics test",
)
def test_local_shell_backend_windows_unc_pushd_failure_never_runs_command(
    tmp_path: Path,
) -> None:
    """An unavailable UNC cwd fails closed before arbitrary code can execute."""
    missing = tmp_path / f"missing-{uuid.uuid4().hex}"
    unc_missing = _windows_admin_unc_path(missing, require_exists=False)
    marker = tmp_path / "must-not-exist.txt"
    command = _python_command(f"from pathlib import Path; Path({str(marker)!r}).touch()")
    backend = LocalShellBackend(root_dir=unc_missing, inherit_env=True)

    result = backend.execute(command)

    assert result.exit_code != 0
    assert not marker.exists()
    assert "Failed to enter the configured UNC working directory" in result.output


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="Windows shell semantics test",
)
def test_local_shell_backend_windows_unc_rejects_unrepresentable_command(
    tmp_path: Path,
) -> None:
    """A command that cannot reach `cmd.exe` is rejected before launch."""
    unc_target = _windows_admin_unc_path(tmp_path)
    backend = LocalShellBackend(root_dir=unc_target, inherit_env=True)

    result = backend.execute("echo before\x00echo after")

    assert result.exit_code == 1
    assert "Cannot safely execute a command containing a NUL character" in result.output


@pytest.mark.skipif(
    sys.platform != "win32",
    reason="Windows shell semantics test",
)
def test_local_shell_backend_windows_command_cwd_and_environment() -> None:
    """Test native Windows command execution, cwd, and environment propagation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(
            root_dir=tmpdir,
            env={"WINDOWS_TEST_VAR": "available"},
        )
        result = backend.execute(
            _python_command("import os; from pathlib import Path; print(os.name); print(Path.cwd()); print(os.environ.get('WINDOWS_TEST_VAR', ''))")
        )

        assert result.exit_code == 0
        assert "nt" in result.output
        assert str(Path(tmpdir).resolve()) in result.output
        assert "available" in result.output


def test_local_shell_backend_inherit_env() -> None:
    """Test that inherit_env=True inherits parent environment."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(root_dir=tmpdir, inherit_env=True)

        # PATH should be available from parent environment
        result = backend.execute(_python_command("import os; print(os.environ.get('PATH', ''))"))

        assert result.exit_code == 0
        assert len(result.output.strip()) > 0  # PATH should not be empty


def test_local_shell_backend_empty_env_by_default() -> None:
    """Test that environment is empty by default (secure default)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(root_dir=tmpdir)

        # Without inherit_env, PATH should not be available
        result = backend.execute(_python_command("import os; print('PATH is: ' + os.environ.get('PATH', ''))"))

        assert result.exit_code == 0
        # PATH should be empty (the string "PATH is: " with no value after)
        assert "PATH is:" in result.output


def test_local_shell_backend_stderr_formatting() -> None:
    """Test that stderr is properly prefixed with [stderr]."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(root_dir=tmpdir, inherit_env=True)

        # Command that outputs to stderr
        result = backend.execute(_python_command("import sys; sys.stderr.write('error message\\n')"))

        assert result.exit_code == 0
        assert "[stderr]" in result.output
        assert "error message" in result.output


async def test_local_shell_backend_async_execute() -> None:
    """Test async execute method."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(root_dir=tmpdir, inherit_env=True)

        result = await backend.aexecute(_python_command("print('async test')"))

        assert isinstance(result, ExecuteResponse)
        assert result.exit_code == 0
        assert "async test" in result.output


async def test_local_shell_backend_async_filesystem_operations() -> None:
    """Test async filesystem operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalShellBackend(root_dir=tmpdir, virtual_mode=True)

        # Async write
        write_result = await backend.awrite("/async_test.txt", "async content")
        assert write_result.error is None

        # Async read
        content = await backend.aread("/async_test.txt")
        assert content.file_data is not None
        assert "async content" in content.file_data["content"]

        # Async edit
        edit_result = await backend.aedit("/async_test.txt", "async", "modified")
        assert edit_result.error is None

        # Verify
        content = await backend.aread("/async_test.txt")
        assert content.file_data is not None
        assert "modified content" in content.file_data["content"]


class TestLocalShellVirtualModeDefaultDeprecation:
    """`virtual_mode=None` (omitted) emits a deprecation; explicit values do not."""

    def test_omitted_virtual_mode_warns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            be = LocalShellBackend(root_dir=tmpdir)

        deprecations = [w for w in captured if issubclass(w.category, DeprecationWarning)]
        assert len(deprecations) == 1
        assert deprecations[0].category is LangChainDeprecationWarning
        assert "virtual_mode" in str(deprecations[0].message)
        # Default falls back to `False` for backwards compatibility.
        assert be.virtual_mode is False

    def test_explicit_virtual_mode_does_not_warn(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            LocalShellBackend(root_dir=tmpdir, virtual_mode=False)
            LocalShellBackend(root_dir=tmpdir, virtual_mode=True)

        deprecations = [w for w in captured if issubclass(w.category, DeprecationWarning) and "virtual_mode" in str(w.message)]
        assert deprecations == []
