"""`LocalShellBackend`: Filesystem backend with unrestricted local shell execution.

This backend extends `FilesystemBackend` to add shell command execution on
the local host system. It provides NO sandboxing or isolation - all operations
run directly on the host machine with full system access.
"""

from __future__ import annotations

import ctypes
import json
import ntpath
import os
import selectors
import signal
import subprocess
import sys
import threading
import time
import uuid
import weakref
from collections.abc import Callable
from contextlib import suppress
from ctypes import wintypes
from pathlib import Path
from typing import cast

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol

DEFAULT_EXECUTE_TIMEOUT = 120
"""Default timeout in seconds for shell command execution."""

_WINDOWS_TERMINATE_TIMEOUT = 5.0
"""Maximum seconds spent reaping Windows process-tree cleanup helpers."""

_POSIX_TERMINATE_TIMEOUT = 5.0
"""Maximum seconds spent draining and reaping a terminated POSIX shell."""

_POSIX_OWNER_DEATH_TERM_TIMEOUT = 1.0
"""Grace period before the owner watchdog escalates from SIGTERM to SIGKILL."""

_POSIX_WATCHDOG_READY_TIMEOUT = 5.0
"""Maximum seconds allowed for the POSIX owner watchdog to become ready."""

_POSIX_WATCHDOG_EXIT_TIMEOUT = 5.0
"""Maximum seconds spent reaping the POSIX owner watchdog."""

_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_WINDOWS_CREATE_SUSPENDED = 0x00000004
_WINDOWS_WAIT_OBJECT_0 = 0
_PROCESS_GROUP_POLL_INTERVAL = 0.05
_OWNER_WATCHDOG_POLL_INTERVAL = 0.1
_WINDOWS_DRIVE_DESIGNATOR_LENGTH = 2
_WINDOWS_UNC_PUSHD_ERROR = "Error: Failed to enter the configured UNC working directory. Verify that the share exists and access is permitted."

_PosixKillpg = Callable[[int, int], None]

_POSIX_SHELL_LAUNCHER = """\
import json
import os
import signal
import sys

gate_fd = int(sys.argv[1])
environment_fd = int(sys.argv[2])

environment_payload = bytearray()
try:
    while chunk := os.read(environment_fd, 65536):
        environment_payload.extend(chunk)
finally:
    os.close(environment_fd)

environment = dict(json.loads(environment_payload.decode("ascii")))

try:
    gate_token = os.read(gate_fd, 1)
finally:
    os.close(gate_fd)

if gate_token != b"G":
    raise SystemExit(125)

for signal_name in ("SIGPIPE", "SIGXFZ", "SIGXFSZ"):
    signal_number = getattr(signal, signal_name, None)
    if signal_number is not None:
        signal.signal(signal_number, signal.SIG_DFL)

os.execve("/bin/sh", ["/bin/sh", "-c", sys.argv[3]], environment)
"""

_POSIX_OWNER_WATCHDOG = """\
import os
import selectors
import signal
import sys
import time

control_fd = int(sys.argv[1])
ready_fd = int(sys.argv[2])
process_group_id = int(sys.argv[3])
owner_pid = int(sys.argv[4])
term_timeout = float(sys.argv[5])
poll_interval = float(sys.argv[6])

def group_exists() -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True

cleanup_requested = True
try:
    with selectors.DefaultSelector() as selector:
        selector.register(control_fd, selectors.EVENT_READ)
        try:
            os.write(ready_fd, b"R")
        finally:
            os.close(ready_fd)
            ready_fd = -1
        while os.getppid() == owner_pid:
            if not group_exists():
                cleanup_requested = False
                break
            if selector.select(poll_interval):
                cleanup_requested = os.read(control_fd, 1) != b"C"
                break
finally:
    if ready_fd >= 0:
        os.close(ready_fd)
    os.close(control_fd)

if not cleanup_requested:
    raise SystemExit(0)

try:
    os.killpg(process_group_id, signal.SIGTERM)
except ProcessLookupError:
    raise SystemExit(0)
except OSError:
    pass

deadline = time.monotonic() + term_timeout
while group_exists() and time.monotonic() < deadline:
    time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))

if group_exists():
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass
"""

_WINDOWS_UNC_COMMAND_LAUNCHER = """\
import os
import subprocess
import sys

command_variable = sys.argv[1]
command_processor = sys.argv[2]
cwd_variable = sys.argv[3]
system_root_variable = sys.argv[4]

command = os.environ.pop(command_variable)
os.environ.pop(cwd_variable, None)
if system_root_variable != "-":
    os.environ.pop("SystemRoot", None)
    os.environ.pop(system_root_variable, None)

command_line = (
    subprocess.list2cmdline([command_processor])
    + ' /d /e:on /v:off /s /c "'
    + command
    + '"'
)
raise SystemExit(
    subprocess.call(
        command_line,
        executable=command_processor,
        shell=False,
        env=os.environ,
    )
)
"""


def _set_windows_job_kill_on_close(handle: int, *, enabled: bool) -> bool:
    """Configure whether closing a Windows Job Object terminates its processes."""

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ("read_operation_count", ctypes.c_ulonglong),
            ("write_operation_count", ctypes.c_ulonglong),
            ("other_operation_count", ctypes.c_ulonglong),
            ("read_transfer_count", ctypes.c_ulonglong),
            ("write_transfer_count", ctypes.c_ulonglong),
            ("other_transfer_count", ctypes.c_ulonglong),
        ]

    class _BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("per_process_user_time_limit", ctypes.c_longlong),
            ("per_job_user_time_limit", ctypes.c_longlong),
            ("limit_flags", wintypes.DWORD),
            ("minimum_working_set_size", ctypes.c_size_t),
            ("maximum_working_set_size", ctypes.c_size_t),
            ("active_process_limit", wintypes.DWORD),
            ("affinity", ctypes.c_size_t),
            ("priority_class", wintypes.DWORD),
            ("scheduling_class", wintypes.DWORD),
        ]

    class _ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("basic_limit_information", _BasicLimitInformation),
            ("io_info", _IoCounters),
            ("process_memory_limit", ctypes.c_size_t),
            ("job_memory_limit", ctypes.c_size_t),
            ("peak_process_memory_used", ctypes.c_size_t),
            ("peak_job_memory_used", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.SetInformationJobObject.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    )
    kernel32.SetInformationJobObject.restype = wintypes.BOOL

    limits = _ExtendedLimitInformation()
    if enabled:
        limits.basic_limit_information.limit_flags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    return bool(
        kernel32.SetInformationJobObject(
            handle,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        )
    )


def _close_windows_handle(handle: int) -> bool:
    """Close a native Windows handle."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    return bool(kernel32.CloseHandle(handle))


def _windows_job_has_processes(handle: int) -> bool:
    """Return whether a Windows Job Object has active processes."""

    class _BasicAccountingInformation(ctypes.Structure):
        _fields_ = [
            ("total_user_time", ctypes.c_longlong),
            ("total_kernel_time", ctypes.c_longlong),
            ("this_period_total_user_time", ctypes.c_longlong),
            ("this_period_total_kernel_time", ctypes.c_longlong),
            ("total_page_fault_count", wintypes.DWORD),
            ("total_processes", wintypes.DWORD),
            ("active_processes", wintypes.DWORD),
            ("total_terminated_processes", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.QueryInformationJobObject.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    )
    kernel32.QueryInformationJobObject.restype = wintypes.BOOL
    accounting = _BasicAccountingInformation()
    queried = kernel32.QueryInformationJobObject(
        handle,
        1,
        ctypes.byref(accounting),
        ctypes.sizeof(accounting),
        None,
    )
    if not queried:
        return True
    return accounting.active_processes > 0


def _terminate_and_wait_windows_job(handle: int) -> bool:
    """Terminate every process in a Job Object and wait for it to empty."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.TerminateJobObject.argtypes = (
        wintypes.HANDLE,
        wintypes.UINT,
    )
    kernel32.TerminateJobObject.restype = wintypes.BOOL
    kernel32.WaitForSingleObject.argtypes = (
        wintypes.HANDLE,
        wintypes.DWORD,
    )
    kernel32.WaitForSingleObject.restype = wintypes.DWORD

    if not kernel32.TerminateJobObject(handle, 1):
        return False
    wait_milliseconds = int(_WINDOWS_TERMINATE_TIMEOUT * 1000)
    return kernel32.WaitForSingleObject(handle, wait_milliseconds) == _WINDOWS_WAIT_OBJECT_0


class _WindowsJobObject:
    """Own a Windows Job Object containing one subprocess tree."""

    def __init__(self, handle: int) -> None:
        self._handle: int | None = handle

    @classmethod
    def create_for_process_handle(
        cls,
        process_handle: int,
    ) -> _WindowsJobObject:
        """Create a kill-on-close job and assign a suspended process."""
        if os.name != "nt":
            msg = "Windows Job Objects are unavailable on this platform"
            raise OSError(msg)

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = (
            ctypes.c_void_p,
            wintypes.LPCWSTR,
        )
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.AssignProcessToJobObject.argtypes = (
            wintypes.HANDLE,
            wintypes.HANDLE,
        )
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL

        job_handle = kernel32.CreateJobObjectW(None, None)
        if not job_handle:
            error = ctypes.get_last_error()
            msg = "CreateJobObjectW failed"
            raise OSError(error, msg)

        def configure_and_assign() -> None:
            if not _set_windows_job_kill_on_close(job_handle, enabled=True):
                error = ctypes.get_last_error()
                msg = "SetInformationJobObject failed"
                raise OSError(error, msg)
            if not kernel32.AssignProcessToJobObject(job_handle, process_handle):
                error = ctypes.get_last_error()
                msg = "AssignProcessToJobObject failed"
                raise OSError(error, msg)

        try:
            configure_and_assign()
            return cls(job_handle)
        except BaseException:
            _close_windows_handle(job_handle)
            raise

    def terminate(self) -> bool:
        """Terminate the full tree, wait for it to exit, and close the Job."""
        handle = self._handle
        if handle is None:
            return True
        self._handle = None
        terminated = False
        try:
            terminated = _terminate_and_wait_windows_job(handle)
        finally:
            closed = _close_windows_handle(handle)
        return terminated and closed

    def has_processes(self) -> bool:
        """Return whether the Job still contains at least one process."""
        handle = self._handle
        if handle is None:
            return False
        return _windows_job_has_processes(handle)

    def close(self) -> None:
        """Release the job without terminating processes after normal exit."""
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        _set_windows_job_kill_on_close(handle, enabled=False)
        _close_windows_handle(handle)


class _PosixOwnerGuard:
    """Own descendants through a specific watchdog instead of a reusable PGID."""

    def __init__(
        self,
        control_fd: int,
        watchdog: subprocess.Popen[str],
        process_group_id: int,
    ) -> None:
        self._control_fd: int | None = control_fd
        self._watchdog: subprocess.Popen[str] | None = watchdog
        self._process_group_id: int | None = process_group_id
        self._lock = threading.Lock()

    def is_alive(self) -> bool:
        """Return whether the watchdog subprocess is still running."""
        with self._lock:
            watchdog = self._watchdog
            return watchdog is not None and watchdog.poll() is None

    def has_processes(self) -> bool:
        """Return whether the live watchdog's command group has members."""
        with self._lock:
            return self._has_processes_locked()

    def _has_processes_locked(self) -> bool:
        """Probe command-group membership while ownership state is stable."""
        watchdog = self._watchdog
        process_group_id = self._process_group_id
        if watchdog is None or process_group_id is None or watchdog.poll() is not None:
            return False

        process_group_api = _posix_process_group_api()
        if process_group_api is None:
            return True
        killpg, _, _ = process_group_api
        if not _posix_process_group_exists(killpg, process_group_id):
            return False
        return watchdog.poll() is None

    def close(self) -> None:
        """Tell the watchdog that normal owner-side cleanup completed."""
        self._finish(b"C")

    def terminate(self) -> None:
        """Ask the still-owned watchdog to terminate its process group."""
        self._finish(b"T")

    def _finish(self, action: bytes) -> None:
        """Send one bounded, idempotent watchdog action and reap that watchdog."""
        with self._lock:
            if action == b"T" and not self._has_processes_locked():
                action = b"C"
            control_fd = self._control_fd
            watchdog = self._watchdog
            self._control_fd = None
            self._watchdog = None
            self._process_group_id = None

        if control_fd is not None:
            if watchdog is not None and watchdog.poll() is None:
                with suppress(OSError):
                    os.write(control_fd, action)
            with suppress(OSError):
                os.close(control_fd)
        if watchdog is not None:
            _reap_process_root(
                watchdog,
                timeout=_POSIX_WATCHDOG_EXIT_TIMEOUT,
            )


class _OwnedShellProcess:
    """Retain one completed command's descendant ownership until shutdown."""

    def __init__(
        self,
        process: subprocess.Popen[str],
        windows_job: _WindowsJobObject | None,
        posix_owner_guard: _PosixOwnerGuard | None,
    ) -> None:
        self._process: subprocess.Popen[str] | None = process
        self._windows_job: _WindowsJobObject | None = windows_job
        self._posix_owner_guard: _PosixOwnerGuard | None = posix_owner_guard

    def has_processes(self) -> bool:
        """Return whether any descendant remains after the command root exited."""
        windows_job = self._windows_job
        if windows_job is not None:
            return windows_job.has_processes()

        posix_owner_guard = self._posix_owner_guard
        if posix_owner_guard is not None:
            return posix_owner_guard.has_processes()

        process = self._process
        if process is None or os.name != "posix":
            return False
        process_group_api = _posix_process_group_api()
        if process_group_api is None:
            return False
        killpg, _, _ = process_group_api
        return _posix_process_group_exists(killpg, process.pid)

    def release(self) -> None:
        """Release ownership after confirming that the process tree is empty."""
        windows_job = self._windows_job
        posix_owner_guard = self._posix_owner_guard
        self._process = None
        self._windows_job = None
        self._posix_owner_guard = None

        if windows_job is not None:
            with suppress(OSError):
                windows_job.close()
        if posix_owner_guard is not None:
            posix_owner_guard.close()

    def terminate(self) -> None:
        """Terminate any remaining descendants and release all ownership."""
        process = self._process
        windows_job = self._windows_job
        posix_owner_guard = self._posix_owner_guard
        self._process = None
        self._windows_job = None
        self._posix_owner_guard = None

        if posix_owner_guard is not None:
            try:
                posix_owner_guard.terminate()
            finally:
                if process is not None:
                    _reap_process_root(
                        process,
                        timeout=_POSIX_TERMINATE_TIMEOUT,
                    )
            return

        if process is not None:
            _terminate_shell_process(process, windows_job)
        elif windows_job is not None:
            with suppress(OSError):
                windows_job.terminate()


class _ShellProcessRegistry:
    """Thread-safe ownership registry for successful background descendants."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._processes: list[_OwnedShellProcess] = []
        self._closed = False

    def register(self, process: _OwnedShellProcess) -> None:
        """Retain a live process tree after pruning completed ownership."""
        releasable: list[_OwnedShellProcess] = []
        terminate = False
        with self._lock:
            if self._closed:
                terminate = True
            else:
                retained: list[_OwnedShellProcess] = []
                for owned_process in self._processes:
                    if owned_process.has_processes():
                        retained.append(owned_process)
                    else:
                        releasable.append(owned_process)
                if process.has_processes():
                    retained.append(process)
                else:
                    releasable.append(process)
                self._processes = retained

        for owned_process in releasable:
            owned_process.release()
        if terminate:
            process.terminate()

    def close(self) -> None:
        """Terminate every retained process tree exactly once."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            processes = self._processes
            self._processes = []

        for process in reversed(processes):
            with suppress(Exception):
                if process.has_processes():
                    process.terminate()
                else:
                    process.release()


def _windows_popen_process_handle(process: subprocess.Popen[str]) -> int:
    """Return the native process handle retained by Windows `Popen`."""
    handle = getattr(process, "_handle", None)
    if not isinstance(handle, int):
        msg = "Windows subprocess process handle is unavailable"
        raise OSError(msg)
    return handle


def _resume_windows_process(process_handle: int) -> None:
    """Resume every thread in a process created with `CREATE_SUSPENDED`."""
    ntdll = ctypes.WinDLL("ntdll")
    ntdll.NtResumeProcess.argtypes = (wintypes.HANDLE,)
    ntdll.NtResumeProcess.restype = wintypes.LONG
    status = int(ntdll.NtResumeProcess(process_handle))
    if status < 0:
        msg = f"NtResumeProcess failed with NTSTATUS 0x{status & 0xFFFFFFFF:08x}"
        raise OSError(msg)


def _windows_taskkill_path() -> str:
    """Return the absolute system `taskkill.exe` path when available."""
    windows_dir = os.environ.get("SYSTEMROOT") or os.environ.get("WINDIR")
    if windows_dir:
        return str(Path(windows_dir) / "System32" / "taskkill.exe")
    return "taskkill.exe"


def _is_windows_drive_absolute(path: str) -> bool:
    """Return whether `path` is an absolute local Windows drive path."""
    drive, tail = ntpath.splitdrive(path)
    return len(drive) == _WINDOWS_DRIVE_DESIGNATOR_LENGTH and drive[1] == ":" and tail.startswith(("\\", "/"))


def _is_windows_unc_path(path: Path) -> bool:
    """Return whether `path` names a UNC share rather than a local drive."""
    value = str(path).replace("/", "\\")
    folded = value.casefold()
    if folded.startswith("\\\\?\\unc\\"):
        return True
    return value.startswith("\\\\") and not value.startswith(("\\\\?\\", "\\\\.\\"))


def _windows_cmd_launch_paths() -> tuple[str, str, str]:
    """Return local paths for `cmd.exe`, its bootstrap cwd, and SystemRoot."""
    system_root: str | None = None
    for candidate in (
        os.environ.get("SYSTEMROOT"),
        os.environ.get("WINDIR"),
    ):
        if candidate and _is_windows_drive_absolute(candidate) and Path(candidate).is_dir():
            system_root = candidate
            break

    cmd_candidates = [os.environ.get("COMSPEC")]
    if system_root is not None:
        cmd_candidates.append(str(Path(system_root) / "System32" / "cmd.exe"))

    for candidate in cmd_candidates:
        if candidate and _is_windows_drive_absolute(candidate) and Path(candidate).is_file():
            if system_root is None:
                inferred_root = str(Path(candidate).parent.parent)
                if _is_windows_drive_absolute(inferred_root) and Path(inferred_root).is_dir():
                    system_root = inferred_root
            if system_root is not None:
                return candidate, str(Path(candidate).parent), system_root

    msg = "Cannot safely launch cmd.exe for UNC shell execution because no absolute local Windows command processor was found."
    raise OSError(msg)


def _windows_env_value(env: dict[str, str], name: str) -> str | None:
    """Read one Windows environment variable case-insensitively."""
    folded_name = name.casefold()
    for key, value in env.items():
        if key.casefold() == folded_name:
            return value
    return None


def _windows_unc_shell_launch(
    command: str,
    *,
    cwd: Path,
    env: dict[str, str],
) -> tuple[str, str, str, dict[str, str]]:
    """Build a fail-closed `cmd.exe` launch for a UNC working directory."""
    cwd_text = str(cwd)
    if "\x00" in command:
        msg = "Cannot safely execute a command containing a NUL character from a UNC working directory."
        raise ValueError(msg)
    if any(character in cwd_text for character in ('"', "\x00", "\r", "\n")):
        msg = "Cannot safely quote the configured UNC working directory for cmd.exe."
        raise ValueError(msg)

    cmd_path, bootstrap_cwd, system_root = _windows_cmd_launch_paths()
    child_env = dict(env)
    token = uuid.uuid4().hex
    cwd_variable = f"DEEPAGENTS_UNC_CWD_{token}"
    command_variable = f"DEEPAGENTS_UNC_COMMAND_{token}"
    child_env[cwd_variable] = cwd_text
    child_env[command_variable] = command

    commands: list[str] = []
    if not _windows_env_value(child_env, "SYSTEMROOT"):
        system_root_variable = f"DEEPAGENTS_SYSTEMROOT_{token}"
        child_env[system_root_variable] = system_root
        commands.append(f'@set "SystemRoot=%{system_root_variable}%"')
    else:
        system_root_variable = None

    commands.append(f'(@pushd "%{cwd_variable}%" || (@echo {_WINDOWS_UNC_PUSHD_ERROR} 1>&2 & @exit /b 1))')
    commands.append(f'@set "{cwd_variable}="')
    commands.append(
        subprocess.list2cmdline(
            [
                sys.executable,
                "-I",
                "-c",
                "import sys;payload=sys.argv.pop(1);exec(bytes.fromhex(payload))",
                _WINDOWS_UNC_COMMAND_LAUNCHER.encode().hex(),
                command_variable,
                cmd_path,
                cwd_variable,
                system_root_variable or "-",
            ]
        )
    )

    wrapper = " && ".join(commands)
    cmd_argv0 = subprocess.list2cmdline([cmd_path])
    command_line = f'{cmd_argv0} /d /e:on /v:off /s /c "{wrapper}"'
    return command_line, cmd_path, bootstrap_cwd, child_env


def _taskkill_windows_process_tree(pid: int) -> None:
    """Run bounded, fixed-argv `taskkill` for one process tree."""
    killer: subprocess.Popen[bytes] | None = None
    try:
        killer = subprocess.Popen(  # noqa: S603
            [
                _windows_taskkill_path(),
                "/PID",
                str(pid),
                "/T",
                "/F",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        killer.communicate(timeout=_WINDOWS_TERMINATE_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired):
        if killer is not None:
            with suppress(OSError):
                killer.kill()
            with suppress(OSError, subprocess.TimeoutExpired):
                killer.communicate(timeout=_WINDOWS_TERMINATE_TIMEOUT)


def _wait_for_windows_process_root_exit(
    process: subprocess.Popen[str],
    *,
    timeout: float,
) -> bool:
    """Wait a bounded interval for the Windows root process to exit."""
    if process.poll() is not None:
        return True
    try:
        process.wait(timeout=timeout)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return process.poll() is not None
    return True


def _terminate_windows_process_tree(
    process: subprocess.Popen[str],
    job: _WindowsJobObject | None,
) -> None:
    """Terminate the owned Job first, using `taskkill` only as a fallback."""
    job_terminated = False
    if job is not None:
        with suppress(OSError):
            job_terminated = job.terminate()

    root_exited = process.poll() is not None
    taskkill_required = job is not None and not job_terminated
    if job_terminated and not root_exited:
        root_exited = _wait_for_windows_process_root_exit(
            process,
            timeout=_WINDOWS_TERMINATE_TIMEOUT,
        )
        taskkill_required = not root_exited
    elif job is None:
        taskkill_required = not root_exited

    if taskkill_required:
        _taskkill_windows_process_tree(process.pid)
        root_exited = _wait_for_windows_process_root_exit(
            process,
            timeout=_WINDOWS_TERMINATE_TIMEOUT,
        )

    if not root_exited:
        with suppress(OSError):
            process.kill()
    _reap_process_root(process, timeout=_WINDOWS_TERMINATE_TIMEOUT)


def _assign_windows_job_and_resume(
    process: subprocess.Popen[str],
) -> _WindowsJobObject:
    """Assign a suspended process to a Job Object, then let it execute."""
    job: _WindowsJobObject | None = None
    try:
        process_handle = _windows_popen_process_handle(process)
        job = _WindowsJobObject.create_for_process_handle(process_handle)
        _resume_windows_process(process_handle)
    except BaseException:
        _terminate_windows_process_tree(process, job)
        raise
    return job


def _posix_process_group_api() -> tuple[_PosixKillpg, int, int] | None:
    """Return typed POSIX process-group primitives when this host provides them."""
    killpg = getattr(os, "killpg", None)
    sigterm = getattr(signal, "SIGTERM", None)
    sigkill = getattr(signal, "SIGKILL", None)
    if not callable(killpg) or not isinstance(sigterm, int) or not isinstance(sigkill, int):
        return None
    return cast("_PosixKillpg", killpg), sigterm, sigkill


def _posix_process_group_exists(killpg: _PosixKillpg, process_group_id: int) -> bool:
    """Return whether a dedicated POSIX shell process group still exists."""
    try:
        killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def _wait_for_posix_process_group_exit(
    process: subprocess.Popen[str],
    killpg: _PosixKillpg,
    process_group_id: int,
    *,
    timeout: float,
) -> bool:
    """Wait a bounded interval for a POSIX shell process group to disappear."""
    deadline = time.monotonic() + timeout
    while True:
        with suppress(OSError):
            process.poll()
        if not _posix_process_group_exists(killpg, process_group_id):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(_PROCESS_GROUP_POLL_INTERVAL, remaining))


def _reap_process_root(
    process: subprocess.Popen[str],
    *,
    timeout: float,
) -> None:
    """Bound root reaping and close captured streams on every cleanup path."""
    for stream in (process.stdout, process.stderr):
        if stream is not None:
            with suppress(OSError):
                stream.close()
    try:
        process.wait(timeout=timeout)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        if process.poll() is None:
            with suppress(OSError):
                process.kill()
        with suppress(OSError, ValueError, subprocess.TimeoutExpired):
            process.wait(timeout=timeout)


def _terminate_posix_process_group(process: subprocess.Popen[str]) -> None:
    """Terminate a dedicated POSIX process group, then reap its root."""
    process_group_api = _posix_process_group_api()
    process_group_signaled = False
    if process_group_api is not None:
        killpg, sigterm, sigkill = process_group_api
        process_group_id = process.pid
        try:
            killpg(process_group_id, sigterm)
        except ProcessLookupError:
            pass
        except OSError:
            pass
        else:
            process_group_signaled = True
            if not _wait_for_posix_process_group_exit(
                process,
                killpg,
                process_group_id,
                timeout=_POSIX_TERMINATE_TIMEOUT,
            ):
                with suppress(ProcessLookupError, OSError):
                    killpg(process_group_id, sigkill)
    if not process_group_signaled and process.poll() is None:
        with suppress(OSError):
            process.kill()

    _reap_process_root(process, timeout=_POSIX_TERMINATE_TIMEOUT)


def _terminate_shell_process(
    process: subprocess.Popen[str],
    windows_job: _WindowsJobObject | None,
) -> None:
    """Terminate the platform process tree and bound root cleanup."""
    if os.name == "nt":
        _terminate_windows_process_tree(process, windows_job)
    elif os.name == "posix":
        _terminate_posix_process_group(process)
    else:
        with suppress(OSError):
            process.kill()
        with suppress(OSError, subprocess.TimeoutExpired):
            process.communicate(timeout=_POSIX_TERMINATE_TIMEOUT)


def _close_file_descriptor(file_descriptor: int | None) -> None:
    """Close one optional file descriptor without masking the active error."""
    if file_descriptor is not None:
        with suppress(OSError):
            os.close(file_descriptor)


def _write_file_descriptor(file_descriptor: int, payload: bytes) -> None:
    """Write an immutable control payload completely to one pipe."""
    remaining = memoryview(payload)
    while remaining:
        written = os.write(file_descriptor, remaining)
        if written <= 0:
            msg = "Failed to write POSIX shell control payload"
            raise OSError(msg)
        remaining = remaining[written:]


def _serialize_posix_environment(env: dict[str, str]) -> bytes:
    """Serialize an exact environment snapshot independently of Python startup."""
    return json.dumps(
        list(env.items()),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")


def _wait_for_posix_watchdog_ready(ready_fd: int) -> None:
    """Require the owner watchdog to acknowledge readiness before shell exec."""
    with selectors.DefaultSelector() as selector:
        selector.register(ready_fd, selectors.EVENT_READ)
        ready = bool(selector.select(_POSIX_WATCHDOG_READY_TIMEOUT))
    if not ready or os.read(ready_fd, 1) != b"R":
        msg = "POSIX shell owner watchdog failed to become ready"
        raise OSError(msg)


def _start_posix_shell_process(  # noqa: PLR0915  # Explicit FD state keeps failure cleanup atomic.
    command: str,
    *,
    cwd: Path,
    env: dict[str, str],
) -> tuple[subprocess.Popen[str], _PosixOwnerGuard]:
    """Gate shell execution until an independent owner watchdog is active."""
    gate_read_fd: int | None = None
    gate_write_fd: int | None = None
    environment_read_fd: int | None = None
    environment_write_fd: int | None = None
    control_read_fd: int | None = None
    control_write_fd: int | None = None
    ready_read_fd: int | None = None
    ready_write_fd: int | None = None
    process: subprocess.Popen[str] | None = None
    watchdog: subprocess.Popen[str] | None = None
    owner_guard: _PosixOwnerGuard | None = None

    try:
        launch_env = dict(env)
        environment_payload = _serialize_posix_environment(launch_env)
        gate_read_fd, gate_write_fd = os.pipe()
        environment_read_fd, environment_write_fd = os.pipe()
        control_read_fd, control_write_fd = os.pipe()
        ready_read_fd, ready_write_fd = os.pipe()

        process = subprocess.Popen(  # noqa: S603
            [
                sys.executable,
                "-I",
                "-c",
                _POSIX_SHELL_LAUNCHER,
                str(gate_read_fd),
                str(environment_read_fd),
                command,
            ],
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            env=launch_env,
            cwd=str(cwd),
            start_new_session=True,
            pass_fds=(gate_read_fd, environment_read_fd),
        )
        _close_file_descriptor(gate_read_fd)
        gate_read_fd = None
        _close_file_descriptor(environment_read_fd)
        environment_read_fd = None

        watchdog = subprocess.Popen(  # noqa: S603
            [
                sys.executable,
                "-I",
                "-c",
                _POSIX_OWNER_WATCHDOG,
                str(control_read_fd),
                str(ready_write_fd),
                str(process.pid),
                str(os.getpid()),
                str(_POSIX_OWNER_DEATH_TERM_TIMEOUT),
                str(_OWNER_WATCHDOG_POLL_INTERVAL),
            ],
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
            pass_fds=(control_read_fd, ready_write_fd),
        )
        _close_file_descriptor(control_read_fd)
        control_read_fd = None
        _close_file_descriptor(ready_write_fd)
        ready_write_fd = None

        _wait_for_posix_watchdog_ready(ready_read_fd)
        _close_file_descriptor(ready_read_fd)
        ready_read_fd = None

        owner_guard = _PosixOwnerGuard(
            control_write_fd,
            watchdog,
            process.pid,
        )
        control_write_fd = None
        watchdog = None

        _write_file_descriptor(environment_write_fd, environment_payload)
        _close_file_descriptor(environment_write_fd)
        environment_write_fd = None
        os.write(gate_write_fd, b"G")
        _close_file_descriptor(gate_write_fd)
        gate_write_fd = None
    except BaseException:
        _close_file_descriptor(gate_write_fd)
        if process is not None:
            _terminate_posix_process_group(process)
        if owner_guard is not None:
            owner_guard.close()
        else:
            _close_file_descriptor(control_write_fd)
            if watchdog is not None:
                _reap_process_root(
                    watchdog,
                    timeout=_POSIX_WATCHDOG_EXIT_TIMEOUT,
                )
        raise
    else:
        return process, owner_guard
    finally:
        _close_file_descriptor(gate_read_fd)
        _close_file_descriptor(environment_read_fd)
        _close_file_descriptor(environment_write_fd)
        _close_file_descriptor(control_read_fd)
        _close_file_descriptor(ready_read_fd)
        _close_file_descriptor(ready_write_fd)


def _start_shell_process(
    command: str,
    *,
    cwd: Path,
    env: dict[str, str],
) -> tuple[
    subprocess.Popen[str],
    _WindowsJobObject | None,
    _PosixOwnerGuard | None,
]:
    """Start a shell process, containing Windows execution before it begins."""
    if os.name == "nt" and _is_windows_unc_path(cwd):
        command_line, executable, bootstrap_cwd, child_env = _windows_unc_shell_launch(
            command,
            cwd=cwd,
            env=env,
        )
        process = subprocess.Popen(  # noqa: S603
            command_line,
            executable=executable,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            env=child_env,
            cwd=bootstrap_cwd,
            creationflags=_WINDOWS_CREATE_SUSPENDED,
        )
    elif os.name == "posix":
        process, owner_guard = _start_posix_shell_process(
            command,
            cwd=cwd,
            env=env,
        )
        return process, None, owner_guard
    else:
        process = subprocess.Popen(  # noqa: S602
            command,
            shell=True,  # Intentional: designed for LLM-controlled shell execution
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            env=env,
            cwd=str(cwd),
            creationflags=_WINDOWS_CREATE_SUSPENDED if os.name == "nt" else 0,
        )
    if os.name == "nt":
        return process, _assign_windows_job_and_resume(process), None
    return process, None, None


def _run_shell_command(
    command: str,
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: int,
    process_registry: _ShellProcessRegistry | None = None,
) -> tuple[str, str, int]:
    """Run one shell command and return captured streams plus its exit code."""
    process: subprocess.Popen[str] | None = None
    windows_job: _WindowsJobObject | None = None
    posix_owner_guard: _PosixOwnerGuard | None = None
    try:
        process, windows_job, posix_owner_guard = _start_shell_process(
            command,
            cwd=cwd,
            env=env,
        )
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        if process is not None:
            _terminate_shell_process(process, windows_job)
        raise
    except BaseException:
        if process is not None:
            _terminate_shell_process(process, windows_job)
        raise
    else:
        return_code = process.returncode if process.returncode is not None else 1
        owned_process = _OwnedShellProcess(
            process,
            windows_job,
            posix_owner_guard,
        )
        windows_job = None
        posix_owner_guard = None
        try:
            if process_registry is None:
                if owned_process.has_processes():
                    owned_process.terminate()
                else:
                    owned_process.release()
            else:
                process_registry.register(owned_process)
        except BaseException:
            owned_process.terminate()
            raise
        return stdout, stderr, return_code
    finally:
        if posix_owner_guard is not None:
            posix_owner_guard.close()


class LocalShellBackend(FilesystemBackend, SandboxBackendProtocol):
    """Filesystem backend with unrestricted local shell command execution.

    This backend extends `FilesystemBackend` to add shell command execution
    capabilities. Commands are executed directly on the host system without any
    sandboxing, process isolation, or security restrictions.

    Windows commands are assigned to an enforced Job Object before execution.
    POSIX descendant cleanup uses a process group plus owner watchdog and is
    best-effort: deliberate re-parenting or calls to `setsid()`/`setpgid()` can
    escape portable process-group containment.

    !!! warning "Security Warning"

        This backend grants agents BOTH direct filesystem access AND
        unrestricted shell execution on your local machine. Use with extreme
        caution and only in appropriate environments.

        **Appropriate use cases:**

        - Local development CLIs (coding assistants, development tools)
        - Personal development environments where you trust the agent's code
        - CI/CD pipelines with proper secret management (see
            security considerations)

        **Inappropriate use cases:**

        - Production environments (e.g., web servers, APIs, multi-tenant systems)
        - Processing untrusted user input or executing untrusted code

        Use `StateBackend`, `StoreBackend`, or extend `BaseSandbox` for production.

        **Security risks:**

        - Agents can execute **arbitrary shell commands** with your
            user's permissions
        - Agents can read **any accessible file**, including secrets (API keys,
            credentials, `.env` files, SSH keys, etc.)
        - Combined with network tools, secrets may be exfiltrated via SSRF attacks
        - File modifications and command execution are **permanent and irreversible**
        - Agents can install packages, modify system files, spawn processes, etc.
        - **No process isolation** - commands run directly on your host system
        - **No resource limits** - commands can consume unlimited CPU, memory, disk

        **Recommended safeguards:**

        Since shell access is unrestricted and can bypass
        filesystem restrictions:

        1. **Enable Human-in-the-Loop (HITL) middleware** to review and
            approve ALL operations before execution. This is
            STRONGLY RECOMMENDED as your primary safeguard when using this backend.
        2. Run in dedicated development environments only - never on shared or
            production systems
        3. Never expose to untrusted users or allow execution of untrusted code
        4. For production environments requiring code execution, extend `BaseSandbox`
            to create a properly isolated backend (Docker containers, VMs, or
            other sandboxed execution environments)

        !!! note

            `virtual_mode=True` and path-based restrictions provide NO security
            with shell access enabled, since commands can access any path on
            the system

    Examples:
        ```python
        from deepagents.backends import LocalShellBackend

        # Create backend with explicit environment
        backend = LocalShellBackend(root_dir="/home/user/project", env={"PATH": "/usr/bin:/bin"})

        # Execute shell commands (runs directly on host)
        result = backend.execute("ls -la")
        print(result.output)
        print(result.exit_code)

        # Use filesystem operations (inherited from FilesystemBackend)
        content = backend.read("/README.md")
        backend.write("/output.txt", "Hello world")

        # Inherit all environment variables
        backend = LocalShellBackend(root_dir="/home/user/project", inherit_env=True)
        ```
    """

    def __init__(
        self,
        root_dir: str | Path | None = None,
        *,
        virtual_mode: bool = True,
        timeout: int = DEFAULT_EXECUTE_TIMEOUT,
        max_output_bytes: int = 100_000,
        env: dict[str, str] | None = None,
        inherit_env: bool = False,
    ) -> None:
        """Initialize local shell backend with filesystem access.

        Args:
            root_dir: Working directory for both filesystem operations and shell commands.

                - If not provided, defaults to the current working directory.
                - Shell commands execute with this as their working directory.
                - When `virtual_mode=False`: Paths are used as-is.

                    Agents can access any file using absolute paths or `..` sequences.
                - When `virtual_mode=True` (default): Acts as a virtual root for filesystem operations.

                    Useful with `CompositeBackend` to support routing file
                    operations across different backend implementations.

                    **Note:** This does NOT restrict shell commands.

            virtual_mode: Enable virtual path mode for filesystem operations.

                When `True` (default), treats `root_dir` as a virtual root filesystem.
                All paths are interpreted relative to `root_dir`
                (e.g., `/file.txt` maps to `{root_dir}/file.txt`).
                Path traversal (`..`, `~`) is blocked.

                **Primary use case:** Working with `CompositeBackend`, which
                routes different path prefixes to different backends. Virtual
                mode allows the `CompositeBackend` to strip route prefixes and
                pass normalized paths to each backend, enabling file operations
                to work correctly across multiple backend implementations.

                **Important:** This only affects filesystem operations.
                Shell commands executed via `execute()` are NOT restricted
                and can access any path.

            timeout: Default maximum time in seconds to wait for shell command execution.

                Defaults to 120 seconds (2 minutes).

                Commands exceeding this timeout will be terminated.

                Can be overridden per-command via the `timeout` parameter
                on `execute()`.

            max_output_bytes: Maximum number of bytes to capture from command output.
                Output exceeding this limit will be truncated.

                Defaults to 100,000 bytes.

            env: Environment variables for shell commands.

                If `None`, starts with an empty environment
                (unless `inherit_env=True`).

            inherit_env: Whether to inherit the parent process's environment variables.

                When `False` (default), only variables in `env` dict are available.

                When `True`, inherits all `os.environ` variables
                and applies `env` overrides.

        Raises:
            ValueError: If timeout is not positive.
        """
        if timeout <= 0:
            msg = f"timeout must be positive, got {timeout}"
            raise ValueError(msg)

        # Initialize parent FilesystemBackend
        super().__init__(
            root_dir=root_dir,
            virtual_mode=virtual_mode,
            max_file_size_mb=10,
        )

        # Store execution parameters
        self._default_timeout = timeout
        self._max_output_bytes = max_output_bytes

        # Build environment based on inherit_env setting
        if inherit_env:
            self._env = os.environ.copy()
            if env is not None:
                self._env.update(env)
        else:
            self._env = env if env is not None else {}

        # Generate unique sandbox ID
        self._sandbox_id = f"local-{uuid.uuid4().hex[:8]}"
        self._process_registry = _ShellProcessRegistry()
        self._process_registry_finalizer = weakref.finalize(
            self,
            self._process_registry.close,
        )

    @property
    def id(self) -> str:
        """Unique identifier for this backend instance.

        Returns:
            String identifier in format "local-{random_hex}".
        """
        return self._sandbox_id

    def close(self) -> None:
        """Terminate retained descendants through platform ownership handles.

        Windows Job containment is enforced. POSIX cleanup is best-effort
        against deliberate process-group or session escape.
        """
        self._process_registry_finalizer()

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        r"""Execute a shell command directly on the host system.

        !!! danger "Unrestricted Execution"

            Commands are executed directly on your host system
            using the system shell. There is **no sandboxing,
            isolation, or security restrictions**. The command runs with
            your user's full permissions and can:

            - Access any file on the filesystem (regardless of `virtual_mode`)
            - Execute any program or script
            - Make network connections
            - Modify system configuration
            - Spawn additional processes
            - Install packages or modify dependencies

            **Always use Human-in-the-Loop (HITL) middleware when using this method.**

        The command is executed using the system shell (`/bin/sh` or equivalent)
        with the working directory set to the backend's `root_dir`.
        Stdout and stderr are combined into a single output stream.

        Descendants remain owned until `close()` when platform containment can
        track them. Windows uses an enforced Job Object. POSIX uses best-effort
        process-group cleanup; descendants can deliberately escape with
        re-parenting, `setsid()`, or `setpgid()`.

        Args:
            command: Shell command string to execute.

                Examples: `"python script.py"`, `"ls -la"`, `"grep pattern file.txt"`

                **Security:** This string is passed directly to the shell.
                Agents can execute arbitrary commands including pipes,
                redirects, command substitution, etc.
            timeout: Maximum time in seconds to wait for this command.

                Overrides the default timeout set at init.

                If `None`, uses the default.

        Returns:
            `ExecuteResponse` containing:
                - `output`: Combined stdout and stderr (stderr lines prefixed with `[stderr]`)
                - `exit_code`: Process exit code (0 for success, non-zero for failure)
                - `truncated`: `True` if output was truncated due to size limits

        Raises:
            ValueError: If per-command timeout is not positive.

        Examples:
            ```python
            # Run a simple command
            result = backend.execute("echo hello")
            assert result.output == "hello\\n"
            assert result.exit_code == 0

            # Handle errors
            result = backend.execute("cat nonexistent.txt")
            assert result.exit_code != 0
            assert "[stderr]" in result.output

            # Check for truncation
            result = backend.execute("cat huge_file.txt")
            if result.truncated:
                print("Output was truncated")

            # Override timeout for long-running commands
            result = backend.execute("make build", timeout=300)

            # Commands run in root_dir, but can access any path
            result = backend.execute("cat /etc/passwd")  # Can read system files!
            ```
        """
        if not command or not isinstance(command, str):
            return ExecuteResponse(
                output="Error: Command must be a non-empty string.",
                exit_code=1,
                truncated=False,
            )

        effective_timeout = timeout if timeout is not None else self._default_timeout
        if effective_timeout <= 0:
            msg = f"timeout must be positive, got {effective_timeout}"
            raise ValueError(msg)

        try:
            stdout, stderr, return_code = _run_shell_command(
                command,
                cwd=self.cwd,
                env=self._env,
                timeout=effective_timeout,
                process_registry=self._process_registry,
            )

            # Combine stdout and stderr
            # Prefix each stderr line with [stderr] for clear attribution.
            # Example: "hello\n[stderr] error: file not found"  # noqa: ERA001
            output_parts = []
            if stdout:
                output_parts.append(stdout)
            if stderr:
                stderr_lines = stderr.strip().split("\n")
                output_parts.extend(f"[stderr] {line}" for line in stderr_lines)

            output = "\n".join(output_parts) if output_parts else "<no output>"

            # Check for truncation
            truncated = False
            if len(output) > self._max_output_bytes:
                output = output[: self._max_output_bytes]
                output += f"\n\n... Output truncated at {self._max_output_bytes} bytes."
                truncated = True

            # Add exit code info if non-zero
            if return_code != 0:
                output = f"{output.rstrip()}\n\nExit code: {return_code}"

            return ExecuteResponse(
                output=output,
                exit_code=return_code,
                truncated=truncated,
            )

        except subprocess.TimeoutExpired:
            if timeout is not None:
                msg = f"Error: Command timed out after {effective_timeout} seconds (custom timeout). The command may be stuck or require more time."
            else:
                msg = f"Error: Command timed out after {effective_timeout} seconds. For long-running commands, re-run using the timeout parameter."
            return ExecuteResponse(
                output=msg,
                exit_code=124,  # Standard timeout exit code
                truncated=False,
            )
        except Exception as e:  # noqa: BLE001
            # Broad exception catch is intentional: we want to catch all execution errors
            # and return a consistent ExecuteResponse rather than propagating exceptions
            return ExecuteResponse(
                output=f"Error executing command ({type(e).__name__}): {e}",
                exit_code=1,
                truncated=False,
            )


__all__ = ["DEFAULT_EXECUTE_TIMEOUT", "LocalShellBackend"]
