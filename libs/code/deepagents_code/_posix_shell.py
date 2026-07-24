"""POSIX process-group ownership through a dedicated watchdog process.

The watchdog is the retained ownership token after a command root exits. It
stops when the original process group disappears, so later cleanup never sends
a signal from the owner process to a bare, potentially reused PGID.

Process-group containment is best-effort: a descendant can deliberately escape
portable cleanup by creating a new process group or session with `setpgid()` or
`setsid()`, including through deliberate re-parenting.
"""

from __future__ import annotations

import asyncio
import json
import os
import selectors
import signal
import subprocess  # noqa: S404
import sys
import time
from contextlib import suppress
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

_POSIX_TERMINATE_TIMEOUT = 5.0
_POSIX_OWNER_DEATH_TERM_TIMEOUT = 1.0
_POSIX_WATCHDOG_READY_TIMEOUT = 5.0
_POSIX_WATCHDOG_EXIT_TIMEOUT = 5.0
_OWNER_WATCHDOG_POLL_INTERVAL = 0.1

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
process_group_id = int(sys.argv[2])
owner_pid = int(sys.argv[3])
term_timeout = float(sys.argv[4])
poll_interval = float(sys.argv[5])

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
        os.write(1, b"R")
        while os.getppid() == owner_pid:
            if not group_exists():
                cleanup_requested = False
                break
            if selector.select(poll_interval):
                cleanup_requested = os.read(control_fd, 1) != b"C"
                break
finally:
    os.close(control_fd)

if not cleanup_requested:
    raise SystemExit(0)

if not group_exists():
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


def _close_file_descriptor(file_descriptor: int | None) -> None:
    """Close an optional file descriptor without masking cleanup."""
    if file_descriptor is not None:
        with suppress(OSError):
            os.close(file_descriptor)


def _posix_process_group_exists(process_group_id: int) -> bool:
    """Return whether a dedicated command process group still has members."""
    killpg = getattr(os, "killpg", None)
    if not callable(killpg):
        return False
    try:
        killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def _close_asyncio_subprocess_transport(process: asyncio.subprocess.Process) -> None:
    """Close a subprocess transport after bounded process reaping."""
    transport = getattr(process, "_transport", None)
    if transport is not None:
        with suppress(Exception):
            transport.close()


async def _reap_process(
    process: asyncio.subprocess.Process,
    *,
    wait_seconds: float,
) -> None:
    """Bound waiting for one subprocess, killing only that subprocess."""
    try:
        await asyncio.wait_for(process.wait(), timeout=wait_seconds)
    except (OSError, ProcessLookupError, TimeoutError):
        if process.returncode is None:
            with suppress(OSError, ProcessLookupError):
                process.kill()
        with suppress(OSError, ProcessLookupError, TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=wait_seconds)
    finally:
        _close_asyncio_subprocess_transport(process)


async def _reap_watchdog(process: asyncio.subprocess.Process) -> None:
    """Bound waiting for one specific watchdog, killing only that watchdog."""
    await _reap_process(
        process,
        wait_seconds=_POSIX_WATCHDOG_EXIT_TIMEOUT,
    )


def _reap_sync_watchdog(
    process: subprocess.Popen[bytes],
    *,
    wait_seconds: float,
) -> bool:
    """Bound waiting for one synchronous watchdog, killing only that watchdog.

    Returns:
        Whether the watchdog exited without being force-killed.
    """
    forced = False
    try:
        process.wait(timeout=wait_seconds)
    except (OSError, subprocess.TimeoutExpired):
        if process.poll() is None:
            forced = True
            with suppress(OSError, ProcessLookupError):
                process.kill()
        with suppress(OSError, subprocess.TimeoutExpired):
            process.wait(timeout=_POSIX_WATCHDOG_EXIT_TIMEOUT)
    finally:
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                with suppress(OSError):
                    stream.close()
    return not forced and process.poll() is not None


class _PosixOwnerGuard:
    """Control and reap one watchdog that owns an original process group."""

    def __init__(
        self,
        control_fd: int,
        watchdog: asyncio.subprocess.Process,
        process_group_id: int,
    ) -> None:
        self._control_fd: int | None = control_fd
        self._watchdog: asyncio.subprocess.Process | None = watchdog
        self._process_group_id: int | None = process_group_id
        self._lock = asyncio.Lock()

    def is_alive(self) -> bool:
        """Return whether the watchdog subprocess is still running."""
        watchdog = self._watchdog
        return watchdog is not None and watchdog.returncode is None

    def has_processes(self) -> bool:
        """Return whether the live watchdog's command group has members."""
        watchdog = self._watchdog
        process_group_id = self._process_group_id
        if (
            watchdog is None
            or process_group_id is None
            or watchdog.returncode is not None
        ):
            return False
        if not _posix_process_group_exists(process_group_id):
            return False
        return self._watchdog is watchdog and watchdog.returncode is None

    async def close(self) -> None:
        """Release a watchdog after confirming no cleanup is needed."""
        await self._finish(b"C")

    async def terminate(self) -> None:
        """Ask the still-owned watchdog to terminate its process group."""
        await self._finish(b"T")

    async def _finish(self, action: bytes) -> None:
        """Send one idempotent action and await the specific watchdog."""
        async with self._lock:
            if action == b"T" and not self.has_processes():
                action = b"C"
            control_fd = self._control_fd
            watchdog = self._watchdog
            self._control_fd = None
            self._watchdog = None
            self._process_group_id = None

            if control_fd is not None:
                if watchdog is not None and watchdog.returncode is None:
                    with suppress(OSError):
                        os.write(control_fd, action)
                _close_file_descriptor(control_fd)
            if watchdog is not None:
                await _reap_watchdog(watchdog)


class _SyncPosixOwnerGuard:
    """Control and reap one synchronous watchdog ownership token."""

    def __init__(
        self,
        control_fd: int,
        watchdog: subprocess.Popen[bytes],
        *,
        wait_timeout: float,
    ) -> None:
        self._control_fd: int | None = control_fd
        self._watchdog: subprocess.Popen[bytes] | None = watchdog
        self._wait_timeout = wait_timeout

    def is_alive(self) -> bool:
        """Return whether the watchdog subprocess is still running."""
        watchdog = self._watchdog
        return watchdog is not None and watchdog.poll() is None

    def close(self) -> bool:
        """Release the watchdog without signaling its process group.

        Returns:
            Whether the live watchdog accepted and completed the action.
        """
        return self._finish(b"C")

    def terminate(
        self,
        *,
        reap: Callable[[], object] | None = None,
    ) -> bool:
        """Ask the still-owned watchdog to terminate its process group.

        Args:
            reap: Optional callback polled while the watchdog runs. Server
                teardown uses this to reap the process-group leader so its
                zombie does not force an unnecessary SIGKILL escalation.

        Returns:
            Whether the live watchdog accepted and completed the action.
        """
        return self._finish(b"T", reap=reap)

    def _finish(
        self,
        action: bytes,
        *,
        reap: Callable[[], object] | None = None,
    ) -> bool:
        """Send one idempotent action and await the specific watchdog.

        Returns:
            Whether the live watchdog accepted and completed the action.
        """
        control_fd = self._control_fd
        watchdog = self._watchdog
        self._control_fd = None
        self._watchdog = None

        action_sent = False
        if control_fd is not None:
            if watchdog is not None and watchdog.poll() is None:
                try:
                    _write_file_descriptor(control_fd, action)
                except OSError:
                    pass
                else:
                    action_sent = True
            _close_file_descriptor(control_fd)

        if watchdog is None:
            return False

        deadline = time.monotonic() + self._wait_timeout
        while watchdog.poll() is None:
            if reap is not None:
                with suppress(OSError, ProcessLookupError):
                    reap()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(_OWNER_WATCHDOG_POLL_INTERVAL, remaining))

        completed = _reap_sync_watchdog(
            watchdog,
            wait_seconds=max(0.0, deadline - time.monotonic()),
        )
        return action_sent and completed


class _PosixOwnerRegistry:
    """Own live command-group watchdogs and reap completed ownership."""

    def __init__(self) -> None:
        self._guards: list[_PosixOwnerGuard] = []
        self._closed = False
        self._lock = asyncio.Lock()

    def __len__(self) -> int:
        """Return the number of retained command groups."""
        return len(self._guards)

    async def retain(self, guard: _PosixOwnerGuard) -> bool:
        """Retain one live command group after pruning completed watchdogs.

        Returns:
            Whether the guard was retained.
        """
        async with self._lock:
            if self._closed:
                await guard.terminate()
                return False

            try:
                await self._prune_locked()
            except BaseException:
                await guard.terminate()
                raise

            if guard.has_processes():
                self._guards.append(guard)
                return True

            await guard.close()
            return False

    async def _prune_locked(self) -> None:
        """Release completed guards while the registry lock is held."""
        retained: list[_PosixOwnerGuard] = []
        first_error: Exception | None = None
        for guard in self._guards:
            if guard.has_processes():
                retained.append(guard)
                continue
            try:
                await guard.close()
            except Exception as exc:  # noqa: BLE001  # Continue independent cleanup.
                if first_error is None:
                    first_error = exc
        self._guards = retained
        if first_error is not None:
            raise first_error

    async def close(self) -> None:
        """Terminate retained groups and reap all watchdogs exactly once."""
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            guards = self._guards
            self._guards = []

        first_error: Exception | None = None
        for guard in reversed(guards):
            try:
                if guard.has_processes():
                    await guard.terminate()
                else:
                    await guard.close()
            except Exception as exc:  # noqa: BLE001  # Continue independent cleanup.
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error


async def _start_posix_owner_guard(process_group_id: int) -> _PosixOwnerGuard:
    """Start and arm a watchdog for an already isolated process group.

    Args:
        process_group_id: Session-leader PID and process-group identifier.

    Returns:
        The armed watchdog ownership handle.

    Raises:
        OSError: If the watchdog cannot start or acknowledge readiness.
    """
    control_read_fd: int | None = None
    control_write_fd: int | None = None
    watchdog: asyncio.subprocess.Process | None = None

    try:
        control_read_fd, control_write_fd = os.pipe()
        watchdog = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-c",
            _POSIX_OWNER_WATCHDOG,
            str(control_read_fd),
            str(process_group_id),
            str(os.getpid()),
            str(_POSIX_OWNER_DEATH_TERM_TIMEOUT),
            str(_OWNER_WATCHDOG_POLL_INTERVAL),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
            pass_fds=(control_read_fd,),
        )
        _close_file_descriptor(control_read_fd)
        control_read_fd = None

        if watchdog.stdout is None:
            msg = "POSIX shell owner watchdog readiness pipe is unavailable"
            raise OSError(msg)
        try:
            ready = await asyncio.wait_for(
                watchdog.stdout.readexactly(1),
                timeout=_POSIX_WATCHDOG_READY_TIMEOUT,
            )
        except (asyncio.IncompleteReadError, TimeoutError) as exc:
            msg = "POSIX shell owner watchdog failed to become ready"
            raise OSError(msg) from exc
        if ready != b"R":
            msg = "POSIX shell owner watchdog sent an invalid readiness token"
            raise OSError(msg)

        owner_guard = _PosixOwnerGuard(
            control_write_fd,
            watchdog,
            process_group_id,
        )
        control_write_fd = None
        watchdog = None
        return owner_guard
    finally:
        _close_file_descriptor(control_read_fd)
        _close_file_descriptor(control_write_fd)
        if watchdog is not None:
            await _reap_watchdog(watchdog)


def _start_sync_posix_owner_guard(
    process_group_id: int,
    *,
    term_timeout: float,
) -> _SyncPosixOwnerGuard:
    """Start and arm a synchronous watchdog for an isolated process group.

    Args:
        process_group_id: Session-leader PID and process-group identifier.
        term_timeout: Grace period before the watchdog escalates to SIGKILL.

    Returns:
        The armed synchronous watchdog ownership handle.

    Raises:
        OSError: If the watchdog cannot start or acknowledge readiness.
    """
    control_read_fd: int | None = None
    control_write_fd: int | None = None
    watchdog: subprocess.Popen[bytes] | None = None

    try:
        control_read_fd, control_write_fd = os.pipe()
        watchdog = subprocess.Popen(  # noqa: S603
            [
                sys.executable,
                "-I",
                "-c",
                _POSIX_OWNER_WATCHDOG,
                str(control_read_fd),
                str(process_group_id),
                str(os.getpid()),
                str(term_timeout),
                str(_OWNER_WATCHDOG_POLL_INTERVAL),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            pass_fds=(control_read_fd,),
        )
        _close_file_descriptor(control_read_fd)
        control_read_fd = None

        if watchdog.stdout is None:
            msg = "POSIX owner watchdog readiness pipe is unavailable"
            raise OSError(msg)
        with selectors.DefaultSelector() as selector:
            selector.register(watchdog.stdout, selectors.EVENT_READ)
            if not selector.select(_POSIX_WATCHDOG_READY_TIMEOUT):
                msg = "POSIX owner watchdog failed to become ready"
                raise OSError(msg)
        ready = watchdog.stdout.read(1)
        if ready != b"R":
            msg = "POSIX owner watchdog sent an invalid readiness token"
            raise OSError(msg)
        watchdog.stdout.close()

        owner_guard = _SyncPosixOwnerGuard(
            control_write_fd,
            watchdog,
            wait_timeout=term_timeout + _POSIX_WATCHDOG_EXIT_TIMEOUT,
        )
        control_write_fd = None
        watchdog = None
        return owner_guard
    finally:
        _close_file_descriptor(control_read_fd)
        if control_write_fd is not None:
            if watchdog is not None and watchdog.poll() is None:
                with suppress(OSError):
                    _write_file_descriptor(control_write_fd, b"C")
            _close_file_descriptor(control_write_fd)
        if watchdog is not None:
            _reap_sync_watchdog(
                watchdog,
                wait_seconds=_POSIX_WATCHDOG_EXIT_TIMEOUT,
            )


def _write_file_descriptor(file_descriptor: int, payload: bytes) -> None:
    """Write an immutable payload completely to one file descriptor.

    Raises:
        OSError: If the descriptor cannot accept the complete payload.
    """
    remaining = memoryview(payload)
    while remaining:
        written = os.write(file_descriptor, remaining)
        if written <= 0:
            msg = "Failed to write POSIX shell control payload"
            raise OSError(msg)
        remaining = remaining[written:]


def _serialize_environment(environment: dict[str, str]) -> bytes:
    """Serialize an exact environment snapshot independently of Python startup.

    Returns:
        ASCII JSON bytes preserving the supplied environment entries.
    """
    return json.dumps(
        list(environment.items()),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")


async def _terminate_starting_process(process: asyncio.subprocess.Process) -> None:
    """Terminate a shell group while its session-leader identity is live."""
    killpg = getattr(os, "killpg", None)
    sigkill = getattr(signal, "SIGKILL", 9)
    if callable(killpg) and isinstance(sigkill, int):
        with suppress(ProcessLookupError, OSError):
            killpg(process.pid, sigkill)
    elif process.returncode is None:
        with suppress(OSError, ProcessLookupError):
            process.kill()
    await _reap_process(
        process,
        wait_seconds=_POSIX_TERMINATE_TIMEOUT,
    )


async def _start_posix_shell_process(
    command: str,
    *,
    cwd: str | None = None,
) -> tuple[asyncio.subprocess.Process, _PosixOwnerGuard]:
    """Gate shell execution until its independent owner watchdog is active.

    Args:
        command: Shell command to execute.
        cwd: Optional working directory for the shell.

    Returns:
        The gated shell process and its armed watchdog ownership handle.

    """
    gate_read_fd: int | None = None
    gate_write_fd: int | None = None
    environment_read_fd: int | None = None
    environment_write_fd: int | None = None
    process: asyncio.subprocess.Process | None = None
    owner_guard: _PosixOwnerGuard | None = None

    try:
        environment = dict(os.environ)
        environment_payload = _serialize_environment(environment)
        gate_read_fd, gate_write_fd = os.pipe()
        environment_read_fd, environment_write_fd = os.pipe()
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-c",
            _POSIX_SHELL_LAUNCHER,
            str(gate_read_fd),
            str(environment_read_fd),
            command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=environment,
            start_new_session=True,
            pass_fds=(gate_read_fd, environment_read_fd),
        )
        _close_file_descriptor(gate_read_fd)
        gate_read_fd = None
        _close_file_descriptor(environment_read_fd)
        environment_read_fd = None

        owner_guard = await _start_posix_owner_guard(process.pid)
        _write_file_descriptor(environment_write_fd, environment_payload)
        _close_file_descriptor(environment_write_fd)
        environment_write_fd = None
        _write_file_descriptor(gate_write_fd, b"G")
        _close_file_descriptor(gate_write_fd)
        gate_write_fd = None
    except BaseException:
        _close_file_descriptor(gate_write_fd)
        if owner_guard is not None:
            await owner_guard.terminate()
        elif process is not None:
            await _terminate_starting_process(process)
        if process is not None and process.returncode is None:
            await _reap_process(
                process,
                wait_seconds=_POSIX_TERMINATE_TIMEOUT,
            )
        raise
    else:
        return process, owner_guard
    finally:
        _close_file_descriptor(gate_read_fd)
        _close_file_descriptor(environment_read_fd)
        _close_file_descriptor(environment_write_fd)


__all__ = [
    "_PosixOwnerGuard",
    "_PosixOwnerRegistry",
    "_SyncPosixOwnerGuard",
    "_start_posix_owner_guard",
    "_start_posix_shell_process",
    "_start_sync_posix_owner_guard",
]
