"""Tests for shared POSIX shell ownership."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from deepagents_code import _posix_shell as posix_shell_module
from deepagents_code._posix_shell import _PosixOwnerGuard, _PosixOwnerRegistry


def _process_state(pid: int) -> str | None:
    """Return one Linux process state without reaping the process."""
    try:
        return (
            (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8").split()[2]
        )
    except (IndexError, OSError):
        return None


def _file_descriptor_count() -> int:
    """Return the current process's Linux file-descriptor count."""
    return len(tuple((Path("/proc") / "self" / "fd").iterdir()))


@pytest.mark.skipif(
    sys.platform == "win32" or not Path("/proc/self/fd").is_dir(),
    reason="Linux foreground-command resource regression",
)
async def test_posix_registry_foreground_commands_do_not_leak_watchdogs(
    tmp_path: Path,
) -> None:
    """Fifty foreground commands leave no retained guards, FDs, or zombies."""
    command_count = 50
    watchdog_pids: list[int] = []
    original_init = _PosixOwnerGuard.__init__

    def capture_watchdog(
        owner_guard: _PosixOwnerGuard,
        control_fd: int,
        watchdog: asyncio.subprocess.Process,
        process_group_id: int,
    ) -> None:
        watchdog_pids.append(watchdog.pid)
        original_init(
            owner_guard,
            control_fd,
            watchdog,
            process_group_id,
        )

    registry = _PosixOwnerRegistry()
    baseline_file_descriptors = _file_descriptor_count()
    maximum_registry_size = 0

    try:
        with patch.object(
            _PosixOwnerGuard,
            "__init__",
            new=capture_watchdog,
        ):
            for _ in range(command_count):
                (
                    process,
                    owner_guard,
                ) = await posix_shell_module._start_posix_shell_process(
                    ":",
                    cwd=str(tmp_path),
                )
                stdout, stderr = await process.communicate()
                retained = await registry.retain(owner_guard)
                posix_shell_module._close_asyncio_subprocess_transport(process)

                assert stdout == b""
                assert stderr == b""
                assert not retained
                maximum_registry_size = max(maximum_registry_size, len(registry))

        await asyncio.sleep(posix_shell_module._OWNER_WATCHDOG_POLL_INTERVAL * 2)
        final_file_descriptors = _file_descriptor_count()
        zombie_watchdogs = [pid for pid in watchdog_pids if _process_state(pid) == "Z"]

        assert len(watchdog_pids) == command_count
        assert maximum_registry_size == 0
        assert len(registry) == 0
        assert final_file_descriptors <= baseline_file_descriptors + 2
        assert zombie_watchdogs == []
    finally:
        await registry.close()


async def test_posix_registry_reaps_dead_watchdog_on_insertion() -> None:
    """A dead candidate closes its control FD and is never retained."""
    watchdog = MagicMock()
    watchdog.returncode = 0
    watchdog.wait = AsyncMock(return_value=0)
    watchdog._transport = MagicMock()
    owner_guard = _PosixOwnerGuard(41, watchdog, 1234)
    registry = _PosixOwnerRegistry()

    with (
        patch.object(posix_shell_module.os, "write") as write,
        patch.object(posix_shell_module.os, "close") as close,
    ):
        retained = await registry.retain(owner_guard)
        await registry.close()
        await registry.close()

    assert not retained
    assert len(registry) == 0
    write.assert_not_called()
    close.assert_called_once_with(41)
    watchdog.wait.assert_awaited_once_with()
    watchdog._transport.close.assert_called_once_with()


async def test_posix_registry_prunes_completed_guard_before_insertion() -> None:
    """Insertion releases existing and candidate guards whose groups are empty."""
    existing = MagicMock()
    existing.has_processes.side_effect = [True, False]
    existing.close = AsyncMock()
    candidate = MagicMock()
    candidate.has_processes.return_value = False
    candidate.close = AsyncMock()
    registry = _PosixOwnerRegistry()

    assert await registry.retain(existing)
    assert not await registry.retain(candidate)

    assert len(registry) == 0
    existing.close.assert_awaited_once_with()
    candidate.close.assert_awaited_once_with()


async def test_posix_registry_shutdown_closes_group_that_already_exited() -> None:
    """Shutdown sends no terminate action after a retained group disappears."""
    watchdog = MagicMock()
    watchdog.returncode = None
    watchdog.wait = AsyncMock(return_value=0)
    watchdog._transport = MagicMock()
    owner_guard = _PosixOwnerGuard(43, watchdog, 1234)
    registry = _PosixOwnerRegistry()

    with (
        patch.object(
            posix_shell_module,
            "_posix_process_group_exists",
            side_effect=[True, False],
        ),
        patch.object(posix_shell_module.os, "write") as write,
        patch.object(posix_shell_module.os, "close") as close,
    ):
        assert await registry.retain(owner_guard)
        await registry.close()

    assert len(registry) == 0
    assert write.call_args_list == [call(43, b"C")]
    close.assert_called_once_with(43)
    watchdog.wait.assert_awaited_once_with()
    watchdog._transport.close.assert_called_once_with()
