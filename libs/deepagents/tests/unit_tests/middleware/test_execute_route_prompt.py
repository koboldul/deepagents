"""Tests for the shell-vs-virtual-path prompt section (issue #3050).

`execute` runs on the default backend's host shell, so routed virtual paths
(e.g. `/common/`) don't exist there. Instead of rewriting commands — which can't
be done correctly for arbitrary shell — the middleware tells the model how to
translate each route's virtual prefix to its host path so it forms the correct
command itself.

A route gets a host mapping only when its files live on the same filesystem the
default's shell runs in: a `LocalShellBackend` default (local shell) paired with
a `FilesystemBackend` route (local disk). A remote/sandbox default runs its shell
elsewhere, so local filesystem routes are not reachable and must be classified as
shell-inaccessible. These tests cover that matrix.
"""

import os
import subprocess
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest
from langgraph.store.memory import InMemoryStore

from deepagents.backends.composite import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.local_shell import LocalShellBackend
from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol
from deepagents.backends.state import StateBackend
from deepagents.backends.store import StoreBackend
from deepagents.middleware.filesystem import _route_host_path_prompt

_NO_HOST_HEADING = "Virtual mounts without a host path mapping"


def _store() -> StoreBackend:
    return StoreBackend(store=InMemoryStore(), namespace=lambda _rt: ("ns",))


def _local_shell() -> LocalShellBackend:
    """A local-shell default whose shell shares the local filesystem with routes."""
    return LocalShellBackend(virtual_mode=True)


class _RemoteSandbox(SandboxBackendProtocol, StoreBackend):
    """A sandbox-capable default that is NOT a LocalShellBackend (e.g. remote).

    Its shell runs in a separate filesystem, so local filesystem routes are not
    reachable from it.
    """

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        return ExecuteResponse(output="", exit_code=0, truncated=False)

    @property
    def id(self) -> str:
        return "remote_sandbox"


def test_returns_empty_for_non_composite_backend() -> None:
    assert _route_host_path_prompt(StateBackend()) == ""


def test_returns_empty_when_no_routes() -> None:
    comp = CompositeBackend(default=_local_shell(), routes={})
    assert _route_host_path_prompt(comp) == ""


def test_posix_maps_virtual_route_with_forward_slashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    monkeypatch.setattr(route, "cwd", PurePosixPath("/srv/agent work"))
    comp = CompositeBackend(default=_local_shell(), routes={"/common/": route})

    prompt = _route_host_path_prompt(comp)

    assert "## Shell paths vs. virtual paths" in prompt
    assert "- `/common/` -> `/srv/agent work/`" in prompt
    assert "`/common/dir/x.py` -> `/srv/agent work/dir/x.py`" in prompt
    assert "\\" not in prompt


def test_windows_drive_prompt_uses_backslashes_and_cmd_quoting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    monkeypatch.setattr(route, "cwd", PureWindowsPath("C:/agent work/common"))
    comp = CompositeBackend(default=_local_shell(), routes={"/common/": route})

    prompt = _route_host_path_prompt(comp)

    assert r"- `/common/` -> `C:\agent work\common\`" in prompt
    assert r'`/common/dir/x.py` -> `"C:\agent work\common\dir\x.py"`' in prompt
    assert "C:/agent work/common" not in prompt
    assert r"`\common\dir\x.py`" not in prompt

    selected_backend, selected_path = comp._get_backend_and_key("/common/dir/x.py")
    assert selected_backend is route
    assert selected_path == "/dir/x.py"


def test_windows_percent_path_prompt_uses_cmd_literal_percent_segments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    monkeypatch.setattr(
        route,
        "cwd",
        PureWindowsPath(r"C:\agent %ROUTE_TEST_VAR%\common"),
    )
    comp = CompositeBackend(default=_local_shell(), routes={"/common/": route})

    prompt = _route_host_path_prompt(comp)

    assert r"- `/common/` -> `C:\agent %ROUTE_TEST_VAR%\common\`" in prompt
    assert r'`/common/dir/x.py` -> `"C:\agent "^%"ROUTE_TEST_VAR"^%"\common\dir\x.py"`' in prompt
    assert r'"C:\agent %ROUTE_TEST_VAR%\common\dir\x.py"' not in prompt
    assert "preventing `%NAME%` environment expansion" in prompt


def test_windows_unc_prompt_uses_backslashes_and_share_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    monkeypatch.setattr(
        route,
        "cwd",
        PureWindowsPath(r"\\server\share\agent work"),
    )
    comp = CompositeBackend(default=_local_shell(), routes={"/common/": route})

    prompt = _route_host_path_prompt(comp)

    assert r"- `/common/` -> `\\server\share\agent work\`" in prompt
    assert r'`/common/dir/x.py` -> `"\\server\share\agent work\dir\x.py"`' in prompt
    assert "//server/share" not in prompt


@pytest.mark.skipif(sys.platform != "win32", reason="requires cmd.exe")
def test_windows_mapped_path_example_is_valid_cmd_syntax(tmp_path: Path) -> None:
    route_root = tmp_path / "route & shell"
    target = route_root / "dir" / "x.py"
    target.parent.mkdir(parents=True)
    target.write_text("cmd path ok\n", encoding="utf-8")
    route = FilesystemBackend(root_dir=str(route_root), virtual_mode=True)
    comp = CompositeBackend(default=_local_shell(), routes={"/common/": route})

    prompt = _route_host_path_prompt(comp)
    quoted_target = f'"{target}"'

    assert f"`{quoted_target}`" in prompt
    cmd = os.environ.get("COMSPEC", "cmd.exe")
    command_line = f'"{cmd}" /d /c type {quoted_target}'
    result = subprocess.run(  # noqa: S603  # Fixed cmd.exe argv validates rendered syntax.
        command_line,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.strip() == "cmd path ok"


@pytest.mark.skipif(sys.platform != "win32", reason="requires cmd.exe")
def test_windows_percent_mapped_path_is_literal_in_native_cmd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variable_name = "DEEPAGENTS_ROUTE_TEST_VAR"
    route_root = tmp_path / f"route %{variable_name}% & shell"
    target = route_root / "dir" / "x.py"
    target.parent.mkdir(parents=True)
    target.write_text("literal percent path\n", encoding="utf-8")

    expanded_target = tmp_path / "route expanded-value & shell" / "dir" / "x.py"
    expanded_target.parent.mkdir(parents=True)
    expanded_target.write_text("expanded path\n", encoding="utf-8")
    monkeypatch.setenv(variable_name, "expanded-value")

    route = FilesystemBackend(root_dir=str(route_root), virtual_mode=True)
    comp = CompositeBackend(default=_local_shell(), routes={"/common/": route})

    prompt = _route_host_path_prompt(comp)
    cmd_safe_target = '"' + str(target).replace("%", '"^%"') + '"'

    assert f"`{cmd_safe_target}`" in prompt
    assert '"^%"' in cmd_safe_target
    cmd = os.environ.get("COMSPEC", "cmd.exe")
    command_line = f'"{cmd}" /d /v:off /c type {cmd_safe_target}'
    result = subprocess.run(  # noqa: S603  # Native cmd.exe verifies literal percent handling.
        command_line,
        capture_output=True,
        text=True,
        check=False,
        env=dict(os.environ),
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.strip() == "literal percent path"


def test_routes_without_host_path_marked_inaccessible() -> None:
    comp = CompositeBackend(default=_local_shell(), routes={"/memories/": _store()})

    prompt = _route_host_path_prompt(comp)

    # A store mount has no host path, so it appears under the no-mapping section
    # and is never presented as a host path mapping — even with a local default.
    assert _NO_HOST_HEADING in prompt
    assert "`/memories/`" in prompt
    assert " -> " not in prompt


def test_posix_non_virtual_filesystem_route_maps_to_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Non-virtual routes strip the prefix and use the remaining absolute path
    # as-is on the host (root_dir ignored), so the prefix maps to the filesystem
    # root `/`.
    route = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    monkeypatch.setattr(route, "cwd", PurePosixPath("/srv/legacy"))
    comp = CompositeBackend(default=_local_shell(), routes={"/common/": route})

    prompt = _route_host_path_prompt(comp)

    assert "- `/common/` -> `/`" in prompt
    assert "`/common/dir/x.py` -> `/dir/x.py`" in prompt


def test_windows_non_virtual_route_maps_to_drive_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    monkeypatch.setattr(route, "cwd", PureWindowsPath("D:/legacy/root"))
    comp = CompositeBackend(default=_local_shell(), routes={"/common/": route})

    prompt = _route_host_path_prompt(comp)

    assert r"- `/common/` -> `D:\`" in prompt
    assert r'`/common/dir/x.py` -> `"D:\dir\x.py"`' in prompt


def test_non_trailing_route_prefix_renders_with_slash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    monkeypatch.setattr(route, "cwd", PurePosixPath("/srv/data"))
    comp = CompositeBackend(default=_local_shell(), routes={"/data": route})

    prompt = _route_host_path_prompt(comp)

    assert "- `/data/` -> `/srv/data/`" in prompt
    assert "`/data/dir/x.py`" in prompt
    assert "/datadir" not in prompt


def test_non_virtual_route_not_mapped_under_remote_sandbox(tmp_path: Path) -> None:
    # Under a remote sandbox default, even a non-virtual local route is on local
    # disk and unreachable from the sandbox shell -> no mapping.
    route = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
    comp = CompositeBackend(
        default=_RemoteSandbox(store=InMemoryStore(), namespace=lambda _rt: ("default",)),
        routes={"/common/": route},
    )

    prompt = _route_host_path_prompt(comp)

    assert " -> " not in prompt
    assert _NO_HOST_HEADING in prompt
    assert "`/common/`" in prompt


def test_mix_of_host_and_non_host_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fs = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    monkeypatch.setattr(fs, "cwd", PurePosixPath("/srv/common"))
    comp = CompositeBackend(
        default=_local_shell(),
        routes={"/common/": fs, "/memories/": _store()},
    )

    prompt = _route_host_path_prompt(comp)

    assert "- `/common/` -> `/srv/common/`" in prompt
    assert _NO_HOST_HEADING in prompt
    assert "`/memories/`" in prompt


def test_remote_sandbox_default_suppresses_host_mappings(tmp_path: Path) -> None:
    # The same virtual-mode FilesystemBackend route that maps under a local-shell
    # default must NOT get a host mapping under a remote/sandbox default: its files
    # are on local disk, unreachable from the sandbox shell.
    route = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    comp = CompositeBackend(
        default=_RemoteSandbox(store=InMemoryStore(), namespace=lambda _rt: ("default",)),
        routes={"/common/": route},
    )

    prompt = _route_host_path_prompt(comp)

    assert " -> " not in prompt  # no host mapping emitted
    assert _NO_HOST_HEADING in prompt
    assert "`/common/`" in prompt
