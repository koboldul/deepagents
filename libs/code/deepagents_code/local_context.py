"""Middleware for injecting runtime and project context into the system prompt.

Local dcode sessions collect context directly from the captured project
directory with Python and fixed-argument subprocess calls. Remote Linux
sandboxes retain the backend-executed Bash detection script.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
import stat
import subprocess  # noqa: S404  # All probes use fixed argv and shell=False.
import sys
import threading
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import (
    IO,
    TYPE_CHECKING,
    Annotated,
    Any,
    NotRequired,
    Protocol,
    cast,
    runtime_checkable,
)

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
)

from deepagents_code.unicode_security import sanitize_control_chars

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from deepagents.backends.protocol import ExecuteResponse
    from deepagents.middleware.summarization import SummarizationEvent
    from langgraph.runtime import Runtime

    from deepagents_code.mcp_tools import MCPServerInfo


_TOOL_NAME_DISPLAY_LIMIT = 10
"""Maximum number of tool names shown per MCP server in the system prompt."""

_DETECT_SCRIPT_TIMEOUT = 30
"""Timeout in seconds for the environment detection script."""

_MCP_ERROR_DETAIL_LIMIT = 200
"""Max characters of an MCP server error surfaced in the system prompt."""

_TRACING_PROJECT_NAME_LIMIT = 200
"""Max characters of a LangSmith project name surfaced in the system prompt."""

_LOCAL_COMMAND_TIMEOUT = 5
"""Timeout in seconds for local context subprocess probes."""

_LOCAL_COMMAND_OUTPUT_LIMIT = 1_000_000
"""Maximum captured characters from a local context subprocess probe."""

_LOCAL_COMMAND_READ_CHUNK_SIZE = 64 * 1024
"""Chunk size used while draining bounded local subprocess output."""

_LOCAL_FILE_LIMIT = 20
"""Maximum top-level project entries shown in local context."""

_LOCAL_TREE_DEPTH = 3
"""Maximum directory depth shown in the local context preview."""

_LOCAL_TREE_LINE_LIMIT = 22
"""Maximum directory-preview lines shown before the truncation marker."""

_LOCAL_MAKEFILE_LINE_LIMIT = 20
"""Maximum Makefile lines shown in local context."""

_WINDOWS_DEFAULT_PATHEXT = (".COM", ".EXE", ".BAT", ".CMD")
"""Executable suffixes used when Windows does not define `PATHEXT`."""

_QUOTED_PATH_ENTRY_MIN_LENGTH = 2
"""Minimum length of a PATH entry wrapped in matching quote characters."""

_GIT_PROBE_ENVIRONMENT_KEYS = frozenset(
    {
        "COMSPEC",
        "HOMEDRIVE",
        "HOMEPATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "PATH",
        "PATHEXT",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "USERPROFILE",
        "WINDIR",
    }
)
"""Environment variables safe and necessary for fixed local Git probes."""

_LOCAL_EXCLUDED_NAMES = frozenset(
    {
        ".coverage",
        ".eggs",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
    }
)


@dataclass(frozen=True)
class _CommandResult:
    """Bounded result from a fixed-argument local subprocess probe."""

    returncode: int
    stdout: str
    truncated: bool


class _BoundedTextBuffer:
    """Drain a text stream while retaining at most a fixed character count."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._chunks: list[str] = []
        self._length = 0
        self.failed = False

    def drain(self, stream: IO[str]) -> None:
        """Read `stream` to EOF while discarding text beyond the buffer limit."""
        try:
            while chunk := stream.read(_LOCAL_COMMAND_READ_CHUNK_SIZE):
                remaining = self._limit - self._length
                if remaining > 0:
                    retained = chunk[:remaining]
                    self._chunks.append(retained)
                    self._length += len(retained)
        except (OSError, ValueError):
            self.failed = True
        finally:
            with suppress(OSError, ValueError):
                stream.close()

    def getvalue(self) -> str:
        """Return the retained text."""
        return "".join(self._chunks)


@dataclass(frozen=True)
class _GitContext:
    """Git details reused by multiple local context sections."""

    root: Path | None = None
    summary: str | None = None


@dataclass(frozen=True)
class _SafePath:
    """A path whose components were checked without following links."""

    path: Path
    metadata: os.stat_result


@dataclass(frozen=True)
class _VisibleEntry:
    """A visible directory entry classified from no-follow metadata."""

    path: Path
    metadata: os.stat_result

    @property
    def is_directory(self) -> bool:
        """Whether the entry is a normal directory."""
        return not _is_link_or_reparse(self.metadata) and stat.S_ISDIR(
            self.metadata.st_mode
        )


def _windows_executable_names(command: str) -> tuple[str, ...]:
    raw_extensions = os.environ.get("PATHEXT")
    extensions = (
        raw_extensions.split(os.pathsep)
        if raw_extensions
        else list(_WINDOWS_DEFAULT_PATHEXT)
    )
    normalized: list[str] = []
    seen: set[str] = set()
    for extension in extensions:
        if not extension:
            continue
        value = extension if extension.startswith(".") else f".{extension}"
        if not re.fullmatch(r"\.[A-Za-z0-9]+", value):
            continue
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            normalized.append(value)

    suffix = Path(command).suffix.casefold()
    if suffix:
        return (command,) if suffix in seen else ()
    return tuple(f"{command}{extension}" for extension in normalized)


def _is_within_path(path: Path, boundary: Path) -> bool:
    """Return whether `path` is equal to or contained by `boundary`."""
    return path == boundary or path.is_relative_to(boundary)


def _is_link_or_reparse(metadata: os.stat_result) -> bool:
    """Return whether no-follow metadata describes a link or reparse point."""
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(
        reparse_flag and getattr(metadata, "st_file_attributes", 0) & reparse_flag
    )


def _is_windows_remote_path(path: Path) -> bool:
    """Reject UNC and device-namespace paths without probing them.

    Returns:
        `True` when `path` uses a Windows remote or device namespace.
    """
    if os.name != "nt":
        return False
    return os.fspath(path).replace("/", "\\").startswith("\\\\")


def _windows_lstat_local_path(path: Path) -> os.stat_result | None:
    """Walk a Windows path with `lstat`, stopping before any reparse target.

    Returns:
        Final no-follow metadata, or `None` when the path is unsafe.
    """
    if _is_windows_remote_path(path):
        return None
    try:
        absolute_path = path.absolute()
    except (OSError, RuntimeError):
        return None
    if _is_windows_remote_path(absolute_path) or not absolute_path.anchor:
        return None

    current = Path(absolute_path.anchor)
    try:
        metadata = current.lstat()
        for part in absolute_path.parts[1:]:
            current /= part
            metadata = current.lstat()
            if current != absolute_path and _is_link_or_reparse(metadata):
                return None
    except OSError:
        return None
    return metadata


def _resolve_trusted_directory(path: Path) -> Path | None:
    """Resolve a local directory without traversing Windows reparse points.

    Returns:
        The resolved local directory, or `None` when it is unsafe.
    """
    if _is_windows_remote_path(path):
        return None
    try:
        absolute_path = path.expanduser().absolute()
        if os.name == "nt":
            metadata = _windows_lstat_local_path(absolute_path)
            if (
                metadata is None
                or _is_link_or_reparse(metadata)
                or not stat.S_ISDIR(metadata.st_mode)
            ):
                return None
        resolved_path = absolute_path.resolve(strict=True)
        resolved_metadata = resolved_path.lstat()
    except (OSError, RuntimeError):
        return None
    if (
        _is_windows_remote_path(resolved_path)
        or _is_link_or_reparse(resolved_metadata)
        or not stat.S_ISDIR(resolved_metadata.st_mode)
    ):
        return None
    return resolved_path


def _safe_path_metadata(
    path: Path,
    *,
    boundary: Path,
    allow_final_link: bool = False,
) -> _SafePath | None:
    """Inspect a contained path without following link or reparse components.

    Returns:
        Guarded path metadata, or `None` when the path is unsafe.
    """
    if _is_windows_remote_path(path) or _is_windows_remote_path(boundary):
        return None
    try:
        lexical_boundary = boundary.expanduser().absolute()
        lexical_path = path.expanduser().absolute()
        relative_path = lexical_path.relative_to(lexical_boundary)
    except (OSError, RuntimeError, ValueError):
        return None

    resolved_boundary = _resolve_trusted_directory(lexical_boundary)
    if resolved_boundary is None:
        return None

    current = resolved_boundary
    metadata: os.stat_result
    if not relative_path.parts:
        try:
            metadata = current.lstat()
        except OSError:
            return None
    else:
        for index, part in enumerate(relative_path.parts):
            current /= part
            try:
                metadata = current.lstat()
            except OSError:
                return None
            final = index == len(relative_path.parts) - 1
            if _is_link_or_reparse(metadata):
                if final and allow_final_link:
                    return _SafePath(path=current, metadata=metadata)
                return None
            if not final and not stat.S_ISDIR(metadata.st_mode):
                return None

    try:
        resolved_path = current.resolve(strict=True)
        current_metadata = current.lstat()
    except (OSError, RuntimeError):
        return None
    if (
        _is_windows_remote_path(resolved_path)
        or not resolved_path.is_relative_to(resolved_boundary)
        or _is_link_or_reparse(current_metadata)
        or not os.path.samestat(metadata, current_metadata)
    ):
        return None
    return _SafePath(path=resolved_path, metadata=current_metadata)


def _safe_is_file(path: Path, *, boundary: Path) -> bool:
    """Return whether a contained path is a normal regular file."""
    result = _safe_path_metadata(path, boundary=boundary)
    return result is not None and stat.S_ISREG(result.metadata.st_mode)


def _safe_is_dir(path: Path, *, boundary: Path) -> bool:
    """Return whether a contained path is a normal directory."""
    result = _safe_path_metadata(path, boundary=boundary)
    return result is not None and stat.S_ISDIR(result.metadata.st_mode)


def _resolve_containing_directory(path: Path, *, child: Path) -> Path | None:
    """Resolve a trusted directory only when it contains `child`.

    Returns:
        The resolved containing directory, or `None` when it is unsafe.
    """
    resolved_path = _resolve_trusted_directory(path)
    resolved_child = _resolve_trusted_directory(child)
    if (
        resolved_path is None
        or resolved_child is None
        or not resolved_child.is_relative_to(resolved_path)
    ):
        return None
    return resolved_path


def _resolve_path_executable(
    command: str,
    *,
    project_root: Path | None = None,
) -> str | None:
    """Resolve a bare executable from trusted absolute `PATH` entries.

    Empty and relative entries are ignored on every platform. Candidates whose
    path or resolved target is controlled by the current directory or active
    project are rejected before any subprocess can execute them.

    Returns:
        The absolute executable path, or `None` when no safe candidate exists.
    """
    if not command or command in {".", ".."} or "/" in command or "\\" in command:
        return None

    raw_path = os.environ.get("PATH")
    if not raw_path:
        return None

    windows = os.name == "nt"
    executable_names = _windows_executable_names(command) if windows else (command,)
    if not executable_names:
        return None

    current_directory = _resolve_trusted_directory(Path.cwd())
    if current_directory is None:
        return None
    rejected_roots = [current_directory]
    if project_root is not None:
        if _is_windows_remote_path(project_root):
            return None
        resolved_project_root = _resolve_trusted_directory(project_root)
        if resolved_project_root is None:
            try:
                resolved_project_root = project_root.expanduser().absolute()
            except (OSError, RuntimeError):
                return None
        if resolved_project_root not in rejected_roots:
            rejected_roots.append(resolved_project_root)

    for raw_directory in raw_path.split(os.pathsep):
        if not raw_directory:
            continue
        path_entry = raw_directory
        if (
            windows
            and len(path_entry) >= _QUOTED_PATH_ENTRY_MIN_LENGTH
            and path_entry[0] == path_entry[-1] == '"'
        ):
            path_entry = path_entry[1:-1]
        directory = Path(path_entry)
        if not directory.is_absolute():
            continue
        try:
            absolute_directory = directory.absolute()
        except (OSError, RuntimeError):
            continue
        if _is_windows_remote_path(absolute_directory):
            continue
        if windows:
            resolved_directory = _resolve_trusted_directory(absolute_directory)
            if resolved_directory is None:
                continue
        else:
            try:
                resolved_directory = absolute_directory.resolve(strict=False)
            except (OSError, RuntimeError):
                continue
        if any(
            _is_within_path(absolute_directory, root)
            or _is_within_path(resolved_directory, root)
            for root in rejected_roots
        ):
            continue

        for executable_name in executable_names:
            candidate = resolved_directory / executable_name
            if windows:
                candidate_result = _safe_path_metadata(
                    candidate,
                    boundary=resolved_directory,
                )
                if candidate_result is None:
                    continue
                metadata = candidate_result.metadata
                resolved_candidate = candidate_result.path
            else:
                try:
                    metadata = candidate.stat()
                    resolved_candidate = candidate.resolve(strict=True)
                except (OSError, RuntimeError):
                    continue
            if not stat.S_ISREG(metadata.st_mode) or not os.access(candidate, os.X_OK):
                continue
            if any(
                _is_within_path(candidate, root)
                or _is_within_path(resolved_candidate, root)
                for root in rejected_roots
            ):
                continue
            return str(candidate)
    return None


def _run_fixed_command(
    argv: tuple[str, ...],
    *,
    cwd: Path,
    output_limit: int = _LOCAL_COMMAND_OUTPUT_LIMIT,
    env: dict[str, str] | None = None,
) -> _CommandResult | None:
    """Run a bounded fixed-argument command without invoking a shell.

    Returns:
        Bounded command output, or `None` if the probe cannot run safely.
    """
    try:
        process = subprocess.Popen(  # noqa: S603  # argv is fixed by callers
            argv,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            env=env,
        )
    except OSError:
        return None

    if process.stdout is None:
        with suppress(OSError):
            process.kill()
        with suppress(OSError, subprocess.SubprocessError):
            process.wait(timeout=_LOCAL_COMMAND_TIMEOUT)
        return None

    capture = _BoundedTextBuffer(output_limit + 1)
    reader = threading.Thread(
        target=capture.drain,
        args=(process.stdout,),
        name="dcode-local-context-output",
        daemon=True,
    )
    reader.start()
    try:
        returncode = process.wait(timeout=_LOCAL_COMMAND_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        with suppress(OSError):
            process.kill()
        with suppress(OSError, subprocess.SubprocessError):
            process.wait(timeout=_LOCAL_COMMAND_TIMEOUT)
        with suppress(OSError, ValueError):
            process.stdout.close()
        reader.join(timeout=_LOCAL_COMMAND_TIMEOUT)
        return None

    reader.join(timeout=_LOCAL_COMMAND_TIMEOUT)
    if reader.is_alive() or capture.failed:
        with suppress(OSError, ValueError):
            process.stdout.close()
        reader.join(timeout=_LOCAL_COMMAND_TIMEOUT)
        return None

    stdout = capture.getvalue()
    truncated = len(stdout) > output_limit
    if truncated:
        stdout = stdout[:output_limit]
    return _CommandResult(
        returncode=returncode,
        stdout=stdout.strip(),
        truncated=truncated,
    )


def _git_probe_environment() -> dict[str, str]:
    """Build a minimal Git environment without user-controlled injection hooks.

    Returns:
        Environment containing only required system values and fixed Git controls.
    """
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in _GIT_PROBE_ENVIRONMENT_KEYS
    }
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return environment


def _run_fixed_git_command(
    git: str,
    arguments: tuple[str, ...],
    *,
    cwd: Path,
) -> _CommandResult | None:
    """Run a fixed Git probe with execution-capable configuration disabled.

    Returns:
        Bounded probe output, or `None` when Git cannot run safely.
    """
    argv = (
        git,
        "--no-pager",
        "-c",
        "core.fsmonitor=false",
        "-c",
        f"core.hooksPath={os.devnull}",
        "-c",
        "diff.external=",
        "-c",
        "pager.status=false",
        "-c",
        "submodule.recurse=false",
        *arguments,
    )
    return _run_fixed_command(argv, cwd=cwd, env=_git_probe_environment())


def _read_open_text_prefix(
    file: IO[str],
    *,
    line_limit: int,
    char_limit: int,
) -> tuple[list[str], bool]:
    lines: list[str] = []
    characters = 0
    truncated = False
    for line in file:
        if len(lines) >= line_limit or characters + len(line) > char_limit:
            truncated = True
            break
        lines.append(line.rstrip("\r\n"))
        characters += len(line)
    return lines, truncated


def _read_text_prefix_no_follow(
    path: Path,
    *,
    boundary: Path,
    line_limit: int,
    char_limit: int = 20_000,
) -> tuple[list[str], bool] | None:
    """Read a regular file only when its opened inode stays inside `boundary`.

    Returns:
        Captured lines and truncation state, or `None` for an unsafe path.
    """
    file_descriptor: int | None = None
    directory_descriptor: int | None = None
    try:
        safe_path = _safe_path_metadata(path, boundary=boundary)
        if safe_path is None or not stat.S_ISREG(safe_path.metadata.st_mode):
            return None

        flags = (
            os.O_RDONLY
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        if os.open in os.supports_dir_fd:
            directory_flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            directory_descriptor = os.open(safe_path.path.parent, directory_flags)
            file_descriptor = os.open(
                safe_path.path.name,
                flags,
                dir_fd=directory_descriptor,
            )
        else:
            file_descriptor = os.open(safe_path.path, flags)

        opened_metadata = os.fstat(file_descriptor)
        current_path = _safe_path_metadata(path, boundary=boundary)
        if (
            not stat.S_ISREG(opened_metadata.st_mode)
            or current_path is None
            or not stat.S_ISREG(current_path.metadata.st_mode)
            or not os.path.samestat(safe_path.metadata, opened_metadata)
            or not os.path.samestat(current_path.metadata, opened_metadata)
            or current_path.path != safe_path.path
        ):
            return None

        with os.fdopen(
            file_descriptor,
            encoding="utf-8",
            errors="replace",
        ) as file:
            file_descriptor = None
            return _read_open_text_prefix(
                file,
                line_limit=line_limit,
                char_limit=char_limit,
            )
    except (OSError, RuntimeError, ValueError):
        return None
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        if directory_descriptor is not None:
            os.close(directory_descriptor)


def _read_detection_text(path: Path, *, boundary: Path) -> str:
    """Read enough project metadata for lightweight marker detection.

    Returns:
        Bounded project text joined with newlines.
    """
    result = _read_text_prefix_no_follow(
        path,
        boundary=boundary,
        line_limit=400,
        char_limit=40_000,
    )
    if result is None:
        return ""
    lines, _ = result
    return "\n".join(lines)


def _safe_display(value: str, *, max_length: int = 500) -> str:
    """Sanitize a filesystem or subprocess value before prompt insertion.

    Returns:
        Sanitized text safe for prompt insertion.
    """
    return sanitize_control_chars(value, max_length=max_length)


def _visible_entries(directory: Path, *, root: Path) -> list[_VisibleEntry]:
    """Return deterministic visible children excluding generated directories."""
    directory_result = _safe_path_metadata(directory, boundary=root)
    if directory_result is None or not stat.S_ISDIR(directory_result.metadata.st_mode):
        return []
    try:
        paths = list(directory_result.path.iterdir())
    except OSError:
        return []

    entries: list[_VisibleEntry] = []
    for path in paths:
        if path.name in _LOCAL_EXCLUDED_NAMES or (
            path.name.startswith(".") and path.name != ".deepagents"
        ):
            continue
        result = _safe_path_metadata(path, boundary=root, allow_final_link=True)
        if result is not None:
            entries.append(_VisibleEntry(path=result.path, metadata=result.metadata))

    return sorted(
        entries,
        key=lambda entry: (
            not entry.is_directory,
            entry.path.name.casefold(),
            entry.path.name,
        ),
    )


def _collect_git_context(root: Path) -> _GitContext:
    """Collect git root, branch or commit, and available main branches.

    Returns:
        Git context summary data, or an empty context when git is unavailable.
    """
    git = _resolve_path_executable("git", project_root=root)
    if git is None:
        return _GitContext()

    inside = _run_fixed_git_command(
        git,
        ("rev-parse", "--is-inside-work-tree"),
        cwd=root,
    )
    if inside is None or inside.returncode != 0 or inside.stdout != "true":
        return _GitContext()

    root_result = _run_fixed_git_command(
        git,
        ("rev-parse", "--show-toplevel"),
        cwd=root,
    )
    git_root = (
        _resolve_containing_directory(Path(root_result.stdout), child=root)
        if root_result is not None
        and root_result.returncode == 0
        and root_result.stdout
        else None
    )

    branch_result = _run_fixed_git_command(
        git,
        ("symbolic-ref", "--quiet", "--short", "HEAD"),
        cwd=root,
    )
    if (
        branch_result is not None
        and branch_result.returncode == 0
        and branch_result.stdout
    ):
        summary = f"Current branch `{_safe_display(branch_result.stdout)}`"
    else:
        commit_result = _run_fixed_git_command(
            git,
            ("rev-parse", "--short", "HEAD"),
            cwd=root,
        )
        if (
            commit_result is None
            or commit_result.returncode != 0
            or not commit_result.stdout
        ):
            return _GitContext(root=git_root)
        summary = f"Detached HEAD at `{_safe_display(commit_result.stdout)}`"

    main_branches = []
    for branch in ("main", "master"):
        result = _run_fixed_git_command(
            git,
            ("show-ref", "--verify", "--quiet", f"refs/heads/{branch}"),
            cwd=root,
        )
        if result is not None and result.returncode == 0:
            main_branches.append(f"`{branch}`")
    if main_branches:
        summary += f", {', '.join(main_branches)} available"

    return _GitContext(root=git_root, summary=summary)


def _collect_project_section(root: Path, git_root: Path | None) -> str:
    """Describe project language, monorepo markers, root, and environments.

    Returns:
        A markdown project summary, or `""` when nothing relevant is found.
    """
    language = ""
    if _safe_is_file(root / "pyproject.toml", boundary=root) or _safe_is_file(
        root / "setup.py",
        boundary=root,
    ):
        language = "python"
    elif _safe_is_file(root / "package.json", boundary=root):
        language = "javascript/typescript"
    elif _safe_is_file(root / "Cargo.toml", boundary=root):
        language = "rust"
    elif _safe_is_file(root / "go.mod", boundary=root):
        language = "go"
    elif _safe_is_file(root / "pom.xml", boundary=root) or _safe_is_file(
        root / "build.gradle",
        boundary=root,
    ):
        language = "java"

    monorepo = (
        _safe_is_file(root / "lerna.json", boundary=root)
        or _safe_is_file(root / "pnpm-workspace.yaml", boundary=root)
        or _safe_is_dir(root / "packages", boundary=root)
        or (
            _safe_is_dir(root / "libs", boundary=root)
            and _safe_is_dir(root / "apps", boundary=root)
        )
        or _safe_is_dir(root / "workspaces", boundary=root)
    )
    environments = [
        name
        for name in (".venv", "venv", "node_modules")
        if _safe_is_dir(root / name, boundary=root)
    ]
    if (
        not language
        and not monorepo
        and not environments
        and (git_root is None or git_root == root)
    ):
        return ""

    lines = ["**Project**:"]
    if language:
        lines.append(f"- Language: {language}")
    if git_root is not None and git_root != root:
        lines.append(f"- Project root: `{_safe_display(str(git_root))}`")
    if monorepo:
        lines.append("- Monorepo: yes")
    if environments:
        lines.append(f"- Environments: {', '.join(environments)}")
    return "\n".join(lines)


def _collect_package_manager_section(root: Path) -> str:
    """Detect Python and Node package managers from project files.

    Returns:
        A markdown package-manager summary, or `""` when nothing is detected.
    """
    managers: list[str] = []
    pyproject = _read_detection_text(root / "pyproject.toml", boundary=root)
    if _safe_is_file(root / "uv.lock", boundary=root) or "[tool.uv]" in pyproject:
        managers.append("Python: uv")
    elif (
        _safe_is_file(
            root / "poetry.lock",
            boundary=root,
        )
        or "[tool.poetry]" in pyproject
    ):
        managers.append("Python: poetry")
    elif _safe_is_file(root / "Pipfile.lock", boundary=root) or _safe_is_file(
        root / "Pipfile",
        boundary=root,
    ):
        managers.append("Python: pipenv")
    elif _safe_is_file(
        root / "pyproject.toml",
        boundary=root,
    ) or _safe_is_file(root / "requirements.txt", boundary=root):
        managers.append("Python: pip")

    if _safe_is_file(root / "bun.lockb", boundary=root) or _safe_is_file(
        root / "bun.lock",
        boundary=root,
    ):
        managers.append("Node: bun")
    elif _safe_is_file(root / "pnpm-lock.yaml", boundary=root):
        managers.append("Node: pnpm")
    elif _safe_is_file(root / "yarn.lock", boundary=root):
        managers.append("Node: yarn")
    elif _safe_is_file(
        root / "package-lock.json",
        boundary=root,
    ) or _safe_is_file(root / "package.json", boundary=root):
        managers.append("Node: npm")

    return f"**Package Manager**: {', '.join(managers)}" if managers else ""


def _collect_runtime_section(root: Path) -> str:
    """Report the application Python and a Node runtime available on PATH.

    Returns:
        A markdown runtime summary with the detected interpreters.
    """
    runtimes = [
        (
            "Application Python "
            f"{sys.version_info.major}.{sys.version_info.minor}."
            f"{sys.version_info.micro}"
        )
    ]
    node = _resolve_path_executable("node", project_root=root)
    if node is not None:
        result = _run_fixed_command((node, "--version"), cwd=root)
        if result is not None and result.returncode == 0 and result.stdout:
            runtimes.append(f"Node {_safe_display(result.stdout.removeprefix('v'))}")
    return f"**Detected Runtimes**: {', '.join(runtimes)}"


def _extract_gh_json_fields(help_text: str) -> str:
    """Extract and normalize the JSON FIELDS section from `gh search --help`.

    Returns:
        Normalized JSON field names, or `""` when the help text has none.
    """
    fields: list[str] = []
    in_fields = False
    for line in help_text.splitlines():
        if line.strip() == "JSON FIELDS":
            in_fields = True
            continue
        if in_fields and not line.strip():
            break
        if in_fields:
            fields.extend(line.strip().split())
    return _safe_display(" ".join(fields), max_length=2_000)


def _collect_gh_section(root: Path) -> str:
    """Report JSON fields exposed by installed GitHub CLI search commands.

    Returns:
        A markdown GitHub CLI summary, or `""` when `gh` is unavailable.
    """
    gh = _resolve_path_executable("gh", project_root=root)
    if gh is None:
        return ""

    field_sets: dict[str, str] = {}
    for search_type in ("prs", "issues"):
        result = _run_fixed_command((gh, "search", search_type, "--help"), cwd=root)
        if result is not None and result.returncode == 0:
            fields = _extract_gh_json_fields(result.stdout)
            if fields:
                field_sets[search_type] = fields
    if not field_sets:
        return ""

    lines = ["**GitHub CLI**:"]
    for search_type in ("prs", "issues"):
        fields = field_sets.get(search_type)
        if fields:
            lines.append(f"- `gh search {search_type} --json` fields: {fields}")
    if "mergedAt" not in field_sets.get("prs", "").replace(",", " ").split():
        lines.extend(
            [
                "- `gh search prs --json` does not expose `mergedAt`;",
                "  use `gh pr view --json mergedAt` per PR for merge timestamps.",
            ]
        )
    return "\n".join(lines)


def _has_dependency_group(pyproject: str, group: str) -> bool:
    """Return whether a bounded `pyproject.toml` prefix defines a dependency group."""
    in_dependency_groups = False
    for raw_line in pyproject.splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            in_dependency_groups = line == "[dependency-groups]"
        elif in_dependency_groups and "=" in line:
            key = line.partition("=")[0].strip().strip("'\"")
            if key == group:
                return True
    return False


def _collect_test_command_section(root: Path) -> str:
    """Infer the most likely locally available project test command.

    Returns:
        A markdown test-command hint, or `""` when no likely command exists.
    """
    makefile_result = _read_text_prefix_no_follow(
        root / "Makefile",
        boundary=root,
        line_limit=500,
    )
    makefile = makefile_result[0] if makefile_result is not None else []
    pyproject_path = root / "pyproject.toml"
    pyproject = _read_detection_text(pyproject_path, boundary=root)
    unit_tests = root / "tests" / "unit_tests"
    if (
        any(line.startswith(("test:", "tests:")) for line in makefile)
        and _resolve_path_executable("make", project_root=root) is not None
    ):
        command = "make test"
    elif _safe_is_file(pyproject_path, boundary=root) and (
        "[tool.pytest" in pyproject
        or _safe_is_file(root / "pytest.ini", boundary=root)
        or _safe_is_dir(root / "tests", boundary=root)
        or _safe_is_dir(root / "test", boundary=root)
    ):
        is_uv_project = (
            _safe_is_file(root / "uv.lock", boundary=root)
            or "[tool.uv]" in pyproject
            or "[project]" in pyproject
        )
        target = " tests/unit_tests/" if _safe_is_dir(unit_tests, boundary=root) else ""
        if (
            _resolve_path_executable("uv", project_root=root) is not None
            and is_uv_project
            and _has_dependency_group(pyproject, "test")
        ):
            command = f"uv run --group test pytest{target}"
        else:
            command = f"pytest{target}"
    elif _safe_is_file(
        root / "package.json",
        boundary=root,
    ) and '"test"' in _read_detection_text(root / "package.json", boundary=root):
        command = "npm test"
    else:
        return ""
    return f"**Run Tests**: `{command}`"


def _collect_files_section(root: Path) -> str:
    """List a bounded set of top-level project files and directories.

    Returns:
        A markdown file listing, or `""` when the directory is empty.
    """
    entries = _visible_entries(root, root=root)
    if not entries:
        return ""
    shown = entries[:_LOCAL_FILE_LIMIT]
    heading = (
        f"**Files** (showing {len(shown)} of {len(entries)}):"
        if len(shown) < len(entries)
        else f"**Files** ({len(entries)}):"
    )
    lines = [heading]
    for entry in shown:
        suffix = "/" if entry.is_directory else ""
        lines.append(f"- {_safe_display(entry.path.name)}{suffix}")
    return "\n".join(lines)


def _collect_tree_section(root: Path) -> str:
    """Build a deterministic bounded directory preview without external tools.

    Returns:
        A markdown tree preview, or `""` when there are no visible entries.
    """
    resolved_root = _resolve_trusted_directory(root)
    if resolved_root is None:
        return ""

    lines = ["."]
    truncated = False

    def visit(directory: Path, prefix: str, depth: int) -> None:
        nonlocal truncated
        if truncated or depth > _LOCAL_TREE_DEPTH:
            return
        entries = _visible_entries(directory, root=resolved_root)
        for index, entry in enumerate(entries):
            if len(lines) >= _LOCAL_TREE_LINE_LIMIT:
                truncated = True
                return
            last = index == len(entries) - 1
            connector = "└── " if last else "├── "
            lines.append(f"{prefix}{connector}{_safe_display(entry.path.name)}")
            if entry.is_directory:
                visit(
                    entry.path,
                    prefix + ("    " if last else "│   "),
                    depth + 1,
                )

    visit(resolved_root, "", 1)
    if len(lines) == 1:
        return ""
    if truncated:
        lines.append("... (more lines truncated)")
    return "\n".join(["**Tree** (3 levels):", "```text", *lines, "```"])


def _collect_makefile_section(root: Path, git_root: Path | None) -> str:
    """Show a bounded Makefile preview from the cwd or containing git root.

    Returns:
        A markdown Makefile preview, or `""` when no Makefile is found.
    """
    resolved_root = _resolve_trusted_directory(root)
    if resolved_root is None:
        return ""
    makefile = resolved_root / "Makefile"
    result = _read_text_prefix_no_follow(
        makefile,
        boundary=resolved_root,
        line_limit=_LOCAL_MAKEFILE_LINE_LIMIT,
    )
    if result is None and git_root is not None:
        resolved_git_root = _resolve_containing_directory(
            git_root,
            child=resolved_root,
        )
        if resolved_git_root is not None and resolved_git_root != resolved_root:
            makefile = resolved_git_root / "Makefile"
            result = _read_text_prefix_no_follow(
                makefile,
                boundary=resolved_git_root,
                line_limit=_LOCAL_MAKEFILE_LINE_LIMIT,
            )
    if result is None:
        return ""
    lines, truncated = result
    if not lines:
        return ""
    display_path = "Makefile" if makefile.parent == resolved_root else str(makefile)
    body = [
        (
            f"**Makefile** (`{_safe_display(display_path)}`, "
            f"first {_LOCAL_MAKEFILE_LINE_LIMIT} lines):"
        ),
        "```makefile",
        *lines,
    ]
    if truncated:
        body.append("... (truncated)")
    body.append("```")
    return "\n".join(body)


def _collect_local_context(local_root: Path) -> str:
    """Collect bounded context directly from a local project directory.

    Returns:
        A combined markdown context block for the local project.
    """
    root = _resolve_trusted_directory(local_root)
    if root is None:
        return ""
    git_context = _collect_git_context(root)
    sections = [
        "## Local Context",
        f"**Current Directory**: `{_safe_display(str(root))}`",
        _collect_project_section(root, git_context.root),
        _collect_package_manager_section(root),
        _collect_runtime_section(root),
        f"**Git**: {git_context.summary}" if git_context.summary else "",
        _collect_gh_section(root),
        _collect_test_command_section(root),
        _collect_files_section(root),
        _collect_tree_section(root),
        _collect_makefile_section(root, git_context.root),
    ]
    return "\n\n".join(section for section in sections if section)


def _sanitize_error_detail(error: str | None) -> str:
    """Make an untrusted MCP error string safe to embed in the system prompt.

    The error originates from exception text or MCP config-file contents, so it
    is untrusted input flowing into the system prompt (prompt-injection and
    log-forging risk). Strip hidden/deceptive Unicode, flatten control
    characters and newlines to spaces so the value cannot break out of its
    single bullet line or inject fake instruction lines, collapse runs of
    whitespace, and bound the length.

    Args:
        error: Raw error message, or `None`.

    Returns:
        A single-line, length-bounded, sanitized string. Falls back to
        `"unknown error"` when no usable message remains.
    """
    if not error:
        return "unknown error"
    sanitized = sanitize_control_chars(error, max_length=_MCP_ERROR_DETAIL_LIMIT)
    return sanitized or "unknown error"


def _sanitize_tracing_project_name(project: str) -> str:
    """Make an untrusted LangSmith project name safe for the system prompt.

    Project names can originate from a workspace `.env` file or process
    environment. Flatten hidden/control characters and bound the length before
    embedding them in prompt bullets so a crafted value cannot inject extra
    prompt lines.

    Args:
        project: Raw LangSmith project name.

    Returns:
        A single-line, length-bounded, sanitized project name. Falls back to
        `"unknown project"` when no usable text remains.
    """
    sanitized = sanitize_control_chars(project, max_length=_TRACING_PROJECT_NAME_LIMIT)
    return sanitized or "unknown project"


def _quote_tracing_project_name(project: str) -> str:
    """JSON-quote a sanitized LangSmith project name for prompt insertion.

    Args:
        project: Sanitized LangSmith project name.

    Returns:
        JSON string literal for the project name.
    """
    return json.dumps(project, ensure_ascii=False)


def _build_mcp_context(servers: list[MCPServerInfo]) -> str:
    """Format MCP server/tool inventory for the system prompt.

    Args:
        servers: List of connected MCP server metadata.

    Returns:
        Formatted markdown string, or `""` if no servers.
    """
    if not servers:
        return ""

    total_tools = sum(len(s.tools) for s in servers)
    lines = [f"**MCP Servers** ({len(servers)} servers, {total_tools} tools):"]

    for server in servers:
        if not server.tools:
            # `status`/`error` always exist on the frozen dataclass; the
            # `__post_init__` invariant guarantees a non-`ok` status carries a
            # non-`None` error. The error is untrusted (exception/config text),
            # so it is sanitized and isolated in an `<error>` delimiter before
            # reaching the prompt.
            if server.status == "error":
                detail = _sanitize_error_detail(server.error)
                lines.append(
                    f"- **{server.name}** ({server.transport}): "
                    f"FAILED TO LOAD — <error>{detail}</error>. "
                    "Treat this integration as temporarily unavailable; "
                    "tell the user the server failed to load and suggest "
                    "restarting the MCP server."
                )
            elif server.status == "unauthenticated":
                detail = _sanitize_error_detail(server.error)
                lines.append(
                    f"- **{server.name}** ({server.transport}): "
                    f"NEEDS LOGIN — <error>{detail}</error>. "
                    "This integration requires authentication before its "
                    "tools are available; tell the user and suggest running "
                    "`/mcp` to log in."
                )
            elif server.status == "disabled":
                lines.append(
                    f"- **{server.name}** ({server.transport}): (disabled by user)"
                )
            else:
                # `ok` with no tools (genuinely empty). `awaiting_reconnect` is a
                # transient UI-only status that never reaches this function (the
                # middleware is always built from a fresh preload), but it would
                # also render benignly here.
                lines.append(
                    f"- **{server.name}** ({server.transport}): (no tools registered)"
                )
            continue

        names = [t.name for t in server.tools]
        if len(names) > _TOOL_NAME_DISPLAY_LIMIT:
            shown = ", ".join(names[:_TOOL_NAME_DISPLAY_LIMIT])
            remaining = len(names) - _TOOL_NAME_DISPLAY_LIMIT
            lines.append(
                f"- **{server.name}** ({server.transport}): "
                f"{shown}, and {remaining} more"
            )
        else:
            lines.append(
                f"- **{server.name}** ({server.transport}): {', '.join(names)}"
            )

    return "\n".join(lines)


def _build_tracing_context(
    agent_project: str | None,
    user_project: str | None,
) -> str:
    """Format LangSmith tracing project names for the system prompt.

    Surfaces both projects so the agent can look up the right traces with the
    LangSmith MCP server or CLI: the project its own runs are traced to, and
    the user's original project that shell commands trace to. The
    shell-command line is shown only when the user's project differs from the
    agent's (after sanitizing both), avoiding a redundant duplicate line.

    Args:
        agent_project: Project receiving the agent's own traces, or `None`
            when LangSmith tracing is not enabled.
        user_project: User's original `LANGSMITH_PROJECT`, used by code the
            agent runs in the shell.

    Returns:
        Formatted markdown string, or `""` when tracing is disabled.
    """
    if not agent_project:
        return ""

    safe_agent_project = _sanitize_tracing_project_name(agent_project)
    quoted_agent_project = _quote_tracing_project_name(safe_agent_project)
    lines = [
        "**LangSmith Tracing**:",
        f"- Agent traces: project {quoted_agent_project}",
    ]
    if user_project:
        safe_user_project = _sanitize_tracing_project_name(user_project)
        if safe_user_project != safe_agent_project:
            quoted_user_project = _quote_tracing_project_name(safe_user_project)
            lines.append(f"- Shell-command traces: project {quoted_user_project}")
    return "\n".join(lines)


@runtime_checkable
class _ExecutableBackend(Protocol):
    """Any backend that supports `execute(command) -> ExecuteResponse`."""

    def execute(
        self, command: str, *, timeout: int | None = None
    ) -> ExecuteResponse: ...


@runtime_checkable
class _AsyncExecutableBackend(Protocol):
    """Any backend that provides an async `aexecute` method."""

    async def aexecute(
        self,
        command: str,
        *,
        timeout: int | None = None,  # noqa: ASYNC109  # Timeout is forwarded to backend, not used as asyncio timeout
    ) -> ExecuteResponse: ...


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context detection script
#
# Outputs markdown describing the current working environment. Each section
# is guarded so that missing tools or unsupported environments are silently
# skipped -- external tools like git, tree, python3, and node are checked
# with `command -v` before use.
#
# The script is built from section functions so each piece can be tested
# independently. Independent sections run as parallel background subshells;
# see build_detect_script() for the orchestration logic.
# ---------------------------------------------------------------------------


def _section_header() -> str:
    """CWD line and Git metadata used by other sections.

    Returns:
        Bash snippet that prints the header and sets `CWD`, `IN_GIT`, and `ROOT`.
    """
    return r"""CWD="$(pwd -P)"
echo "## Local Context"
echo ""
echo "**Current Directory**: \`${CWD}\`"
echo ""

# --- Check git once ---
GIT_BIN="$(command -v git 2>/dev/null || true)"
safe_git() {
  env -i \
    PATH="${PATH-}" \
    HOME="${HOME-}" \
    TMPDIR="${TMPDIR-}" \
    TMP="${TMP-}" \
    TEMP="${TEMP-}" \
    LANG="${LANG-}" \
    LC_ALL="${LC_ALL-}" \
    LC_CTYPE="${LC_CTYPE-}" \
    GIT_CONFIG_GLOBAL=/dev/null \
    GIT_CONFIG_NOSYSTEM=1 \
    GIT_TERMINAL_PROMPT=0 \
    "$GIT_BIN" --no-pager \
      -c core.fsmonitor=false \
      -c core.hooksPath=/dev/null \
      -c diff.external= \
      -c pager.status=false \
      -c submodule.recurse=false \
      "$@"
}
IN_GIT=false
ROOT=""
if [ -n "$GIT_BIN" ]; then
  GIT_INFO="$(safe_git rev-parse --is-inside-work-tree --show-toplevel 2>/dev/null)"
  GIT_MODE="${GIT_INFO%%$'\n'*}"
  case "$GIT_MODE" in
    true)
      IN_GIT=true
      ROOT="${GIT_INFO#*$'\n'}"
      ;;
    false) IN_GIT=true ;;  # Bare repository or the Git directory itself.
  esac
fi"""


def _section_project() -> str:
    """Language, monorepo, project-root display, virtual-env detection.

    Returns:
        Bash snippet (requires `CWD` and `ROOT` from header).
    """
    return r"""# --- Project ---
PROJ_LANG=""
[ -f pyproject.toml ] || [ -f setup.py ] && PROJ_LANG="python"
[ -z "$PROJ_LANG" ] && [ -f package.json ] && PROJ_LANG="javascript/typescript"
[ -z "$PROJ_LANG" ] && [ -f Cargo.toml ] && PROJ_LANG="rust"
[ -z "$PROJ_LANG" ] && [ -f go.mod ] && PROJ_LANG="go"
[ -z "$PROJ_LANG" ] && { [ -f pom.xml ] || [ -f build.gradle ]; } && PROJ_LANG="java"

MONOREPO=false
{ [ -f lerna.json ] || [ -f pnpm-workspace.yaml ] \
  || [ -d packages ] || { [ -d libs ] && [ -d apps ]; } \
  || [ -d workspaces ]; } && MONOREPO=true

ENVS=""
{ [ -d .venv ] || [ -d venv ]; } && ENVS=".venv"
[ -d node_modules ] && ENVS="${ENVS:+${ENVS}, }node_modules"

HAS_PROJECT=false
{ [ -n "$PROJ_LANG" ] || { [ -n "$ROOT" ] && [ "$ROOT" != "$CWD" ]; } \
  || $MONOREPO || [ -n "$ENVS" ]; } && HAS_PROJECT=true

if $HAS_PROJECT; then
  echo "**Project**:"
  [ -n "$PROJ_LANG" ] && echo "- Language: ${PROJ_LANG}"
  [ -n "$ROOT" ] && [ "$ROOT" != "$CWD" ] && echo "- Project root: \`${ROOT}\`"
  $MONOREPO && echo "- Monorepo: yes"
  [ -n "$ENVS" ] && echo "- Environments: ${ENVS}"
  echo ""
fi"""


def _section_package_managers() -> str:
    """Python and Node package manager detection.

    Returns:
        Bash snippet (standalone).
    """
    return r"""# --- Package managers ---
PKG=""
if [ -f uv.lock ]; then PKG="Python: uv"
elif [ -f poetry.lock ]; then PKG="Python: poetry"
elif [ -f Pipfile.lock ] || [ -f Pipfile ]; then PKG="Python: pipenv"
elif [ -f pyproject.toml ]; then
  if grep -q '\[tool\.uv\]' pyproject.toml 2>/dev/null; then PKG="Python: uv"
  elif grep -q '\[tool\.poetry\]' pyproject.toml 2>/dev/null; then PKG="Python: poetry"
  else PKG="Python: pip"
  fi
elif [ -f requirements.txt ]; then PKG="Python: pip"
fi

NODE_PKG=""
if [ -f bun.lockb ] || [ -f bun.lock ]; then NODE_PKG="Node: bun"
elif [ -f pnpm-lock.yaml ]; then NODE_PKG="Node: pnpm"
elif [ -f yarn.lock ]; then NODE_PKG="Node: yarn"
elif [ -f package-lock.json ] || [ -f package.json ]; then NODE_PKG="Node: npm"
fi
[ -n "$NODE_PKG" ] && PKG="${PKG:+${PKG}, }${NODE_PKG}"
[ -n "$PKG" ] && echo "**Package Manager**: ${PKG}" && echo ""
"""


def _section_runtimes() -> str:
    """Python and Node runtime version detection.

    Returns:
        Bash snippet (standalone).
    """
    return r"""# --- Runtimes ---
_RT_TMP="${_DCT:-}"
_RT_CLEANUP=false
if [ -z "$_RT_TMP" ]; then
  _RT_TMP="$(mktemp -d)" || exit 1
  _RT_CLEANUP=true
fi

HAS_PYTHON=false
if command -v python3 >/dev/null 2>&1; then
  python3 --version > "$_RT_TMP/runtime_python" 2>/dev/null &
  HAS_PYTHON=true
fi
HAS_NODE=false
if command -v node >/dev/null 2>&1; then
  node --version > "$_RT_TMP/runtime_node" 2>/dev/null &
  HAS_NODE=true
fi
wait

RT=""
if $HAS_PYTHON && [ -s "$_RT_TMP/runtime_python" ]; then
  IFS= read -r PV < "$_RT_TMP/runtime_python"
  PV="${PV#* }"
  PV="${PV%% *}"
  [ -n "$PV" ] && RT="Python ${PV}"
fi
if $HAS_NODE && [ -s "$_RT_TMP/runtime_node" ]; then
  IFS= read -r NV < "$_RT_TMP/runtime_node"
  NV="${NV#v}"
  [ -n "$NV" ] && RT="${RT:+${RT}, }Node ${NV}"
fi
$_RT_CLEANUP && rm -rf "$_RT_TMP"
[ -n "$RT" ] && echo "**Detected Runtimes**: ${RT}" && echo ""
"""


def _section_git() -> str:
    """Git branch or detached HEAD commit and main branches.

    Returns:
        Bash snippet (requires `IN_GIT` from header).
    """
    return r"""# --- Git ---
if $IN_GIT; then
  BRANCH="$(safe_git rev-parse --abbrev-ref HEAD 2>/dev/null)"
  if [ "$BRANCH" = "HEAD" ]; then
    COMMIT="$(safe_git rev-parse --short HEAD 2>/dev/null)"
    GT="**Git**: Detached HEAD at \`${COMMIT}\`"
  else
    GT="**Git**: Current branch \`${BRANCH}\`"
  fi

  MAINS=""
  for b in $(safe_git for-each-ref --format='%(refname:short)' \
      refs/heads/main refs/heads/master 2>/dev/null); do
    case "$b" in
      main) MAINS="${MAINS:+${MAINS}, }\`main\`" ;;
      master) MAINS="${MAINS:+${MAINS}, }\`master\`" ;;
    esac
  done
  [ -n "$MAINS" ] && GT="${GT}, ${MAINS} available"

  FILTER_OVERRIDES=()
  while IFS= read -r FILTER_KEY; do
    FILTER_NAME="${FILTER_KEY#filter.}"
    FILTER_NAME="${FILTER_NAME%.*}"
    [ -n "$FILTER_NAME" ] || continue
    FILTER_OVERRIDES+=(
      -c "filter.${FILTER_NAME}.clean="
      -c "filter.${FILTER_NAME}.process="
      -c "filter.${FILTER_NAME}.required=false"
    )
  done < <(
    safe_git config --name-only \
      --get-regexp '^filter\..*\.(clean|process|required)$' 2>/dev/null
  )
  DC=0
  while IFS= read -r STATUS_LINE; do
    [ -n "$STATUS_LINE" ] && DC=$((DC + 1))
  done < <(
    safe_git "${FILTER_OVERRIDES[@]}" status --porcelain 2>/dev/null
  )
  if [ "$DC" -gt 0 ]; then
    if [ "$DC" -eq 1 ]; then GT="${GT}, 1 uncommitted change"
    else GT="${GT}, ${DC} uncommitted changes"
    fi
  fi

  echo "$GT"
  echo ""
fi"""


def _section_gh_cli() -> str:
    """GitHub CLI search JSON-field affordances from the installed `gh`.

    Returns:
        Bash snippet (standalone).
    """
    return r"""# --- GitHub CLI ---
if command -v gh >/dev/null 2>&1; then
  _gh_json_fields() {
    gh search "$1" --help 2>/dev/null \
      | awk '
        /^JSON FIELDS/ { in_fields = 1; next }
        in_fields && /^$/ { exit }
        in_fields {
          sub(/^[[:space:]]+/, "")
          gsub(/[[:space:]]+/, " ")
          fields = fields (fields ? " " : "") $0
        }
        END {
          sub(/^ /, "", fields)
          sub(/ $/, "", fields)
          if (fields != "") print fields
        }
      '
  }

  _GH_TMP="${_DCT:-}"
  _GH_CLEANUP=false
  if [ -z "$_GH_TMP" ]; then
    _GH_TMP="$(mktemp -d)" || exit 1
    _GH_CLEANUP=true
  fi
  _gh_json_fields prs > "$_GH_TMP/gh_prs_fields" &
  _gh_json_fields issues > "$_GH_TMP/gh_issues_fields" &
  wait

  GH_PRS_FIELDS=""
  GH_ISSUES_FIELDS=""
  [ -s "$_GH_TMP/gh_prs_fields" ] \
    && IFS= read -r GH_PRS_FIELDS < "$_GH_TMP/gh_prs_fields"
  [ -s "$_GH_TMP/gh_issues_fields" ] \
    && IFS= read -r GH_ISSUES_FIELDS < "$_GH_TMP/gh_issues_fields"
  $_GH_CLEANUP && rm -rf "$_GH_TMP"
  if [ -n "$GH_PRS_FIELDS" ] || [ -n "$GH_ISSUES_FIELDS" ]; then
    echo "**GitHub CLI**:"
    [ -n "$GH_PRS_FIELDS" ] \
      && echo "- \`gh search prs --json\` fields: ${GH_PRS_FIELDS}"
    [ -n "$GH_ISSUES_FIELDS" ] \
      && echo "- \`gh search issues --json\` fields: ${GH_ISSUES_FIELDS}"
    case ",$GH_PRS_FIELDS," in
      *mergedAt*) ;;
      *) echo "- \`gh search prs --json\` does not expose \`mergedAt\`;"
         echo "  use \`gh pr view --json mergedAt\` per PR for merge timestamps." ;;
    esac
    echo ""
  fi
fi"""


def _section_test_command() -> str:
    """Test command detection (make test / pytest / npm test).

    Returns:
        Bash snippet (standalone).
    """
    return r"""# --- Test command ---
TC=""
if [ -f Makefile ] && [ ! -L Makefile ] \
    && grep -qE '^tests?:' Makefile 2>/dev/null; then TC="make test"
elif [ -f pyproject.toml ]; then
  if grep -q '\[tool\.pytest' pyproject.toml 2>/dev/null \
      || [ -f pytest.ini ] || [ -d tests ] || [ -d test ]; then
    TC="pytest"
  fi
elif [ -f package.json ] \
    && grep -q '"test"' package.json 2>/dev/null; then
  TC="npm test"
fi
[ -n "$TC" ] && echo "**Run Tests**: \`${TC}\`" && echo ""
"""


def _section_files() -> str:
    """Directory listing (filtered, capped at 20).

    Returns:
        Bash snippet (standalone).
    """
    return r"""# --- Files ---
FILE_SUMMARY=$(
  { ls -1 2>/dev/null; [ -e .deepagents ] && echo .deepagents; } |
  sort -u |
  awk '
    BEGIN {
      excluded["node_modules"] = excluded["__pycache__"] = 1
      excluded[".pytest_cache"] = excluded[".mypy_cache"] = 1
      excluded[".ruff_cache"] = excluded[".tox"] = 1
      excluded[".coverage"] = excluded[".eggs"] = 1
      excluded["dist"] = excluded["build"] = 1
    }
    !($0 in excluded) {
      total++
      if (shown < 20) files[++shown] = $0
    }
    END {
      print total + 0
      print shown + 0
      for (i = 1; i <= shown; i++) print files[i]
    }
  '
)
TOTAL="${FILE_SUMMARY%%$'\n'*}"
FILE_DETAILS="${FILE_SUMMARY#*$'\n'}"
SHOWN="${FILE_DETAILS%%$'\n'*}"
SHOWN_FILES="${FILE_DETAILS#*$'\n'}"

if [ "$TOTAL" -gt 0 ]; then
  if [ "$SHOWN" -lt "$TOTAL" ]; then
    echo "**Files** (showing ${SHOWN} of ${TOTAL}):"
  else
    echo "**Files** (${TOTAL}):"
  fi
  while IFS= read -r f; do
    if [ -d "$f" ]; then echo "- ${f}/"
    else echo "- ${f}"
    fi
  done <<< "$SHOWN_FILES"
  echo ""
fi"""


def _section_tree() -> str:
    """`tree -L 3` output.

    Returns:
        Bash snippet (standalone).
    """
    return r"""# --- Tree ---
if command -v tree >/dev/null 2>&1; then
  TREE_EXCL='node_modules|.venv|__pycache__|.pytest_cache'
  TREE_EXCL="${TREE_EXCL}|.git|.mypy_cache|.ruff_cache"
  TREE_EXCL="${TREE_EXCL}|.tox|.coverage|.eggs|dist|build"
  T_PREVIEW=$(tree -L 3 --noreport --dirsfirst \
    -I "$TREE_EXCL" 2>/dev/null | sed -n '1,22p;23{p;q;}')
  if [ -n "$T_PREVIEW" ]; then
    PREVIEW_LINES=$(printf '%s\n' "$T_PREVIEW" | awk 'END { print NR }')
    T="$T_PREVIEW"
    TREE_TRUNCATED=false
    if [ "$PREVIEW_LINES" -gt 22 ]; then
      T=$(printf '%s\n' "$T_PREVIEW" | sed -n '1,22p')
      TREE_TRUNCATED=true
    fi
    echo "**Tree** (3 levels):"
    echo '```text'
    echo "$T"
    $TREE_TRUNCATED && echo "... (more lines truncated)"
    echo '```'
    echo ""
  fi
fi"""


def _section_makefile() -> str:
    """First 20 lines of Makefile (falls back to git root in monorepos).

    Returns:
        Bash snippet (requires `ROOT` and `CWD` from `_section_header`).
    """
    return r"""# --- Makefile ---
MK=""
MK_ROOT=""
if [ -f Makefile ] && [ ! -L Makefile ]; then
  MK="Makefile"
  MK_ROOT="$CWD"
elif [ -n "$ROOT" ] && [ "$ROOT" != "$CWD" ] \
    && [ -f "${ROOT}/Makefile" ] && [ ! -L "${ROOT}/Makefile" ]; then
  MK="${ROOT}/Makefile"
  MK_ROOT="$ROOT"
fi
if [ -n "$MK" ]; then
  MK_PARENT="$(cd -P -- "$(dirname -- "$MK")" 2>/dev/null && pwd -P)"
  SAFE_ROOT="$(cd -P -- "$MK_ROOT" 2>/dev/null && pwd -P)"
  [ -n "$MK_PARENT" ] && [ "$MK_PARENT" = "$SAFE_ROOT" ] || MK=""
fi
if [ -n "$MK" ]; then
  echo "**Makefile** (\`${MK}\`, first 20 lines):"
  echo '```makefile'
  awk 'NR <= 20 { print; next } { print "... (truncated)"; exit }' "$MK"
  echo '```'
fi"""


def build_detect_script() -> str:
    """Concatenate all section functions into the full detection script.

    Independent sections run as parallel background jobs writing to temp
    files, then results are concatenated in the original display order.
    The header (sets `CWD`, `IN_GIT`, and `ROOT`) and project section run first
    because later sections depend on their variables.

    Returns:
        Complete bash heredoc ready for `backend.execute()`.
    """
    # Header (sets CWD, IN_GIT, ROOT) + project run synchronously for others
    serial_prefix = f"{_section_header()}\n{_section_project()}"

    # These sections are independent — run them in parallel.
    # Subshells inherit parent variables (IN_GIT, ROOT, CWD) via fork.
    # Individual exit codes are not tracked because sections legitimately
    # exit non-zero when they have nothing to report (e.g. no runtimes).
    parallel_sections = [
        ("02_pkgmgr", _section_package_managers()),
        ("03_runtimes", _section_runtimes()),
        ("04_git", _section_git()),
        ("05_gh_cli", _section_gh_cli()),
        ("06_testcmd", _section_test_command()),
        ("07_files", _section_files()),
        ("08_tree", _section_tree()),
        ("09_makefile", _section_makefile()),
    ]

    # Build parallel wrapper: each section runs in a subshell writing to a
    # temp file. Section stderr is discarded to prevent noise leakage.
    parallel_setup = "_DCT=$(mktemp -d) || exit 1\ntrap 'rm -rf \"$_DCT\"' EXIT"
    parallel_block = "\n".join(
        f'(\n{body}\n) > "$_DCT/{name}" 2>/dev/null &'
        for name, body in parallel_sections
    )
    cat_line = "cat " + " ".join(f'"$_DCT/{name}"' for name, _ in parallel_sections)

    body = f"{serial_prefix}\n{parallel_setup}\n{parallel_block}\nwait\n{cat_line}"
    return f"bash <<'__DETECT_CONTEXT_EOF__'\n{body}\n__DETECT_CONTEXT_EOF__\n"


DETECT_CONTEXT_SCRIPT = build_detect_script()

# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


class LocalContextState(AgentState):
    """State for local context middleware."""

    _local_context: NotRequired[Annotated[str, PrivateStateAttr]]
    """Private formatted local context cached for prompt injection.

    The context is intentionally stored in private state rather than recomputed
    before every model call: volatile sections such as file lists and directory
    trees would otherwise churn the system prompt and reduce provider
    prompt-cache hits across a conversation.
    """

    _local_context_refreshed_at_cutoff: NotRequired[Annotated[int, PrivateStateAttr]]
    """Cutoff index of the summarization event we last refreshed for.

    Stored in LangGraph checkpointed state (isolated per thread) and private
    (not exposed to subagents via `PrivateStateAttr`). Used to avoid redundant
    re-runs of the detection script for the same summarization event.
    """


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class LocalContextMiddleware(AgentMiddleware):
    """Inject runtime and project context into the system prompt.

    Local sessions collect context directly from `local_root`. Remote sessions
    run the Bash detection script through the backend. Context is collected on
    first interaction and after each summarization event, cached in state, and
    appended to every model request.
    """

    state_schema = LocalContextState

    def __init__(
        self,
        backend: object,
        *,
        local_root: str | Path | None = None,
        mcp_server_info: list[MCPServerInfo] | None = None,
        tracing_project: str | None = None,
        user_tracing_project: str | None = None,
    ) -> None:
        """Initialize local or backend-based context collection.

        Args:
            backend: Backend used for remote context collection. It is not
                invoked when `local_root` is supplied.
            local_root: Captured local project working directory. When set,
                context is collected in-process instead of through the backend.
            mcp_server_info: MCP server metadata to include in the system prompt.
            tracing_project: LangSmith project the agent's own runs trace to, or
                `None` when tracing is disabled (the tracing section is omitted).
            user_tracing_project: User's original `LANGSMITH_PROJECT` used by
                shell commands the agent runs.
        """
        self.backend = backend
        self._local_root = Path(local_root) if local_root is not None else None
        tracing_context = _build_tracing_context(tracing_project, user_tracing_project)
        mcp_context = _build_mcp_context(mcp_server_info or [])
        self._static_context = "\n\n".join(
            context for context in (tracing_context, mcp_context) if context
        )

    @staticmethod
    def _handle_detect_result(result: ExecuteResponse) -> str | None:
        """Validate detection script output and normalize it for state storage.

        Args:
            result: Execution result from the backend.

        Returns:
            Stripped script output, or `None` on failure/empty output.
        """
        output = result.output.strip() if result.output else ""
        if result.exit_code is None or result.exit_code != 0:
            logger.warning(
                "Local context detection script %s; "
                "context will be omitted. Output: %.200s",
                f"exited with code {result.exit_code}"
                if result.exit_code is not None
                else "did not report an exit code",
                output or "(empty)",
            )
            return None
        if not output:
            logger.debug(
                "Local context detection script succeeded but produced no output"
            )
        return output or None

    def _run_detect_script(self) -> str | None:
        """Collect local context or run the remote detection script.

        Returns:
            Formatted context, or `None` on failure/empty output.
        """
        if self._local_root is not None:
            try:
                return _collect_local_context(self._local_root)
            except Exception:
                logger.warning(
                    "Local context collection failed for %s; context will be "
                    "omitted from system prompt",
                    self._local_root,
                    exc_info=True,
                )
                return None

        backend = self.backend
        if not isinstance(backend, _ExecutableBackend):
            logger.debug(
                "Skipping sync local context detection; backend %s only "
                "supports async execution",
                type(backend).__name__,
            )
            return None
        try:
            result = backend.execute(
                DETECT_CONTEXT_SCRIPT, timeout=_DETECT_SCRIPT_TIMEOUT
            )
        except NotImplementedError:
            # Expected for async-only backends (e.g. HarborSandbox) that
            # define a stub execute() raising NotImplementedError.
            logger.debug(
                "Backend %s does not support sync execute; "
                "context detection deferred to async path",
                type(backend).__name__,
            )
            return None
        except Exception:
            logger.warning(
                "Local context detection failed (backend: %s); context will "
                "be omitted from system prompt",
                type(backend).__name__,
                exc_info=True,
            )
            return None

        return LocalContextMiddleware._handle_detect_result(result)

    # override - state parameter is intentionally narrowed from
    # AgentState to LocalContextState for type safety within this middleware.
    def before_agent(  # ty: ignore[invalid-method-override]
        self,
        state: LocalContextState,
        runtime: Runtime,  # noqa: ARG002  # Required by interface but not used in local context
    ) -> dict[str, Any] | None:
        """Run context detection on first interaction and refresh after summarization.

        On the first invocation, runs the detection script and stores the result.
        After a summarization event (indicated by a new `_summarization_event`
        in state), re-runs the script to capture any environment changes that
        occurred during the session.

        Args:
            state: Current agent state.
            runtime: Runtime context.

        Returns:
            State update with `_local_context` populated on success. On a
                post-summarization refresh failure, returns a state update
                recording the cutoff (without `_local_context`) to prevent
                retry loops.

                Returns `None` if context is already set and no refresh is
                needed, or if initial detection fails.
        """
        # --- Post-summarization refresh ---
        # _summarization_event is a private field from SummarizationState.
        # At runtime the merged state dict contains all middleware fields;
        # accessed as untyped dict value because LocalContextState does not
        # (and should not) redeclare it.
        raw_event = state.get("_summarization_event")
        if raw_event is not None:
            event: SummarizationEvent = raw_event
            cutoff = event.get("cutoff_index")
            refreshed_cutoff = state.get("_local_context_refreshed_at_cutoff")
            if cutoff != refreshed_cutoff:
                output = self._run_detect_script()
                if output:
                    return {
                        "_local_context": output,
                        "_local_context_refreshed_at_cutoff": cutoff,
                    }
                # Script failed — record cutoff to avoid retry loop,
                # keep existing `_local_context`.
                return {"_local_context_refreshed_at_cutoff": cutoff}

        # --- Initial detection (first invocation) ---
        if state.get("_local_context"):
            return None

        output = self._run_detect_script()
        if output:
            return {"_local_context": output}
        return None

    async def _arun_detect_script(self) -> str | None:
        """Collect context asynchronously without blocking the event loop.

        Local collection and sync-only remote backends run in a worker thread.
        Async-capable remote backends use `aexecute`.

        Returns:
            Stripped script output, or `None` on failure/empty output.
        """
        if self._local_root is not None:
            try:
                return await asyncio.to_thread(
                    _collect_local_context,
                    self._local_root,
                )
            except Exception:
                logger.warning(
                    "Async local context collection failed for %s; context "
                    "will be omitted from system prompt",
                    self._local_root,
                    exc_info=True,
                )
                return None

        backend = self.backend
        if not (
            isinstance(backend, _AsyncExecutableBackend)
            and inspect.iscoroutinefunction(backend.aexecute)
        ):
            try:
                return await asyncio.to_thread(self._run_detect_script)
            except Exception:
                logger.warning(
                    "Local context detection via sync fallback failed "
                    "(backend: %s); context will be omitted from system prompt",
                    type(backend).__name__,
                    exc_info=True,
                )
                return None
        try:
            result = await backend.aexecute(
                DETECT_CONTEXT_SCRIPT, timeout=_DETECT_SCRIPT_TIMEOUT
            )
        except Exception:
            logger.warning(
                "Local context detection failed (backend: %s); context will "
                "be omitted from system prompt",
                type(backend).__name__,
                exc_info=True,
            )
            return None

        return LocalContextMiddleware._handle_detect_result(result)

    async def abefore_agent(  # ty: ignore[invalid-method-override]
        self,
        state: LocalContextState,
        runtime: Runtime,  # noqa: ARG002  # Required by interface but not used in local context
    ) -> dict[str, Any] | None:
        """Async variant of `before_agent` for use in async execution contexts.

        Args:
            state: Current agent state.
            runtime: Runtime context.

        Returns:
            State update with `_local_context` populated on success. On a
                post-summarization refresh failure, returns a state update
                recording the cutoff (without `_local_context`) to prevent
                retry loops.

                Returns `None` if context is already set and no refresh is
                needed, or if initial detection fails.
        """
        raw_event = state.get("_summarization_event")
        if raw_event is not None:
            event: SummarizationEvent = raw_event
            cutoff = event.get("cutoff_index")
            refreshed_cutoff = state.get("_local_context_refreshed_at_cutoff")
            if cutoff != refreshed_cutoff:
                output = await self._arun_detect_script()
                if output:
                    return {
                        "_local_context": output,
                        "_local_context_refreshed_at_cutoff": cutoff,
                    }
                return {"_local_context_refreshed_at_cutoff": cutoff}

        if state.get("_local_context"):
            return None

        output = await self._arun_detect_script()
        if output:
            return {"_local_context": output}
        return None

    def _get_modified_request(self, request: ModelRequest) -> ModelRequest | None:
        """Append local context and MCP info to the system prompt if available.

        Args:
            request: The model request to potentially modify.

        Returns:
            Modified request with context appended, or `None`.
        """
        state = cast("LocalContextState", request.state)
        local_context = state.get("_local_context", "")
        system_prompt = request.system_prompt or ""

        if local_context:
            if self._static_context:
                prompt_parts = (system_prompt, local_context, self._static_context)
            else:
                prompt_parts = (system_prompt, local_context)
        elif self._static_context:
            prompt_parts = (system_prompt, self._static_context)
        else:
            return None

        return request.override(system_prompt="\n\n".join(prompt_parts))

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject local context into system prompt.

        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        modified_request = self._get_modified_request(request)
        return handler(modified_request or request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Inject local context into system prompt (async).

        Args:
            request: The model request being processed.
            handler: The async handler function to call with the modified request.

        Returns:
            The model response from the handler.
        """
        modified_request = self._get_modified_request(request)
        return await handler(modified_request or request)


__all__ = ["LocalContextMiddleware"]
