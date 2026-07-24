"""Shared path-safety and size limits for read-only repository inspection.

Both the goal-criteria agent's `_RepositoryToolBudgetMiddleware` and the rubric
grader's read-only tools let an LLM sub-agent inspect working-directory files.
They must apply identical guarantees: reads stay confined to the repository
root, symlink escapes are rejected in sandboxes and local filesystems, and every
result is size bounded so a single tool call cannot blow the sub-agent's context
budget.

`RepositoryBounds` centralizes that logic so the middleware and the grader tool
wrappers share one implementation. It is intentionally framework-agnostic: it
operates on tool names and argument dicts and returns either a bounded value or
an error-message string, leaving `ToolMessage`/`Command` construction and the
per-run call budget to the caller.
"""

from __future__ import annotations

import ast
import asyncio
import base64
import json
import logging
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import SandboxBackendProtocol

if TYPE_CHECKING:
    from collections.abc import Sequence

    from deepagents.backends.protocol import BackendProtocol, FileInfo

logger = logging.getLogger(__name__)

REPOSITORY_TOOL_CALL_LIMIT = 25
REPOSITORY_READ_LINE_LIMIT = 120
REPOSITORY_READ_BYTE_LIMIT = 256_000
REPOSITORY_DIRECTORY_ENTRY_LIMIT = 200
REPOSITORY_GLOB_MATCH_LIMIT = 200
REPOSITORY_GREP_MATCH_LIMIT = 100
REPOSITORY_TOOL_RESULT_LIMIT = 12_000
REPOSITORY_TOOL_NAMES = frozenset({"ls", "read_file", "glob", "grep"})
REPOSITORY_PATH_RESULT_PREFIX = "__DEEPAGENTS_REPOSITORY_PATH__"

REPOSITORY_PATH_ERROR = "Repository path is unavailable."
REPOSITORY_UNAVAILABLE_ERROR = (
    "Repository is temporarily unavailable; the path could not be verified."
)
REPOSITORY_SIZE_ERROR = "Repository file exceeds the size limit."
REPOSITORY_LISTING_ERROR = "Repository directory exceeds the listing limit."
REPOSITORY_READ_ONLY_ERROR = "Repository inspection is limited to read-only tools."

# Backend faults that should degrade to a bounded, logged "path unavailable"
# error rather than crashing the sub-agent.
_BACKEND_ERRORS: tuple[type[BaseException], ...] = (
    NotImplementedError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


class RepositoryBounds:
    """Path-safety and size limits for read-only repository inspection tools."""

    def __init__(
        self,
        backend: BackendProtocol,
        *,
        root: str = "/",
        source_root: str | None = None,
    ) -> None:
        """Initialize repository bounds rooted at an absolute backend path.

        Args:
            backend: Server-side repository backend used by filesystem tools.
            root: Absolute backend path that bounds repository reads.
            source_root: Optional host path translated to virtual paths beneath
                `root`. Only valid for virtual filesystem backends.

        Raises:
            ValueError: If a root is unsafe or `source_root` is incompatible
                with the backend.
        """
        self._virtual_paths = getattr(backend, "virtual_mode", False) is True
        windows_root = PureWindowsPath(root)
        self._source_windows_root = (
            windows_root
            if source_root is None
            and self._virtual_paths
            and windows_root.is_absolute()
            else None
        )
        self._source_posix_root: PurePosixPath | None = None
        if source_root is not None:
            if not self._virtual_paths:
                msg = "Repository source_root requires a virtual filesystem backend."
                raise ValueError(msg)
            source_windows_root = PureWindowsPath(source_root)
            if source_windows_root.is_absolute():
                self._source_windows_root = source_windows_root
            else:
                source_posix_root = PurePosixPath(source_root)
                if (
                    not source_posix_root.is_absolute()
                    or ".." in source_posix_root.parts
                    or "~" in source_root
                ):
                    msg = (
                        "Repository source_root must be an absolute contained path: "
                        f"{source_root!r}"
                    )
                    raise ValueError(msg)
                self._source_posix_root = source_posix_root
        if self._source_windows_root is not None:
            path: PurePosixPath | PureWindowsPath = PurePosixPath("/")
        elif windows_root.is_absolute():
            path = windows_root
        else:
            path = PurePosixPath(root.replace("\\", "/"))
        if not path.is_absolute() or ".." in path.parts or "~" in root:
            msg = f"Repository root must be an absolute contained path: {root!r}"
            raise ValueError(msg)
        self._backend = backend
        self._root = str(path)
        self._root_path = path
        self._windows_paths = isinstance(path, PureWindowsPath)
        self._filesystem = backend if isinstance(backend, FilesystemBackend) else None
        self._sandbox = (
            backend
            if self._filesystem is None and isinstance(backend, SandboxBackendProtocol)
            else None
        )
        self._filesystem_root: Path | None = None
        if self._filesystem is not None:
            try:
                self._filesystem_root = self._resolve_filesystem_path(self._root)
            except _BACKEND_ERRORS:
                logger.warning(
                    "Could not resolve the local repository root; local repository "
                    "paths will be unavailable",
                    exc_info=True,
                )

    @property
    def root(self) -> str:
        """Absolute path that bounds repository reads."""
        return self._root

    def safe_path(self, raw_path: str) -> bool:
        """Return whether an explicit repository path is absolute and contained."""
        return self.normalize_path(raw_path) is not None

    def normalize_path(self, raw_path: str) -> str | None:
        """Return the contained backend path, translating Windows host paths.

        Virtual filesystem backends expose POSIX paths even when their host root
        is a native host path. Direct host paths under that configured root are
        translated to the equivalent virtual path before backend calls.

        Returns:
            A normalized contained backend path, or `None` when unsafe.
        """
        if self._source_windows_root is not None:
            windows_path = PureWindowsPath(raw_path)
            if windows_path.is_absolute():
                root_parts = self._source_windows_root.parts
                path_parts = windows_path.parts
                if (
                    ".." in path_parts
                    or "~" in raw_path
                    or len(path_parts) < len(root_parts)
                    or any(
                        part.casefold() != root.casefold()
                        for part, root in zip(
                            path_parts[: len(root_parts)],
                            root_parts,
                            strict=True,
                        )
                    )
                ):
                    return None
                return str(PurePosixPath("/", *path_parts[len(root_parts) :]))
            if "\\" in raw_path:
                return None

        if self._source_posix_root is not None:
            source_path = PurePosixPath(raw_path)
            source_root = self._source_posix_root
            if ".." in source_path.parts or "~" in raw_path:
                return None
            if source_path == source_root or source_root in source_path.parents:
                relative = source_path.relative_to(source_root)
                return str(PurePosixPath("/", *relative.parts))

        if self._windows_paths:
            path: PurePosixPath | PureWindowsPath = PureWindowsPath(raw_path)
        else:
            path = PurePosixPath(raw_path)
        root = self._root_path
        if (
            not path.is_absolute()
            or ".." in path.parts
            or "~" in raw_path
            or not (path == root or root in path.parents)
        ):
            return None
        return str(path)

    @staticmethod
    def safe_pattern(pattern: str) -> bool:
        """Return whether a relative or absolute glob pattern cannot traverse."""
        path = PurePosixPath(pattern.replace("\\", "/"))
        return ".." not in path.parts and "~" not in pattern

    def _resolve_filesystem_path(self, raw_path: str) -> Path:
        """Resolve a backend path to its canonical local filesystem target.

        Returns:
            The canonical host path used by the local filesystem backend.

        Raises:
            RuntimeError: If called for a backend without local filesystem access.
        """
        if self._filesystem is None:
            msg = "Local filesystem backend is unavailable."
            raise RuntimeError(msg)
        if self._filesystem.virtual_mode:
            return (self._filesystem.cwd / raw_path.lstrip("/")).resolve(strict=False)
        return Path(raw_path).resolve(strict=False)

    def _filesystem_contains(self, raw_path: str) -> bool:
        """Return whether a local path canonically resolves below the root."""
        if self._filesystem is None:
            return True
        if self._filesystem_root is None:
            return False
        try:
            resolved = self._resolve_filesystem_path(raw_path)
        except _BACKEND_ERRORS:
            logger.warning(
                "Local repository containment check failed; treating the path as "
                "unavailable",
                exc_info=True,
            )
            return False
        return (
            resolved == self._filesystem_root
            or self._filesystem_root in resolved.parents
        )

    def _containment_command(self, raw_path: str) -> str:
        """Build a sandbox command that checks the canonical repository boundary.

        Returns:
            A command that emits a private success marker only for contained paths.
        """
        payload = base64.b64encode(json.dumps([self._root, raw_path]).encode()).decode()
        return (
            'python3 -c "import base64,json,os;'
            f"values=json.loads(base64.b64decode('{payload}'));"
            "root=os.path.realpath(values[0]);path=os.path.realpath(values[1]);"
            "contained=os.path.commonpath([root,path])==root;"
            f"print('{REPOSITORY_PATH_RESULT_PREFIX}'+str(int(contained)))\""
        )

    def sandbox_contains(self, raw_path: str) -> bool:
        """Return whether the backend resolves a path below the repository root.

        For sandbox and local-filesystem backends this performs a canonical
        (symlink-resolving) containment check. For any other backend there is
        no canonical check available and this returns `True`, so callers must
        still apply `safe_path` for the lexical guard.
        """
        if self._sandbox is None:
            return self._filesystem_contains(raw_path)
        try:
            result = self._sandbox.execute(self._containment_command(raw_path))
        except _BACKEND_ERRORS:
            logger.warning(
                "Repository containment check failed; treating the path as unavailable",
                exc_info=True,
            )
            return False
        return result.exit_code in {None, 0} and any(
            line == f"{REPOSITORY_PATH_RESULT_PREFIX}1"
            for line in result.output.splitlines()
        )

    async def asandbox_contains(self, raw_path: str) -> bool:
        """Asynchronously check canonical backend repository containment.

        For sandbox and local-filesystem backends this performs a canonical
        (symlink-resolving) containment check. For any other backend there is
        no canonical check available and this returns `True`, so callers must
        still apply `safe_path` for the lexical guard.

        Returns:
            `True` when the backend resolves the path below the repository
            root, or when no canonical check is available for the backend.
        """
        if self._sandbox is None:
            if self._filesystem is None:
                return True
            return await asyncio.to_thread(self._filesystem_contains, raw_path)
        try:
            result = await self._sandbox.aexecute(self._containment_command(raw_path))
        except _BACKEND_ERRORS:
            logger.warning(
                "Repository containment check failed; treating the path as unavailable",
                exc_info=True,
            )
            return False
        return result.exit_code in {None, 0} and any(
            line == f"{REPOSITORY_PATH_RESULT_PREFIX}1"
            for line in result.output.splitlines()
        )

    def entry_size(
        self,
        entries: Sequence[FileInfo] | None,
        normalized_path: str,
    ) -> int | None:
        """Return the reported byte size of a backend entry, if present.

        Malformed entries (not a mapping, or missing/non-string `path`) are
        skipped rather than raising, so a single bad entry cannot fail an
        otherwise valid preflight.

        Returns:
            The entry's integer size, or `None` when unknown.
        """
        expected = self.normalize_path(normalized_path)
        if expected is None:
            return None
        case_insensitive = self._windows_paths or self._source_windows_root is not None
        for item in entries or []:
            raw = item.get("path") if isinstance(item, dict) else None
            if not isinstance(raw, str):
                continue
            actual = self.normalize_path(raw)
            if actual is not None and (
                actual.casefold() == expected.casefold()
                if case_insensitive
                else actual == expected
            ):
                size = item.get("size")
                return size if isinstance(size, int) else None
        return None

    def _validate_search_paths(
        self,
        name: str,
        args: dict[str, Any],
    ) -> str | None:
        """Validate optional paths and path-like patterns for search tools.

        Returns:
            A path error message, or `None` when every explicit path is contained.
        """
        path = args.get("path")
        if path is not None and (not isinstance(path, str) or not self.safe_path(path)):
            return REPOSITORY_PATH_ERROR

        patterns = [args.get("pattern")] if name == "glob" else [args.get("glob")]
        if any(
            pattern is not None
            and (not isinstance(pattern, str) or not self.safe_pattern(pattern))
            for pattern in patterns
        ):
            return REPOSITORY_PATH_ERROR
        return None

    def preflight(self, name: str, args: dict[str, Any]) -> str | None:
        """Reject malformed paths and backend entries that exceed hard limits.

        Rejects any tool name outside the read-only inspection set so the
        read-only guarantee fails closed rather than depending on callers to
        wire only read-only tools.

        Returns:
            A bounded error message, or `None` when preflight succeeds.
        """
        if name not in REPOSITORY_TOOL_NAMES:
            # Read-only invariant: only the four read-only inspection tools are
            # ever validated here. Reject anything else (e.g. a mis-wired write
            # tool) instead of falling through and validating it as a path
            # operation.
            return REPOSITORY_READ_ONLY_ERROR
        if name in {"glob", "grep"}:
            error = self._validate_search_paths(name, args)
            if error is not None:
                return error
            raw_path = args.get("path")
            if raw_path is None:
                raw_path = self._root
            if not isinstance(raw_path, str):
                return REPOSITORY_PATH_ERROR
            normalized_path = self.normalize_path(raw_path)
            if normalized_path is None or not self.sandbox_contains(normalized_path):
                return REPOSITORY_PATH_ERROR
            return None

        key = "file_path" if name == "read_file" else "path"
        raw_path = args.get(key)
        if not isinstance(raw_path, str):
            return None

        normalized_path = self.normalize_path(raw_path)
        if normalized_path is None:
            return REPOSITORY_PATH_ERROR
        if not self.sandbox_contains(normalized_path):
            return REPOSITORY_PATH_ERROR
        path: PurePosixPath | PureWindowsPath
        if self._windows_paths:
            path = PureWindowsPath(normalized_path)
        else:
            path = PurePosixPath(normalized_path)

        # Scope the guard to the backend call itself: a backend that raises
        # (outage, serialization fault) reports a distinct "temporarily
        # unavailable" error, so the grader/user can tell an infrastructure
        # fault apart from a genuinely absent or out-of-bounds path (which
        # returns REPOSITORY_PATH_ERROR). The size/entry bookkeeping below is
        # deliberately left outside the guard so a defect there surfaces as a
        # real crash rather than silently degrading every run.
        try:
            result = self._backend.ls(
                normalized_path if name == "ls" else str(path.parent)
            )
        except _BACKEND_ERRORS:
            logger.warning(
                "Repository preflight failed for tool %r; treating the "
                "repository as temporarily unavailable",
                name,
                exc_info=True,
            )
            return REPOSITORY_UNAVAILABLE_ERROR
        if result.error is not None:
            return REPOSITORY_PATH_ERROR
        if name == "ls":
            if len(result.entries or []) > REPOSITORY_DIRECTORY_ENTRY_LIMIT:
                return REPOSITORY_LISTING_ERROR
        else:  # read_file
            size = self.entry_size(result.entries, normalized_path)
            if size is not None and size > REPOSITORY_READ_BYTE_LIMIT:
                return REPOSITORY_SIZE_ERROR
        return None

    async def apreflight(self, name: str, args: dict[str, Any]) -> str | None:
        """Asynchronously enforce repository path and metadata limits.

        Rejects any tool name outside the read-only inspection set so the
        read-only guarantee fails closed rather than depending on callers to
        wire only read-only tools.

        Returns:
            A bounded error message, or `None` when preflight succeeds.
        """
        if name not in REPOSITORY_TOOL_NAMES:
            # Read-only invariant: only the four read-only inspection tools are
            # ever validated here. Reject anything else (e.g. a mis-wired write
            # tool) instead of falling through and validating it as a path
            # operation.
            return REPOSITORY_READ_ONLY_ERROR
        if name in {"glob", "grep"}:
            error = self._validate_search_paths(name, args)
            if error is not None:
                return error
            raw_path = args.get("path")
            if raw_path is None:
                raw_path = self._root
            if not isinstance(raw_path, str):
                return REPOSITORY_PATH_ERROR
            normalized_path = self.normalize_path(raw_path)
            if normalized_path is None or not await self.asandbox_contains(
                normalized_path
            ):
                return REPOSITORY_PATH_ERROR
            return None

        key = "file_path" if name == "read_file" else "path"
        raw_path = args.get(key)
        if not isinstance(raw_path, str):
            return None

        normalized_path = self.normalize_path(raw_path)
        if normalized_path is None:
            return REPOSITORY_PATH_ERROR
        if not await self.asandbox_contains(normalized_path):
            return REPOSITORY_PATH_ERROR
        path: PurePosixPath | PureWindowsPath
        if self._windows_paths:
            path = PureWindowsPath(normalized_path)
        else:
            path = PurePosixPath(normalized_path)

        try:
            result = await self._backend.als(
                normalized_path if name == "ls" else str(path.parent)
            )
        except _BACKEND_ERRORS:
            logger.warning(
                "Repository preflight failed for tool %r; treating the "
                "repository as temporarily unavailable",
                name,
                exc_info=True,
            )
            return REPOSITORY_UNAVAILABLE_ERROR
        if result.error is not None:
            return REPOSITORY_PATH_ERROR
        if name == "ls":
            if len(result.entries or []) > REPOSITORY_DIRECTORY_ENTRY_LIMIT:
                return REPOSITORY_LISTING_ERROR
        elif name == "read_file":
            size = self.entry_size(result.entries, normalized_path)
            if size is not None and size > REPOSITORY_READ_BYTE_LIMIT:
                return REPOSITORY_SIZE_ERROR
        return None

    def clamp_args(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Clamp repository-tool arguments that directly control result size.

        Returns:
            A new args dict with bounded read lines or grep matches, and a
            repository-root default for search paths.
        """
        clamped = dict(args)
        if name == "read_file":
            limit = clamped.get("limit", REPOSITORY_READ_LINE_LIMIT)
            if not isinstance(limit, int) or isinstance(limit, bool):
                limit = REPOSITORY_READ_LINE_LIMIT
            clamped["limit"] = max(1, min(limit, REPOSITORY_READ_LINE_LIMIT))
        elif name in {"glob", "grep"} and clamped.get("path") is None:
            clamped["path"] = self._root
        if name == "grep":
            count = clamped.get("max_count", REPOSITORY_GREP_MATCH_LIMIT)
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                count = REPOSITORY_GREP_MATCH_LIMIT
            clamped["max_count"] = min(count, REPOSITORY_GREP_MATCH_LIMIT)
        path_key = "file_path" if name == "read_file" else "path"
        raw_path = clamped.get(path_key)
        if isinstance(raw_path, str):
            normalized_path = self.normalize_path(raw_path)
            if normalized_path is not None:
                clamped[path_key] = normalized_path
        return clamped

    @staticmethod
    def bounded_glob_content(content: str) -> str:
        """Limit a filesystem glob's rendered path count when it is parseable.

        Returns:
            Glob output containing no more than the configured number of paths.
        """
        body, separator, notes = content.partition("\n\n")
        try:
            paths = ast.literal_eval(body)
        except (SyntaxError, ValueError):
            return content
        if not isinstance(paths, list) or not all(
            isinstance(path, str) for path in paths
        ):
            return content
        if len(paths) <= REPOSITORY_GLOB_MATCH_LIMIT:
            return content
        marker = (
            "[Glob results limited to the first "
            f"{REPOSITORY_GLOB_MATCH_LIMIT} matches.]"
        )
        bounded = str(paths[:REPOSITORY_GLOB_MATCH_LIMIT])
        suffix = f"\n\n{notes}" if separator and notes else ""
        return f"{bounded}\n\n{marker}{suffix}"

    def bound_text(self, name: str, content: str) -> str:
        """Return a size-bounded repository tool result body.

        Returns:
            The bounded content, with glob output additionally match-limited.
        """
        if name == "glob":
            content = self.bounded_glob_content(content)
        if len(content) > REPOSITORY_TOOL_RESULT_LIMIT:
            marker = "\n[Repository tool result shortened to the context limit.]"
            content = content[: REPOSITORY_TOOL_RESULT_LIMIT - len(marker)] + marker
        return content
