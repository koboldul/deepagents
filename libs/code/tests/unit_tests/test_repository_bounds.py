"""Unit tests for the shared repository-inspection bounds."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from deepagents.backends.protocol import LsResult

from deepagents_code._repository_bounds import REPOSITORY_PATH_ERROR, RepositoryBounds

if TYPE_CHECKING:
    from pathlib import Path

    from deepagents.backends.protocol import FileInfo


def _backend(*, size: int = 10) -> MagicMock:
    backend = MagicMock()
    backend.ls.return_value = LsResult(
        entries=[{"path": "/src.py", "is_dir": False, "size": size}]
    )
    return backend


class TestRepositoryBoundsConstruction:
    """The root is validated and normalized at construction time."""

    @pytest.mark.parametrize("root", ["relative", "/a/../b", "~/x"])
    def test_rejects_unsafe_root(self, root: str) -> None:
        with pytest.raises(ValueError, match="absolute contained path"):
            RepositoryBounds(_backend(), root=root)

    def test_virtual_backend_translates_windows_root(self) -> None:
        backend = _backend()
        backend.virtual_mode = True

        bounds = RepositoryBounds(backend, root=r"C:\Users\Owner\Repository")

        assert bounds.root == "/"


class TestSafePath:
    """Explicit paths must be absolute, non-traversing, and under the root."""

    @pytest.mark.parametrize(
        "path", ["../etc/passwd", "~/secrets", "relative/x", "/a/../b"]
    )
    def test_unsafe_paths_are_rejected(self, path: str) -> None:
        bounds = RepositoryBounds(_backend(), root="/workspace")
        assert bounds.safe_path(path) is False

    @pytest.mark.parametrize(
        "path",
        [
            r"c:\Users\Owner\Repository\README.md",
            r"C:\users\owner\repository\README.md",
        ],
    )
    def test_windows_paths_are_case_insensitive(self, path: str) -> None:
        bounds = RepositoryBounds(
            _backend(),
            root=r"C:\Users\Owner\Repository",
        )

        assert bounds.safe_path(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            r"C:\Users\Owner\Repository-escape\secret.txt",
            r"D:\Users\Owner\Repository\secret.txt",
            r"C:\Users\Owner\Repository\..\secret.txt",
        ],
    )
    def test_windows_paths_reject_escapes(self, path: str) -> None:
        bounds = RepositoryBounds(
            _backend(),
            root=r"C:\Users\Owner\Repository",
        )

        assert bounds.safe_path(path) is False


class TestClampArgs:
    """Read/search arguments are clamped to hard limits."""

    def test_virtual_windows_path_is_translated(self) -> None:
        backend = _backend()
        backend.virtual_mode = True
        bounds = RepositoryBounds(
            backend,
            root=r"C:\Users\Owner\Repository",
        )

        clamped = bounds.clamp_args(
            "read_file",
            {"file_path": r"c:\users\owner\repository\README.md"},
        )

        assert clamped["file_path"] == "/README.md"

    def test_virtual_source_root_translates_native_and_virtual_paths(
        self,
        tmp_path: Path,
    ) -> None:
        backend = _backend()
        backend.virtual_mode = True
        source_root = tmp_path / "repository"
        bounds = RepositoryBounds(backend, source_root=str(source_root))

        native = bounds.clamp_args(
            "read_file",
            {"file_path": str(source_root / "README.md")},
        )
        virtual = bounds.clamp_args(
            "read_file",
            {"file_path": "/README.md"},
        )

        assert native["file_path"] == "/README.md"
        assert virtual["file_path"] == "/README.md"


class TestBoundText:
    """Result bodies are size and match bounded."""


class TestPreflight:
    """Preflight enforces path safety and backend metadata limits."""

    def test_rejects_local_symlink_outside_root(self, tmp_path: Path) -> None:
        from deepagents.backends.filesystem import FilesystemBackend

        repository = tmp_path / "repository"
        repository.mkdir()
        secret = tmp_path / "secret.txt"
        secret.write_text("secret")
        link = repository / "proof.txt"
        link.symlink_to(secret)
        backend = FilesystemBackend(root_dir=repository, virtual_mode=False)
        bounds = RepositoryBounds(backend, root=str(repository))

        assert (
            bounds.preflight("read_file", {"file_path": str(link)})
            == REPOSITORY_PATH_ERROR
        )

    async def test_async_rejects_local_symlink_outside_root(
        self, tmp_path: Path
    ) -> None:
        from deepagents.backends.filesystem import FilesystemBackend

        repository = tmp_path / "repository"
        repository.mkdir()
        secret = tmp_path / "secret.txt"
        secret.write_text("secret")
        link = repository / "proof.txt"
        link.symlink_to(secret)
        backend = FilesystemBackend(root_dir=repository, virtual_mode=False)
        bounds = RepositoryBounds(backend, root=str(repository))

        assert (
            await bounds.apreflight("read_file", {"file_path": str(link)})
            == REPOSITORY_PATH_ERROR
        )

    def test_allows_local_symlink_within_root(self, tmp_path: Path) -> None:
        from deepagents.backends.filesystem import FilesystemBackend

        repository = tmp_path / "repository"
        repository.mkdir()
        target = repository / "target.txt"
        target.write_text("safe")
        link = repository / "proof.txt"
        link.symlink_to(target)
        backend = FilesystemBackend(root_dir=repository, virtual_mode=False)
        bounds = RepositoryBounds(backend, root=str(repository))

        assert bounds.preflight("read_file", {"file_path": str(link)}) is None

    def test_virtual_windows_path_reaches_backend_as_posix(
        self,
        tmp_path: Path,
    ) -> None:
        from deepagents.backends.filesystem import FilesystemBackend

        repository = tmp_path / "repository"
        repository.mkdir()
        (repository / "README.md").write_text(
            "criteria context",
            encoding="utf-8",
        )
        backend = FilesystemBackend(root_dir=repository, virtual_mode=True)
        bounds = RepositoryBounds(
            backend,
            root=r"C:\Users\Owner\Repository",
        )
        raw_path = r"c:\users\owner\repository\README.md"

        assert bounds.preflight("read_file", {"file_path": raw_path}) is None
        clamped = bounds.clamp_args("read_file", {"file_path": raw_path})
        result = backend.read(clamped["file_path"])

        assert result.error is None
        assert result.file_data is not None
        assert result.file_data["content"] == "criteria context"

    def test_virtual_windows_path_rejects_drive_escape(self) -> None:
        backend = _backend()
        backend.virtual_mode = True
        bounds = RepositoryBounds(
            backend,
            root=r"C:\Users\Owner\Repository",
        )

        assert (
            bounds.preflight(
                "read_file",
                {"file_path": (r"C:\Users\Owner\Repository-escape\README.md")},
            )
            == REPOSITORY_PATH_ERROR
        )

    def test_windows_entry_size_matches_case_insensitively(self) -> None:
        bounds = RepositoryBounds(
            _backend(),
            root=r"C:\Users\Owner\Repository",
        )
        entries: list[FileInfo] = [
            {
                "path": r"C:\Users\Owner\Repository\README.md",
                "is_dir": False,
                "size": 42,
            }
        ]

        assert (
            bounds.entry_size(
                entries,
                r"c:\users\owner\repository\readme.md",
            )
            == 42
        )
