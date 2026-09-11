"""Unit tests for rubric (`RubricMiddleware`) CLI wiring."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from deepagents_code._server_config import ServerConfig
from deepagents_code.goal_state_limits import RUBRIC_CHAR_LIMIT
from deepagents_code.main import _resolve_rubric_text


class TestResolveRubricText:
    """`_resolve_rubric_text` literal/file/@path resolution."""

    def test_none_when_unset(self) -> None:
        assert _resolve_rubric_text(None) is None

    def test_literal(self) -> None:
        assert _resolve_rubric_text("tests pass; minimal") == "tests pass; minimal"

    def test_literal_is_stripped(self) -> None:
        assert _resolve_rubric_text("  do X  ") == "do X"

    def test_empty_literal_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            _resolve_rubric_text("   ")

    def test_at_path_in_rubric(self, tmp_path: Path) -> None:
        f = tmp_path / "rubric.md"
        f.write_text("from at-path", encoding="utf-8")
        assert _resolve_rubric_text(f"@{f}") == "from at-path"

    def test_at_prefix_always_treated_as_path(self) -> None:
        # Documents the one-way ambiguity: any `@`-prefixed value is read as a
        # file path, so a literal rubric beginning with `@` is unreachable and
        # surfaces a read error rather than being used verbatim.
        with pytest.raises(ValueError, match="Could not read rubric file"):
            _resolve_rubric_text("@tests pass; minimal diff")

    def test_bare_at_sign_rejected(self) -> None:
        # `@` with no path (e.g. an empty shell glob) must still error rather
        # than silently resolving to the current directory.
        with pytest.raises(ValueError, match="Could not read rubric file"):
            _resolve_rubric_text("@")

    def test_at_path_expands_tilde(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Exercises the `.expanduser()` call, otherwise uncovered.
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        (tmp_path / "rubric.md").write_text("tilde criteria", encoding="utf-8")
        assert _resolve_rubric_text("@~/rubric.md") == "tilde criteria"

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Could not read rubric file"):
            _resolve_rubric_text(f"@{tmp_path / 'nope.md'}")

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "rubric.md"
        f.write_text("   \n", encoding="utf-8")
        with pytest.raises(ValueError, match="is empty"):
            _resolve_rubric_text(f"@{f}")

    def test_oversized_literal_and_file_are_rejected(self, tmp_path: Path) -> None:
        """Both CLI rubric forms fail before an agent session can start."""
        oversized = "x" * (RUBRIC_CHAR_LIMIT + 1)
        path = tmp_path / "large.md"
        path.write_text(oversized, encoding="utf-8")

        for value in (oversized, f"@{path}"):
            with pytest.raises(ValueError, match="maximum is 12,000"):
                _resolve_rubric_text(value)


class TestRubricGating:
    """Rubric flags require `-n`/piped stdin; the guard lives in `cli_main`."""


class TestServerConfigRubric:
    """Rubric grader settings round-trip through env serialization."""

    def test_from_cli_args_forwards_rubric_settings(self) -> None:
        config = ServerConfig.from_cli_args(
            project_context=None,
            model_name=None,
            model_params=None,
            assistant_id="agent",
            auto_approve=False,
            sandbox_type="none",
            sandbox_id=None,
            sandbox_snapshot_name=None,
            sandbox_setup=None,
            enable_shell=True,
            enable_ask_user=False,
            rubric_model="openai:gpt-5.1",
            rubric_max_iterations=7,
            mcp_config_path=None,
            no_mcp=False,
            trust_project_mcp=None,
            interactive=True,
        )
        assert config.rubric_model == "openai:gpt-5.1"
        assert config.rubric_max_iterations == 7
