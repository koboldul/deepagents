"""Unit tests for skill-name collision (override) debug logging."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from deepagents.backends.filesystem import FilesystemBackend

from deepagents_code.plugins.adapters.skills_middleware import PluginSkillsMiddleware
from deepagents_code.skills.load import list_skills

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


_MERGE_LOGGER = "deepagents_code.skills.merge"


def _create_skill(skill_dir: Path, name: str, description: str) -> None:
    """Create a minimal skill directory with a valid `SKILL.md`."""
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(f"""---
name: {name}
description: {description}
---
Content
""")


def _override_records(
    caplog: pytest.LogCaptureFixture, level: int = logging.DEBUG
) -> list[logging.LogRecord]:
    """Return the merge-helper override records emitted at exactly `level`."""
    return [
        record
        for record in caplog.records
        if record.name == _MERGE_LOGGER and record.levelno == level
    ]


def _args(record: logging.LogRecord) -> tuple[object, ...]:
    """Return a record's positional log args, asserting they form a tuple."""
    args = record.args
    assert isinstance(args, tuple)
    return args


class TestMergeSkillHelper:
    """Directly exercise `merge_skill`."""


class TestListSkillsCollisionLogging:
    """Exercise collision logging through the CLI `list_skills` discovery path."""

    def test_project_overrides_user_logs_debug(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A project skill overriding a user skill logs a DEBUG override."""
        user_dir = tmp_path / "user_skills"
        project_dir = tmp_path / "project_skills"
        _create_skill(user_dir / "shared-skill", "shared-skill", "User version")
        _create_skill(project_dir / "shared-skill", "shared-skill", "Project version")

        with caplog.at_level(logging.DEBUG, logger=_MERGE_LOGGER):
            skills = list_skills(
                user_skills_dir=user_dir, project_skills_dir=project_dir
            )

        assert len(skills) == 1
        assert skills[0]["description"] == "Project version"

        records = _override_records(caplog)
        assert len(records) == 1
        name, prev_path, prev_label, new_path, new_label = _args(records[0])
        assert name == "shared-skill"
        assert prev_label == "user"
        assert new_label == "project"
        assert user_dir.as_posix() in str(prev_path)
        assert project_dir.as_posix() in str(new_path)

    def test_distinct_skills_do_not_log(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Distinct skill names across sources produce no override log."""
        user_dir = tmp_path / "user_skills"
        project_dir = tmp_path / "project_skills"
        _create_skill(user_dir / "user-skill", "user-skill", "User skill")
        _create_skill(project_dir / "project-skill", "project-skill", "Project skill")

        with caplog.at_level(logging.DEBUG, logger=_MERGE_LOGGER):
            skills = list_skills(
                user_skills_dir=user_dir, project_skills_dir=project_dir
            )

        assert len(skills) == 2
        assert _override_records(caplog, logging.DEBUG) == []
        assert _override_records(caplog, logging.WARNING) == []


class TestMiddlewareCollisionLogging:
    """Exercise collision logging through `PluginSkillsMiddleware` (sync + async).

    These lock in the new three-way `zip(self.sources, self.source_labels,
    self._namespaces, ...)` wiring: both entry points must merge through
    `merge_skill` and log overrides identically.
    """

    @staticmethod
    def _middleware(user_dir: Path, project_dir: Path) -> PluginSkillsMiddleware:
        """Build a middleware over two colliding, non-namespaced sources."""
        _create_skill(user_dir / "review", "review", "User review")
        _create_skill(project_dir / "review", "review", "Project review")
        return PluginSkillsMiddleware(
            backend=FilesystemBackend(virtual_mode=False),
            sources=[(str(user_dir), "User"), (str(project_dir), "Project")],
            system_prompt=None,
        )

    @staticmethod
    def _namespaced_middleware(dir_a: Path, dir_b: Path) -> PluginSkillsMiddleware:
        """Build a middleware over two colliding plugin (namespaced) sources.

        Both sources share the `myplugin` namespace and a `review` skill, so
        both qualify to `myplugin:review` and drive the namespaced (`else`)
        loop branch through `load_namespaced_skills` — the branch a plain-source
        collision never reaches.
        """
        _create_skill(dir_a / "review", "review", "Plugin A review")
        _create_skill(dir_b / "review", "review", "Plugin B review")
        return PluginSkillsMiddleware(
            backend=FilesystemBackend(virtual_mode=False),
            sources=[
                (str(dir_a), "Plugin A", "myplugin"),
                (str(dir_b), "Plugin B", "myplugin"),
            ],
            system_prompt=None,
        )
