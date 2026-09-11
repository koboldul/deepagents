"""Tests for formatting module."""

from __future__ import annotations

import locale
from datetime import datetime
from typing import TYPE_CHECKING

from deepagents_code.formatting import format_message_timestamp, uses_24_hour_clock

if TYPE_CHECKING:
    import pytest


class TestFormatDuration:
    """Tests for format_duration() helper."""


class TestFormatMessageTimestamp:
    """Tests for format_message_timestamp() helper."""

    def test_today_omits_date(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Today's messages show only the time, not the date."""
        monkeypatch.setattr(
            "deepagents_code.formatting.uses_24_hour_clock", lambda: False
        )
        today = (
            datetime.now()
            .astimezone()
            .replace(hour=12, minute=0, second=5, microsecond=0)
        )
        assert format_message_timestamp(today.timestamp()) == "12:00:05 PM"

    def test_other_day_includes_date(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Messages from other days keep the leading date."""
        monkeypatch.setattr(
            "deepagents_code.formatting.uses_24_hour_clock", lambda: False
        )
        timestamp = datetime(2024, 1, 1, 12, 0, 5).astimezone().timestamp()
        assert format_message_timestamp(timestamp) == "Jan 1, 12:00:05 PM"

    def test_24_hour_clock_drops_am_pm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A 24-hour clock renders time without an AM/PM suffix."""
        monkeypatch.setattr(
            "deepagents_code.formatting.uses_24_hour_clock", lambda: True
        )
        timestamp = datetime(2024, 1, 1, 13, 0, 5).astimezone().timestamp()
        assert format_message_timestamp(timestamp) == "Jan 1, 13:00:05"

    def test_midnight_12_hour_renders_as_12_am(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The 12-hour branch renders midnight as `12:..:.. AM`, not `0:`."""
        monkeypatch.setattr(
            "deepagents_code.formatting.uses_24_hour_clock", lambda: False
        )
        timestamp = datetime(2024, 1, 1, 0, 0, 5).astimezone().timestamp()
        assert format_message_timestamp(timestamp) == "Jan 1, 12:00:05 AM"


class TestUses24HourClock:
    """Tests for the system 12-/24-hour clock detection."""

    @staticmethod
    def _clear_cache() -> None:
        uses_24_hour_clock.cache_clear()

    def test_macos_force_24_hour_preference_wins(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On macOS the explicit 24-hour preference is honored over locale."""
        import deepagents_code.formatting as fmt

        self._clear_cache()
        monkeypatch.setattr(fmt.sys, "platform", "darwin")
        monkeypatch.setattr(fmt, "macos_force_24_hour_time", lambda: True)
        try:
            assert fmt.uses_24_hour_clock() is True
        finally:
            self._clear_cache()

    def test_macos_force_12_hour_preference_wins(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On macOS an explicit 12-hour preference overrides the locale."""
        import deepagents_code.formatting as fmt

        self._clear_cache()
        monkeypatch.setattr(fmt.sys, "platform", "darwin")
        monkeypatch.setattr(fmt, "macos_force_24_hour_time", lambda: False)
        try:
            assert fmt.uses_24_hour_clock() is False
        finally:
            self._clear_cache()
