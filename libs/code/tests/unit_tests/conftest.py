"""Shared fixtures for unit tests."""

from __future__ import annotations

import contextlib
import ipaddress
import os
import socket
from typing import TYPE_CHECKING, Protocol, cast

import pytest
from pytest_socket import SocketBlockedError
from typing_extensions import Buffer

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable
    from contextlib import AbstractContextManager
    from pathlib import Path


_UPDATE_CHECK_SELF_MANAGED_MARK = "self_managed_update_check"
_WINDOWS_XDIST_MAX_WORKERS = 8
_REAL_SOCKET = socket.socket
_REAL_BIND = socket.socket.bind
_REAL_CONNECT = socket.socket.connect
_REAL_CONNECT_EX = socket.socket.connect_ex
_REAL_SEND = socket.socket.send
_REAL_SENDALL = socket.socket.sendall
_REAL_SENDTO = socket.socket.sendto


class _RawAcceptSocket(Protocol):
    """Socket exposing the low-level accept primitive."""

    def _accept(self) -> tuple[int, object]: ...


_SocketAddress = str | Buffer | tuple[object, ...]


class _SocketSendMsg(Protocol):
    """Bound-compatible descriptor for platforms exposing `socket.sendmsg`."""

    def __call__(
        self,
        socket_instance: socket.socket,
        buffers: Iterable[Buffer],
        ancdata: Iterable[tuple[int, int, Buffer]] = (),
        flags: int = 0,
        address: _SocketAddress | None = None,
        /,
    ) -> int: ...


class _SocketSendfile(Protocol):
    """Descriptor shape used to call the platform `socket.sendfile` method."""

    def __call__(
        self,
        socket_instance: socket.socket,
        file: object,
        offset: int = 0,
        count: int | None = None,
        /,
    ) -> int: ...


_REAL_SENDFILE = cast("_SocketSendfile", socket.socket.sendfile)
_REAL_SENDMSG = cast(
    "_SocketSendMsg | None",
    getattr(socket.socket, "sendmsg", None),
)
_REAL_SENDMSG_AFALG = getattr(socket.socket, "sendmsg_afalg", None)


def _is_loopback_address(
    family: socket.AddressFamily | int,
    address: object,
) -> bool:
    unix_family = getattr(socket, "AF_UNIX", None)
    if unix_family is not None and family == unix_family:
        return True
    if family not in {socket.AF_INET, socket.AF_INET6}:
        return False
    if not isinstance(address, tuple) or not address:
        return False
    host = address[0]
    if not isinstance(host, str):
        return False
    if host.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.split("%", maxsplit=1)[0]).is_loopback
    except ValueError:
        return False


def _require_loopback_address(
    family: socket.AddressFamily | int,
    address: object,
) -> None:
    """Reject a non-loopback destination before entering the socket API."""
    if not _is_loopback_address(family, address):
        raise SocketBlockedError


def _require_loopback_peer(socket_instance: socket.socket) -> None:
    """Guard connected send APIs that do not carry a destination argument."""
    unix_family = getattr(socket, "AF_UNIX", None)
    if unix_family is not None and socket_instance.family == unix_family:
        return
    if socket_instance.family not in {socket.AF_INET, socket.AF_INET6}:
        raise SocketBlockedError
    try:
        peer = socket_instance.getpeername()
    except OSError:
        raise SocketBlockedError from None
    _require_loopback_address(socket_instance.family, peer)


class _WindowsLoopbackSocket(_REAL_SOCKET):
    """Socket that permits only local IPC while pytest networking is disabled."""

    def bind(self, address: _SocketAddress) -> None:
        _require_loopback_address(self.family, address)
        _REAL_BIND(self, address)

    def connect(self, address: _SocketAddress) -> None:
        _require_loopback_address(self.family, address)
        _REAL_CONNECT(self, address)

    def connect_ex(self, address: _SocketAddress) -> int:
        _require_loopback_address(self.family, address)
        return _REAL_CONNECT_EX(self, address)

    def send(self, data: Buffer, flags: int = 0) -> int:
        _require_loopback_peer(self)
        return _REAL_SEND(self, data, flags)

    def sendall(self, data: Buffer, flags: int = 0) -> None:
        _require_loopback_peer(self)
        _REAL_SENDALL(self, data, flags)

    def sendto(
        self,
        data: Buffer,
        flags_or_address: int | _SocketAddress,
        address: _SocketAddress | None = None,
        /,
    ) -> int:
        destination = flags_or_address if address is None else address
        _require_loopback_address(self.family, destination)
        if address is None:
            return _REAL_SENDTO(
                self,
                data,
                cast("_SocketAddress", flags_or_address),
            )
        return _REAL_SENDTO(self, data, cast("int", flags_or_address), address)

    def sendfile(
        self,
        file: object,
        offset: int = 0,
        count: int | None = None,
    ) -> int:
        _require_loopback_peer(self)
        return _REAL_SENDFILE(self, file, offset, count)


def _guarded_sendmsg(
    socket_instance: _WindowsLoopbackSocket,
    buffers: Iterable[Buffer],
    ancdata: Iterable[tuple[int, int, Buffer]] = (),
    flags: int = 0,
    address: _SocketAddress | None = None,
) -> int:
    """Apply the loopback policy to platforms exposing `socket.sendmsg`."""
    sendmsg = _REAL_SENDMSG
    if sendmsg is None:
        msg = "socket.sendmsg is unavailable"
        raise NotImplementedError(msg)
    if address is None:
        _require_loopback_peer(socket_instance)
        return sendmsg(socket_instance, buffers, ancdata, flags)
    _require_loopback_address(socket_instance.family, address)
    return sendmsg(socket_instance, buffers, ancdata, flags, address)


def _guarded_sendmsg_afalg(
    socket_instance: _WindowsLoopbackSocket,
    *args: object,
    **kwargs: object,
) -> int:
    """Block Linux AF_ALG sends, which cannot represent loopback IPC."""
    del socket_instance, args, kwargs
    raise SocketBlockedError


if _REAL_SENDMSG is not None:
    setattr(  # noqa: B010  # method exists only on supporting platforms
        _WindowsLoopbackSocket,
        "sendmsg",
        _guarded_sendmsg,
    )
if _REAL_SENDMSG_AFALG is not None:
    setattr(  # noqa: B010  # method exists only on supporting platforms
        _WindowsLoopbackSocket,
        "sendmsg_afalg",
        _guarded_sendmsg_afalg,
    )


def _windows_loopback_socketpair(
    family: int = socket.AF_INET,
    type: int = socket.SOCK_STREAM,  # noqa: A002  # matches socket.socketpair
    proto: int = 0,
) -> tuple[socket.socket, socket.socket]:
    """Create the private loopback pair required by Windows asyncio loops.

    Returns:
        Connected loopback sockets without reopening general network access.
    """
    if family != socket.AF_INET or type != socket.SOCK_STREAM:
        msg = "Windows test socketpair only supports IPv4 stream sockets"
        raise ValueError(msg)
    listener = _REAL_SOCKET(family, type, proto)
    client: socket.socket | None = None
    server: socket.socket | None = None
    try:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        client = _REAL_SOCKET(family, type, proto)
        _REAL_CONNECT(client, listener.getsockname())
        raw_listener = cast("_RawAcceptSocket", listener)
        descriptor, _ = raw_listener._accept()
        server = _REAL_SOCKET(family, type, proto, fileno=descriptor)
    except BaseException:
        if client is not None:
            client.close()
        if server is not None:
            server.close()
        raise
    else:
        return client, server
    finally:
        listener.close()


if os.name == "nt":
    setattr(  # noqa: B010  # platform stub rejects the Windows-only replacement
        socket,
        "socketpair",
        _windows_loopback_socketpair,
    )


@pytest.hookimpl(trylast=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    """Permit Windows loopback IPC while preserving the global network block."""
    if os.name != "nt" or not item.config.getoption("disable_socket"):
        return
    if (
        item.config.getoption("force_enable_socket")
        or item.get_closest_marker("enable_socket")
        or "socket_enabled" in getattr(item, "fixturenames", ())
    ):
        return
    setattr(  # noqa: B010  # platform stub rejects the test-only replacement
        socket,
        "socket",
        _WindowsLoopbackSocket,
    )


def pytest_xdist_auto_num_workers(config: pytest.Config) -> int | None:
    """Cap Windows auto workers to avoid process and loopback-port saturation."""
    del config
    if os.name != "nt":
        return None
    return min(_WINDOWS_XDIST_MAX_WORKERS, os.cpu_count() or 1)


def _self_manages_update_check(request: pytest.FixtureRequest) -> bool:
    """Return whether the test owns app-startup update-check setup."""
    return request.node.get_closest_marker(_UPDATE_CHECK_SELF_MANAGED_MARK) is not None


@contextlib.contextmanager
def _isolated_model_cache_warmup(cache_root: Path) -> Generator[None, None, None]:
    """Warm model caches without consulting user configuration or credentials."""
    from deepagents_code import config, model_config

    config_dir = cache_root / ".deepagents"
    config_dir.mkdir(parents=True, exist_ok=True)
    path_patches = pytest.MonkeyPatch()
    path_patches.setattr(model_config, "DEFAULT_CONFIG_DIR", config_dir)
    path_patches.setattr(
        model_config, "DEFAULT_CONFIG_PATH", config_dir / "config.toml"
    )
    path_patches.setattr(model_config, "DEFAULT_STATE_DIR", config_dir / ".state")
    path_patches.setattr(config, "_GLOBAL_DOTENV_PATH", config_dir / ".env")

    discovery_var = "DEEPAGENTS_CODE_OLLAMA_DISCOVERY"
    original_discovery = os.environ.get(discovery_var)
    os.environ[discovery_var] = "0"
    model_config.clear_caches()
    try:
        with contextlib.suppress(Exception):
            model_config.get_available_models()
            model_config.get_model_profiles()
        _clear_path_dependent_model_caches()
        yield
    finally:
        try:
            model_config.clear_caches()
        finally:
            try:
                path_patches.undo()
            finally:
                try:
                    model_config.clear_caches()
                finally:
                    if original_discovery is None:
                        os.environ.pop(discovery_var, None)
                    else:
                        os.environ[discovery_var] = original_discovery


def _clear_path_dependent_model_caches() -> None:
    """Discard config-derived results while retaining warmed provider metadata."""
    from deepagents_code import model_config

    model_config._available_models_cache = None
    model_config._default_config_cache = None
    model_config._profiles_cache = None
    model_config._profiles_override_cache = None
    model_config.invalidate_thread_config_cache()


@pytest.fixture(autouse=True, scope="session")
def _warm_model_caches(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[None, None, None]:
    """Pre-populate model-config caches once per xdist worker.

    Tests like the model-selector UI tests call `get_available_models()` and
    `get_model_profiles()` during widget init. Without warm provider metadata,
    the first invocation in each worker process pays ~800-1200 ms of disk I/O
    via `importlib.util`. Config-derived result caches are discarded before
    each test so a patched config path is always read by that test.

    Keep Ollama discovery disabled so ordinary UI tests do not probe a local
    daemon. Tests that cover discovery delete the override before calling it.

    Tests that explicitly need a clean cache (e.g. `test_model_config.py`) use
    their own function-scoped `clear_caches()` fixture which overrides this.
    """
    with _isolated_model_cache_warmup(tmp_path_factory.mktemp("model-cache")):
        yield


@pytest.fixture(autouse=True)
def _reset_path_dependent_model_caches() -> Generator[None, None, None]:
    """Prevent a prior test's config path from affecting the next test."""
    _clear_path_dependent_model_caches()
    try:
        yield
    finally:
        _clear_path_dependent_model_caches()


@pytest.fixture
def windows_loopback_socket_type() -> type[socket.socket]:
    """Expose the guarded socket class to direct harness tests."""
    return _WindowsLoopbackSocket


@pytest.fixture
def isolated_model_cache_warmup() -> Callable[
    [Path],
    AbstractContextManager[None],
]:
    """Expose isolated cache warming to direct harness tests."""
    return _isolated_model_cache_warmup


@pytest.fixture(autouse=True)
def _restore_os_environ() -> Generator[None, None, None]:
    """Snapshot and restore `os.environ` around every test.

    Production code under test (`_ensure_bootstrap`, `_load_dotenv`,
    `_apply_default_langsmith_project`) writes to `os.environ` directly. When a
    test clears a variable with `monkeypatch.delenv(name, raising=False)` that
    was already absent, monkeypatch records no undo entry — so a later direct
    write by that code survives teardown and leaks into subsequent tests (e.g.
    a dotenv-reload test leaking `DEEPAGENTS_CODE_OPENAI_API_KEY` into a gateway
    key-mismatch test). Defined before the other autouse fixtures so it tears
    down last, leaving `os.environ` pristine no matter how a key was set.

    Restores by diffing against the snapshot rather than a blanket
    `clear()`/`update()`, so a test that never touches `os.environ` (the vast
    majority) triggers zero `putenv` calls on teardown.
    """
    snapshot = dict(os.environ)
    try:
        yield
    finally:
        for key in [key for key in os.environ if key not in snapshot]:
            del os.environ[key]
        for key, value in snapshot.items():
            if os.environ.get(key) != value:
                os.environ[key] = value


@pytest.fixture(autouse=True)
def _clear_langsmith_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent LangSmith env vars loaded from .env from leaking into tests.

    `deepagents_code.config` loads dotenv lazily on first `settings` access
    (via `_ensure_bootstrap()` / `_load_dotenv()`, which reads values with
    `dotenv.dotenv_values()`) and may inject `LANGSMITH_*` variables from a
    local `.env` file. Because those mutations persist in `os.environ`, they
    can be present before this fixture runs, causing spurious failures in unit
    tests that run with `--disable-socket` because the LangSmith client
    attempts real HTTP requests.

    Each test that *needs* LangSmith variables should set them explicitly via
    `monkeypatch.setenv` or `patch.dict("os.environ", ...)`.
    """
    for key in (
        "LANGSMITH_API_KEY",
        "LANGCHAIN_API_KEY",
        "LANGSMITH_TRACING",
        "LANGCHAIN_TRACING_V2",
        "LANGSMITH_ENDPOINT",
        "LANGCHAIN_ENDPOINT",
        "LANGSMITH_PROJECT",
        "DEEPAGENTS_CODE_LANGSMITH_PROJECT",
        "DEEPAGENTS_CODE_LANGSMITH_REDACT",
        "DEEPAGENTS_CODE_LANGSMITH_API_KEY",
        "DEEPAGENTS_CODE_LANGCHAIN_API_KEY",
        "DEEPAGENTS_CODE_LANGSMITH_TRACING",
        "DEEPAGENTS_CODE_LANGCHAIN_TRACING_V2",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def _disable_langsmith_batching(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent test-created LangSmith clients from starting ingestion threads."""
    from langsmith import Client

    original_init = cast("Callable[..., None]", Client.__init__)

    def _init(self: Client, *args: object, **kwargs: object) -> None:
        kwargs["auto_batch_tracing"] = False
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(Client, "__init__", _init)


@pytest.fixture(autouse=True)
def _clear_tavily_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent a Tavily key loaded from .env from leaking into tests.

    Like `LANGSMITH_*`, the lazy dotenv load on first `settings` access (see
    `_clear_langsmith_env`) may inject `TAVILY_API_KEY` from a developer's
    local `.env`.
    A leaked key flips `settings.has_tavily` to `True`, which silently changes
    onboarding behavior: the launch sequence short-circuits the Tavily step on
    a dev machine but runs it on CI, so a test that reaches the step passes
    locally yet hangs (real screen push) or writes a credential on CI.

    Each test that *needs* a Tavily key should set it explicitly via
    `monkeypatch.setenv` or patch `settings.has_tavily`.
    """
    for key in ("TAVILY_API_KEY", "DEEPAGENTS_CODE_TAVILY_API_KEY"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def _clear_project_mcp_trust_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent developer MCP trust decisions from changing unit-test behavior.

    These may already be present in `os.environ` before any fixture runs:
    `deepagents_code.config` loads dotenv lazily on first `settings` access
    (via `_ensure_bootstrap()` / `_load_dotenv()`) and injects them from the
    developer's global `~/.deepagents/.env`. The dangerous allowlist adds
    project-agnostic trust decisions, so leaving it set changes trust-list and
    selective-project-trust assertions.
    Removing them here (rather than relying on each test) keeps the MCP,
    model-config, and main suites hermetic. `_isolate_global_dotenv` below
    prevents a later dotenv reread (e.g. via `/reload`) from restoring them.
    """
    for key in (
        "DEEPAGENTS_CODE_DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS",
        "DEEPAGENTS_CODE_DISABLED_PROJECT_MCP_SERVERS",
        "DEEPAGENTS_CODE_ENABLED_PROJECT_MCP_SERVERS",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def _clear_debug_notifications_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent the debug-notifications override from changing suppression tests.

    `DEEPAGENTS_CODE_DEBUG_NOTIFICATIONS` may be loaded from the developer's
    global `~/.deepagents/.env` when `deepagents_code.config` loads dotenv
    lazily on first `settings` access, before fixtures run. When set, the
    notification suppression path skips persistence -- it removes the entry
    without calling `suppress_warning` -- so tests asserting that a suppression
    is persisted (or that a failed suppression keeps the row) break. Tests that
    exercise the debug path set it explicitly.
    """
    monkeypatch.delenv("DEEPAGENTS_CODE_DEBUG_NOTIFICATIONS", raising=False)


@pytest.fixture(autouse=True)
def _clear_provider_base_url_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent provider base-URL env vars from leaking into tests.

    A developer machine provisioned with the LangSmith gateway exports
    `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL`, which `get_base_url` now reads as
    a fallback. Clear them (and the `DEEPAGENTS_CODE_` overrides) so base-URL
    tests are deterministic. Tests that need a value set it explicitly.
    """
    for key in (
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_API_URL",
        "BASETEN_BASE_URL",
        "BASETEN_API_BASE",
        "GOOGLE_GEMINI_BASE_URL",
        "DEEPAGENTS_CODE_OPENAI_BASE_URL",
        "DEEPAGENTS_CODE_ANTHROPIC_BASE_URL",
        "DEEPAGENTS_CODE_BASETEN_BASE_URL",
        "DEEPAGENTS_CODE_BASETEN_API_BASE",
        "DEEPAGENTS_CODE_GOOGLE_GEMINI_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def _clear_onboarding_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent local debug onboarding env vars from affecting tests."""
    monkeypatch.delenv("DEEPAGENTS_CODE_DEBUG_ONBOARDING", raising=False)


@pytest.fixture(autouse=True)
def _clear_update_env(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent update debug/loop-guard and toggle env vars from affecting tests.

    `DEEPAGENTS_CODE_DEBUG_UPDATE` short-circuits the install path, and the
    internal `DEEPAGENTS_CODE_RESTARTED_AFTER_UPDATE` sentinel suppresses
    auto-update to break a restart loop. The sentinel can leak in not just from a
    developer shell but from a prior test exercising the production code that sets
    it, so it is cleared unconditionally. `DEEPAGENTS_CODE_AUTO_UPDATE` is read
    directly from the environment by `is_auto_update_enabled`, so a developer who
    exports it would otherwise make auto-update tests fail or pass spuriously.

    Most unit tests should not run the app startup PyPI update check at all: it
    performs DNS in a background worker, which pytest-socket reports under
    `--disable-socket` even when the app swallows the failure. Set the production
    opt-out env var by default so subprocess tests inherit the same no-network
    behavior. Tests marked `self_managed_update_check` cover the update-check
    gate directly, so they opt out of this default below.
    """
    monkeypatch.delenv("DEEPAGENTS_CODE_DEBUG_UPDATE", raising=False)
    monkeypatch.delenv("DEEPAGENTS_CODE_RESTARTED_AFTER_UPDATE", raising=False)
    monkeypatch.delenv("DEEPAGENTS_CODE_AUTO_UPDATE", raising=False)

    if _self_manages_update_check(request):
        monkeypatch.delenv("DEEPAGENTS_CODE_NO_UPDATE_CHECK", raising=False)
    else:
        monkeypatch.setenv("DEEPAGENTS_CODE_NO_UPDATE_CHECK", "1")


@pytest.fixture(autouse=True)
def _disable_app_startup_update_checks(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep app startup tests from racing PyPI or user update config.

    Tests marked `self_managed_update_check` manage the gate themselves, so leave
    `is_update_check_enabled` untouched for them.
    """
    if _self_manages_update_check(request):
        return
    monkeypatch.setattr(
        "deepagents_code.update_check.is_update_check_enabled",
        lambda: False,
    )


@pytest.fixture(autouse=True)
def _clear_behavior_override_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent developer behavior overrides from changing default-path tests."""
    for key in (
        "DEEPAGENTS_CODE_CURSOR_STYLE",
        "DEEPAGENTS_CODE_EXPERIMENTAL",
        "DEEPAGENTS_CODE_MEMORY_AUTO_SAVE",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def _clear_external_event_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent local alpha event-listener env vars from affecting tests."""
    monkeypatch.delenv("DEEPAGENTS_CODE_EXTERNAL_EVENT_SOCKET", raising=False)
    monkeypatch.delenv("DEEPAGENTS_CODE_EXTERNAL_EVENT_SOCKET_PATH", raising=False)


@pytest.fixture(autouse=True)
def _disable_terminal_escape(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop tests from leaking terminal control sequences to the real terminal.

    Production code constructs `DeepAgentsApp` and exercises the spinner / theme
    paths, which emit `OSC 11` (background color) and `OSC 9;4` (taskbar
    progress) via `terminal_escape.write_terminal_escape`. That writer targets
    `/dev/tty`, which pytest does not capture, so running the suite from inside
    a real terminal (e.g. an editable install) visibly recolors the developer's
    session. Opting out keeps the run inert. `test_terminal_escape.py` clears
    this var in its own fixture so its assertions still exercise the real path.
    """
    monkeypatch.setenv("DEEPAGENTS_CODE_NO_TERMINAL_ESCAPE", "1")


@pytest.fixture(autouse=True)
def _register_theme_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make app-specific CSS variables available to all test `App` instances.

    Production code defines these in `DeepAgentsApp.get_theme_variable_defaults`
    but many tests use lightweight `App[None]` subclasses that lack the override.
    Patching the base class ensures custom mode variables resolve everywhere
    without requiring each test app to opt in.
    """
    from textual.app import App

    from deepagents_code.theme import get_css_variable_defaults

    original = App.get_theme_variable_defaults
    custom = get_css_variable_defaults(dark=True)

    def _with_custom_vars(self: App) -> dict[str, str]:
        base = original(self)
        base.update(custom)
        return base

    monkeypatch.setattr(App, "get_theme_variable_defaults", _with_custom_vars)


@pytest.fixture(autouse=True)
def _provide_app_context() -> Generator[None]:
    """Set Textual's `active_app` context var for sync widget tests.

    Many unit tests construct widgets and call `compose()` directly without a
    running Textual app. Widget code that calls `self.app` (e.g., for
    theme-aware color lookups) needs a valid app in the context. This fixture
    provides a minimal `App` instance with the default LangChain theme
    registered so that `get_theme_colors()` returns the LC brand palette
    (matching `DARK_COLORS`).
    """
    from textual._context import active_app
    from textual.app import App
    from textual.theme import Theme

    from deepagents_code import theme

    app = App()
    c = theme.DARK_COLORS
    app.register_theme(
        Theme(
            name="langchain",
            primary=c.primary,
            secondary=c.secondary,
            accent=c.accent,
            foreground=c.foreground,
            background=c.background,
            surface=c.surface,
            panel=c.panel,
            warning=c.warning,
            error=c.error,
            success=c.success,
            dark=True,
        )
    )
    app.theme = "langchain"
    token = active_app.set(app)
    try:
        yield
    finally:
        active_app.reset(token)


@pytest.fixture(autouse=True)
def _isolate_global_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the global dotenv path at a nonexistent temp file.

    `deepagents_code.config._GLOBAL_DOTENV_PATH` defaults to the developer's
    real `~/.deepagents/.env`. Code paths like `/reload` call
    `_load_dotenv(refresh_loaded=True)`, which rereads that file and can restore
    ambient variables the isolation fixtures cleared (e.g.
    `DEEPAGENTS_CODE_EXPERIMENTAL` or the MCP trust vars). Redirecting it to a
    guaranteed-absent path under `tmp_path` makes dotenv rereads inert without
    touching production behavior. Tests that exercise global dotenv loading set
    this attribute explicitly after this fixture runs.

    This only stops a reread from *restoring* cleared vars; a var already
    loaded into `os.environ` before this fixture runs is removed only if it is
    on one of the `_clear_*` denylists above. New config vars that must not
    leak from the developer's environment need to be added there.
    """
    monkeypatch.setattr(
        "deepagents_code.config._GLOBAL_DOTENV_PATH",
        tmp_path / "nonexistent-global.env",
    )


@pytest.fixture(autouse=True)
def _isolate_state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect app-managed state away from the developer's real data."""
    state_dir = tmp_path / ".state"
    monkeypatch.setattr("deepagents_code.model_config.DEFAULT_STATE_DIR", state_dir)

    from deepagents_code import sessions

    monkeypatch.setattr(sessions, "_db_path", None)
    sessions._message_count_cache.clear()
    sessions._initial_prompt_cache.clear()
    sessions._recent_threads_cache.clear()


@pytest.fixture(autouse=True)
def _isolate_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect ChatInput history to a temp file.

    Without this, every test that mounts a `ChatInput` widget writes to the
    real `~/.deepagents/.state/history.jsonl`, causing duplicate/stale
    entries that persist across test runs and branch switches.
    """
    monkeypatch.setattr(
        "deepagents_code.tui.widgets.chat_input._default_history_path",
        lambda: tmp_path / "history.jsonl",
    )


@pytest.fixture(autouse=True)
def _clear_kitty_kbd_probe_cache() -> None:
    """Reset the `functools.cache` on the kitty-keyboard-protocol probe.

    The probe is cached for the lifetime of the process in production,
    but stale state leaks across tests that patch the probe function or
    rely on platform-specific behaviour. Clearing on every test keeps
    results deterministic regardless of file order or `pytest-xdist`
    sharding.
    """
    from deepagents_code.terminal_capabilities import supports_kitty_keyboard_protocol

    supports_kitty_keyboard_protocol.cache_clear()
