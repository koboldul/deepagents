"""Direct tests for the unit-test harness."""

from __future__ import annotations

import asyncio
import io
import os
import socket
from typing import TYPE_CHECKING, Protocol, cast

import pytest
from pytest_socket import SocketBlockedError

from deepagents_code import auth_store, config, model_config

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager
    from pathlib import Path


class _SendMsg(Protocol):
    """Callable shape needed by the optional `socket.sendmsg` test."""

    def __call__(
        self,
        buffers: list[bytes],
        ancdata: list[tuple[int, int, bytes]],
        flags: int,
        address: tuple[str, int],
        /,
    ) -> int: ...


class _SendMsgAfalg(Protocol):
    """Callable shape needed by the optional `socket.sendmsg_afalg` test."""

    def __call__(self, buffers: list[bytes], /) -> int: ...


_LOOPBACK_CASES = (
    (socket.AF_INET, "127.0.0.1"),
    (socket.AF_INET6, "::1"),
)
_EXTERNAL_CASES = (
    (socket.AF_INET, "192.0.2.1"),
    (socket.AF_INET6, "2001:db8::1"),
)


def _bind_loopback_or_skip(
    socket_instance: socket.socket,
    family: socket.AddressFamily,
    host: str,
) -> None:
    """Bind loopback, skipping only when the host lacks IPv6 support."""
    try:
        socket_instance.bind((host, 0))
    except OSError as exc:
        if family == socket.AF_INET6:
            pytest.skip(f"IPv6 loopback unavailable: {exc}")
        raise


@pytest.mark.parametrize(("family", "host"), _LOOPBACK_CASES)
def test_windows_loopback_socket_allows_tcp_loopback(
    windows_loopback_socket_type: type[socket.socket],
    family: socket.AddressFamily,
    host: str,
) -> None:
    """TCP loopback remains usable for IPv4 and available IPv6 stacks."""
    with windows_loopback_socket_type(family, socket.SOCK_STREAM) as listener:
        listener.settimeout(2)
        _bind_loopback_or_skip(listener, family, host)
        listener.listen(1)
        with windows_loopback_socket_type(family, socket.SOCK_STREAM) as client:
            client.settimeout(2)
            client.connect(listener.getsockname())
            server, _ = listener.accept()
            with server:
                server.settimeout(2)
                client.sendall(b"tcp-loopback")
                assert server.recv(64) == b"tcp-loopback"


@pytest.mark.parametrize(("family", "host"), _LOOPBACK_CASES)
def test_windows_loopback_socket_allows_udp_loopback(
    windows_loopback_socket_type: type[socket.socket],
    family: socket.AddressFamily,
    host: str,
) -> None:
    """UDP loopback remains usable for IPv4 and available IPv6 stacks."""
    with (
        windows_loopback_socket_type(family, socket.SOCK_DGRAM) as receiver,
        windows_loopback_socket_type(family, socket.SOCK_DGRAM) as sender,
    ):
        receiver.settimeout(2)
        _bind_loopback_or_skip(receiver, family, host)
        assert sender.sendto(b"udp-loopback", receiver.getsockname()) == 12
        payload, _ = receiver.recvfrom(64)
        assert payload == b"udp-loopback"


@pytest.mark.parametrize(("family", "host"), _EXTERNAL_CASES)
@pytest.mark.parametrize("kind", [socket.SOCK_STREAM, socket.SOCK_DGRAM])
def test_windows_loopback_socket_blocks_external_connects(
    windows_loopback_socket_type: type[socket.socket],
    family: socket.AddressFamily,
    host: str,
    kind: socket.SocketKind,
) -> None:
    """TCP and connected UDP cannot target arbitrary external addresses."""
    with windows_loopback_socket_type(family, kind) as client:
        with pytest.raises(SocketBlockedError):
            client.connect((host, 9))
        with pytest.raises(SocketBlockedError):
            client.connect_ex((host, 9))


@pytest.mark.parametrize(("family", "host"), _EXTERNAL_CASES)
def test_windows_loopback_socket_blocks_external_udp_sendto(
    windows_loopback_socket_type: type[socket.socket],
    family: socket.AddressFamily,
    host: str,
) -> None:
    """Both `sendto` overloads reject egress before the OS call."""
    with windows_loopback_socket_type(family, socket.SOCK_DGRAM) as sender:
        with pytest.raises(SocketBlockedError):
            sender.sendto(b"blocked", (host, 9))
        with pytest.raises(SocketBlockedError):
            sender.sendto(b"blocked", 0, (host, 9))


def test_windows_loopback_socket_guards_destinationless_send_apis(
    windows_loopback_socket_type: type[socket.socket],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Connected send APIs validate their peer before writing."""

    def external_peer(_socket_instance: socket.socket) -> tuple[str, int]:
        return ("192.0.2.1", 9)

    monkeypatch.setattr(windows_loopback_socket_type, "getpeername", external_peer)
    with windows_loopback_socket_type(socket.AF_INET, socket.SOCK_STREAM) as sender:
        with pytest.raises(SocketBlockedError):
            sender.send(b"blocked")
        with pytest.raises(SocketBlockedError):
            sender.sendall(b"blocked")
        with pytest.raises(SocketBlockedError), io.BytesIO(b"blocked") as payload:
            sender.sendfile(payload)


def test_windows_loopback_socket_guards_sendmsg_when_available(
    windows_loopback_socket_type: type[socket.socket],
) -> None:
    """`sendmsg` cannot bypass the destination guard on supporting platforms."""
    if not hasattr(windows_loopback_socket_type, "sendmsg"):
        pytest.skip("socket.sendmsg is unavailable")
    with windows_loopback_socket_type(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        sendmsg = cast("_SendMsg", sender.sendmsg)
        with pytest.raises(SocketBlockedError):
            sendmsg([b"blocked"], [], 0, ("192.0.2.1", 9))


def test_windows_loopback_socket_guards_sendmsg_afalg_when_available(
    windows_loopback_socket_type: type[socket.socket],
) -> None:
    """`sendmsg_afalg` is blocked because AF_ALG is not loopback IPC."""
    if not hasattr(windows_loopback_socket_type, "sendmsg_afalg"):
        pytest.skip("socket.sendmsg_afalg is unavailable")
    with windows_loopback_socket_type(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        sendmsg_afalg = cast("_SendMsgAfalg", sender.sendmsg_afalg)
        with pytest.raises(SocketBlockedError):
            sendmsg_afalg([b"blocked"])


@pytest.mark.skipif(os.name != "nt", reason="Windows-only harness integration")
def test_windows_loopback_hook_preserves_constructor_and_socketpair(
    pytestconfig: pytest.Config,
    windows_loopback_socket_type: type[socket.socket],
) -> None:
    """The installed guard permits construction and private socketpair IPC."""
    if not pytestconfig.getoption("disable_socket"):
        pytest.skip("requires --disable-socket")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as socket_instance:
        assert isinstance(socket_instance, windows_loopback_socket_type)
    left, right = socket.socketpair()
    with left, right:
        left.settimeout(2)
        right.settimeout(2)
        left.sendall(b"socketpair")
        assert right.recv(64) == b"socketpair"


@pytest.mark.skipif(os.name != "nt", reason="Windows-only event-loop transport")
def test_windows_loopback_hook_preserves_event_loop(
    pytestconfig: pytest.Config,
) -> None:
    """The Windows event loop can create and use its private wakeup socketpair."""
    if not pytestconfig.getoption("disable_socket"):
        pytest.skip("requires --disable-socket")
    loop = asyncio.new_event_loop()
    try:
        assert loop.run_until_complete(asyncio.sleep(0, result="ready")) == "ready"
    finally:
        loop.close()


def _patch_simulated_user_paths(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
) -> Path:
    """Point every user-level config path at a simulated real home."""
    config_dir = root / ".deepagents"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "config.toml"
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_DIR", config_dir)
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.setattr(model_config, "DEFAULT_STATE_DIR", config_dir / ".state")
    monkeypatch.setattr(config, "_GLOBAL_DOTENV_PATH", config_dir / ".env")
    return config_path


def test_model_cache_warmup_never_reads_simulated_real_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_model_cache_warmup: Callable[[Path], AbstractContextManager[None]],
) -> None:
    """Warmup uses isolated config/auth/global paths and restores them cleared."""
    real_config = _patch_simulated_user_paths(monkeypatch, tmp_path / "real-home")
    real_config.write_text(
        '[models.providers.real_sentinel]\nmodels = ["must-not-be-cached"]\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(model_config, "_get_provider_profile_modules", list)
    model_config.clear_caches()

    warm_root = tmp_path / "warm"
    with isolated_model_cache_warmup(warm_root):
        isolated_dir = warm_root / ".deepagents"
        assert isolated_dir / "config.toml" == model_config.DEFAULT_CONFIG_PATH
        assert auth_store.auth_path() == isolated_dir / ".state" / "auth.json"
        assert isolated_dir / ".env" == config._GLOBAL_DOTENV_PATH
        assert "real_sentinel" not in model_config.get_available_models()

    assert real_config == model_config.DEFAULT_CONFIG_PATH
    try:
        assert "real_sentinel" in model_config.ModelConfig.load().providers
    finally:
        model_config.clear_caches()


def test_model_cache_warmup_allows_isolated_test_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_model_cache_warmup: Callable[[Path], AbstractContextManager[None]],
) -> None:
    """A test path replaces the prewarmed empty config after cache clearing."""
    _patch_simulated_user_paths(monkeypatch, tmp_path / "real-home")
    monkeypatch.setattr(model_config, "_get_provider_profile_modules", list)
    own_config = tmp_path / "test-config.toml"
    own_config.write_text(
        '[models.providers.isolated]\nmodels = ["owned-by-test"]\n',
        encoding="utf-8",
    )

    with (
        isolated_model_cache_warmup(tmp_path / "warm"),
        pytest.MonkeyPatch.context() as config_patch,
    ):
        config_patch.setattr(model_config, "DEFAULT_CONFIG_PATH", own_config)
        assert model_config.get_available_models()["isolated"] == ["owned-by-test"]
