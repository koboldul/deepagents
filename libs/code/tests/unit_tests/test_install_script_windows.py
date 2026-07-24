"""Tests for the native PowerShell install script."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).parents[2]
REPO_ROOT = Path(__file__).parents[4]
SCRIPT = CODE_ROOT / "scripts" / "install.ps1"


def _powershell_runtimes() -> tuple[str, ...]:
    """Return each available native PowerShell runtime once."""
    if sys.platform != "win32":
        return ()
    runtimes: list[str] = []
    seen: set[str] = set()
    for name in ("powershell.exe", "pwsh.exe", "pwsh"):
        executable = shutil.which(name)
        if executable is None:
            continue
        key = os.path.normcase(str(Path(executable).resolve()))
        if key not in seen:
            seen.add(key)
            runtimes.append(executable)
    return tuple(runtimes)


@pytest.fixture(
    params=_powershell_runtimes(),
    ids=lambda runtime: Path(runtime).stem,
)
def powershell(request: pytest.FixtureRequest) -> str:
    """Provide each installed Windows PowerShell implementation."""
    return str(request.param)


def _run_installer(
    powershell: str, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    """Run the native installer under one PowerShell implementation."""
    return subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


def _run_installer_function(
    powershell: str,
    function_name: str,
    body: str,
    *,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    """Load one installer function without executing the installer body."""
    script_path = str(SCRIPT).replace("'", "''")
    command = f"""
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile(
    '{script_path}',
    [ref]$tokens,
    [ref]$errors
)
if ($errors.Count -ne 0) {{
    throw ($errors | ForEach-Object {{ $_.Message }})
}}
$definition = $ast.Find(
    {{
        param($node)
        $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
            $node.Name -eq '{function_name}'
    }},
    $true
)
if ($null -eq $definition) {{
    throw 'Installer function {function_name} was not found.'
}}
. ([ScriptBlock]::Create($definition.Extent.Text))
{body}
"""
    return subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


def _write_legacy_uv(uv_path: Path) -> None:
    uv_path.write_text(
        """@echo off
if "%1 %2 %3"=="tool dir --bin" (
  echo error: unexpected argument '--bin' 1>&2
  exit /b 2
)
if "%1 %2"=="tool update-shell" exit /b 0
if "%1 %2"=="tool install" exit /b 0
echo unexpected uv arguments: %* 1>&2
exit /b 1
""",
        encoding="utf-8",
    )


def _write_dcode(dcode_path: Path, version: str) -> None:
    dcode_path.write_text(
        f"""@echo off
if "%1"=="-v" (
  echo deepagents-code {version}
  exit /b 0
)
exit /b 1
""",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("document", "checkout_command"),
    [
        (
            REPO_ROOT / "README.md",
            r"powershell -ExecutionPolicy Bypass -File .\libs\code\scripts\install.ps1",
        ),
        (
            CODE_ROOT / "README.md",
            r"powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1",
        ),
        (
            CODE_ROOT / "DEVELOPMENT.md",
            r"powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1",
        ),
    ],
)
def test_windows_docs_expose_dcode_in_current_shell(
    document: Path,
    checkout_command: str,
) -> None:
    """Every Windows install path makes plain `dcode` immediately available."""
    contents = document.read_text(encoding="utf-8")
    block_start = contents.index("winget install --id astral-sh.uv -e")
    install_block = contents[block_start : contents.index("```", block_start)]
    commands = (
        "winget install --id astral-sh.uv -e",
        '$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine")',
        "uv tool install -U --prerelease allow deepagents-code",
        "uv tool update-shell",
        "$uvToolBin = (uv tool dir --bin).Trim()",
        '$env:Path = "$uvToolBin;$env:Path"',
        "dcode",
    )
    positions = [install_block.index(command) for command in commands]

    assert positions == sorted(positions)
    assert checkout_command in contents


def test_powershell_installer_has_required_guardrails() -> None:
    """The checked-in installer keeps its required native Windows behavior."""
    contents = SCRIPT.read_text(encoding="utf-8")

    forbidden_remote_execution = (
        r"https://astral\.sh/uv/install\.ps1",
        r"\b(?:irm|iwr|iex)\b",
        r"\bInvoke-(?:Expression|RestMethod|WebRequest)\b",
        r"\[ScriptBlock\]\s*::\s*Create",
        r"\bStart-BitsTransfer\b",
        r"\b(?:DownloadFile|DownloadString)\b",
        r"\b(?:HttpClient|WebClient)\b",
        r"\bSystem\.Net\.WebRequest\b",
        r"\b(?:curl|wget)(?:\.exe)?\b",
    )
    for pattern in forbidden_remote_execution:
        assert re.search(pattern, contents, re.IGNORECASE) is None, pattern

    assert "winget install --id astral-sh.uv -e" in contents
    assert "https://docs.astral.sh/uv/getting-started/installation/" in contents
    assert "DEEPAGENTS_CODE_EXTRAS" in contents
    assert "DEEPAGENTS_CODE_PRERELEASE" in contents
    assert "tool install -U" in contents
    assert "tool dir --bin" in contents
    assert "UV_TOOL_BIN_DIR" in contents
    assert "GetFullPath" in contents
    assert "tool update-shell" in contents
    assert "$previousErrorActionPreference" in contents
    assert "PSNativeCommandUseErrorActionPreference" in contents
    assert "} finally {" in contents
    assert 'Get-Command "uv" `' in contents
    assert "-CommandType Application" in contents
    assert 'Get-Command "dcode"' in contents
    assert "[StringComparison]::OrdinalIgnoreCase" in contents
    assert "shadowed on PATH" in contents
    assert "dcode -v" in contents


@pytest.mark.skipif(sys.platform != "win32", reason="requires native Windows")
@pytest.mark.parametrize(
    "shadow",
    [
        'function uv { "function shadow" }',
        "Set-Alias -Name uv -Value Write-Output",
    ],
    ids=["function", "alias"],
)
def test_find_uv_ignores_non_application_command_shadows(
    tmp_path: Path,
    powershell: str,
    shadow: str,
) -> None:
    """A function or alias cannot hide or impersonate a real `uv.exe`."""
    source_uv = shutil.which("uv.exe")
    if source_uv is None:
        pytest.skip("a real uv.exe is required")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    expected_uv = bin_dir / "uv.exe"
    shutil.copy2(source_uv, expected_uv)

    env = os.environ.copy()
    env["PATH"] = str(bin_dir)
    result = _run_installer_function(
        powershell,
        "Find-Uv",
        f"""
{shadow}
[Console]::Out.Write((Find-Uv))
""",
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert os.path.normcase(result.stdout.strip()) == os.path.normcase(str(expected_uv))


@pytest.mark.skipif(sys.platform != "win32", reason="requires native Windows")
def test_powershell_installer_fails_cleanly_when_uv_is_missing(
    tmp_path: Path,
    powershell: str,
) -> None:
    """A missing uv reports trusted choices without invoking winget."""
    bin_dir = tmp_path / "bin"
    home = tmp_path / "home"
    local_app_data = tmp_path / "local-app-data"
    winget_sentinel = tmp_path / "winget-invoked.txt"
    bin_dir.mkdir()
    home.mkdir()
    local_app_data.mkdir()

    (bin_dir / "winget.cmd").write_text(
        """@echo off
echo invoked>"%FAKE_WINGET_SENTINEL%"
exit /b 0
""",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PATH"] = str(bin_dir)
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
    env["LOCALAPPDATA"] = str(local_app_data)
    env["FAKE_WINGET_SENTINEL"] = str(winget_sentinel)

    result = _run_installer(powershell, env)

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "uv is required but was not found." in output
    assert "winget install --id astral-sh.uv -e" in output
    assert "https://docs.astral.sh/uv/getting-started/installation/" in output
    assert not winget_sentinel.exists()


@pytest.mark.skipif(sys.platform != "win32", reason="requires native Windows")
def test_powershell_installer_uses_existing_uv_with_extras_and_prerelease(
    tmp_path: Path,
    powershell: str,
) -> None:
    """An existing uv installs successfully with validated environment choices."""
    bin_dir = tmp_path / "bin"
    tool_bin = tmp_path / "tool-bin"
    args_file = tmp_path / "uv-args.txt"
    bin_dir.mkdir()
    tool_bin.mkdir()

    (bin_dir / "uv.cmd").write_text(
        """@echo off
if "%1 %2 %3"=="tool dir --bin" (
  echo %FAKE_UV_TOOL_BIN%
  exit /b 0
)
if "%1 %2"=="tool update-shell" exit /b 0
if "%1 %2"=="tool install" (
  echo %*>"%FAKE_UV_ARGS_FILE%"
  exit /b 0
)
echo unexpected uv arguments: %* 1>&2
exit /b 1
""",
        encoding="utf-8",
    )
    (tool_bin / "dcode.cmd").write_text(
        """@echo off
if "%1"=="-v" (
  echo deepagents-code 9.9.9
  exit /b 0
)
exit /b 1
""",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["FAKE_UV_TOOL_BIN"] = str(tool_bin)
    env["FAKE_UV_ARGS_FILE"] = str(args_file)
    env["DEEPAGENTS_CODE_EXTRAS"] = "nvidia,ollama"
    env["DEEPAGENTS_CODE_PRERELEASE"] = "if-necessary"

    result = _run_installer(powershell, env)

    assert result.returncode == 0, result.stderr
    arguments = args_file.read_text(encoding="utf-8")
    assert "tool install -U" in arguments
    assert "--prerelease if-necessary" in arguments
    assert "deepagents-code[nvidia,ollama]" in arguments
    assert "deepagents-code 9.9.9" in result.stdout


@pytest.mark.skipif(sys.platform != "win32", reason="requires native Windows")
def test_powershell_installer_moves_existing_tool_bin_before_old_dcode(
    tmp_path: Path,
    powershell: str,
) -> None:
    """An existing tool-bin PATH entry is moved ahead of an old dcode."""
    uv_bin = tmp_path / "uv-bin"
    old_bin = tmp_path / "old-bin"
    tool_bin = tmp_path / "tool-bin"
    uv_bin.mkdir()
    old_bin.mkdir()
    tool_bin.mkdir()

    (uv_bin / "uv.cmd").write_text(
        """@echo off
if "%1 %2 %3"=="tool dir --bin" (
  echo %FAKE_UV_TOOL_BIN%
  exit /b 0
)
if "%1 %2"=="tool update-shell" exit /b 0
if "%1 %2"=="tool install" exit /b 0
echo unexpected uv arguments: %* 1>&2
exit /b 1
""",
        encoding="utf-8",
    )
    (old_bin / "dcode.cmd").write_text(
        """@echo off
if "%1"=="-v" (
  echo deepagents-code 0.0.1
  exit /b 0
)
exit /b 1
""",
        encoding="utf-8",
    )
    (tool_bin / "dcode.cmd").write_text(
        """@echo off
if "%1"=="-v" (
  echo deepagents-code 9.9.9
  exit /b 0
)
exit /b 1
""",
        encoding="utf-8",
    )

    env = os.environ.copy()
    existing_tool_bin = str(tool_bin).swapcase()
    env["PATH"] = os.pathsep.join(
        [str(uv_bin), str(old_bin), existing_tool_bin, env["PATH"]]
    )
    env["FAKE_UV_TOOL_BIN"] = str(tool_bin)

    result = _run_installer(powershell, env)

    assert result.returncode == 0, result.stderr
    assert "deepagents-code 9.9.9" in result.stdout
    assert "deepagents-code 0.0.1" not in result.stdout


@pytest.mark.skipif(sys.platform != "win32", reason="requires native Windows")
@pytest.mark.parametrize(
    ("selected", "version"),
    [
        ("UV_TOOL_BIN_DIR", "9.9.1"),
        ("XDG_BIN_HOME", "9.9.2"),
        ("XDG_DATA_HOME", "9.9.3"),
        ("HOME", "9.9.4"),
    ],
)
def test_powershell_installer_legacy_tool_bin_precedence(
    tmp_path: Path,
    powershell: str,
    selected: str,
    version: str,
) -> None:
    """Legacy uv uses the first valid normalized executable directory."""
    uv_bin = tmp_path / "uv-bin"
    old_bin = tmp_path / "old-bin"
    uv_tool_bin = tmp_path / "uv-tool-bin"
    xdg_bin = tmp_path / "xdg-bin"
    xdg_root = tmp_path / "xdg-root"
    xdg_data_home = xdg_root / "share"
    xdg_data_bin = xdg_root / "bin"
    home = tmp_path / "home"
    home_bin = home / ".local" / "bin"
    for directory in (
        uv_bin,
        old_bin,
        uv_tool_bin,
        xdg_bin,
        xdg_data_home,
        xdg_data_bin,
        home_bin,
    ):
        directory.mkdir(parents=True)

    _write_legacy_uv(uv_bin / "uv.cmd")
    _write_dcode(old_bin / "dcode.cmd", "0.0.1")
    candidates = {
        "UV_TOOL_BIN_DIR": (uv_tool_bin, "9.9.1"),
        "XDG_BIN_HOME": (xdg_bin, "9.9.2"),
        "XDG_DATA_HOME": (xdg_data_bin, "9.9.3"),
        "HOME": (home_bin, "9.9.4"),
    }
    for candidate, candidate_version in candidates.values():
        _write_dcode(candidate / "dcode.cmd", candidate_version)

    invalid = tmp_path / "not-a-directory"
    invalid.write_text("invalid", encoding="utf-8")
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join([str(uv_bin), str(old_bin), env["PATH"]])
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
    env["UV_TOOL_BIN_DIR"] = str(
        uv_tool_bin / "unused" / ".." if selected == "UV_TOOL_BIN_DIR" else invalid
    )
    env["XDG_BIN_HOME"] = str(
        xdg_bin / "unused" / ".." if selected == "XDG_BIN_HOME" else invalid
    )
    env["XDG_DATA_HOME"] = str(
        xdg_data_home / "unused" / ".." if selected == "XDG_DATA_HOME" else invalid
    )
    if selected == "HOME":
        env["UV_TOOL_BIN_DIR"] = str(invalid)
        env["XDG_BIN_HOME"] = str(invalid)
        env["XDG_DATA_HOME"] = str(tmp_path / "missing-data" / "share")

    result = _run_installer(powershell, env)

    assert result.returncode == 0, result.stderr
    assert f"deepagents-code {version}" in result.stdout
    assert "deepagents-code 0.0.1" not in result.stdout


@pytest.mark.skipif(sys.platform != "win32", reason="requires native Windows")
def test_powershell_installer_falls_back_when_tool_dir_bin_is_unsupported(
    tmp_path: Path,
    powershell: str,
) -> None:
    """Older uv can skip `tool dir --bin`; the installer still succeeds."""
    uv_bin = tmp_path / "uv-bin"
    old_bin = tmp_path / "old-bin"
    tool_bin = tmp_path / "tool-bin"
    home = tmp_path / "home"
    uv_bin.mkdir()
    old_bin.mkdir()
    tool_bin.mkdir()
    (home / ".local").mkdir(parents=True)

    (uv_bin / "uv.cmd").write_text(
        """@echo off
if "%1 %2 %3"=="tool dir --bin" (
  echo error: unexpected argument '--bin' 1>&2
  exit /b 2
)
if "%1 %2"=="tool update-shell" exit /b 0
if "%1 %2"=="tool install" exit /b 0
echo unexpected uv arguments: %* 1>&2
exit /b 1
""",
        encoding="utf-8",
    )
    (old_bin / "dcode.cmd").write_text(
        """@echo off
if "%1"=="-v" (
  echo deepagents-code 0.0.1
  exit /b 0
)
exit /b 1
""",
        encoding="utf-8",
    )
    (tool_bin / "dcode.cmd").write_text(
        """@echo off
if "%1"=="-v" (
  echo deepagents-code 9.9.9
  exit /b 0
)
exit /b 1
""",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PATH"] = os.pathsep.join([str(uv_bin), str(old_bin), env["PATH"]])
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
    env["UV_TOOL_BIN_DIR"] = str(tool_bin / "nested" / "..")

    result = _run_installer(powershell, env)

    assert result.returncode == 0, result.stderr
    assert "deepagents-code 9.9.9" in result.stdout
    assert "deepagents-code 0.0.1" not in result.stdout


@pytest.mark.skipif(sys.platform != "win32", reason="requires native Windows")
def test_powershell_installer_uses_home_local_bin_when_tool_dir_bin_is_unsupported(
    tmp_path: Path,
    powershell: str,
) -> None:
    """Legacy uv falls back to `~/.local/bin` when it cannot report a bin dir."""
    uv_bin = tmp_path / "uv-bin"
    old_bin = tmp_path / "old-bin"
    home = tmp_path / "home"
    uv_bin.mkdir()
    old_bin.mkdir()
    (home / ".local" / "bin").mkdir(parents=True)

    (uv_bin / "uv.cmd").write_text(
        """@echo off
if "%1 %2 %3"=="tool dir --bin" (
  echo error: unexpected argument '--bin' 1>&2
  exit /b 2
)
if "%1 %2"=="tool update-shell" exit /b 0
if "%1 %2"=="tool install" exit /b 0
echo unexpected uv arguments: %* 1>&2
exit /b 1
""",
        encoding="utf-8",
    )
    (old_bin / "dcode.cmd").write_text(
        """@echo off
if "%1"=="-v" (
  echo deepagents-code 0.0.1
  exit /b 0
)
exit /b 1
""",
        encoding="utf-8",
    )
    (home / ".local" / "bin" / "dcode.cmd").write_text(
        """@echo off
if "%1"=="-v" (
  echo deepagents-code 9.9.9
  exit /b 0
)
exit /b 1
""",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PATH"] = os.pathsep.join([str(uv_bin), str(old_bin), env["PATH"]])
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
    env.pop("UV_TOOL_BIN_DIR", None)

    result = _run_installer(powershell, env)

    assert result.returncode == 0, result.stderr
    assert "deepagents-code 9.9.9" in result.stdout
    assert "deepagents-code 0.0.1" not in result.stdout
