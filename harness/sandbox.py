"""bubblewrap sandboxes for agent runs.

Both sandboxes mount the filesystem read-only and hide the user's home
directory and /tmp. That hides the full database (data/benchmark.duckdb, which
holds the holdout period), earlier run logs and Claude Code transcripts. They
re-expose only the `bess` and `harness` packages, the project venv and its
Python interpreter.

- Policy sandbox: runs a submitted policy for holdout evaluation. No network,
  nothing writable except a private /tmp. It sees the submission snapshot
  read-only.
- Agent sandbox (Claude): wraps the Claude Code CLI the Agent SDK drives. The
  CLI needs the network to reach the API, so the network isn't unshared here.
  Bash commands the agent runs are additionally sandboxed by Claude Code itself
  with no network access (see run_agent.py), so it can't fetch the real August
  prices from AEMO.
- Agent bash sandbox (OpenRouter): each bash command the harness's own agent
  loop runs. No network; the only writable paths are the workspace and /tmp,
  which is a per-run directory kept for the whole session.

The policy and agent-bash sandboxes start from an empty environment, so API
keys and tokens in the harness's environment never reach agent code.
"""

import os
import shlex
from pathlib import Path

import claude_agent_sdk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HOME = Path.home()
UV_PYTHON = HOME / ".local" / "share" / "uv"  # interpreter the project venv links to
BUNDLED_CLI = Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude"
VENV_BIN = PROJECT_ROOT / ".venv" / "bin"


def _env() -> dict[str, str]:
    """The minimal environment sandboxed processes get (no API keys or tokens)."""
    return {"PATH": f"{VENV_BIN}:/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp", "LANG": "C.UTF-8",
            "PYTHONUNBUFFERED": "1", "VIRTUAL_ENV": str(PROJECT_ROOT / ".venv")}


def _clean_env() -> list[str]:
    """Drop the harness's environment (API keys, tokens) and set a minimal one."""
    args = ["--clearenv"]
    for k, v in _env().items():
        args += ["--setenv", k, v]
    return args


def _base_args() -> list[str]:
    args = [
        "bwrap",
        "--die-with-parent",
        "--ro-bind", "/", "/",
        "--dev", "/dev",
        "--proc", "/proc",
        "--tmpfs", "/tmp",
        "--tmpfs", str(HOME),
    ]
    for path in [PROJECT_ROOT / "bess", PROJECT_ROOT / "harness", PROJECT_ROOT / ".venv", UV_PYTHON]:
        args += ["--ro-bind", str(path), str(path)]
    return args


def policy_sandbox(snapshot_dir: Path) -> list[str]:
    """Command prefix that runs a process with no network, seeing the
    submission snapshot read-only and working in it."""
    snapshot_dir = Path(snapshot_dir).resolve()
    return _base_args() + _clean_env() + [
        "--unshare-all",
        "--new-session",
        "--ro-bind", str(snapshot_dir), str(snapshot_dir),
        "--chdir", str(snapshot_dir),
    ]


def agent_bash_sandbox(workspace: Path, tmp_dir: Path) -> list[str]:
    """Command prefix for one bash command of the OpenRouter agent: no network,
    the workspace writable and the working directory, everything else as the
    policy sandbox sees it. `tmp_dir` (a per-run directory) is mounted as /tmp,
    so /tmp persists between commands as it does in Claude Code's sandbox."""
    workspace, tmp_dir = Path(workspace).resolve(), Path(tmp_dir).resolve()
    return _base_args() + _clean_env() + [
        "--unshare-all",
        "--new-session",
        "--bind", str(tmp_dir), "/tmp",
        "--bind", str(workspace), str(workspace),
        "--chdir", str(workspace),
    ]


class PersistentSandbox:
    """One agent-bash sandbox kept alive for a whole OpenRouter run, so that,
    as with Claude Code's Bash, a background process an agent starts keeps
    running after the command that started it returns. A holder process
    (`sleep infinity` under agent_bash_sandbox) owns the sandbox's namespaces;
    each command is entered into them with nsenter, with the same minimal
    environment. The holder dies with the harness (bwrap --die-with-parent)."""

    def __init__(self, workspace: Path, tmp_dir: Path):
        self.workspace, self.tmp_dir = Path(workspace).resolve(), Path(tmp_dir).resolve()
        self._holder = None
        self._pid = None

    def _start(self) -> None:
        import subprocess
        import time

        self._holder = subprocess.Popen(agent_bash_sandbox(self.workspace, self.tmp_dir) + ["sleep", "infinity"],
                                        start_new_session=True, stdin=subprocess.DEVNULL,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):  # bwrap forks the sandboxed process; its pid is the one to enter
            kids = Path(f"/proc/{self._holder.pid}/task/{self._holder.pid}/children")
            pids = kids.read_text().split() if kids.exists() else []
            if pids:
                self._pid = pids[-1]
                return
            time.sleep(0.05)
        raise RuntimeError("the agent sandbox didn't start")

    def command(self, command: str) -> list[str]:
        """argv that runs `command` with bash inside the sandbox."""
        if self._holder is None or self._holder.poll() is not None:
            self._start()
        env = [f"{k}={v}" for k, v in _env().items()]
        return ["nsenter", "-t", self._pid, "-U", "-m", "-n", "-p", "-i", "-u", "--preserve-credentials",
                f"--wd={self.workspace}", "env", "-i", *env, "bash", "-c", command]

    def close(self) -> None:
        import signal

        if self._holder is not None and self._holder.poll() is None:
            try:
                os.killpg(self._holder.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self._holder = None


def write_agent_cli_wrapper(path: Path, workspace: Path, config_dir: Path) -> Path:
    """Write an executable that runs the Claude Code CLI inside the agent
    sandbox; pass it to ClaudeAgentOptions(cli_path=...). The workspace is the
    only writable project path. Claude Code keeps its config, session files and
    transcripts in `config_dir` (a per-run directory, never the user's own
    ~/.claude), so the agent can't read other sessions."""
    workspace, config_dir = Path(workspace).resolve(), Path(config_dir).resolve()
    args = _base_args() + [
        "--ro-bind", str(BUNDLED_CLI), str(BUNDLED_CLI),
        "--bind", str(workspace), str(workspace),
        "--bind", str(config_dir), str(config_dir),
        "--setenv", "CLAUDE_CONFIG_DIR", str(config_dir),
        "--chdir", str(workspace),
        str(BUNDLED_CLI),
    ]
    path = Path(path)
    path.write_text("#!/bin/sh\nexec " + " ".join(shlex.quote(a) for a in args) + ' "$@"\n')
    path.chmod(0o755)
    return path
