"""bubblewrap sandboxes for agent runs.

Both sandboxes mount the filesystem read-only and hide the user's home
directory and /tmp. That hides the full database (data/benchmark.duckdb, which
holds the holdout period), earlier run logs and Claude Code transcripts. They
re-expose only the `bess` and `harness` packages, the project venv and its
Python interpreter.

- Policy sandbox: runs a submitted policy for holdout evaluation. No network,
  nothing writable except a private /tmp. It sees the submission snapshot
  read-only.
- Agent sandbox: wraps the Claude Code CLI the Agent SDK drives. The CLI needs
  the network to reach the API, so the network isn't unshared here. Bash
  commands the agent runs are additionally sandboxed by Claude Code itself with
  no network access (see run_agent.py), so it can't fetch the real August
  prices from AEMO.
"""

import shlex
from pathlib import Path

import claude_agent_sdk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HOME = Path.home()
UV_PYTHON = HOME / ".local" / "share" / "uv"  # interpreter the project venv links to
BUNDLED_CLI = Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude"


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
    return _base_args() + [
        "--unshare-all",
        "--new-session",
        "--ro-bind", str(snapshot_dir), str(snapshot_dir),
        "--chdir", str(snapshot_dir),
    ]


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
