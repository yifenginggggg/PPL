"""Synchronous command backends using only the Python standard library.

Subprocess output and arguments are deliberately omitted from diagnostics.
Timeouts kill and reap the child through subprocess.run; cancellation such as
KeyboardInterrupt propagates after subprocess.run has stopped the child.
"""

from __future__ import annotations

from dataclasses import dataclass
import errno
import json
import math
import subprocess
import tempfile
from typing import Protocol


__all__ = [
    "ModelReply", "Backend", "BackendError", "CommandBackend", "ClaudeCLIBackend",
]


@dataclass
class ModelReply:
    text: str
    metadata: dict


class Backend(Protocol):
    def generate(
        self, *, system_prompt: str, payload: dict, model_id: str
    ) -> ModelReply:
        ...


class BackendError(RuntimeError):
    """A backend failed; its message never includes raw process diagnostics."""


def _timeout(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise BackendError("Backend timeout must be a positive finite number.")
    return value


def _json_input(value: dict) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise BackendError("Backend input cannot be encoded as UTF-8 JSON.") from None


def _run_command(
    argv: list[str], request: bytes, timeout: float, *, cwd: str | None = None
) -> str:
    try:
        completed = subprocess.run(
            argv,
            input=request,
            capture_output=True,
            text=False,
            shell=False,
            check=False,
            timeout=timeout,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired:
        raise BackendError("Backend process timed out.") from None
    except OSError as exc:
        code = errno.errorcode.get(exc.errno, "OSERROR")
        raise BackendError(f"Backend process could not run ({code}).") from None
    except (subprocess.SubprocessError, ValueError):
        raise BackendError("Backend process could not run.") from None

    if completed.returncode != 0:
        raise BackendError(
            f"Backend process exited with status {completed.returncode}."
        )
    try:
        return completed.stdout.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise BackendError("Backend stdout is not valid UTF-8.") from None


class CommandBackend:
    """Send one JSON request to argv and return stdout verbatim, without parsing.

    Successful replies have empty metadata. Arguments are passed without a shell.
    """

    def __init__(self, argv: list[str], timeout: float = 600):
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(arg, str) for arg in argv)
            or not argv[0]
        ):
            raise BackendError("Backend argv must be a nonempty list of strings.")
        self.argv = list(argv)
        self.timeout = _timeout(timeout)

    def generate(
        self, *, system_prompt: str, payload: dict, model_id: str
    ) -> ModelReply:
        request = _json_input(
            {"model_id": model_id, "system_prompt": system_prompt, "input": payload}
        )
        return ModelReply(_run_command(self.argv, request, self.timeout), {})


class ClaudeCLIBackend:
    """Run a tool-free Claude CLI call in a fresh, disposable working directory.

    Only the outer CLI JSON envelope is parsed; its result string is unchanged.
    Metadata copies available model, modelUsage, usage, cost, cost_usd,
    total_cost_usd and session_id fields verbatim from that envelope.
    """

    def __init__(
        self, executable: str = "claude", timeout: float = 600, effort: str = "high"
    ):
        if not isinstance(executable, str) or not executable:
            raise BackendError("Claude CLI executable must be a nonempty string.")
        if not isinstance(effort, str) or not effort:
            raise BackendError("Claude CLI effort must be a nonempty string.")
        self.executable = executable
        self.timeout = _timeout(timeout)
        self.effort = effort

    def generate(
        self, *, system_prompt: str, payload: dict, model_id: str
    ) -> ModelReply:
        request = _json_input(payload)
        argv = [
            self.executable,
            "-p",
            "--output-format", "json",
            "--no-session-persistence",
            "--tools", "",
            "--strict-mcp-config",
            "--mcp-config", '{"mcpServers":{}}',
            "--safe-mode",
            "--model", model_id,
            "--effort", self.effort,
            "--system-prompt", system_prompt,
        ]
        try:
            with tempfile.TemporaryDirectory(prefix="profile-eval-claude-") as cwd:
                stdout = _run_command(argv, request, self.timeout, cwd=cwd)
        except OSError:
            raise BackendError("Claude CLI temporary directory failed.") from None

        try:
            envelope = json.loads(stdout)
        except (ValueError, RecursionError):
            raise BackendError("Claude CLI returned an invalid JSON envelope.") from None
        if not isinstance(envelope, dict):
            raise BackendError("Claude CLI returned an invalid result envelope.")
        if envelope.get("is_error"):
            raise BackendError("Claude CLI reported an error.")
        if not isinstance(envelope.get("result"), str):
            raise BackendError("Claude CLI result must be a string.")

        metadata = {
            key: envelope[key]
            for key in (
                "model", "modelUsage", "usage", "cost", "cost_usd",
                "total_cost_usd", "session_id",
            )
            if key in envelope
        }
        return ModelReply(envelope["result"], metadata)
