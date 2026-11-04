"""Focused backend tests; every subprocess invocation is mocked."""

import errno
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from profile_eval import backends
from profile_eval.backends import (
    Backend,
    BackendError,
    ClaudeCLIBackend,
    CommandBackend,
    ModelReply,
)


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.run_patch = patch.object(backends.subprocess, "run")
        self.run = self.run_patch.start()
        self.addCleanup(self.run_patch.stop)
        self.run.return_value = subprocess.CompletedProcess([], 0, b"{}", b"")
        self.request = {
            "system_prompt": "Return JSON.\n保留文字",
            "payload": {"question": "你好", "items": [1, False, None]},
            "model_id": "chosen-model",
        }

    def assert_safe_error(self, error):
        message = str(error)
        self.assertLess(len(message), 160)
        self.assertNotIn("secret-canary", message)
        self.assertNotIn("credential-canary", message)
        self.assertTrue(error.__suppress_context__)

    def test_reply_and_protocol_api(self):
        reply = ModelReply(text="{}", metadata={"model": "chosen-model"})
        self.assertEqual(reply.text, "{}")
        self.assertEqual(reply.metadata, {"model": "chosen-model"})
        self.assertTrue(Backend._is_protocol)

    def test_command_serializes_one_request_and_preserves_stdout(self):
        stdout = ' \r\n{"answer":"你好"}\r\n\t'
        self.run.return_value.stdout = stdout.encode("utf-8")
        argv = ["provider-command", "--literal", "$(never-executed); *"]
        reply = CommandBackend(argv, timeout=12.5).generate(**self.request)
        self.assertEqual(reply, ModelReply(stdout, {}))
        self.run.assert_called_once_with(
            argv,
            input=json.dumps(
                {
                    "model_id": self.request["model_id"],
                    "system_prompt": self.request["system_prompt"],
                    "input": self.request["payload"],
                },
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8"),
            capture_output=True,
            text=False,
            shell=False,
            check=False,
            timeout=12.5,
            cwd=None,
        )

    def test_command_does_not_parse_or_strip_stdout(self):
        for stdout in (b"", b"not json\n", b"```json\n{}\n```\n"):
            with self.subTest(stdout=stdout):
                self.run.return_value.stdout = stdout
                reply = CommandBackend(["provider-command"]).generate(**self.request)
                self.assertEqual(reply.text, stdout.decode("utf-8"))

    def test_command_copies_argv(self):
        argv = ["provider-command", "--original"]
        backend = CommandBackend(argv)
        argv.append("--added-later")
        backend.generate(**self.request)
        self.assertEqual(self.run.call_args.args[0], ["provider-command", "--original"])

    def test_generate_arguments_are_keyword_only(self):
        for backend in (CommandBackend(["provider-command"]), ClaudeCLIBackend()):
            with self.subTest(backend=type(backend).__name__):
                with self.assertRaises(TypeError):
                    backend.generate("system", {}, "chosen-model")
        self.run.assert_not_called()

    def test_invalid_configuration_is_rejected_without_subprocess(self):
        for argv in ([], "provider-command", [""], ["provider-command", 1]):
            with self.subTest(argv=argv):
                with self.assertRaises(BackendError):
                    CommandBackend(argv)
        for timeout in (0, -1, float("inf"), float("nan"), None, True, "600"):
            for factory in (lambda: CommandBackend(["provider-command"], timeout),
                            lambda: ClaudeCLIBackend(timeout=timeout)):
                with self.subTest(timeout=timeout):
                    with self.assertRaises(BackendError):
                        factory()
        for kwargs in ({"executable": ""}, {"executable": None},
                       {"effort": ""}, {"effort": None}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(BackendError):
                    ClaudeCLIBackend(**kwargs)
        self.run.assert_not_called()

    def test_json_input_failures_are_sanitized(self):
        cyclic = {"secret-canary": []}
        cyclic["secret-canary"].append(cyclic)
        for payload in ({"secret-canary": object()}, {"secret-canary": float("nan")},
                        {"secret-canary": "\ud800"}, cyclic):
            for backend in (CommandBackend(["provider-command"]), ClaudeCLIBackend()):
                with self.subTest(payload_type=type(payload).__name__,
                                  backend=type(backend).__name__):
                    with self.assertRaises(BackendError) as caught:
                        backend.generate(system_prompt="system", payload=payload,
                                         model_id="chosen-model")
                    self.assert_safe_error(caught.exception)
        self.run.assert_not_called()

    def test_nonzero_exit_never_discloses_stdout_stderr_or_argv(self):
        for code in (1, 17, -15):
            with self.subTest(code=code):
                self.run.return_value = subprocess.CompletedProcess(
                    ["secret-canary"], code,
                    b"credential-canary", b"Authorization: Bearer secret-canary",
                )
                with self.assertRaises(BackendError) as caught:
                    CommandBackend(["secret-canary"]).generate(**self.request)
                self.assertEqual(str(caught.exception),
                                 f"Backend process exited with status {code}.")
                self.assertNotIn("canary", str(caught.exception))

    def test_invalid_stdout_utf8_is_rejected(self):
        self.run.return_value.stdout = b'{"secret-canary":"\xff"}'
        with self.assertRaises(BackendError) as caught:
            CommandBackend(["provider-command"]).generate(**self.request)
        self.assertEqual(str(caught.exception), "Backend stdout is not valid UTF-8.")
        self.assert_safe_error(caught.exception)

    def test_process_failures_have_sanitized_diagnostics(self):
        failures = (
            FileNotFoundError(errno.ENOENT, "secret-canary", "credential-canary"),
            PermissionError(errno.EACCES, "secret-canary", "credential-canary"),
            OSError("secret-canary"),
            ValueError("secret-canary"),
            subprocess.SubprocessError("secret-canary"),
            subprocess.CalledProcessError(3, ["secret-canary"], b"credential-canary"),
        )
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                self.run.side_effect = failure
                with self.assertRaises(BackendError) as caught:
                    CommandBackend(["secret-canary"]).generate(**self.request)
                self.assert_safe_error(caught.exception)
        self.run.side_effect = failures[0]
        with self.assertRaisesRegex(BackendError, "ENOENT"):
            CommandBackend(["secret-canary"]).generate(**self.request)

    def test_timeout_forwards_limit_and_suppresses_partial_output(self):
        self.run.side_effect = subprocess.TimeoutExpired(
            ["secret-canary"], 0.25, output=b"credential-canary", stderr=b"secret-canary"
        )
        with self.assertRaises(BackendError) as caught:
            CommandBackend(["provider-command"], timeout=0.25).generate(**self.request)
        self.assertEqual(str(caught.exception), "Backend process timed out.")
        self.assert_safe_error(caught.exception)
        self.assertEqual(self.run.call_args.kwargs["timeout"], 0.25)
        self.run.assert_called_once()

    def test_cancellation_propagates_without_retry(self):
        for cancellation in (KeyboardInterrupt(), SystemExit(130)):
            with self.subTest(cancellation=type(cancellation).__name__):
                self.run.reset_mock()
                self.run.side_effect = cancellation
                with self.assertRaises(type(cancellation)):
                    CommandBackend(["provider-command"]).generate(**self.request)
                self.run.assert_called_once()

    def test_claude_exact_command_payload_and_temporary_cwd(self):
        result = ' \r\n{"answer":"你好"}\r\n'
        metadata = {
            "model": "reported-model",
            "modelUsage": {"reported-model": {"inputTokens": 7, "costUSD": 0.01}},
            "usage": {"input_tokens": 7, "output_tokens": 3},
            "cost": 0.01,
            "cost_usd": 0.01,
            "total_cost_usd": 0.01,
            "session_id": "test-session",
        }
        envelope = {"result": result, "is_error": False, **metadata,
                    "unrelated": "excluded"}
        visited = []

        def complete(argv, **kwargs):
            cwd = Path(kwargs["cwd"])
            self.assertTrue(cwd.is_dir())
            self.assertEqual(list(cwd.iterdir()), [])
            visited.append(cwd)
            return subprocess.CompletedProcess(
                argv, 0, json.dumps(envelope, ensure_ascii=False).encode("utf-8"), b""
            )

        self.run.side_effect = complete
        original_cwd = Path.cwd()
        reply = ClaudeCLIBackend().generate(**self.request)
        self.assertEqual(reply, ModelReply(result, metadata))
        argv = [
            "claude", "-p", "--output-format", "json", "--no-session-persistence",
            "--tools", "", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
            "--safe-mode", "--model", "chosen-model", "--effort", "high",
            "--system-prompt", self.request["system_prompt"],
        ]
        self.run.assert_called_once_with(
            argv,
            input=json.dumps(self.request["payload"], ensure_ascii=False,
                             allow_nan=False).encode("utf-8"),
            capture_output=True, text=False, shell=False, check=False,
            timeout=600, cwd=str(visited[0]),
        )
        self.assertFalse(visited[0].exists())
        self.assertEqual(Path.cwd(), original_cwd)

    def test_claude_custom_options_and_fresh_cwd_per_call(self):
        self.run.return_value.stdout = b'{"result":"ok"}'
        backend = ClaudeCLIBackend(executable="/custom/claude", timeout=19, effort="low")
        for _ in range(2):
            backend.generate(**self.request)
        calls = self.run.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(calls[0].kwargs["cwd"], calls[1].kwargs["cwd"])
        for call in calls:
            argv = call.args[0]
            self.assertEqual(argv[0], "/custom/claude")
            self.assertEqual(argv[argv.index("--effort") + 1], "low")
            self.assertEqual(call.kwargs["timeout"], 19)
            self.assertFalse(Path(call.kwargs["cwd"]).exists())
            self.assertNotIn("--fallback-model", argv)
            self.assertNotIn("--dangerously-skip-permissions", argv)
            self.assertNotIn("--permission-mode", argv)

    def test_claude_optional_metadata_and_empty_result(self):
        for envelope in ({"result": ""}, {"result": "ok", "is_error": False},
                         {"result": "ok", "usage": {}, "total_cost_usd": 0,
                          "session_id": None}):
            with self.subTest(envelope=envelope):
                self.run.return_value.stdout = json.dumps(envelope).encode("utf-8")
                reply = ClaudeCLIBackend().generate(**self.request)
                self.assertEqual(reply.text, envelope["result"])
                expected = {key: value for key, value in envelope.items()
                            if key not in ("result", "is_error")}
                self.assertEqual(reply.metadata, expected)

    def test_claude_rejects_error_envelope_before_reading_result(self):
        self.run.return_value.stdout = json.dumps({
            "is_error": True,
            "result": "credential-canary",
            "errors": ["secret-canary"],
        }).encode("utf-8")
        with self.assertRaises(BackendError) as caught:
            ClaudeCLIBackend().generate(**self.request)
        self.assertEqual(str(caught.exception), "Claude CLI reported an error.")
        self.assertNotIn("canary", str(caught.exception))

    def test_claude_rejects_malformed_json_envelopes(self):
        invalid = (
            b"secret-canary", b'```json\n{"result":"ok"}\n```',
            b'{"result":"ok"}\n{"result":"extra"}', b"[]", b"null", b'"secret-canary"',
            b"{}", b'{"result":null}', b'{"result":7}', b'{"result":{}}',
        )
        for stdout in invalid:
            with self.subTest(stdout=stdout):
                self.run.return_value.stdout = stdout
                with self.assertRaises(BackendError) as caught:
                    ClaudeCLIBackend().generate(**self.request)
                self.assertNotIn("canary", str(caught.exception))

    def test_claude_cleans_cwd_on_timeout_failure_and_cancellation(self):
        for failure in (
            subprocess.TimeoutExpired("secret-canary", 0.5, output=b"credential-canary"),
            FileNotFoundError(errno.ENOENT, "secret-canary", "credential-canary"),
            KeyboardInterrupt(), SystemExit(130),
        ):
            with self.subTest(failure=type(failure).__name__):
                visited = []

                def fail(argv, **kwargs):
                    visited.append(Path(kwargs["cwd"]))
                    self.assertTrue(visited[0].is_dir())
                    raise failure

                self.run.reset_mock()
                self.run.side_effect = fail
                expected = BackendError if isinstance(failure, Exception) else type(failure)
                with self.assertRaises(expected):
                    ClaudeCLIBackend(timeout=0.5).generate(**self.request)
                self.run.assert_called_once()
                self.assertFalse(visited[0].exists())

    def test_claude_temp_directory_failure_is_sanitized(self):
        with patch.object(backends.tempfile, "TemporaryDirectory",
                          side_effect=OSError("secret-canary")):
            with self.assertRaises(BackendError) as caught:
                ClaudeCLIBackend().generate(**self.request)
        self.assert_safe_error(caught.exception)
        self.assertEqual(str(caught.exception), "Claude CLI temporary directory failed.")
        self.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
