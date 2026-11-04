from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "pipeline.py"
SPEC = importlib.util.spec_from_file_location("pipeline", MODULE_PATH)
assert SPEC and SPEC.loader
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


class PromptIdentityTests(unittest.TestCase):
    def test_prompt_matches_frozen_hash(self) -> None:
        digest = hashlib.sha256(pipeline.SYSTEM_PROMPT.encode("utf-8")).hexdigest()
        self.assertEqual(digest, pipeline.PROMPT_SHA256)


class EvidenceTests(unittest.TestCase):
    def test_only_user_messages_are_rendered(self) -> None:
        messages = [
            {"role": "system", "content": "persona scaffolding"},
            {"role": "user", "content": "I live in Osaka."},
            {"role": "assistant", "content": "Nice."},
            {"role": "user", "content": "I play tennis."},
        ]
        text = pipeline.render_evidence(messages)
        self.assertIn("USER MESSAGE ID u01\nI live in Osaka.", text)
        self.assertIn("USER MESSAGE ID u02\nI play tennis.", text)
        self.assertNotIn("scaffolding", text)
        self.assertNotIn("Nice", text)
        self.assertNotIn("u03", text)


class ChunkingTests(unittest.TestCase):
    def _messages(self, count: int) -> list[dict[str, str]]:
        return [
            {"role": "user", "content": f"fact number {index} " + "word " * 20}
            for index in range(count)
        ]

    def test_stages_are_contiguous_and_complete(self) -> None:
        messages = self._messages(40)
        stages, info = pipeline.split_stages(
            messages,
            stage_count=8,
            counter=pipeline.TokenCounter(None),
            max_history_tokens=1_000_000,
            max_chunk_tokens=1_000_000,
        )
        self.assertEqual(len(stages), 8)
        flattened = [message for stage in stages for message in stage]
        self.assertEqual(flattened, messages)
        self.assertEqual(info["stage_count"], 8)
        self.assertFalse(info["truncated"])

    def test_history_prefix_is_truncated(self) -> None:
        messages = self._messages(40)
        stages, info = pipeline.split_stages(
            messages,
            stage_count=8,
            counter=pipeline.TokenCounter(None),
            max_history_tokens=300,
            max_chunk_tokens=1_000_000,
        )
        self.assertTrue(info["truncated"])
        kept = [message for stage in stages for message in stage]
        self.assertLess(len(kept), len(messages))
        self.assertEqual(kept, messages[: len(kept)])

    def test_stage_count_cannot_exceed_messages(self) -> None:
        stages, info = pipeline.split_stages(
            self._messages(3),
            stage_count=8,
            counter=pipeline.TokenCounter(None),
            max_history_tokens=1_000_000,
            max_chunk_tokens=1_000_000,
        )
        self.assertEqual(len(stages), 3)
        self.assertEqual(info["stage_count"], 3)

    def test_skewed_lengths_keep_every_stage_nonempty(self) -> None:
        messages = [{"role": "user", "content": "x" * 10_000}]
        messages.extend(
            {"role": "user", "content": f"short {index}"} for index in range(5)
        )
        stages, _info = pipeline.split_stages(
            messages,
            stage_count=6,
            counter=pipeline.TokenCounter(None),
            max_history_tokens=1_000_000,
            max_chunk_tokens=1_000_000,
        )
        self.assertEqual(len(stages), 6)
        self.assertTrue(all(stages))
        self.assertEqual([message for stage in stages for message in stage], messages)


class HistoryLoadingTests(unittest.TestCase):
    def _write(self, payload: object) -> Path:
        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(payload, handle)
        handle.close()
        self.addCleanup(Path(handle.name).unlink, missing_ok=True)
        return Path(handle.name)

    def test_raw_message_dict(self) -> None:
        path = self._write(
            {
                "messages": [
                    {"role": "user", "content": "I am a nurse."},
                    {"role": "assistant", "content": "Understood."},
                ]
            }
        )
        value, pre_chunked = pipeline.load_history(path)
        self.assertFalse(pre_chunked)
        self.assertEqual(len(value), 2)

    def test_system_messages_are_removed(self) -> None:
        path = self._write(
            {
                "history": [
                    {"role": "system", "content": "persona scaffolding"},
                    {"role": "user", "content": "I am a nurse."},
                    {"role": "assistant", "content": "Understood."},
                ]
            }
        )
        value, pre_chunked = pipeline.load_history(path)
        self.assertFalse(pre_chunked)
        self.assertEqual([message["role"] for message in value], ["user", "assistant"])

    def test_personamem_chat_history_key(self) -> None:
        path = self._write(
            {
                "metadata": {"persona_id": 66},
                "chat_history": [
                    {"role": "user", "content": "I am a nurse."},
                ],
            }
        )
        value, pre_chunked = pipeline.load_history(path)
        self.assertFalse(pre_chunked)
        self.assertEqual(value[0]["content"], "I am a nurse.")

    def test_pre_chunked_detection(self) -> None:
        path = self._write(
            [{"messages": [{"role": "user", "content": "I am a nurse."}]}]
        )
        value, pre_chunked = pipeline.load_history(path)
        self.assertTrue(pre_chunked)
        self.assertEqual(len(value), 1)


class OutputTests(unittest.TestCase):
    def test_output_path_must_be_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out"
            path.write_text("not a directory", encoding="utf-8")
            with self.assertRaises(pipeline.PipelineError):
                pipeline._prepare_output(path, overwrite=True)

    def test_overwrite_resets_stale_generated_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out"
            path.mkdir()
            (path / "p9.txt").write_text("stale", encoding="utf-8")
            (path / "manifest.json").write_text("{}", encoding="utf-8")
            (path / "notes.md").write_text("keep", encoding="utf-8")
            pipeline._prepare_output(path, overwrite=True, reset=True)
            self.assertFalse((path / "p9.txt").exists())
            self.assertFalse((path / "manifest.json").exists())
            self.assertTrue((path / "notes.md").exists())


class RunArgumentTests(unittest.TestCase):
    def test_resume_and_dry_run_are_mutually_exclusive(self) -> None:
        args = pipeline._parse_args(
            [
                "--history", "history.json",
                "--output-dir", "output",
                "--model", "model",
                "--resume",
                "--dry-run",
            ]
        )
        with self.assertRaises(pipeline.PipelineError):
            pipeline.run(args)

    def test_nonpositive_limits_are_rejected(self) -> None:
        args = pipeline._parse_args(
            [
                "--history", "history.json",
                "--output-dir", "output",
                "--model", "model",
                "--max-attempts", "0",
            ]
        )
        with self.assertRaises(pipeline.PipelineError):
            pipeline.run(args)

    def test_missing_api_key_does_not_create_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            history = Path(directory) / "history.json"
            history.write_text(
                json.dumps([{"role": "user", "content": "I am a nurse."}]),
                encoding="utf-8",
            )
            output = Path(directory) / "output"
            args = pipeline._parse_args(
                [
                    "--history", str(history),
                    "--output-dir", str(output),
                    "--model", "model",
                    "--api-base-url", "https://example.invalid/v1",
                    "--api-key-env", "PPL_AUDIT_MISSING_KEY",
                ]
            )
            with self.assertRaises(pipeline.PipelineError):
                pipeline.run(args)
            self.assertFalse(output.exists())

    def test_resume_revalidates_accepted_profiles_under_current_limits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            history = Path(directory) / "history.json"
            history.write_text(
                json.dumps([{"role": "user", "content": "I am a nurse."}]),
                encoding="utf-8",
            )
            output = Path(directory) / "output"
            output.mkdir()
            evidence = pipeline.render_evidence(
                [{"role": "user", "content": "I am a nurse."}]
            )
            profile = "The user is a very dedicated nurse who lives in Osaka."
            (output / "p1.txt").write_text(profile + "\n", encoding="utf-8")
            (output / "manifest.json").write_text(
                json.dumps(
                    {
                        "prompt_sha256": pipeline.PROMPT_SHA256,
                        "model": "model",
                        "backend": "openai",
                        "history": {
                            "sha256": pipeline.sha256_text(
                                history.read_text(encoding="utf-8")
                            ),
                            "stage_count": 1,
                            "pre_chunked": False,
                        },
                        "stages": [
                            {
                                "stage": 1,
                                "evidence_sha256": pipeline.sha256_text(evidence),
                                "profile_sha256": pipeline.sha256_text(profile),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            args = pipeline._parse_args(
                [
                    "--history", str(history),
                    "--output-dir", str(output),
                    "--model", "model",
                    "--resume",
                    "--max-output-tokens", "10",
                ]
            )
            with self.assertRaisesRegex(
                pipeline.PipelineError,
                "invalid under the current limits",
            ):
                pipeline.run(args)


class ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.counter = pipeline.TokenCounter(None)

    def test_accepts_prose(self) -> None:
        text = "The user is a nurse who lives in Osaka and plays tennis."
        self.assertEqual(pipeline.validate_profile(text, self.counter, 2048), text)

    def test_rejects_bullets(self) -> None:
        with self.assertRaises(pipeline.ProfileValidationError):
            pipeline.validate_profile("- The user is a nurse.", self.counter, 2048)

    def test_rejects_json(self) -> None:
        with self.assertRaises(pipeline.ProfileValidationError):
            pipeline.validate_profile('{"name": "Oliver"}', self.counter, 2048)

    def test_rejects_prompt_leak(self) -> None:
        with self.assertRaises(pipeline.ProfileValidationError):
            pipeline.validate_profile(
                "Causal update stage: 3. The user is a nurse.", self.counter, 2048
            )

    def test_rejects_unfinished_sentence(self) -> None:
        with self.assertRaises(pipeline.ProfileValidationError):
            pipeline.validate_profile("The user is a nurse who", self.counter, 2048)

    def test_rejects_over_limit(self) -> None:
        with self.assertRaises(pipeline.ProfileValidationError):
            pipeline.validate_profile("word " * 3000 + ".", self.counter, 2048)


class ApiParsingTests(unittest.TestCase):
    def test_chat_completions_shape(self) -> None:
        payload = {"choices": [{"message": {"content": "The user is a nurse."}}]}
        self.assertEqual(pipeline._extract_api_text(payload), "The user is a nurse.")

    def test_responses_shape(self) -> None:
        payload = {"output": [{"content": [{"text": "The user is a nurse."}]}]}
        self.assertEqual(pipeline._extract_api_text(payload), "The user is a nurse.")


if __name__ == "__main__":
    unittest.main()
