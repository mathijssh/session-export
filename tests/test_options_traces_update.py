from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from session_sdk.converters import MessageExtractor
from session_sdk.models import NativeSession, TextMessage, ToolCall
from session_sdk.options import ConversionOptions, conversion_options, current_options
from session_sdk.paths import SessionIdFactory
from session_sdk.stores import ClaudeStore, CodexStore, PiDcpStore, PiStore
from session_sdk.traces import build_trace
from unisessions.cli import main

PI_SESSION = "11111111-2222-4333-8444-555555555555"


def write_pi_with_tools(home: Path) -> Path:
    """A Pi session: a user question, an assistant turn with a tool call and its result, then a reply."""
    path = home / "sessions" / "--C--work--" / f"2026-06-28T04-20-31-629Z_{PI_SESSION}.jsonl"
    records = [
        {"type": "session", "version": 3, "id": PI_SESSION, "timestamp": "2026-06-28T04:20:31.629Z", "cwd": r"C:\work"},
        {"type": "message", "id": "00000001", "parentId": None, "timestamp": "2026-06-28T04:20:32.000Z",
         "message": {"role": "user", "content": [{"type": "text", "text": "List the files"}]}},
        {"type": "message", "id": "00000002", "parentId": "00000001", "timestamp": "2026-06-28T04:20:33.000Z",
         "message": {"role": "assistant", "content": [
             {"type": "thinking", "thinking": "Use ls."},
             {"type": "text", "text": "Checking."},
             {"type": "toolCall", "id": "call-1", "name": "bash", "arguments": {"command": "ls"}}]}},
        {"type": "message", "id": "00000003", "parentId": "00000002", "timestamp": "2026-06-28T04:20:34.000Z",
         "message": {"role": "toolResult", "toolCallId": "call-1", "toolName": "bash",
                     "content": [{"type": "text", "text": "a.txt"}], "isError": False}},
        {"type": "message", "id": "00000004", "parentId": "00000003", "timestamp": "2026-06-28T04:20:35.000Z",
         "message": {"role": "assistant", "content": [{"type": "text", "text": "There is a.txt."}]}},
    ]
    from session_sdk.jsonl import JsonlFile
    JsonlFile(path).write(records, overwrite=True)
    return path


def extracted(options: ConversionOptions) -> list[TextMessage]:
    with tempfile.TemporaryDirectory() as root:
        home = Path(root)
        write_pi_with_tools(home / "pi")
        session = PiStore(home / "pi").load(PI_SESSION)
        with conversion_options(options):
            return MessageExtractor().from_pi(session)


class OptionsTests(unittest.TestCase):
    def test_everything_is_carried_by_default(self) -> None:
        messages = extracted(ConversionOptions())
        assistant = messages[1]
        self.assertEqual(assistant.thinking, "Use ls.")
        self.assertEqual([c.name for c in assistant.tool_calls], ["bash"])
        self.assertEqual(assistant.tool_calls[0].output, "a.txt")

    def test_no_tool_calls_drops_calls_and_outputs_only(self) -> None:
        messages = extracted(ConversionOptions(include_tool_calls=False))
        self.assertEqual(messages[1].tool_calls, ())
        self.assertEqual(messages[1].thinking, "Use ls.")

    def test_no_thinking_drops_reasoning_only(self) -> None:
        messages = extracted(ConversionOptions(include_thinking=False))
        self.assertIsNone(messages[1].thinking)
        self.assertEqual(messages[1].tool_calls[0].name, "bash")

    def test_environment_defaults(self) -> None:
        with mock.patch.dict(os.environ, {"UNISESSIONS_NO_TOOL_CALLS": "1"}):
            self.assertFalse(ConversionOptions.from_env().include_tool_calls)
            self.assertTrue(ConversionOptions.from_env().include_thinking)
        self.assertEqual(current_options(), ConversionOptions())

    def test_cli_switch_applies_to_a_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            home = Path(root)
            write_pi_with_tools(home / "pi")
            status = main(["--pi-agent-home", str(home / "pi"), "--claude-home", str(home / "claude"),
                           "pi-to-claude", PI_SESSION, "--write", "--no-tool-calls", "--no-thinking"])
            self.assertEqual(status, 0)
            text = next((home / "claude").rglob("*.jsonl")).read_text(encoding="utf-8")
            self.assertNotIn("tool_use", text)
            self.assertNotIn("tool_result", text)
            self.assertNotIn('"type": "thinking"', text)


class TraceTests(unittest.TestCase):
    def _trace(self, fmt: str, options: ConversionOptions = ConversionOptions()) -> list[dict]:
        with tempfile.TemporaryDirectory() as root:
            write_pi_with_tools(Path(root))
            session = PiStore(Path(root)).load(PI_SESSION)
            with conversion_options(options):
                messages = MessageExtractor().from_pi(session)
                return build_trace(fmt, session, messages)

    def test_openai_trace_has_tool_calls_and_tool_results(self) -> None:
        conversation = self._trace("openai")[0]["messages"]
        self.assertEqual([m["role"] for m in conversation], ["user", "assistant", "tool", "assistant"])
        assistant = conversation[1]
        self.assertEqual(assistant["content"], "<think>\nUse ls.\n</think>\n\nChecking.")
        call = assistant["tool_calls"][0]
        self.assertEqual(call["id"], "call-1")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"command": "ls"})
        self.assertEqual(conversation[2], {"role": "tool", "tool_call_id": "call-1", "content": "a.txt"})

    def test_sharegpt_trace_uses_function_call_and_observation(self) -> None:
        turns = self._trace("sharegpt")[0]["conversations"]
        self.assertEqual([t["from"] for t in turns], ["human", "gpt", "function_call", "observation", "gpt"])
        self.assertEqual(json.loads(turns[2]["value"]), {"name": "bash", "arguments": {"command": "ls"}})
        self.assertEqual(turns[3]["value"], "a.txt")

    def test_trace_switches_remove_tool_turns_and_think_blocks(self) -> None:
        conversation = self._trace("openai", ConversionOptions(include_thinking=False, include_tool_calls=False))[0]["messages"]
        self.assertEqual([m["role"] for m in conversation], ["user", "assistant", "assistant"])
        self.assertEqual(conversation[1]["content"], "Checking.")
        self.assertNotIn("tool_calls", conversation[1])

    def test_sts_trace_stays_text_only(self) -> None:
        records = self._trace("sts")
        self.assertEqual(records[0]["type"], "session")
        self.assertFalse(any("tool_calls" in r.get("message", {}) for r in records[1:]))


class UpdateModeTests(unittest.TestCase):
    def test_update_skips_unchanged_and_rewrites_after_source_changes(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            home = Path(root)
            source = write_pi_with_tools(home / "pi")
            args = ["--pi-agent-home", str(home / "pi"), "--claude-home", str(home / "claude"), "pi-to-claude", PI_SESSION, "--write"]
            self.assertEqual(main(args), 0)
            destination = next((home / "claude").rglob("*.jsonl"))
            written = destination.stat().st_mtime_ns

            with mock.patch("builtins.print") as printed:
                main(args + ["--on-conflict", "update"])
            self.assertTrue(any("skipping" in str(c.args[0]) for c in printed.call_args_list))

            later = written + 5_000_000_000
            os.utime(source, ns=(later, later))
            with mock.patch("builtins.print") as printed:
                main(args + ["--on-conflict", "update"])
            self.assertFalse(any("skipping" in str(c.args[0]) for c in printed.call_args_list))
            self.assertGreater(destination.stat().st_mtime_ns, written)


class BulkExportTests(unittest.TestCase):
    def test_export_all_writes_every_requested_target(self) -> None:
        from tests.test_conversion import ConversionTests
        with tempfile.TemporaryDirectory() as root:
            home = Path(root)
            codex_home = home / "codex"
            ConversionTests()._write_codex_fixture(codex_home, "01234567-89ab-cdef-0123-456789abcdef")
            args = ["--codex-home", str(codex_home), "--claude-home", str(home / "claude"), "--grok-home", str(home / "grok"),
                    "--pi-agent-home", str(home / "pi"), "--pi-dcp-home", str(home / "dcp"),
                    "export-all", "--write", "--targets", "pi", "claude", "grok"]
            self.assertEqual(main(args), 0)
            self.assertEqual(len(ClaudeStore(home / "claude").list()), 1)
            self.assertEqual(len(PiStore(home / "pi").list()), 1)
            from session_sdk.stores import GrokStore
            self.assertEqual(len(GrokStore(home / "grok").list()), 1)


if __name__ == "__main__":
    unittest.main()
