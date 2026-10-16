from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from session_sdk.converters import MessageExtractor
from session_sdk.stores import GrokStore

# Tables from superagent-ai/grok-cli src/storage/migrations.ts (the parts this reader uses).
DDL = """
CREATE TABLE workspaces (id TEXT PRIMARY KEY, scope_key TEXT NOT NULL UNIQUE, canonical_path TEXT NOT NULL,
  git_root TEXT, display_name TEXT NOT NULL, last_seen_at TEXT NOT NULL) STRICT;
CREATE TABLE sessions (id TEXT PRIMARY KEY, workspace_id TEXT NOT NULL, title TEXT, recap_text TEXT,
  recap_model TEXT, recap_updated_at TEXT, model TEXT NOT NULL, mode TEXT NOT NULL, cwd_at_start TEXT NOT NULL,
  cwd_last TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL) STRICT;
CREATE TABLE messages (session_id TEXT NOT NULL, seq INTEGER NOT NULL, role TEXT NOT NULL,
  message_json TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY (session_id, seq)) STRICT;
"""

SESSION = "grok-cli-session-1"
CWD = r"C:\work\demo"


def make_grok_db(home: Path) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    path = home / "grok.db"
    connection = sqlite3.connect(path)
    try:
        connection.executescript(DDL)
        connection.execute("INSERT INTO workspaces VALUES ('ws-1', 'k', ?, NULL, 'demo', '2026-06-01T10:00:00Z')", (CWD,))
        connection.execute(
            "INSERT INTO sessions VALUES (?, 'ws-1', 'Demo', NULL, NULL, NULL, 'grok-4', 'build', ?, ?, 'done', '2026-06-01T10:00:00Z', '2026-06-01T10:05:00Z')",
            (SESSION, CWD, CWD),
        )
        messages = [
            {"role": "user", "content": "List the files"},
            {
                "role": "assistant",
                "content": [
                    {"type": "reasoning", "text": "I should run ls."},
                    {"type": "text", "text": "Running it now."},
                    {"type": "tool-call", "toolCallId": "tc-1", "toolName": "bash", "input": {"command": "ls"}},
                    {"type": "tool-call", "toolCallId": "tc-2", "toolName": "read_file", "input": {"path": "b.txt"}},
                ],
            },
            {
                "role": "tool",
                "content": [
                    {"type": "tool-result", "toolCallId": "tc-1", "toolName": "bash",
                     "output": {"type": "json", "value": {"success": True, "output": "a.txt"}}},
                    {"type": "tool-result", "toolCallId": "tc-2", "toolName": "read_file",
                     "output": {"type": "error-text", "value": "no such file"}},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "a.txt is there, b.txt is missing."}]},
        ]
        for seq, message in enumerate(messages, start=1):
            connection.execute(
                "INSERT INTO messages VALUES (?, ?, ?, ?, ?)",
                (SESSION, seq, message["role"], json.dumps(message), f"2026-06-01T10:0{seq}:00Z"),
            )
        connection.commit()
    finally:
        connection.close()
    return path


class GrokCliStoreTests(unittest.TestCase):
    def test_lists_and_loads_grok_cli_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            home = Path(root) / ".grok"
            make_grok_db(home)
            store = GrokStore(home)
            summaries = store.list()
            self.assertEqual([s.session_id for s in summaries], [SESSION])
            self.assertEqual(summaries[0].cwd, CWD)
            self.assertEqual(summaries[0].message_count, 3)  # user and assistant rows; tool results are not chat
            session = store.load(SESSION)
            self.assertEqual(session.provider, "grok")
            self.assertEqual(len(session.records), 4)
            reloaded = store.load_path(summaries[0].path)
            self.assertEqual(reloaded.session_id, SESSION)

    def test_extraction_keeps_text_thinking_calls_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            home = Path(root) / ".grok"
            make_grok_db(home)
            session = GrokStore(home).load(SESSION)
            messages = MessageExtractor().from_grok(session)
            self.assertEqual([m.role for m in messages], ["user", "assistant", "assistant"])
            assistant = messages[1]
            self.assertEqual(assistant.text, "Running it now.")  # reasoning text must not leak into the reply
            self.assertEqual(assistant.thinking, "I should run ls.")
            self.assertEqual([(c.id, c.name) for c in assistant.tool_calls], [("tc-1", "bash"), ("tc-2", "read_file")])
            self.assertEqual(assistant.tool_calls[0].arguments, {"command": "ls"})
            self.assertEqual((assistant.tool_calls[0].output, assistant.tool_calls[0].is_error), ("a.txt", False))
            self.assertEqual((assistant.tool_calls[1].output, assistant.tool_calls[1].is_error), ("no such file", True))
            self.assertEqual(messages[2].text, "a.txt is there, b.txt is missing.")

    def test_grok_build_envelopes_still_extract(self) -> None:
        records = [{"timestamp": 1, "method": "session/update", "params": {"sessionId": "s", "update": {
            "sessionUpdate": "agent_message_chunk", "messageId": "m", "content": {"type": "text", "text": "hi"}}}}]
        session = type("S", (), {"records": records, "timestamp": "2026-06-01T10:00:00Z"})()
        messages = MessageExtractor().from_grok(session)  # type: ignore[arg-type]
        self.assertEqual([m.text for m in messages], ["hi"])


if __name__ == "__main__":
    unittest.main()
