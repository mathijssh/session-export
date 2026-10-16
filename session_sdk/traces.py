"""Trace format builders for exporting sessions to HuggingFace, OpenAI, and ShareGPT trace formats.

Three trace formats are supported:

1. **STS-Format** (Session Trace Simple Format) — HuggingFace Hub's native trace viewer format.
   JSONL with a session header line followed by message lines. Text only: the format does not
   define tool events.
   https://huggingface.co/docs/hub/session-traces-format

2. **OpenAI fine-tuning format** — ``{"messages": [...]}`` JSONL, one conversation per line.
   Assistant turns carry ``tool_calls`` and each tool result is a ``{"role": "tool"}`` message
   with the matching ``tool_call_id``, as the chat fine-tuning docs describe.
   https://platform.openai.com/docs/guides/fine-tuning

3. **ShareGPT format** — ``{"conversations": [...]}`` JSONL with ``human``, ``gpt``, ``function_call``
   and ``observation`` turns, as LLaMA-Factory reads them.
   https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md

Reasoning is written as a ``<think>...</think>`` block at the start of the assistant text, the usual
convention in training data. Both switches in ``session_sdk.options`` apply, so a trace can leave
out thinking or tool calls.

All builders consume the same ``list[TextMessage]`` produced by ``MessageExtractor``.
"""

from __future__ import annotations

import json

from session_sdk.json_types import JsonObject
from session_sdk.models import NativeSession, TextMessage, ToolCall
from session_sdk.options import current_options


def _assistant_text(message: TextMessage) -> str:
    """Reasoning first, in a ``<think>`` block, then the visible text."""
    parts: list[str] = []
    if message.thinking and current_options().include_thinking:
        parts.append(f"<think>\n{message.thinking}\n</think>")
    if message.text:
        parts.append(message.text)
    return "\n\n".join(parts)


def _tool_output(call: ToolCall) -> str:
    return call.output if call.output is not None else "(no output recorded)"


class STSTraceBuilder:
    """Build HuggingFace Session Trace Simple Format (STS-Format) JSONL.

    Output: one JSON object per line. First line is the session header,
    subsequent lines are message envelopes.

    Example::

        {"type":"session","harness":"codex","id":"abc123","name":"Fix the bug"}
        {"type":"message","message":{"role":"user","content":"Fix the bug"}}
        {"type":"message","message":{"role":"assistant","content":"Looking at it now","timestamp":1719900000000,"model":"gpt-4o"}}
    """

    def build(self, session: NativeSession, messages: list[TextMessage]) -> list[JsonObject]:
        header: JsonObject = {
            "type": "session",
            "harness": session.provider,
            "id": session.session_id,
        }
        if session.cwd:
            header["cwd"] = session.cwd
        records: list[JsonObject] = [header]
        for message in messages:
            if message.is_contextual:
                continue
            if message.is_compaction:
                # Compaction summaries become system messages with a marker
                records.append({
                    "type": "message",
                    "message": {
                        "role": "system",
                        "content": f"[compaction summary]\n{message.text}",
                    },
                })
                continue
            msg: JsonObject = {
                "role": message.role,
                "content": _assistant_text(message) if message.role == "assistant" else message.text,
            }
            if message.model:
                msg["model"] = message.model
            if message.timestamp:
                msg["timestamp"] = _to_epoch_ms(message.timestamp)
            records.append({"type": "message", "message": msg})
        return records


class OpenAITraceBuilder:
    """Build OpenAI fine-tuning conversation format JSONL.

    Output: one JSON object per line, each containing a ``messages`` array
    with the full conversation. Designed for ``client.files.create(purpose='fine-tune')``.

    Example::

        {"messages":[{"role":"user","content":"Fix the bug"},{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"bash","arguments":"{\\"command\\": \\"ls\\"}"}}]},{"role":"tool","tool_call_id":"c1","content":"a.txt"},{"role":"assistant","content":"Done."}]}
    """

    def build(self, session: NativeSession, messages: list[TextMessage]) -> list[JsonObject]:
        conversation: list[JsonObject] = []
        for message in messages:
            if message.is_contextual:
                continue
            if message.is_compaction:
                conversation.append({
                    "role": "system",
                    "content": f"[compaction summary]\n{message.text}",
                })
                continue
            if message.role != "assistant":
                conversation.append({"role": message.role, "content": message.text})
                continue
            entry: JsonObject = {"role": "assistant", "content": _assistant_text(message) or None}
            if message.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)},
                    }
                    for call in message.tool_calls
                ]
            conversation.append(entry)
            for call in message.tool_calls:
                conversation.append({"role": "tool", "tool_call_id": call.id, "content": _tool_output(call)})
        return [{"messages": conversation}]


class ShareGPTTraceBuilder:
    """Build ShareGPT conversation format JSONL for LLaMA-Factory.

    Output: one JSON object per line with a ``conversations`` array. Roles:
    user -> ``human``, assistant -> ``gpt``, each tool call -> ``function_call`` (JSON with
    ``name`` and ``arguments``), each tool result -> ``observation``.

    LLaMA-Factory expects human and observation turns at odd positions and gpt and function_call
    turns at even positions. Tool turns follow the order they happened in, so a turn with both
    text and tool calls can break that pattern. The output is still valid ShareGPT.

    Example::

        {"conversations":[{"from":"human","value":"Fix the bug"},{"from":"function_call","value":"{\\"name\\": \\"bash\\", \\"arguments\\": {\\"command\\": \\"ls\\"}}"},{"from":"observation","value":"a.txt"},{"from":"gpt","value":"Done."}]}
    """

    def build(self, session: NativeSession, messages: list[TextMessage]) -> list[JsonObject]:
        conversations: list[JsonObject] = []
        for message in messages:
            if message.is_contextual:
                continue
            if message.is_compaction:
                conversations.append({
                    "from": "system",
                    "value": f"[compaction summary]\n{message.text}",
                })
                continue
            if message.role != "assistant":
                conversations.append({"from": "human" if message.role == "user" else message.role, "value": message.text})
                continue
            text = _assistant_text(message)
            if text:
                conversations.append({"from": "gpt", "value": text})
            for call in message.tool_calls:
                conversations.append({
                    "from": "function_call",
                    "value": json.dumps({"name": call.name, "arguments": call.arguments}, ensure_ascii=False),
                })
                conversations.append({"from": "observation", "value": _tool_output(call)})
        return [{"conversations": conversations}]


# Trace format registry
TRACE_BUILDERS: dict[str, type] = {
    "sts": STSTraceBuilder,
    "openai": OpenAITraceBuilder,
    "sharegpt": ShareGPTTraceBuilder,
}

TRACE_FORMATS = tuple(TRACE_BUILDERS.keys())


def build_trace(
    format: str,
    session: NativeSession,
    messages: list[TextMessage],
) -> list[JsonObject]:
    """Build trace records for the given format.

    Args:
        format: One of ``"sts"``, ``"openai"``, ``"sharegpt"``.
        session: The source NativeSession.
        messages: Extracted TextMessages from MessageExtractor.

    Returns:
        List of JSON objects to write as JSONL lines.

    Raises:
        ValueError: If the format is not recognized.
    """
    builder_cls = TRACE_BUILDERS.get(format)
    if builder_cls is None:
        raise ValueError(f"Unknown trace format: {format!r}. Supported: {', '.join(TRACE_FORMATS)}")
    return builder_cls().build(session, messages)


def _to_epoch_ms(timestamp: str) -> int:
    """Convert ISO timestamp string to epoch milliseconds."""
    from session_sdk.paths import iso_to_epoch_ms
    return iso_to_epoch_ms(timestamp)
