"""Conversion options: which parts of a session are carried across.

Everything is carried by default. Two switches turn parts off:

- ``include_thinking``: reasoning text (the thinking field on each message).
- ``include_tool_calls``: tool invocations and their outputs.

The CLI reads the defaults from the environment (``UNISESSIONS_NO_THINKING=1``,
``UNISESSIONS_NO_TOOL_CALLS=1``) and from ``--no-thinking`` / ``--no-tool-calls``.
SDK code can scope options with ``conversion_options(...)``. The extractors apply
the options when they return messages, so every builder and trace format sees the
same filtered history.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator

_TRUE = {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class ConversionOptions:
    include_thinking: bool = True
    include_tool_calls: bool = True

    @classmethod
    def from_env(cls) -> ConversionOptions:
        return cls(
            include_thinking=os.environ.get("UNISESSIONS_NO_THINKING", "").strip().lower() not in _TRUE,
            include_tool_calls=os.environ.get("UNISESSIONS_NO_TOOL_CALLS", "").strip().lower() not in _TRUE,
        )


_current: ContextVar[ConversionOptions] = ContextVar("unisessions_conversion_options", default=ConversionOptions())


def current_options() -> ConversionOptions:
    return _current.get()


@contextmanager
def conversion_options(options: ConversionOptions) -> Iterator[ConversionOptions]:
    """Apply ``options`` to every extraction inside the block."""
    token = _current.set(options)
    try:
        yield options
    finally:
        _current.reset(token)
