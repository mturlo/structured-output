"""Thin Anthropic adapter. Wraps messages.create() and exposes a uniform result."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from anthropic import Anthropic

from .config import Settings


@dataclass
class ToolCall:
    name: str
    input: dict[str, Any]
    tool_use_id: str


@dataclass
class CompletionResult:
    text: str
    tool_calls: list[ToolCall]
    stop_reason: str | None
    input_tokens: int
    output_tokens: int
    latency_ms: int
    raw: Any = field(repr=False, default=None)


def parse_response(response: Any, latency_ms: int) -> CompletionResult:
    """Extract text and tool_use blocks from an Anthropic Message response."""
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in response.content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text_parts.append(block.text)
        elif block_type == "tool_use":
            tool_calls.append(
                ToolCall(
                    name=block.name,
                    input=dict(block.input) if block.input else {},
                    tool_use_id=block.id,
                )
            )
    usage = response.usage
    return CompletionResult(
        text="".join(text_parts),
        tool_calls=tool_calls,
        stop_reason=response.stop_reason,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        latency_ms=latency_ms,
        raw=response,
    )


class AnthropicClient:
    """Thin wrapper. Centralises model, max_tokens, retry-free single shot."""

    def __init__(self, settings: Settings, client: Anthropic | None = None) -> None:
        self.settings = settings
        self._client = client or Anthropic(api_key=settings.anthropic_api_key)

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> CompletionResult:
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "max_tokens": max_tokens or self.settings.max_tokens,
            "messages": messages,
        }
        if system is not None:
            kwargs["system"] = system
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        start = time.perf_counter()
        response = self._client.messages.create(**kwargs)
        latency_ms = int((time.perf_counter() - start) * 1000)
        return parse_response(response, latency_ms)
