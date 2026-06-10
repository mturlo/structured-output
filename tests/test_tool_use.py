"""Tests for technique 2 (tool use). Scripted FakeClient — no API."""

from __future__ import annotations

from dataclasses import dataclass, field

from structured_output.provider import ToolCall
from structured_output.techniques.tool_use import (
    TECHNIQUE,
    TOOL_NAME,
    extract_tool_use,
)


@dataclass
class FakeCompletion:
    text: str = ""
    tool_calls: list = field(default_factory=list)
    input_tokens: int = 100
    output_tokens: int = 200
    latency_ms: int = 150
    stop_reason: str = "tool_use"
    raw: object = None


class FakeClient:
    def __init__(self, responses: list[FakeCompletion]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, messages, *, system=None, tools=None, tool_choice=None, **kw):
        self.calls.append(
            {"messages": messages, "system": system, "tools": tools, "tool_choice": tool_choice}
        )
        if not self._responses:
            raise AssertionError("FakeClient exhausted")
        return self._responses.pop(0)


VALID_INPUT = {
    "id": "s1",
    "title": "Test",
    "questions": [{"type": "open", "id": "q1", "text": "Why?"}],
    "routing": [],
}

INVALID_INPUT = {
    "id": "s1",
    "title": "Test",
    "questions": [
        {"type": "open", "id": "q1", "text": "A"},
        {"type": "open", "id": "q1", "text": "Dup"},
    ],
    "routing": [],
}


def _tool_response(input_data: dict, tool_use_id: str = "toolu_1") -> FakeCompletion:
    return FakeCompletion(
        tool_calls=[ToolCall(name=TOOL_NAME, input=input_data, tool_use_id=tool_use_id)],
    )


def test_first_call_shape():
    client = FakeClient([_tool_response(VALID_INPUT)])
    extract_tool_use(client, brief="b")
    call = client.calls[0]
    assert call["system"].startswith("You are a market-research")
    assert call["tools"][0]["name"] == TOOL_NAME
    assert "properties" in call["tools"][0]["input_schema"]
    assert call["tool_choice"] == {"type": "auto"}
    assert len(call["messages"]) == 1
    assert call["messages"][0]["role"] == "user"


def test_success_first_try():
    client = FakeClient([_tool_response(VALID_INPUT)])
    result = extract_tool_use(client, brief="b")
    assert result.technique == TECHNIQUE
    assert result.succeeded
    assert result.retries == 0
    assert result.parsed.id == "s1"
    assert result.attempts[0].error is None


def test_retry_on_validation_error_with_tool_result():
    client = FakeClient(
        [
            _tool_response(INVALID_INPUT, tool_use_id="toolu_a"),
            _tool_response(VALID_INPUT, tool_use_id="toolu_b"),
        ]
    )
    result = extract_tool_use(client, brief="b")
    assert result.succeeded
    assert result.retries == 1
    assert "Schema validation error" in result.attempts[0].error

    # Second call: assistant has tool_use, user has tool_result with matching id
    second = client.calls[1]
    msgs = second["messages"]
    assert len(msgs) == 3
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"][0]["type"] == "tool_use"
    assert msgs[1]["content"][0]["id"] == "toolu_a"
    assert msgs[2]["role"] == "user"
    assert msgs[2]["content"][0]["type"] == "tool_result"
    assert msgs[2]["content"][0]["tool_use_id"] == "toolu_a"
    assert msgs[2]["content"][0]["is_error"] is True


def test_model_responds_with_text_instead_of_tool():
    client = FakeClient(
        [
            FakeCompletion(text="Sure, here's the questionnaire as JSON..."),
            _tool_response(VALID_INPUT),
        ]
    )
    result = extract_tool_use(client, brief="b")
    assert result.succeeded
    assert "did not call" in result.attempts[0].error
    # Second user turn is a plain text nudge (no tool_use_id to reference)
    second_msgs = client.calls[1]["messages"]
    assert second_msgs[2]["role"] == "user"
    assert isinstance(second_msgs[2]["content"], str)
    assert TOOL_NAME in second_msgs[2]["content"]


def test_exhaust_retries():
    client = FakeClient([_tool_response(INVALID_INPUT)] * 4)
    result = extract_tool_use(client, brief="b", max_retries=3)
    assert not result.succeeded
    assert len(result.attempts) == 4
    assert result.retries == 3


def test_totals_aggregate():
    client = FakeClient(
        [_tool_response(INVALID_INPUT), _tool_response(VALID_INPUT)]
    )
    result = extract_tool_use(client, brief="b")
    assert result.total_input_tokens == 200
    assert result.total_output_tokens == 400
    assert result.total_latency_ms == 300
