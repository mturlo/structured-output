"""Tests for technique 1 (prompted JSON). Uses a fake client — no API."""

from __future__ import annotations

import json
from dataclasses import dataclass

from structured_output.extraction import ExtractionResult
from structured_output.techniques.prompted_json import (
    TECHNIQUE,
    _strip_to_json,
    extract_prompted_json,
)

# ---------- _strip_to_json ----------

def test_strip_to_json_bare():
    assert _strip_to_json('{"a": 1}') == '{"a": 1}'


def test_strip_to_json_with_fences():
    text = '```json\n{"a": 1}\n```'
    assert _strip_to_json(text) == '{"a": 1}'


def test_strip_to_json_with_preamble():
    text = 'Here you go:\n{"a": 1}\nlet me know'
    assert _strip_to_json(text) == '{"a": 1}'


def test_strip_to_json_nested():
    payload = '{"a": {"b": 1}}'
    assert _strip_to_json(f"```\n{payload}\n```") == payload


# ---------- fake client ----------

@dataclass
class FakeCompletion:
    text: str
    input_tokens: int = 100
    output_tokens: int = 200
    latency_ms: int = 150
    tool_calls: list = None
    stop_reason: str = "end_turn"
    raw: object = None


class FakeClient:
    """Returns scripted responses in order. Records calls."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, messages, *, system=None, **kwargs):
        self.calls.append({"messages": messages, "system": system, **kwargs})
        if not self._responses:
            raise AssertionError("FakeClient ran out of scripted responses")
        return FakeCompletion(text=self._responses.pop(0))


# ---------- happy path ----------

VALID_JSON = json.dumps(
    {
        "id": "s1",
        "title": "Test",
        "questions": [{"type": "open", "id": "q1", "text": "Why?"}],
        "routing": [],
    }
)


def test_extract_first_try_success():
    client = FakeClient([VALID_JSON])
    result = extract_prompted_json(client, brief="some brief")
    assert isinstance(result, ExtractionResult)
    assert result.technique == TECHNIQUE
    assert result.succeeded
    assert result.retries == 0
    assert result.parsed.id == "s1"
    assert len(client.calls) == 1
    assert client.calls[0]["system"].startswith("You are a market-research")


def test_first_call_has_no_assistant_turn():
    client = FakeClient([VALID_JSON])
    extract_prompted_json(client, brief="b")
    msgs = client.calls[0]["messages"]
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert "Research brief" in msgs[0]["content"]
    assert "Questionnaire" in msgs[0]["content"]  # schema embedded


# ---------- retry on parse error ----------

def test_retry_on_invalid_json_then_success():
    client = FakeClient(["not json at all", VALID_JSON])
    result = extract_prompted_json(client, brief="b", max_retries=3)
    assert result.succeeded
    assert result.retries == 1
    assert "JSON parse error" in result.attempts[0].error
    assert result.attempts[1].error is None
    # second call should include the assistant turn + retry user turn
    second_msgs = client.calls[1]["messages"]
    assert len(second_msgs) == 3
    assert second_msgs[1]["role"] == "assistant"
    assert second_msgs[2]["role"] == "user"
    assert "failed validation" in second_msgs[2]["content"]


# ---------- retry on validation error ----------

INVALID_SEMANTIC = json.dumps(
    {
        "id": "s1",
        "title": "Test",
        "questions": [
            {"type": "open", "id": "q1", "text": "Why?"},
            {"type": "open", "id": "q1", "text": "Dup id"},
        ],
        "routing": [],
    }
)


def test_retry_on_validation_error_then_success():
    client = FakeClient([INVALID_SEMANTIC, VALID_JSON])
    result = extract_prompted_json(client, brief="b")
    assert result.succeeded
    assert result.retries == 1
    assert "Schema validation error" in result.attempts[0].error


# ---------- exhaustion ----------

def test_exhaust_retries_returns_failure():
    client = FakeClient(["bad"] * 4)  # 1 initial + 3 retries
    result = extract_prompted_json(client, brief="b", max_retries=3)
    assert not result.succeeded
    assert result.parsed is None
    assert len(result.attempts) == 4
    assert result.retries == 3
    assert all(a.error is not None for a in result.attempts)
    assert len(result.validation_errors) == 4


def test_no_retries_when_max_retries_zero():
    client = FakeClient(["bad"])
    result = extract_prompted_json(client, brief="b", max_retries=0)
    assert not result.succeeded
    assert len(result.attempts) == 1


def test_totals_aggregate_across_attempts():
    client = FakeClient(["bad", VALID_JSON])
    result = extract_prompted_json(client, brief="b")
    assert result.total_input_tokens == 200
    assert result.total_output_tokens == 400
    assert result.total_latency_ms == 300


# ---------- model output with markdown fences ----------

def test_handles_fenced_response():
    fenced = f"```json\n{VALID_JSON}\n```"
    client = FakeClient([fenced])
    result = extract_prompted_json(client, brief="b")
    assert result.succeeded
