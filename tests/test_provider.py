"""Test provider wrapper without hitting the API."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from structured_output.config import Settings
from structured_output.provider import AnthropicClient, parse_response


def _fake_response(content_blocks, stop_reason="end_turn", input_tokens=10, output_tokens=20):
    return SimpleNamespace(
        content=content_blocks,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def test_parse_response_text_only():
    response = _fake_response(
        [SimpleNamespace(type="text", text="Hello world")],
    )
    result = parse_response(response, latency_ms=42)
    assert result.text == "Hello world"
    assert result.tool_calls == []
    assert result.stop_reason == "end_turn"
    assert result.input_tokens == 10
    assert result.output_tokens == 20
    assert result.latency_ms == 42


def test_parse_response_tool_use():
    response = _fake_response(
        [
            SimpleNamespace(type="text", text="Calling tool..."),
            SimpleNamespace(
                type="tool_use",
                name="extract_questionnaire",
                id="toolu_123",
                input={"title": "Test"},
            ),
        ],
        stop_reason="tool_use",
    )
    result = parse_response(response, latency_ms=99)
    assert result.text == "Calling tool..."
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.name == "extract_questionnaire"
    assert call.input == {"title": "Test"}
    assert call.tool_use_id == "toolu_123"
    assert result.stop_reason == "tool_use"


def test_parse_response_multiple_text_blocks_concatenate():
    response = _fake_response(
        [
            SimpleNamespace(type="text", text="Part 1. "),
            SimpleNamespace(type="text", text="Part 2."),
        ],
    )
    result = parse_response(response, latency_ms=1)
    assert result.text == "Part 1. Part 2."


def test_client_passes_args_to_sdk():
    sdk = MagicMock()
    sdk.messages.create.return_value = _fake_response(
        [SimpleNamespace(type="text", text="ok")]
    )
    settings = Settings(anthropic_api_key="sk-test", model="claude-x", max_tokens=500)
    client = AnthropicClient(settings, client=sdk)

    result = client.complete(
        [{"role": "user", "content": "hi"}],
        system="you are helpful",
        tools=[{"name": "t", "input_schema": {"type": "object"}}],
        tool_choice={"type": "tool", "name": "t"},
    )

    sdk.messages.create.assert_called_once()
    kwargs = sdk.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-x"
    assert kwargs["max_tokens"] == 500
    assert kwargs["system"] == "you are helpful"
    assert kwargs["tools"][0]["name"] == "t"
    assert kwargs["tool_choice"]["name"] == "t"
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert result.text == "ok"


def test_client_omits_optional_args_when_none():
    sdk = MagicMock()
    sdk.messages.create.return_value = _fake_response(
        [SimpleNamespace(type="text", text="ok")]
    )
    settings = Settings(anthropic_api_key="sk-test")
    client = AnthropicClient(settings, client=sdk)
    client.complete([{"role": "user", "content": "hi"}])

    kwargs = sdk.messages.create.call_args.kwargs
    assert "system" not in kwargs
    assert "tools" not in kwargs
    assert "tool_choice" not in kwargs
