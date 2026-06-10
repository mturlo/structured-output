"""Technique 2: tool use.

Define an `extract_questionnaire` tool whose input_schema is the Questionnaire
JSON schema. tool_choice=auto — model decides to call it. Parse the tool_use
block's input as the Questionnaire. On failure, send back a tool_result with
is_error=True and let the model retry.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from ..extraction import AttemptRecord, ExtractionResult
from ..provider import AnthropicClient, CompletionResult
from ..schema import Questionnaire

TECHNIQUE = "tool_use"
TOOL_NAME = "extract_questionnaire"

SYSTEM_PROMPT = (
    "You are a market-research questionnaire designer. Read the brief and call the "
    f"`{TOOL_NAME}` tool with a complete Questionnaire that matches the tool's input "
    "schema. Do not respond with prose."
)

USER_TEMPLATE = """Research brief:
<brief>
{brief}
</brief>

Call the {tool} tool with the extracted Questionnaire."""


def _tool_definition() -> dict[str, Any]:
    return {
        "name": TOOL_NAME,
        "description": "Submit the extracted market-research Questionnaire.",
        "input_schema": Questionnaire.model_json_schema(),
    }


def _tool_choice() -> dict[str, Any]:
    return {"type": "auto"}


def _parse_tool_call(
    response: CompletionResult,
) -> tuple[Questionnaire | None, str | None, str | None]:
    """Return (parsed, error, tool_use_id). tool_use_id is None if no tool was called."""
    matching = [c for c in response.tool_calls if c.name == TOOL_NAME]
    if not matching:
        return None, f"Model did not call {TOOL_NAME}. Got text instead.", None
    call = matching[0]
    try:
        return Questionnaire.model_validate(call.input), None, call.tool_use_id
    except ValidationError as e:
        return None, f"Schema validation error: {e.errors(include_url=False)}", call.tool_use_id


def extract_tool_use(
    client: AnthropicClient,
    brief: str,
    *,
    max_retries: int = 3,
) -> ExtractionResult:
    user_msg = USER_TEMPLATE.format(brief=brief, tool=TOOL_NAME)
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_msg}]
    tools = [_tool_definition()]
    tool_choice = _tool_choice()
    result = ExtractionResult(technique=TECHNIQUE, parsed=None)

    for attempt_idx in range(max_retries + 1):
        response = client.complete(
            messages, system=SYSTEM_PROMPT, tools=tools, tool_choice=tool_choice
        )
        parsed, error, tool_use_id = _parse_tool_call(response)
        result.attempts.append(
            AttemptRecord(
                text=response.text,
                error=error,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                latency_ms=response.latency_ms,
            )
        )
        if parsed is not None:
            result.parsed = parsed
            return result
        if attempt_idx == max_retries:
            break

        assistant_content: list[dict[str, Any]] = []
        if response.text:
            assistant_content.append({"type": "text", "text": response.text})
        for call in response.tool_calls:
            assistant_content.append(
                {
                    "type": "tool_use",
                    "id": call.tool_use_id,
                    "name": call.name,
                    "input": call.input,
                }
            )
        messages.append({"role": "assistant", "content": assistant_content})

        if tool_use_id is not None:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "is_error": True,
                            "content": (
                                f"Your tool input failed validation:\n{error}\n\n"
                                f"Call {TOOL_NAME} again with corrected input."
                            ),
                        }
                    ],
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"You must call the {TOOL_NAME} tool. Do not respond with text. "
                        "Call it now with the extracted Questionnaire."
                    ),
                }
            )

    return result
