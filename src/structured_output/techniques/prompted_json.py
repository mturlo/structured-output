"""Technique 1: prompted JSON.

Plain prompt with embedded JSON schema. Model returns text. We parse + validate.
On failure, feed the error back as a follow-up user turn and retry, capped.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from ..extraction import AttemptRecord, ExtractionResult
from ..provider import AnthropicClient
from ..schema import Questionnaire

TECHNIQUE = "prompted_json"

SYSTEM_PROMPT = """You are a market-research questionnaire designer. Read the research \
brief and output a complete Questionnaire as a single JSON object that matches the \
provided JSON schema exactly. Output ONLY the JSON object — no prose, no markdown \
fences, no explanation."""

USER_TEMPLATE = """Research brief:
<brief>
{brief}
</brief>

Target JSON schema:
<schema>
{schema}
</schema>

Return one JSON object that validates against the schema. Output JSON only."""


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _strip_to_json(text: str) -> str:
    """Best-effort extraction of a JSON object from model output."""
    text = text.strip()
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _parse_and_validate(text: str) -> tuple[Questionnaire | None, str | None]:
    payload = _strip_to_json(text)
    try:
        data: Any = json.loads(payload)
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e.msg} at line {e.lineno} col {e.colno}"
    try:
        return Questionnaire.model_validate(data), None
    except ValidationError as e:
        return None, f"Schema validation error: {e.errors(include_url=False)}"


def extract_prompted_json(
    client: AnthropicClient,
    brief: str,
    *,
    max_retries: int = 3,
) -> ExtractionResult:
    """Run prompted-JSON extraction with up to max_retries follow-ups (total = 1 + retries)."""
    schema_json = json.dumps(Questionnaire.model_json_schema(), indent=2)
    user_msg = USER_TEMPLATE.format(brief=brief, schema=schema_json)
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_msg}]
    result = ExtractionResult(technique=TECHNIQUE, parsed=None)

    for attempt_idx in range(max_retries + 1):
        response = client.complete(messages, system=SYSTEM_PROMPT)
        parsed, error = _parse_and_validate(response.text)
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
        messages.append({"role": "assistant", "content": response.text})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Your previous response failed validation:\n{error}\n\n"
                    "Return a corrected JSON object that validates against the schema. "
                    "Output JSON only."
                ),
            }
        )

    return result
