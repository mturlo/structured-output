# structured-output

Project 03 of [ai-dojo](https://github.com/mturlo/ai-dojo). Compare three structured-output techniques on a market-research **questionnaire extraction** task.

## Techniques

1. **Prompted JSON** — plain prompt + schema dump, parse + retry
2. **Tool use** — Anthropic `tools` / OpenAI function calling
3. **Native `response_format`** — OpenAI `json_schema` strict; Anthropic falls back to tool-use-as-json

## Stack

Python 3.14 · uv · Pydantic v2 · instructor · anthropic · openai · pytest · rich

Models: `claude-haiku-4-5-20251001`, `gpt-4o-mini`.

## Layout

```
src/structured_output/   # package
tests/                   # pytest
fixtures/briefs/         # hand-written research briefs (*.md)
runs/                    # sweep artifacts (JSON, gitignored)
notes/                   # technique comparison writeup
```

## Setup

```sh
uv sync
cp .env.example .env  # fill ANTHROPIC_API_KEY, OPENAI_API_KEY
```

## Usage

```sh
uv run extract --brief fixtures/briefs/01.md --technique tool_use --provider anthropic
uv run sweep --briefs fixtures/briefs/ --out runs/
```

## Success criteria

See [project brief](https://github.com/mturlo/ai-dojo/blob/main/projects/03-structured-output.md).
