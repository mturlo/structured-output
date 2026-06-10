# structured-output

Project 03 of [ai-dojo](https://github.com/mturlo/ai-dojo). Compare three structured-output techniques on a market-research **questionnaire extraction** task.

## Techniques

1. **Prompted JSON** — plain prompt + schema dump, parse + retry
2. **Tool use** — Anthropic `tools` with `tool_choice`, parse from `tool_use` block
3. **Strict tool use** — Anthropic tool with full JSON schema + forced `tool_choice`, the closest analogue to OpenAI's strict `json_schema` response_format

## Stack

Python 3.14 · uv · Pydantic v2 · instructor · anthropic · pytest · rich

Model: `claude-haiku-4-5-20251001` (Anthropic only — OpenAI dropped).

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
cp .env.example .env  # fill ANTHROPIC_API_KEY
```

## Usage

```sh
uv run extract --brief fixtures/briefs/01-coffee-subscription.md --technique tool_use
uv run sweep --briefs fixtures/briefs/ --out runs/
```

## Success criteria

See [project brief](https://github.com/mturlo/ai-dojo/blob/main/projects/03-structured-output.md).
