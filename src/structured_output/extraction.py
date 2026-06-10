"""Shared types for technique results."""

from __future__ import annotations

from dataclasses import dataclass, field

from .schema import Questionnaire


@dataclass
class AttemptRecord:
    text: str
    error: str | None
    input_tokens: int
    output_tokens: int
    latency_ms: int


@dataclass
class ExtractionResult:
    technique: str
    parsed: Questionnaire | None
    attempts: list[AttemptRecord] = field(default_factory=list)

    @property
    def retries(self) -> int:
        """Retries beyond the first attempt. 0 = success on first try."""
        return max(0, len(self.attempts) - 1)

    @property
    def succeeded(self) -> bool:
        return self.parsed is not None

    @property
    def total_input_tokens(self) -> int:
        return sum(a.input_tokens for a in self.attempts)

    @property
    def total_output_tokens(self) -> int:
        return sum(a.output_tokens for a in self.attempts)

    @property
    def total_latency_ms(self) -> int:
        return sum(a.latency_ms for a in self.attempts)

    @property
    def validation_errors(self) -> list[str]:
        return [a.error for a in self.attempts if a.error is not None]
