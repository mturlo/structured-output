"""Questionnaire domain schema. Pydantic v2 discriminated union over question types."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

END = "END"


class ResponseOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1, description="Stable code, unique within question.")
    label: str = Field(min_length=1)
    exclusive: bool = False


class _QuestionBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    required: bool = True


class SingleChoiceQuestion(_QuestionBase):
    type: Literal["single_choice"] = "single_choice"
    options: list[ResponseOption] = Field(min_length=2)

    @field_validator("options")
    @classmethod
    def _unique_codes(cls, v: list[ResponseOption]) -> list[ResponseOption]:
        codes = [o.code for o in v]
        if len(set(codes)) != len(codes):
            raise ValueError("Duplicate option codes")
        return v


class MultiChoiceQuestion(_QuestionBase):
    type: Literal["multi_choice"] = "multi_choice"
    options: list[ResponseOption] = Field(min_length=2)
    min_selections: int = Field(default=1, ge=0)
    max_selections: int | None = Field(default=None, ge=1)

    @field_validator("options")
    @classmethod
    def _unique_codes(cls, v: list[ResponseOption]) -> list[ResponseOption]:
        codes = [o.code for o in v]
        if len(set(codes)) != len(codes):
            raise ValueError("Duplicate option codes")
        return v

    @model_validator(mode="after")
    def _check_selection_bounds(self) -> MultiChoiceQuestion:
        if self.max_selections is not None:
            if self.min_selections > self.max_selections:
                raise ValueError("min_selections > max_selections")
            if self.max_selections > len(self.options):
                raise ValueError("max_selections exceeds option count")
        return self


class GridQuestion(_QuestionBase):
    type: Literal["grid"] = "grid"
    rows: list[ResponseOption] = Field(min_length=1)
    columns: list[ResponseOption] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_codes(self) -> GridQuestion:
        for label, items in (("rows", self.rows), ("columns", self.columns)):
            codes = [o.code for o in items]
            if len(set(codes)) != len(codes):
                raise ValueError(f"Duplicate {label} codes")
        return self


class OpenQuestion(_QuestionBase):
    type: Literal["open"] = "open"
    max_length: int | None = Field(default=None, ge=1)


Question = Annotated[
    SingleChoiceQuestion | MultiChoiceQuestion | GridQuestion | OpenQuestion,
    Field(discriminator="type"),
]


class RoutingRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    from_question_id: str = Field(min_length=1)
    condition: str = Field(
        min_length=1,
        description="Option code or simple expression that triggers the jump.",
    )
    to_question_id: str = Field(
        min_length=1, description="Target question id, or 'END' to terminate."
    )


class Questionnaire(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    questions: list[Question] = Field(min_length=1)
    routing: list[RoutingRule] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_consistency(self) -> Questionnaire:
        ids = [q.id for q in self.questions]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate question ids")
        id_set = set(ids)
        for rule in self.routing:
            if rule.from_question_id not in id_set:
                raise ValueError(f"Routing from unknown question {rule.from_question_id!r}")
            if rule.to_question_id != END and rule.to_question_id not in id_set:
                raise ValueError(f"Routing to unknown question {rule.to_question_id!r}")
        return self
