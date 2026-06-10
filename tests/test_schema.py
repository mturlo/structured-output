import pytest
from pydantic import TypeAdapter, ValidationError

from structured_output.schema import (
    END,
    GridQuestion,
    MultiChoiceQuestion,
    OpenQuestion,
    Question,
    Questionnaire,
    ResponseOption,
    RoutingRule,
    SingleChoiceQuestion,
)


def opt(code: str, label: str | None = None) -> ResponseOption:
    return ResponseOption(code=code, label=label or f"Label {code}")


# ---------- SingleChoice ----------

def test_single_choice_ok():
    q = SingleChoiceQuestion(id="q1", text="Pick one", options=[opt("a"), opt("b")])
    assert q.type == "single_choice"
    assert len(q.options) == 2


def test_single_choice_needs_two_options():
    with pytest.raises(ValidationError):
        SingleChoiceQuestion(id="q1", text="x", options=[opt("a")])


def test_single_choice_duplicate_codes():
    with pytest.raises(ValidationError, match="Duplicate option codes"):
        SingleChoiceQuestion(id="q1", text="x", options=[opt("a"), opt("a", "B")])


# ---------- MultiChoice ----------

def test_multi_choice_ok():
    q = MultiChoiceQuestion(
        id="q1", text="Pick many",
        options=[opt("a"), opt("b"), opt("c")],
        min_selections=1, max_selections=2,
    )
    assert q.max_selections == 2


def test_multi_choice_bounds_violated():
    with pytest.raises(ValidationError, match="min_selections > max_selections"):
        MultiChoiceQuestion(
            id="q1", text="x",
            options=[opt("a"), opt("b")],
            min_selections=2, max_selections=1,
        )


def test_multi_choice_max_exceeds_options():
    with pytest.raises(ValidationError, match="max_selections exceeds option count"):
        MultiChoiceQuestion(
            id="q1", text="x",
            options=[opt("a"), opt("b")],
            max_selections=5,
        )


# ---------- Grid ----------

def test_grid_ok():
    q = GridQuestion(
        id="q1", text="Rate items",
        rows=[opt("r1"), opt("r2")],
        columns=[opt("c1"), opt("c2"), opt("c3")],
    )
    assert q.type == "grid"


def test_grid_duplicate_row_codes():
    with pytest.raises(ValidationError, match="Duplicate rows codes"):
        GridQuestion(
            id="q1", text="x",
            rows=[opt("r"), opt("r", "Other")],
            columns=[opt("c")],
        )


# ---------- Open ----------

def test_open_ok():
    q = OpenQuestion(id="q1", text="Why?", max_length=500)
    assert q.max_length == 500


def test_open_no_max_length():
    q = OpenQuestion(id="q1", text="Why?")
    assert q.max_length is None


# ---------- Discriminated union ----------

def test_discriminated_union_parse():
    adapter = TypeAdapter(Question)
    parsed = adapter.validate_python(
        {"type": "open", "id": "q1", "text": "Why?", "required": False}
    )
    assert isinstance(parsed, OpenQuestion)
    assert parsed.required is False


def test_discriminated_union_unknown_type():
    adapter = TypeAdapter(Question)
    with pytest.raises(ValidationError):
        adapter.validate_python({"type": "ranking", "id": "q1", "text": "x"})


# ---------- Questionnaire ----------

def _q_open(qid: str) -> OpenQuestion:
    return OpenQuestion(id=qid, text=f"Q {qid}")


def test_questionnaire_ok():
    q = Questionnaire(
        id="study-1", title="Test", questions=[_q_open("q1"), _q_open("q2")],
        routing=[RoutingRule(from_question_id="q1", condition="a", to_question_id="q2")],
    )
    assert len(q.questions) == 2


def test_questionnaire_duplicate_question_ids():
    with pytest.raises(ValidationError, match="Duplicate question ids"):
        Questionnaire(
            id="s", title="t",
            questions=[_q_open("q1"), _q_open("q1")],
        )


def test_questionnaire_routing_unknown_from():
    with pytest.raises(ValidationError, match="Routing from unknown question"):
        Questionnaire(
            id="s", title="t", questions=[_q_open("q1")],
            routing=[RoutingRule(from_question_id="qX", condition="a", to_question_id="q1")],
        )


def test_questionnaire_routing_unknown_to():
    with pytest.raises(ValidationError, match="Routing to unknown question"):
        Questionnaire(
            id="s", title="t", questions=[_q_open("q1")],
            routing=[RoutingRule(from_question_id="q1", condition="a", to_question_id="qX")],
        )


def test_questionnaire_routing_to_end_ok():
    q = Questionnaire(
        id="s", title="t", questions=[_q_open("q1")],
        routing=[RoutingRule(from_question_id="q1", condition="a", to_question_id=END)],
    )
    assert q.routing[0].to_question_id == END


def test_questionnaire_extra_field_forbidden():
    with pytest.raises(ValidationError):
        Questionnaire.model_validate(
            {"id": "s", "title": "t", "questions": [
                {"type": "open", "id": "q1", "text": "x"}
            ], "junk": 1}
        )


# ---------- JSON schema export ----------

def test_questionnaire_json_schema_generates():
    schema = Questionnaire.model_json_schema()
    assert schema["title"] == "Questionnaire"
    assert "questions" in schema["properties"]
