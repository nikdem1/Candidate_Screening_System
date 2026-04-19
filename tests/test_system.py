import json
import tempfile
from pathlib import Path

import pytest

from candidate_screening_system import (
    CandidateScreeningSystem,
    InputLoader,
    InterviewInput,
    InterviewTurn,
    RuleBasedEvaluator,
)


def test_load_valid_json() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(
            {
                "candidate_id": "test",
                "vacancy_title": "Test",
                "vacancy_description": "Description",
                "position_level": "junior",
                "interview_turns": [{"question": "Q?", "answer": "A"}],
                "metadata": {},
            },
            f,
            ensure_ascii=False,
        )
        f.flush()
        interview = InputLoader.load_json(f.name)
    assert interview.candidate_id == "test"
    assert len(interview.interview_turns) == 1
    Path(f.name).unlink()


def test_empty_interview() -> None:
    interview = InterviewInput(
        candidate_id="test",
        vacancy_title="Test",
        vacancy_description="Desc",
        position_level="junior",
        interview_turns=[],
        transcript="",
    )
    evaluator = RuleBasedEvaluator()
    system = CandidateScreeningSystem(evaluator)
    with pytest.raises(ValueError):
        system.run(interview)


def test_rule_based_on_demo() -> None:
    demo = InterviewInput(
        candidate_id="demo",
        vacancy_title="Менеджер по работе с клиентами",
        vacancy_description=(
            "Компания ищет менеджера по работе с клиентами начального уровня. "
            "Важны грамотная устная речь, работа с возражениями, клиентский сервис."
        ),
        position_level="junior",
        interview_turns=[
            InterviewTurn(
                question="Расскажите о себе и опыте работы.",
                answer="Последние полтора года я работал администратором в сервисной компании.",
            ),
            InterviewTurn(
                question="Почему вас заинтересовала эта вакансия?",
                answer="Мне интересна работа с клиентами и развитие в продажах.",
            ),
        ],
    )
    evaluator = RuleBasedEvaluator()
    evaluation = evaluator.evaluate(demo)
    assert 0 <= evaluation.score <= 100
    assert isinstance(evaluation.decision, str)
    assert evaluation.decision.strip()
    assert len(evaluation.skill_signals) == 5


def test_report_generation(tmp_path: Path) -> None:
    from candidate_screening_system import CandidateEvaluation

    eval_obj = CandidateEvaluation(
        candidate_id="test",
        vacancy_title="Test",
        score=70.0,
        decision="условно рекомендован",
        confidence=0.7,
        strengths=["s1"],
        weaknesses=["w1"],
        risks=["r1"],
        recommendation="rec",
        explanation="expl",
        skill_signals=[],
        question_scores=[],
        evaluator_name="test",
    )
    system = CandidateScreeningSystem(RuleBasedEvaluator())
    system.save_outputs(eval_obj, tmp_path)
    files = list(tmp_path.glob("*"))
    assert any(f.suffix == ".json" for f in files)
    assert any(f.suffix == ".txt" for f in files)
