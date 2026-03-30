import json
import tempfile
from pathlib import Path

import pytest

from candidate_screening_system import (
    InputLoader,
    RuleBasedEvaluator,
    CandidateScreeningSystem,
    InterviewInput,
    InterviewTurn,
)


def test_load_valid_json():
    """Проверяет загрузку корректного JSON."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({
            "candidate_id": "test",
            "vacancy_title": "Test",
            "vacancy_description": "Description",
            "position_level": "junior",
            "interview_turns": [
                {"question": "Q?", "answer": "A"}
            ],
            "metadata": {}
        }, f)
        f.flush()
        interview = InputLoader.load_json(f.name)
    assert interview.candidate_id == "test"
    assert len(interview.interview_turns) == 1
    Path(f.name).unlink()


def test_empty_interview():
    """Пустое интервью должно вызывать исключение."""
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
    with pytest.raises(ValueError, match="пустое"):
        system.run(interview)


def test_rule_based_on_demo():
    """Запуск rule‑based evaluator на демо-данных (встроенных в код)."""
    # Создаём интервью аналогичное демо
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
                answer="Последние полтора года я работал администратором..."
            ),
            InterviewTurn(
                question="Почему вас заинтересовала эта вакансия?",
                answer="Мне интересна работа с клиентами и развитие в продажах."
            )
        ]
    )
    evaluator = RuleBasedEvaluator()
    evaluation = evaluator.evaluate(demo)
    assert 0 <= evaluation.score <= 100
    assert evaluation.decision in ("рекомендован", "условно рекомендован", "не рекомендован")
    assert len(evaluation.skill_signals) == 5  # пять критериев


def test_report_generation(tmp_path):
    """Проверяет, что отчёты создаются без ошибок."""
    from candidate_screening_system import ReportGenerator, CandidateEvaluation

    # Минимальная валидная оценка
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
    gen = ReportGenerator()
    json_path = tmp_path / "report.json"
    txt_path = tmp_path / "report.txt"
    # Используем CandidateScreeningSystem для сохранения (он использует ReportGenerator)
    system = CandidateScreeningSystem(RuleBasedEvaluator())
    system.save_outputs(eval_obj, tmp_path)  # сохраняет с временной меткой, но проверим наличие
    # Проверим, что создано хотя бы два файла
    files = list(tmp_path.glob("*"))
    assert any(f.suffix == ".json" for f in files)
    assert any(f.suffix == ".txt" for f in files)
