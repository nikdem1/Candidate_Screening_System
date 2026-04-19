#!/usr/bin/env python

"""
Prototype of a candidate screening system for audio interview analysis.

What this script can do:
1. Accept text transcripts or question-answer interview structures.
2. Evaluate a candidate against a vacancy description.
3. Produce a structured HR report in JSON and TXT formats.
4. Optionally call an external Yandex-compatible LLM endpoint if credentials are provided.
5. Work fully offline with a deterministic rule-based evaluator.

This is designed as a defendable academic prototype: the program is complete,
runnable, and testable, but it does not pretend to be a production-grade
neural network trained on a large proprietary dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import textwrap
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from urllib import error, request


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger("candidate_screening_system")


def setup_console_output() -> None:
    """Configure console streams to avoid encoding crashes on Windows terminals."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                # Non-critical: keep default encoding if stream can't be reconfigured.
                pass


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class InterviewTurn:
    question: str
    answer: str


@dataclass
class InterviewInput:
    candidate_id: str
    vacancy_title: str
    vacancy_description: str
    position_level: str
    interview_turns: List[InterviewTurn] = field(default_factory=list)
    transcript: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def merged_transcript(self) -> str:
        if self.transcript.strip():
            return self.transcript.strip()
        blocks: List[str] = []
        for idx, turn in enumerate(self.interview_turns, start=1):
            blocks.append(f"Вопрос {idx}: {turn.question.strip()}")
            blocks.append(f"Ответ {idx}: {turn.answer.strip()}")
        return "\n".join(blocks).strip()


@dataclass
class SkillSignal:
    name: str
    score: float
    evidence: List[str]
    rationale: str


@dataclass
class CandidateEvaluation:
    candidate_id: str
    vacancy_title: str
    score: float
    decision: str
    confidence: float
    strengths: List[str]
    weaknesses: List[str]
    risks: List[str]
    recommendation: str
    explanation: str
    skill_signals: List[SkillSignal]
    question_scores: List[Dict[str, Any]]
    evaluator_name: str


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

STOPWORDS_RU = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а",
    "то", "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же",
    "вы", "за", "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня",
    "еще", "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг",
    "ли", "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас",
    "нибудь", "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего",
    "ей", "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы",
    "тебя", "их", "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз",
    "тоже", "себе", "под", "будет", "ж", "тогда", "кто", "этот", "того", "потому",
    "этого", "какой", "совсем", "ним", "здесь", "этом", "один", "почти", "мой",
    "тем", "чтобы", "нее", "сейчас", "были", "куда", "зачем", "всех", "можно",
    "при", "наконец", "два", "об", "другой", "хоть", "после", "над", "больше",
    "тот", "через", "эти", "нас", "про", "всего", "них", "какая", "много",
}

FILLERS_RU = {
    "ээ", "эм", "ну", "как бы", "типа", "короче", "в общем", "наверное", "может быть"
}

POSITIVE_MARKERS = {
    "опыт", "ответственность", "клиент", "команда", "проект", "результат", "улучш",
    "решил", "достиг", "продаж", "crm", "аналит", "обуч", "работал", "добился",
    "коммуника", "обратн", "конфликт", "задач", "план", "цель", "мотивац",
}

NEGATIVE_MARKERS = {
    "не знаю", "не умею", "сложно", "не работал", "не сталкивался", "не уверен",
    "затрудняюсь", "не помню", "без опыта",
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def sentence_split(text: str) -> List[str]:
    pieces = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in pieces if p.strip()]


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-zА-Яа-яЁё0-9+#-]+", text.lower())


def extract_keywords(text: str, max_keywords: int = 20) -> List[str]:
    freq: Dict[str, int] = {}
    for token in tokenize(text):
        if len(token) < 3:
            continue
        if token in STOPWORDS_RU:
            continue
        freq[token] = freq.get(token, 0) + 1
    ranked = sorted(freq.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _ in ranked[:max_keywords]]


def find_sentences_with_keywords(text: str, keywords: Iterable[str], limit: int = 3) -> List[str]:
    kws = [k.lower() for k in keywords if k]
    results: List[str] = []
    for sent in sentence_split(text):
        lowered = sent.lower()
        if any(k in lowered for k in kws):
            results.append(sent)
        if len(results) >= limit:
            break
    return results


def safe_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use UTF-8 with BOM so Windows editors reliably detect Cyrillic.
    path.write_text(content, encoding="utf-8-sig")


def safe_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

class InputLoader:
    """Loads interview payloads from JSON files."""

    @staticmethod
    def load_json(path: str | Path) -> InterviewInput:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        with file_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        interview_turns = [
            InterviewTurn(question=item["question"], answer=item["answer"])
            for item in raw.get("interview_turns", [])
        ]

        return InterviewInput(
            candidate_id=raw["candidate_id"],
            vacancy_title=raw.get("vacancy_title", "Не указано"),
            vacancy_description=raw["vacancy_description"],
            position_level=raw.get("position_level", "junior"),
            interview_turns=interview_turns,
            transcript=raw.get("transcript", ""),
            metadata=raw.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Transcript processing
# ---------------------------------------------------------------------------

class TranscriptProcessor:
    def validate(self, interview: InterviewInput) -> None:
        text = interview.merged_transcript()
        if not text:
            raise ValueError("Интервью пустое: отсутствует transcript и interview_turns")
        if len(text) < 50:
            raise ValueError("Текст интервью слишком короткий для анализа")

    def summarize_transcript(self, interview: InterviewInput) -> Dict[str, Any]:
        text = interview.merged_transcript()
        tokens = tokenize(text)
        sentences = sentence_split(text)
        keywords = extract_keywords(text, max_keywords=15)

        filler_count = 0
        lowered = text.lower()
        for filler in FILLERS_RU:
            filler_count += lowered.count(filler)

        return {
            "chars": len(text),
            "words": len(tokens),
            "sentences": len(sentences),
            "keywords": keywords,
            "filler_count": filler_count,
            "avg_sentence_length": round((len(tokens) / max(len(sentences), 1)), 2),
        }


# ---------------------------------------------------------------------------
# Vacancy analysis
# ---------------------------------------------------------------------------

class VacancyAnalyzer:
    def extract_requirements(self, vacancy_title: str, vacancy_description: str) -> Dict[str, Any]:
        text = f"{vacancy_title}. {vacancy_description}"
        keywords = extract_keywords(text, max_keywords=25)
        grouped = {
            "communication": [k for k in keywords if k.startswith(("комм", "клиен", "перег", "през"))],
            "sales": [k for k in keywords if k.startswith(("прод", "crm", "ворон", "сделк", "план"))],
            "analytics": [k for k in keywords if k.startswith(("анал", "данн", "отчет", "метрик"))],
            "teamwork": [k for k in keywords if k.startswith(("команд", "сотруд", "взаим"))],
            "stress": [k for k in keywords if k.startswith(("стресс", "конфликт", "нагруз"))],
        }
        return {
            "keywords": keywords,
            "grouped": grouped,
        }


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

class BaseEvaluator:
    name = "base"

    def evaluate(self, interview: InterviewInput) -> CandidateEvaluation:
        raise NotImplementedError


class RuleBasedEvaluator(BaseEvaluator):
    name = "rule_based_v1"

    def __init__(self) -> None:
        self.processor = TranscriptProcessor()
        self.vacancy_analyzer = VacancyAnalyzer()

    def evaluate(self, interview: InterviewInput) -> CandidateEvaluation:
        self.processor.validate(interview)
        transcript = interview.merged_transcript()
        transcript_info = self.processor.summarize_transcript(interview)
        vacancy_info = self.vacancy_analyzer.extract_requirements(
            interview.vacancy_title,
            interview.vacancy_description,
        )

        communication = self._score_communication(transcript, transcript_info)
        relevance = self._score_relevance(transcript, vacancy_info)
        experience = self._score_experience(transcript)
        structure = self._score_structure(transcript_info)
        motivation = self._score_motivation(transcript)

        weighted_score = (
            communication.score * 0.24
            + relevance.score * 0.28
            + experience.score * 0.20
            + structure.score * 0.14
            + motivation.score * 0.14
        )

        question_scores = self._score_by_question(interview)
        score = round(clamp(weighted_score, 0.0, 100.0), 2)
        decision = self._decision_from_score(score)
        confidence = round(self._confidence_from_input(transcript_info, question_scores), 2)

        strengths, weaknesses, risks = self._build_findings(
            communication,
            relevance,
            experience,
            structure,
            motivation,
            question_scores,
        )

        explanation = self._build_explanation(
            interview=interview,
            score=score,
            decision=decision,
            communication=communication,
            relevance=relevance,
            experience=experience,
            structure=structure,
            motivation=motivation,
        )
        recommendation = self._build_recommendation(score, strengths, risks)

        return CandidateEvaluation(
            candidate_id=interview.candidate_id,
            vacancy_title=interview.vacancy_title,
            score=score,
            decision=decision,
            confidence=confidence,
            strengths=strengths,
            weaknesses=weaknesses,
            risks=risks,
            recommendation=recommendation,
            explanation=explanation,
            skill_signals=[communication, relevance, experience, structure, motivation],
            question_scores=question_scores,
            evaluator_name=self.name,
        )

    def _score_communication(self, transcript: str, info: Dict[str, Any]) -> SkillSignal:
        score = 55.0
        avg_sentence = float(info["avg_sentence_length"])
        filler_count = int(info["filler_count"])
        if 8 <= avg_sentence <= 22:
            score += 12
        elif avg_sentence < 5:
            score -= 10
        else:
            score += 4
        score -= min(filler_count * 2.5, 20)
        if any(word in transcript.lower() for word in ["клиент", "команда", "объясня", "договар", "обратная связь"]):
            score += 10
        evidence = find_sentences_with_keywords(transcript, ["клиент", "команда", "объяс", "договар"], limit=3)
        rationale = (
            "Оценка основана на длине и связности фраз, количестве речевых паразитов и наличии маркеров коммуникации."
        )
        return SkillSignal("Коммуникация", round(clamp(score, 0, 100), 2), evidence, rationale)

    def _score_relevance(self, transcript: str, vacancy_info: Dict[str, Any]) -> SkillSignal:
        score = 45.0
        transcript_tokens = set(tokenize(transcript))
        vacancy_keywords = vacancy_info["keywords"]
        overlap = [kw for kw in vacancy_keywords if kw in transcript_tokens]
        score += min(len(overlap) * 4.2, 40)
        if any(marker in transcript.lower() for marker in ["crm", "план продаж", "работа с клиентами", "возражени"]):
            score += 10
        evidence = find_sentences_with_keywords(transcript, overlap[:5], limit=3)
        rationale = (
            "Оценка показывает, насколько ответы кандидата содержательно пересекаются с требованиями вакансии."
        )
        return SkillSignal("Релевантность вакансии", round(clamp(score, 0, 100), 2), evidence, rationale)

    def _score_experience(self, transcript: str) -> SkillSignal:
        score = 40.0
        lowered = transcript.lower()
        patterns = [
            r"\b(\d+)\s+(?:год|года|лет|месяц|месяца|месяцев)\b",
            r"работал",
            r"занимался",
            r"отвечал",
            r"вел",
            r"реализовал",
            r"проект",
        ]
        hits = 0
        for pattern in patterns:
            hits += len(re.findall(pattern, lowered))
        score += min(hits * 6, 36)
        if "без опыта" in lowered:
            score -= 20
        evidence = find_sentences_with_keywords(transcript, ["работал", "занимался", "проект", "отвечал"], limit=3)
        rationale = "Оценка строится по наличию описанного опыта, обязанностей и завершенных задач."
        return SkillSignal("Практический опыт", round(clamp(score, 0, 100), 2), evidence, rationale)

    def _score_structure(self, info: Dict[str, Any]) -> SkillSignal:
        score = 50.0
        sentence_count = int(info["sentences"])
        word_count = int(info["words"])
        if sentence_count >= 5:
            score += 10
        if word_count >= 180:
            score += 12
        elif word_count < 90:
            score -= 12
        if len(info["keywords"]) >= 8:
            score += 10
        rationale = "Оценка отражает развернутость и структурированность ответов кандидата."
        evidence = [
            f"Количество слов: {word_count}",
            f"Количество предложений: {sentence_count}",
            f"Ключевые слова: {', '.join(info['keywords'][:6])}",
        ]
        return SkillSignal("Структурированность ответа", round(clamp(score, 0, 100), 2), evidence, rationale)

    def _score_motivation(self, transcript: str) -> SkillSignal:
        score = 45.0
        lowered = transcript.lower()
        positive_hits = sum(1 for marker in POSITIVE_MARKERS if marker in lowered)
        negative_hits = sum(1 for marker in NEGATIVE_MARKERS if marker in lowered)
        score += min(positive_hits * 4.5, 30)
        score -= min(negative_hits * 8, 24)
        if any(phrase in lowered for phrase in ["хочу развиваться", "интересна вакансия", "готов учиться", "мотивирует"]):
            score += 12
        evidence = find_sentences_with_keywords(transcript, ["хочу", "интерес", "готов", "развив"], limit=3)
        rationale = "Оценка отражает заинтересованность кандидата в позиции и готовность к развитию."
        return SkillSignal("Мотивация", round(clamp(score, 0, 100), 2), evidence, rationale)

    def _score_by_question(self, interview: InterviewInput) -> List[Dict[str, Any]]:
        scores: List[Dict[str, Any]] = []
        if not interview.interview_turns:
            # fallback: split transcript into pseudo-questions if structured data missing
            transcript = interview.merged_transcript()
            chunks = sentence_split(transcript)
            if not chunks:
                return []
            pseudo_answers = [" ".join(chunks[i:i + 2]) for i in range(0, len(chunks), 2)]
            for idx, answer in enumerate(pseudo_answers, start=1):
                answer_score = self._quick_answer_score(answer)
                scores.append({
                    "question_index": idx,
                    "question": f"Фрагмент {idx}",
                    "score": answer_score,
                    "comment": self._question_comment(answer_score),
                })
            return scores

        for idx, turn in enumerate(interview.interview_turns, start=1):
            answer_score = self._quick_answer_score(turn.answer)
            scores.append({
                "question_index": idx,
                "question": turn.question,
                "score": answer_score,
                "comment": self._question_comment(answer_score),
            })
        return scores

    def _quick_answer_score(self, answer: str) -> float:
        tokens = tokenize(answer)
        score = 45.0
        if len(tokens) > 30:
            score += 10
        if len(tokens) > 55:
            score += 10
        if any(marker in answer.lower() for marker in POSITIVE_MARKERS):
            score += 12
        if any(marker in answer.lower() for marker in NEGATIVE_MARKERS):
            score -= 14
        return round(clamp(score, 0, 100), 2)

    def _question_comment(self, score: float) -> str:
        if score >= 80:
            return "Сильный, развернутый и содержательный ответ."
        if score >= 65:
            return "Достаточно хороший ответ, но есть запас по конкретике."
        if score >= 50:
            return "Средний ответ, желательно больше примеров и детализации."
        return "Слабый или слишком общий ответ."

    def _confidence_from_input(self, info: Dict[str, Any], question_scores: List[Dict[str, Any]]) -> float:
        confidence = 0.55
        if info["words"] >= 180:
            confidence += 0.12
        if info["sentences"] >= 8:
            confidence += 0.08
        if len(question_scores) >= 4:
            confidence += 0.08
        variance_penalty = 0.0
        if question_scores:
            avg = sum(item["score"] for item in question_scores) / len(question_scores)
            spread = sum(abs(item["score"] - avg) for item in question_scores) / len(question_scores)
            variance_penalty = min(spread / 250, 0.10)
        confidence -= variance_penalty
        return clamp(confidence, 0.45, 0.92)

    def _decision_from_score(self, score: float) -> str:
        if score >= 78:
            return "рекомендован"
        if score >= 60:
            return "условно рекомендован"
        return "не рекомендован"

    def _build_findings(
        self,
        communication: SkillSignal,
        relevance: SkillSignal,
        experience: SkillSignal,
        structure: SkillSignal,
        motivation: SkillSignal,
        question_scores: List[Dict[str, Any]],
    ) -> Tuple[List[str], List[str], List[str]]:
        strengths: List[str] = []
        weaknesses: List[str] = []
        risks: List[str] = []

        mapping = [communication, relevance, experience, structure, motivation]
        for signal in mapping:
            if signal.score >= 72:
                strengths.append(f"Высокий показатель по критерию «{signal.name.lower()}»." )
            elif signal.score < 55:
                weaknesses.append(f"Недостаточный уровень по критерию «{signal.name.lower()}»." )

        low_questions = [q for q in question_scores if q["score"] < 55]
        high_questions = [q for q in question_scores if q["score"] >= 75]
        if high_questions:
            strengths.append("Есть сильные ответы на отдельные вопросы интервью.")
        if low_questions:
            weaknesses.append("Некоторые ответы недостаточно конкретны или поверхностны.")
            risks.append("На следующем этапе стоит дополнительно проверить проблемные темы интервью.")

        if experience.score < 55:
            risks.append("Риск недостаточного практического опыта для быстрого входа в должность.")
        if relevance.score < 60:
            risks.append("Есть риск неполного соответствия профилю вакансии.")
        if communication.score < 60:
            risks.append("Может потребоваться дополнительная оценка коммуникативных навыков.")
        if motivation.score < 55:
            risks.append("Мотивация к позиции выражена недостаточно явно.")

        if not strengths:
            strengths.append("Кандидат дал достаточный объем данных для первичной оценки.")
        if not weaknesses:
            weaknesses.append("Критических слабых мест на этапе первичной автоматизированной оценки не выявлено.")
        if not risks:
            risks.append("Рекомендуется стандартная очная верификация результатов автоматической оценки.")

        return strengths[:5], weaknesses[:5], risks[:5]

    def _build_explanation(
        self,
        interview: InterviewInput,
        score: float,
        decision: str,
        communication: SkillSignal,
        relevance: SkillSignal,
        experience: SkillSignal,
        structure: SkillSignal,
        motivation: SkillSignal,
    ) -> str:
        top = sorted(
            [communication, relevance, experience, structure, motivation],
            key=lambda x: x.score,
            reverse=True,
        )
        low = sorted(
            [communication, relevance, experience, structure, motivation],
            key=lambda x: x.score,
        )
        return (
            f"Кандидат {interview.candidate_id} по вакансии «{interview.vacancy_title}» получил {score} балла, "
            f"итоговое решение: {decision}. Наиболее сильные стороны анализа: "
            f"{top[0].name.lower()} ({top[0].score}) и {top[1].name.lower()} ({top[1].score}). "
            f"Наиболее уязвимые зоны: {low[0].name.lower()} ({low[0].score}) и {low[1].name.lower()} ({low[1].score}). "
            f"Оценка сформирована на основе содержания интервью, пересечения с требованиями вакансии, "
            f"развернутости ответов и наличия маркеров практического опыта и мотивации."
        )

    def _build_recommendation(self, score: float, strengths: List[str], risks: List[str]) -> str:
        if score >= 78:
            return (
                "Рекомендуется перевод кандидата на следующий этап отбора: предметное интервью с HR или руководителем, "
                "с проверкой кейсов и подтверждением заявленного опыта."
            )
        if score >= 60:
            return (
                "Рекомендуется дополнительное уточняющее интервью. Необходимо адресно проверить риски: "
                f"{risks[0].lower()}"
            )
        return (
            "На текущем этапе кандидат не рекомендуется к дальнейшему рассмотрению без повторного интервью или "
            "существенного уточнения опыта и мотивации."
        )


class YandexLLMEvaluator(BaseEvaluator):
    """
    Optional evaluator. Requires environment variables:
    YANDEX_LLM_URL
    YANDEX_LLM_API_KEY

    It expects a completion-like JSON API. Since actual cloud account settings vary,
    the response parser is intentionally tolerant and falls back to the rule-based
    evaluator if the call fails.
    """

    name = "yandex_llm_adapter"

    def __init__(self, fallback: Optional[BaseEvaluator] = None):
        self.url = os.getenv("YANDEX_LLM_URL", "").strip()
        self.api_key = os.getenv("YANDEX_LLM_API_KEY", "").strip()
        self.folder_id = os.getenv("YANDEX_FOLDER_ID", "").strip()
        self.model_uri = os.getenv("YANDEX_MODEL_URI", "").strip()
        self.fallback = fallback or RuleBasedEvaluator()

    def is_configured(self) -> bool:
        return bool(self.url and self.api_key)

    def evaluate(self, interview: InterviewInput) -> CandidateEvaluation:
        if not self.is_configured():
            LOGGER.warning("Yandex LLM не настроен, используется резервный офлайн-оценщик.")
            return self.fallback.evaluate(interview)

        prompt = self._build_prompt(interview)
        payload = self._build_payload(prompt)

        try:
            response_json = self._post_json(payload)
            parsed = self._parse_response(interview, response_json)
            if parsed is None:
                LOGGER.warning("Не удалось корректно распарсить ответ LLM, используется резервный оценщик.")
                return self.fallback.evaluate(interview)
            return parsed
        except Exception as exc:  # broad by design for resilient fallback
            LOGGER.exception("Ошибка обращения к LLM: %s", exc)
            return self.fallback.evaluate(interview)

    def _build_prompt(self, interview: InterviewInput) -> str:
        return textwrap.dedent(
            f"""
            Ты выступаешь как HR-аналитик.
            Проанализируй интервью кандидата и верни результат строго в JSON.

            Требования вакансии:
            {interview.vacancy_title}
            {interview.vacancy_description}

            Уровень позиции: {interview.position_level}
            Идентификатор кандидата: {interview.candidate_id}

            Текст интервью:
            {interview.merged_transcript()}

            Верни JSON с полями:
            score (0-100),
            decision,
            confidence (0-1),
            strengths (list),
            weaknesses (list),
            risks (list),
            recommendation,
            explanation,
            skill_signals (list of objects with name, score, evidence, rationale),
            question_scores (list of objects with question_index, question, score, comment)
            """
        ).strip()

    def _build_payload(self, prompt: str) -> Dict[str, Any]:
        if self.model_uri:
            model_uri = self.model_uri
        elif self.folder_id:
            model_uri = f"gpt://{self.folder_id}/yandexgpt/latest"
        else:
            model_uri = "yandexgpt/latest"
        return {
            "modelUri": model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": 0.2,
                "maxTokens": 2500,
            },
            "messages": [
                {"role": "system", "text": "Ты анализируешь кандидатов на вакансии."},
                {"role": "user", "text": prompt},
            ],
        }

    def _post_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(self.url, data=body, method="POST")
        req.add_header("Authorization", f"Api-Key {self.api_key}")
        req.add_header("Content-Type", "application/json")
        with request.urlopen(req, timeout=90) as resp:
            content = resp.read().decode("utf-8")
        return json.loads(content)

    def _parse_response(self, interview: InterviewInput, response_json: Dict[str, Any]) -> Optional[CandidateEvaluation]:
        text_candidates: List[str] = []
        if isinstance(response_json.get("result"), str):
            text_candidates.append(response_json["result"])
        alt_paths = [
            response_json.get("text"),
            response_json.get("output_text"),
        ]
        for item in alt_paths:
            if isinstance(item, str):
                text_candidates.append(item)
        # common Yandex response formats
        try:
            alternatives = response_json.get("result", {}).get("alternatives", [])
            for alt in alternatives:
                message = alt.get("message", {})
                if isinstance(message.get("text"), str):
                    text_candidates.append(message["text"])
        except AttributeError:
            pass

        for candidate_text in text_candidates:
            try:
                parsed = json.loads(candidate_text)
                return self._evaluation_from_json(interview, parsed)
            except json.JSONDecodeError:
                match = re.search(r"\{.*\}", candidate_text, flags=re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group(0))
                        return self._evaluation_from_json(interview, parsed)
                    except json.JSONDecodeError:
                        continue
        return None

    def _evaluation_from_json(self, interview: InterviewInput, payload: Dict[str, Any]) -> CandidateEvaluation:
        skill_signals = [
            SkillSignal(
                name=item.get("name", "Неизвестный критерий"),
                score=float(item.get("score", 0.0)),
                evidence=list(item.get("evidence", [])),
                rationale=item.get("rationale", ""),
            )
            for item in payload.get("skill_signals", [])
            if isinstance(item, dict)
        ]
        return CandidateEvaluation(
            candidate_id=interview.candidate_id,
            vacancy_title=interview.vacancy_title,
            score=float(payload.get("score", 0.0)),
            decision=str(payload.get("decision", "не рекомендован")),
            confidence=float(payload.get("confidence", 0.5)),
            strengths=list(payload.get("strengths", [])),
            weaknesses=list(payload.get("weaknesses", [])),
            risks=list(payload.get("risks", [])),
            recommendation=str(payload.get("recommendation", "")),
            explanation=str(payload.get("explanation", "")),
            skill_signals=skill_signals,
            question_scores=list(payload.get("question_scores", [])),
            evaluator_name=self.name,
        )


# ---------------------------------------------------------------------------
# Neural evaluator
# ---------------------------------------------------------------------------

class NeuralEvaluator(BaseEvaluator):
    name = "neural_tfidf_v1"

    def __init__(self, model_path: str | Path, use_rule_insights: bool = True):
        try:
            import joblib  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "joblib is not installed. Install ML deps: pip install -r requirements-ml.txt"
            ) from exc

        bundle = joblib.load(Path(model_path))
        self.decision_model = bundle["decision_model"]
        self.score_model = bundle["score_model"]
        self.processor = TranscriptProcessor()
        self.use_rule_insights = use_rule_insights
        self.rule_evaluator = RuleBasedEvaluator() if use_rule_insights else None

    @staticmethod
    def _to_model_text(interview: InterviewInput) -> str:
        chunks: List[str] = []
        for turn in interview.interview_turns:
            if turn.question.strip():
                chunks.append(f"Q: {turn.question.strip()}")
            if turn.answer.strip():
                chunks.append(f"A: {turn.answer.strip()}")
        if not chunks:
            merged = interview.merged_transcript().strip()
            if merged:
                chunks.append(merged)
        return " ".join(chunks).strip()

    @staticmethod
    def _decision_from_score(score: float) -> str:
        if score >= 80:
            return "рекомендован"
        if score >= 60:
            return "условно рекомендован"
        return "не рекомендован"

    def evaluate(self, interview: InterviewInput) -> CandidateEvaluation:
        self.processor.validate(interview)
        model_text = self._to_model_text(interview)

        predicted_decision = str(self.decision_model.predict([model_text])[0])
        predicted_score = float(self.score_model.predict([model_text])[0])
        score = round(clamp(predicted_score, 0, 100), 2)
        decision = predicted_decision or self._decision_from_score(score)

        confidence = 0.65
        if hasattr(self.decision_model, "predict_proba"):
            try:
                probs = self.decision_model.predict_proba([model_text])[0]
                confidence = round(float(max(probs)), 2)
            except Exception:
                confidence = 0.65

        strengths: List[str] = []
        weaknesses: List[str] = []
        risks: List[str] = []
        recommendation = "Рекомендуется дополнительное интервью."
        explanation = (
            "Итоговые score и decision получены ML-моделью, обученной "
            "на размеченных интервью."
        )
        skill_signals: List[SkillSignal] = []
        question_scores: List[Dict[str, Any]] = []

        if self.rule_evaluator is not None:
            rule_eval = self.rule_evaluator.evaluate(interview)
            strengths = rule_eval.strengths
            weaknesses = rule_eval.weaknesses
            risks = rule_eval.risks
            recommendation = rule_eval.recommendation
            skill_signals = rule_eval.skill_signals
            question_scores = rule_eval.question_scores
            explanation = (
                "Итоговые score и decision получены нейросетью; "
                "детализация по критериям добавлена из rule-based оценщика."
            )

        return CandidateEvaluation(
            candidate_id=interview.candidate_id,
            vacancy_title=interview.vacancy_title,
            score=score,
            decision=decision,
            confidence=confidence,
            strengths=strengths,
            weaknesses=weaknesses,
            risks=risks,
            recommendation=recommendation,
            explanation=explanation,
            skill_signals=skill_signals,
            question_scores=question_scores,
            evaluator_name=self.name,
        )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

class ReportGenerator:
    def to_json(self, evaluation: CandidateEvaluation) -> Dict[str, Any]:
        return {
            **asdict(evaluation),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }

    def to_text(self, evaluation: CandidateEvaluation) -> str:
        skill_lines = []
        for signal in evaluation.skill_signals:
            evidence = "; ".join(signal.evidence[:3]) if signal.evidence else "Без явных цитат"
            skill_lines.append(
                f"- {signal.name}: {signal.score}\n"
                f"  Обоснование: {signal.rationale}\n"
                f"  Подтверждения: {evidence}"
            )

        question_lines = []
        for item in evaluation.question_scores:
            question_lines.append(
                f"- Вопрос {item.get('question_index')}: {item.get('score')} | {item.get('comment')}\n"
                f"  Тема: {item.get('question')}"
            )

        return textwrap.dedent(
            f"""
            HR-ОТЧЕТ ПО КАНДИДАТУ
            ====================
            Кандидат: {evaluation.candidate_id}
            Вакансия: {evaluation.vacancy_title}
            Итоговый балл: {evaluation.score}
            Решение: {evaluation.decision}
            Уверенность модели: {evaluation.confidence}
            Оценщик: {evaluation.evaluator_name}

            Сильные стороны:
            {self._bullet_block(evaluation.strengths)}

            Слабые стороны:
            {self._bullet_block(evaluation.weaknesses)}

            Риски:
            {self._bullet_block(evaluation.risks)}

            Пояснение:
            {evaluation.explanation}

            Рекомендация:
            {evaluation.recommendation}

            Детализация по критериям:
            {chr(10).join(skill_lines) if skill_lines else '- Нет данных'}

            Детализация по вопросам:
            {chr(10).join(question_lines) if question_lines else '- Нет данных'}
            """
        ).strip()

    @staticmethod
    def _bullet_block(items: List[str]) -> str:
        if not items:
            return "- Нет данных"
        return "\n".join(f"- {item}" for item in items)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class CandidateScreeningSystem:
    def __init__(self, evaluator: BaseEvaluator, report_generator: Optional[ReportGenerator] = None):
        self.evaluator = evaluator
        self.report_generator = report_generator or ReportGenerator()
        self.processor = TranscriptProcessor()

    def run(self, interview: InterviewInput) -> CandidateEvaluation:
        self.processor.validate(interview)
        return self.evaluator.evaluate(interview)

    def save_outputs(self, evaluation: CandidateEvaluation, output_dir: str | Path) -> Tuple[Path, Path]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{evaluation.candidate_id}_{stamp}"
        json_path = output_path / f"{base_name}.json"
        txt_path = output_path / f"{base_name}.txt"

        safe_write_json(json_path, self.report_generator.to_json(evaluation))
        safe_write_text(txt_path, self.report_generator.to_text(evaluation))
        return json_path, txt_path


# ---------------------------------------------------------------------------
# Demo data generation
# ---------------------------------------------------------------------------

SAMPLE_INTERVIEW = {
    "candidate_id": "candidate_001",
    "vacancy_title": "Менеджер по работе с клиентами",
    "vacancy_description": (
        "Компания ищет менеджера по работе с клиентами начального уровня. "
        "Важны грамотная устная речь, работа с возражениями, клиентский сервис, "
        "умение работать в CRM, дисциплина, обучаемость и ориентация на результат."
    ),
    "position_level": "junior",
    "interview_turns": [
        {
            "question": "Расскажите о себе и опыте работы.",
            "answer": (
                "Последние полтора года я работал администратором в сервисной компании, "
                "где принимал входящие обращения, консультировал клиентов и вел записи в CRM. "
                "Часто приходилось объяснять условия услуг, решать спорные ситуации и передавать сложные кейсы старшему менеджеру."
            ),
        },
        {
            "question": "Почему вас заинтересовала эта вакансия?",
            "answer": (
                "Мне интересна работа с клиентами и развитие в продажах. Я хочу перейти в более структурированную компанию, "
                "где можно расти по KPI, учиться работать с возражениями и влиять на результат."
            ),
        },
        {
            "question": "Как вы ведете себя в конфликтной ситуации с клиентом?",
            "answer": (
                "Сначала я стараюсь спокойно выслушать клиента, уточнить факты и показать, что понял его проблему. "
                "После этого предлагаю варианты решения и фиксирую договоренности. В моей прошлой работе это помогало снижать напряжение и сохранять лояльность клиента."
            ),
        },
        {
            "question": "Есть ли у вас опыт работы с планами и метриками?",
            "answer": (
                "Да, у нас были показатели по скорости ответа, качеству консультаций и количеству закрытых обращений. "
                "Я еженедельно смотрел отчетность и старался улучшать свои показатели."
            ),
        },
    ],
    "metadata": {
        "source": "demo",
        "language": "ru",
    },
}


def create_demo_files(target_dir: str | Path) -> Tuple[Path, Path]:
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    interview_path = target / "sample_interview.json"
    vacancy_path = target / "sample_vacancy.txt"

    safe_write_json(interview_path, SAMPLE_INTERVIEW)
    safe_write_text(vacancy_path, SAMPLE_INTERVIEW["vacancy_description"])
    return interview_path, vacancy_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Система первичного отбора кандидатов на Python"
    )
    parser.add_argument("--input", help="Путь к JSON-файлу интервью")
    parser.add_argument("--output-dir", default="./output", help="Каталог для отчетов")
    parser.add_argument(
        "--engine",
        choices=["rule", "yandex", "neural"],
        default="rule",
        help="Тип оценщика: rule, yandex или neural",
    )
    parser.add_argument(
        "--model-path",
        default="training/model_bundle.joblib",
        help="Путь к обученной neural-модели (для --engine neural)",
    )
    parser.add_argument(
        "--no-rule-insights",
        action="store_true",
        help="Для neural: отключить детализацию из rule-based оценщика",
    )

    parser.add_argument("--transcribe", action="store_true", help="Расшифровать аудио/видео через Whisper")
    parser.add_argument("--transcribe-only", action="store_true", help="Только расшифровка, без оценки")
    parser.add_argument("--audio", help="Путь к одному аудиофайлу интервью")
    parser.add_argument("--video", help="Путь к одному видеофайлу интервью (автоконвертация в аудио)")
    parser.add_argument("--source-url", help="HTTP(S)-ссылка на внешний аудио/видео файл")
    parser.add_argument(
        "--downloads-dir",
        default="incoming_media",
        help="Единая папка для загруженных внешних медиафайлов",
    )
    parser.add_argument(
        "--validate-source-only",
        action="store_true",
        help="Проверить внешнюю ссылку и формат без скачивания/транскрибации",
    )
    parser.add_argument("--audio-dir", help="Папка с question_1/answer_1 и т.д.")
    parser.add_argument("--audio-language", default="ru", help="Язык медиа для Whisper")
    parser.add_argument("--audio-model-size", default="small", help="Размер Whisper-модели")
    parser.add_argument("--question-prefix", default="question_", help="Префикс аудиовопросов")
    parser.add_argument("--answer-prefix", default="answer_", help="Префикс аудиоответов")
    parser.add_argument("--candidate-id", help="ID кандидата для транскрипции")
    parser.add_argument("--vacancy-title", help="Название вакансии для транскрипции")
    parser.add_argument("--vacancy-description", help="Описание вакансии для транскрипции")
    parser.add_argument("--position-level", default="junior", help="Уровень позиции для транскрипции")

    parser.add_argument("--demo", action="store_true", help="Создать демонстрационные входные файлы")
    parser.add_argument("--verbose", action="store_true", help="Подробный лог")
    return parser


def ui_step(message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def run_with_heartbeat(
    label: str,
    work: Callable[[], Any],
    heartbeat_seconds: float = 7.0,
) -> Any:
    stop_event = threading.Event()
    started_at = time.time()

    def _heartbeat() -> None:
        while not stop_event.wait(heartbeat_seconds):
            elapsed = int(time.time() - started_at)
            ui_step(f"{label}... в процессе ({elapsed} сек)")

    thread = threading.Thread(target=_heartbeat, daemon=True)
    thread.start()
    try:
        return work()
    finally:
        stop_event.set()
        thread.join(timeout=0.2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    setup_console_output()
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    if args.demo:
        interview_path, vacancy_path = create_demo_files(args.output_dir)
        LOGGER.info("Созданы demo-файлы: %s и %s", interview_path, vacancy_path)
        print(f"Demo interview saved to: {interview_path}")
        print(f"Demo vacancy saved to: {vacancy_path}")
        if not args.input:
            return 0

    if args.validate_source_only:
        if not args.source_url:
            parser.error("Для --validate-source-only требуется --source-url")
        from external_media_loader import validate_external_source

        ui_step("Этап 1/2: Проверка доступности ссылки и формата")
        validation = validate_external_source(args.source_url)
        print(f"Source URL: {validation.url}")
        print(
            "Reachability check: "
            f"{'OK' if validation.reachable else 'FAILED'}"
            + (f" (HTTP {validation.http_status})" if validation.http_status else "")
        )
        print(f"Detected format: {validation.extension or 'unknown'}")
        print("Conversion/transcription: " f"{'ALLOWED' if validation.format_supported else 'NOT ALLOWED'}")
        print(f"Details: {validation.reason}")
        ui_step("Этап 2/2: Проверка завершена")
        return 0 if validation.reachable and validation.format_supported else 1

    generated_input_path: Optional[Path] = None
    if args.transcribe:
        downloaded_media_path: Optional[Path] = None
        if args.source_url:
            from external_media_loader import download_external_media

            ui_step("Этап 1/5: Проверка внешней ссылки")
            last_download_report = {"percent": -1, "ts": 0.0}

            def _download_progress(downloaded: int, total: Optional[int]) -> None:
                now = time.time()
                if total and total > 0:
                    percent = int(downloaded * 100 / total)
                    if percent != last_download_report["percent"] and (
                        percent % 5 == 0 or downloaded >= total
                    ):
                        ui_step(
                            "Этап 2/5: Загрузка файла "
                            f"{percent}% ({downloaded / (1024 * 1024):.1f}/{total / (1024 * 1024):.1f} MB)"
                        )
                        last_download_report["percent"] = percent
                elif now - float(last_download_report["ts"]) >= 1.0:
                    ui_step(
                        "Этап 2/5: Загрузка файла "
                        f"{downloaded / (1024 * 1024):.1f} MB"
                    )
                    last_download_report["ts"] = now

            download_result = download_external_media(
                url=args.source_url,
                downloads_dir=args.downloads_dir,
                progress_callback=_download_progress,
            )
            validation = download_result.validation
            print(f"Source URL: {validation.url}")
            print(
                "Reachability check: "
                f"{'OK' if validation.reachable else 'FAILED'}"
                + (f" (HTTP {validation.http_status})" if validation.http_status else "")
            )
            print(f"Detected format: {validation.extension or 'unknown'}")
            print(f"Validation result: {'OK' if validation.format_supported else 'FAILED'}")
            print(f"Details: {validation.reason}")
            if not download_result.downloaded_path:
                parser.error("Невозможно продолжить: внешняя ссылка недоступна или формат не поддерживается.")
            downloaded_media_path = download_result.downloaded_path
            print(f"Downloaded to: {downloaded_media_path}")
        else:
            ui_step("Этап 1/5: Внешняя загрузка не требуется, используем локальный источник")

        if not args.candidate_id:
            parser.error("Для --transcribe требуется --candidate-id")
        if not args.vacancy_title:
            parser.error("Для --transcribe требуется --vacancy-title")
        if not args.vacancy_description:
            parser.error("Для --transcribe требуется --vacancy-description")
        selected_media_sources = (
            int(bool(args.audio))
            + int(bool(args.video))
            + int(bool(args.audio_dir))
            + int(bool(args.source_url))
        )
        if selected_media_sources != 1:
            parser.error(
                "Для --transcribe укажите ровно один источник: "
                "--audio или --video или --source-url или --audio-dir"
            )

        from audio_transcriber import AudioTranscriber, build_interview_payload

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        transcriber = AudioTranscriber(model_size=args.audio_model_size)

        transcript = ""
        interview_payload: Dict[str, Any]
        single_media_path = args.audio or args.video or (str(downloaded_media_path) if downloaded_media_path else None)
        if single_media_path:
            ui_step("Этап 3/5: Расшифровка медиа")
            transcript = run_with_heartbeat(
                "Расшифровка",
                lambda: transcriber.transcribe_file(single_media_path, language=args.audio_language),
            )
            interview_payload = build_interview_payload(
                candidate_id=args.candidate_id,
                vacancy_title=args.vacancy_title,
                vacancy_description=args.vacancy_description,
                position_level=args.position_level,
                transcript=transcript,
            )
        else:
            ui_step("Этап 3/5: Расшифровка пар question/answer")
            turns = transcriber.transcribe_question_answer_dir(
                args.audio_dir,
                question_prefix=args.question_prefix,
                answer_prefix=args.answer_prefix,
                language=args.audio_language,
            )
            transcript_lines: List[str] = []
            for idx, turn in enumerate(turns, start=1):
                transcript_lines.append(f"Вопрос {idx}: {turn.question}")
                transcript_lines.append(f"Ответ {idx}: {turn.answer}")
            transcript = "\n".join(transcript_lines).strip()
            interview_payload = build_interview_payload(
                candidate_id=args.candidate_id,
                vacancy_title=args.vacancy_title,
                vacancy_description=args.vacancy_description,
                position_level=args.position_level,
                turns=turns,
                transcript=transcript,
            )

        txt_path = output_dir / f"{args.candidate_id}_transcript.txt"
        json_path = output_dir / f"{args.candidate_id}.json"
        ui_step("Этап 4/5: Сохранение результатов транскрибации")
        safe_write_text(txt_path, transcript)
        safe_write_json(json_path, interview_payload)
        generated_input_path = json_path
        print(f"Transcript TXT: {txt_path}")
        print(f"Interview JSON: {json_path}")

        if args.transcribe_only:
            return 0

    input_path = args.input
    if generated_input_path is not None:
        input_path = str(generated_input_path)

    if not input_path:
        parser.error("Нужно передать --input path/to/interview.json или использовать --demo")

    try:
        ui_step("Этап 5/5: Оценка кандидата и генерация отчета")
        interview = InputLoader.load_json(input_path)
        evaluator: BaseEvaluator
        if args.engine == "yandex":
            evaluator = YandexLLMEvaluator(fallback=RuleBasedEvaluator())
        elif args.engine == "neural":
            evaluator = NeuralEvaluator(
                model_path=args.model_path,
                use_rule_insights=not args.no_rule_insights,
            )
        else:
            evaluator = RuleBasedEvaluator()

        system = CandidateScreeningSystem(evaluator=evaluator)
        evaluation = run_with_heartbeat(
            "Оценка кандидата",
            lambda: system.run(interview),
            heartbeat_seconds=5.0,
        )
        json_path, txt_path = run_with_heartbeat(
            "Формирование отчета",
            lambda: system.save_outputs(evaluation, args.output_dir),
            heartbeat_seconds=5.0,
        )

        print("Анализ завершен успешно.")
        print(f"Score: {evaluation.score}")
        print(f"Decision: {evaluation.decision}")
        print(f"JSON report: {json_path}")
        print(f"TXT report: {txt_path}")
        return 0
    except Exception as exc:
        LOGGER.exception("Ошибка выполнения: %s", exc)
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
