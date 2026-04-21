#!/usr/bin/env python
"""Prepare labeled training JSON from raw interview text."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


HEADER_RE = re.compile(
    r"^\s*Кандидат\s+(?P<idx>\d+)\s*:\s*(?P<name>[^(\n]+?)\s*(?:\((?P<level>[^)]+)\))?\s*$",
    flags=re.MULTILINE,
)
QUESTION_RE = re.compile(r"^\s*Интервьюер:\s*(?P<q>.+?)\s*$", flags=re.MULTILINE)
NUMBERED_QUESTION_RE = re.compile(r"^\s*(?P<num>\d+)\.\s*(?P<q>.+?)\s*$")
MARKER_RE = re.compile(r"-\((?P<label>[^)]*?)\)-\s*$", flags=re.IGNORECASE)


@dataclass
class ParsedTurn:
    question: str
    answer: str
    answer_label: Optional[str]
    answer_label_raw: Optional[str]


def split_candidate_blocks(text: str) -> List[Tuple[re.Match[str], str]]:
    matches = list(HEADER_RE.finditer(text))
    blocks: List[Tuple[re.Match[str], str]] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        blocks.append((match, block))
    return blocks


def normalize_answer_label(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    lowered = raw.lower()
    if "strong" in lowered or "силь" in lowered:
        return "strong"
    if "weak" in lowered or "слаб" in lowered:
        return "weak"
    if "mid" in lowered or "сред" in lowered or "погранич" in lowered:
        return "mid"
    return None


def parse_level(level_raw: Optional[str]) -> str:
    if not level_raw:
        return "mid"
    lowered = level_raw.lower()
    if "силь" in lowered:
        return "strong"
    if "слаб" in lowered:
        return "weak"
    if "сред" in lowered or "погранич" in lowered:
        return "mid"
    return "mid"


def parse_turns(block: str) -> List[ParsedTurn]:
    turns: List[ParsedTurn] = []
    question_matches = list(QUESTION_RE.finditer(block))
    for idx, match in enumerate(question_matches):
        q_start = match.start()
        q_end = match.end()
        next_start = (
            question_matches[idx + 1].start()
            if idx + 1 < len(question_matches)
            else len(block)
        )
        question = match.group("q").strip()
        answer_chunk = block[q_end:next_start].strip()
        if not answer_chunk:
            continue

        lines = [ln.strip() for ln in answer_chunk.splitlines() if ln.strip()]
        if not lines:
            continue

        first_line = lines[0]
        if ":" in first_line:
            speaker, rest = first_line.split(":", 1)
            if 0 < len(speaker.split()) <= 5:
                lines[0] = rest.strip()

        answer_text = " ".join(lines).strip()
        label_raw: Optional[str] = None
        marker_match = MARKER_RE.search(answer_text)
        if marker_match:
            label_raw = marker_match.group("label").strip()
            answer_text = MARKER_RE.sub("", answer_text).strip(" -")

        turns.append(
            ParsedTurn(
                question=question,
                answer=answer_text,
                answer_label=normalize_answer_label(label_raw),
                answer_label_raw=label_raw,
            )
        )

    if turns:
        return turns
    return parse_turns_numbered(block)


def parse_turns_numbered(block: str) -> List[ParsedTurn]:
    lines = [ln.rstrip() for ln in block.splitlines()]
    turns: List[ParsedTurn] = []
    i = 0

    while i < len(lines):
        current = lines[i].strip()
        q_match = NUMBERED_QUESTION_RE.match(current)
        if not q_match:
            i += 1
            continue

        question = q_match.group("q").strip()
        i += 1
        answer_lines: List[str] = []
        while i < len(lines):
            probe = lines[i].strip()
            if NUMBERED_QUESTION_RE.match(probe):
                break
            if probe:
                answer_lines.append(probe)
            i += 1

        if not answer_lines:
            continue

        answer_text = " ".join(answer_lines).strip()
        label_raw: Optional[str] = None
        marker_match = MARKER_RE.search(answer_text)
        if marker_match:
            label_raw = marker_match.group("label").strip()
            answer_text = MARKER_RE.sub("", answer_text).strip(" -")

        turns.append(
            ParsedTurn(
                question=question,
                answer=answer_text,
                answer_label=normalize_answer_label(label_raw),
                answer_label_raw=label_raw,
            )
        )

    return turns


def decision_from_score(score: float) -> str:
    if score >= 80:
        return "рекомендован"
    if score >= 60:
        return "условно рекомендован"
    return "не рекомендован"


def default_score_from_level(level: str) -> float:
    if level == "strong":
        return 85.0
    if level == "mid":
        return 65.0
    if level == "weak":
        return 30.0
    return 55.0


def score_from_answer_labels(turns: List[ParsedTurn]) -> Optional[float]:
    label_to_score = {"strong": 90.0, "mid": 65.0, "weak": 35.0}
    values = [
        label_to_score[turn.answer_label]
        for turn in turns
        if turn.answer_label in label_to_score
    ]
    if not values:
        return None
    return round(sum(values) / len(values), 2)


def build_record(
    idx: int,
    name: str,
    level_raw: Optional[str],
    turns: List[ParsedTurn],
    vacancy_title: str,
    vacancy_description: str,
    position_level: str,
) -> Dict[str, Any]:
    level = parse_level(level_raw)
    base_score = default_score_from_level(level)
    label_score = score_from_answer_labels(turns)
    score = label_score if label_score is not None else base_score
    decision = decision_from_score(score)

    interview_turns = [{"question": t.question, "answer": t.answer} for t in turns]
    answer_labels = [t.answer_label for t in turns if t.answer_label]
    answer_labels_raw = [t.answer_label_raw for t in turns if t.answer_label_raw]

    return {
        "candidate_id": f"candidate_{idx:03d}",
        "candidate_name": name.strip(),
        "vacancy_title": vacancy_title,
        "vacancy_description": vacancy_description,
        "position_level": position_level,
        "interview_turns": interview_turns,
        "target_score": round(float(score), 2),
        "target_decision": decision,
        "target_class": 2 if decision == "рекомендован" else 1 if decision == "условно рекомендован" else 0,
        "metadata": {
            "source": "Unlabeled_Data.txt",
            "candidate_level_raw": level_raw,
            "candidate_level_normalized": level,
            "answer_labels": answer_labels,
            "answer_labels_raw": answer_labels_raw,
        },
    }


def prepare_dataset(
    *,
    input_path: Path,
    output_path: Path,
    vacancy_title: str,
    vacancy_description: str,
    position_level: str,
    export_interviews_dir: Optional[Path],
) -> Dict[str, Any]:
    text = input_path.read_text(encoding="utf-8", errors="replace")
    blocks = split_candidate_blocks(text)

    dataset: List[Dict[str, Any]] = []
    skipped = 0
    for header, block in blocks:
        idx = int(header.group("idx"))
        name = (header.group("name") or "").strip()
        level_raw = (header.group("level") or "").strip() or None
        turns = parse_turns(block)
        if not turns:
            skipped += 1
            continue

        record = build_record(
            idx=idx,
            name=name or f"candidate_{idx}",
            level_raw=level_raw,
            turns=turns,
            vacancy_title=vacancy_title,
            vacancy_description=vacancy_description,
            position_level=position_level,
        )
        dataset.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    exported = 0
    if export_interviews_dir:
        export_interviews_dir.mkdir(parents=True, exist_ok=True)
        for item in dataset:
            interview_payload = {
                "candidate_id": item["candidate_id"],
                "vacancy_title": item["vacancy_title"],
                "vacancy_description": item["vacancy_description"],
                "position_level": item["position_level"],
                "interview_turns": item["interview_turns"],
                "metadata": {
                    "source": "training_prepared",
                    "candidate_name": item["candidate_name"],
                },
            }
            target = export_interviews_dir / f"{item['candidate_id']}.json"
            target.write_text(
                json.dumps(interview_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            exported += 1

    return {
        "input": str(input_path),
        "output": str(output_path),
        "candidates_detected": len(blocks),
        "records_created": len(dataset),
        "records_skipped": skipped,
        "interview_json_exported": exported,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse raw interview text and build training_data.json."
    )
    parser.add_argument(
        "--input",
        default="Unlabeled_Data.txt",
        help="Path to raw text data.",
    )
    parser.add_argument(
        "--output",
        default="training/training_data.json",
        help="Path to output JSON dataset.",
    )
    parser.add_argument(
        "--vacancy-title",
        default="Менеджер по работе с клиентами",
        help="Vacancy title for generated records.",
    )
    parser.add_argument(
        "--vacancy-description",
        default=(
            "Компания ищет менеджера по работе с клиентами начального уровня. "
            "Важны коммуникация, работа с возражениями, клиентский сервис, "
            "умение работать в CRM и ориентация на результат."
        ),
        help="Vacancy description for generated records.",
    )
    parser.add_argument(
        "--position-level",
        default="junior",
        help="Position level field value.",
    )
    parser.add_argument(
        "--export-interviews-dir",
        default="training/interviews",
        help="Directory for per-candidate interview JSON files.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    stats = prepare_dataset(
        input_path=Path(args.input),
        output_path=Path(args.output),
        vacancy_title=args.vacancy_title,
        vacancy_description=args.vacancy_description,
        position_level=args.position_level,
        export_interviews_dir=Path(args.export_interviews_dir)
        if args.export_interviews_dir
        else None,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
