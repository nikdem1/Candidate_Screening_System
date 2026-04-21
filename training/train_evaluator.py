#!/usr/bin/env python
"""Train baseline neural/text model for candidate decision and score prediction."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


@dataclass
class TrainMetrics:
    samples_total: int
    samples_train: int
    samples_test: int
    class_accuracy: float
    class_f1_macro: float
    score_mae: float
    score_rmse: float


def _interview_to_text(item: Dict[str, Any]) -> str:
    turns = item.get("interview_turns", [])
    chunks: List[str] = []
    for turn in turns:
        q = str(turn.get("question", "")).strip()
        a = str(turn.get("answer", "")).strip()
        if q:
            chunks.append(f"Q: {q}")
        if a:
            chunks.append(f"A: {a}")
    return " ".join(chunks).strip()


def _load_dataset(path: Path) -> Tuple[List[str], List[str], List[float]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    texts: List[str] = []
    decisions: List[str] = []
    scores: List[float] = []

    for item in raw:
        text = _interview_to_text(item)
        if not text:
            continue
        decision = str(item.get("target_decision", "")).strip()
        score = float(item.get("target_score", 0.0))
        if not decision:
            continue
        texts.append(text)
        decisions.append(decision)
        scores.append(score)

    if not texts:
        raise ValueError("No valid records found in training data.")
    return texts, decisions, scores


def _decision_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    lowercase=True,
                    sublinear_tf=True,
                    min_df=1,
                    max_features=40000,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def _score_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    lowercase=True,
                    sublinear_tf=True,
                    min_df=1,
                    max_features=40000,
                ),
            ),
            ("reg", Ridge(alpha=1.0)),
        ]
    )


def train_and_save(
    *,
    data_path: Path,
    model_out: Path,
    metrics_out: Path,
    test_size: float,
    seed: int,
) -> TrainMetrics:
    texts, decisions, scores = _load_dataset(data_path)
    labels = np.array(decisions)
    score_values = np.array(scores, dtype=float)

    unique_labels = sorted(set(decisions))
    stratify = labels if all((labels == lbl).sum() > 1 for lbl in unique_labels) else None

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        texts,
        labels,
        score_values,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )

    decision_model = _decision_model()
    decision_model.fit(X_train, y_train)
    decision_pred = decision_model.predict(X_test)

    score_model = _score_model()
    score_model.fit(X_train, s_train)
    score_pred = score_model.predict(X_test)
    score_pred = np.clip(score_pred, 0, 100)

    metrics = TrainMetrics(
        samples_total=len(texts),
        samples_train=len(X_train),
        samples_test=len(X_test),
        class_accuracy=float(accuracy_score(y_test, decision_pred)),
        class_f1_macro=float(f1_score(y_test, decision_pred, average="macro")),
        score_mae=float(mean_absolute_error(s_test, score_pred)),
        score_rmse=float(np.sqrt(mean_squared_error(s_test, score_pred))),
    )

    bundle = {
        "decision_model": decision_model,
        "score_model": score_model,
        "labels": unique_labels,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "train_metrics": asdict(metrics),
        "source_data": str(data_path),
    }
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_out)

    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        **asdict(metrics),
        "labels": unique_labels,
        "model_path": str(model_out),
        "data_path": str(data_path),
    }
    metrics_out.write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train candidate screening evaluator model.")
    parser.add_argument("--data", default="training/training_data.json", help="Path to dataset JSON.")
    parser.add_argument("--model-out", default="training/model_bundle.joblib", help="Path to save model bundle.")
    parser.add_argument("--metrics-out", default="training/metrics.json", help="Path to save metrics JSON.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metrics = train_and_save(
        data_path=Path(args.data),
        model_out=Path(args.model_out),
        metrics_out=Path(args.metrics_out),
        test_size=float(args.test_size),
        seed=int(args.seed),
    )
    print(json.dumps(asdict(metrics), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
