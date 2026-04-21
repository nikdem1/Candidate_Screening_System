#!/usr/bin/env python
"""Build ASR manifests for Whisper fine-tuning from local audio/text pairs."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4"}


@dataclass
class Sample:
    audio: str
    text: str
    language: str


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def collect_samples(
    *,
    audio_dir: Path,
    transcript_dir: Path | None,
    transcript_ext: str,
    language: str,
    recursive: bool,
) -> List[Sample]:
    pattern = "**/*" if recursive else "*"
    samples: List[Sample] = []

    for path in audio_dir.glob(pattern):
        if not path.is_file():
            continue
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue

        if transcript_dir is None:
            text_path = path.with_suffix(transcript_ext)
        else:
            rel = path.relative_to(audio_dir)
            text_path = (transcript_dir / rel).with_suffix(transcript_ext)

        if not text_path.exists():
            continue
        text = _read_text(text_path)
        if not text:
            continue
        samples.append(
            Sample(audio=str(path.resolve()), text=text, language=language)
        )

    return samples


def split_samples(
    samples: Sequence[Sample],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    if not samples:
        return [], [], []
    data = list(samples)
    rng = random.Random(seed)
    rng.shuffle(data)

    total = len(data)
    n_val = int(total * val_ratio)
    n_test = int(total * test_ratio)
    n_train = total - n_val - n_test
    if n_train <= 0:
        raise ValueError(
            "Train split is empty. Reduce --val-ratio/--test-ratio or add more samples."
        )

    train = data[:n_train]
    val = data[n_train:n_train + n_val]
    test = data[n_train + n_val:]
    return train, val, test


def write_jsonl(path: Path, samples: Sequence[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            payload: Dict[str, str] = {
                "audio": s.audio,
                "text": s.text,
                "language": s.language,
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build train/val/test JSONL manifests for Whisper fine-tuning."
    )
    parser.add_argument("--audio-dir", required=True, help="Directory with audio files.")
    parser.add_argument(
        "--transcript-dir",
        help=(
            "Directory with transcript text files. "
            "If omitted, transcript is expected near audio with the same stem."
        ),
    )
    parser.add_argument(
        "--transcript-ext",
        default=".txt",
        help="Transcript extension, default: .txt",
    )
    parser.add_argument("--language", default="ru", help="Language tag for all samples.")
    parser.add_argument("--recursive", action="store_true", help="Scan audio dir recursively.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--out-dir",
        default="training/asr_data",
        help="Output dir for train.jsonl/val.jsonl/test.jsonl.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    audio_dir = Path(args.audio_dir)
    transcript_dir = Path(args.transcript_dir) if args.transcript_dir else None

    samples = collect_samples(
        audio_dir=audio_dir,
        transcript_dir=transcript_dir,
        transcript_ext=args.transcript_ext,
        language=args.language,
        recursive=bool(args.recursive),
    )
    train, val, test = split_samples(
        samples=samples,
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        seed=int(args.seed),
    )

    out_dir = Path(args.out_dir)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    test_path = out_dir / "test.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)
    write_jsonl(test_path, test)

    print(
        json.dumps(
            {
                "total_samples": len(samples),
                "train_samples": len(train),
                "val_samples": len(val),
                "test_samples": len(test),
                "train_manifest": str(train_path),
                "val_manifest": str(val_path),
                "test_manifest": str(test_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
