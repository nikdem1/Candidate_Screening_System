#!/usr/bin/env python
"""Generate bootstrap synthetic ASR data from interview JSON files using TTS."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import List

import pyttsx3


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def select_utterance(interview: dict, max_chars: int) -> str:
    turns = interview.get("interview_turns", [])
    chunks: List[str] = []
    for turn in turns:
        answer = normalize_text(str(turn.get("answer", "")))
        if answer:
            chunks.append(answer)
        if sum(len(c) for c in chunks) >= max_chars:
            break
    text = " ".join(chunks).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].strip()
    return text


def write_tts_wav(engine: pyttsx3.Engine, text: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    engine.save_to_file(text, str(out_path))
    engine.runAndWait()
    timeout_s = 30
    start = time.time()
    while not out_path.exists():
        if time.time() - start > timeout_s:
            raise TimeoutError(f"TTS did not create file in time: {out_path}")
        time.sleep(0.1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic ASR dataset (audio+txt) from interview JSON files."
    )
    parser.add_argument(
        "--interviews-dir",
        default="training/interviews",
        help="Directory with interview JSON files.",
    )
    parser.add_argument(
        "--out-audio-dir",
        default="training/asr_raw/audio",
        help="Output dir for generated WAV files.",
    )
    parser.add_argument(
        "--out-text-dir",
        default="training/asr_raw/text",
        help="Output dir for transcript TXT files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="How many interview files to process.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=350,
        help="Max transcript length per sample.",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=165,
        help="TTS speech rate.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    interviews_dir = Path(args.interviews_dir)
    out_audio_dir = Path(args.out_audio_dir)
    out_text_dir = Path(args.out_text_dir)
    out_audio_dir.mkdir(parents=True, exist_ok=True)
    out_text_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(interviews_dir.glob("*.json"))[: args.limit]
    if not files:
        raise FileNotFoundError(f"No JSON interviews found in {interviews_dir}")

    engine = pyttsx3.init()
    engine.setProperty("rate", args.rate)

    created = 0
    skipped = 0
    for idx, path in enumerate(files, start=1):
        payload = json.loads(path.read_text(encoding="utf-8"))
        text = select_utterance(payload, max_chars=args.max_chars)
        if len(text) < 10:
            skipped += 1
            continue

        sample_id = payload.get("candidate_id") or f"sample_{idx:03d}"
        wav_path = out_audio_dir / f"{sample_id}.wav"
        txt_path = out_text_dir / f"{sample_id}.txt"

        write_tts_wav(engine, text, wav_path)
        txt_path.write_text(text, encoding="utf-8")
        created += 1

    print(
        json.dumps(
            {
                "processed": len(files),
                "created": created,
                "skipped": skipped,
                "audio_dir": str(out_audio_dir),
                "text_dir": str(out_text_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
