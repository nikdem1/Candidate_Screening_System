#!/usr/bin/env python
"""Transcribe audio using a fine-tuned Whisper checkpoint."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe audio with a fine-tuned Whisper model."
    )
    parser.add_argument("--model-dir", required=True, help="Path to fine-tuned model dir.")
    parser.add_argument("--audio", required=True, help="Audio file path.")
    parser.add_argument("--language", default="ru", help="Language code.")
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    parser.add_argument("--output", help="Optional output text file path.")
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=30.0,
        help="Chunk size in seconds for long-form decoding.",
    )
    parser.add_argument(
        "--min-rms",
        type=float,
        default=0.003,
        help="Skip near-silence chunks with RMS below this threshold.",
    )
    return parser


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _iter_chunks(audio: np.ndarray, sr: int, chunk_seconds: float) -> list[np.ndarray]:
    chunk_size = max(1, int(sr * chunk_seconds))
    return [audio[i : i + chunk_size] for i in range(0, len(audio), chunk_size)]


def main() -> int:
    args = build_parser().parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(args.model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_dir).to(device)

    audio_array, _ = librosa.load(args.audio, sr=16000, mono=True)
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=args.language,
        task=args.task,
    )
    texts: list[str] = []
    for chunk in _iter_chunks(audio_array, 16000, args.chunk_seconds):
        if chunk.size == 0:
            continue
        rms = float(np.sqrt(np.mean(np.square(chunk))))
        if rms < args.min_rms:
            continue
        inputs = processor(
            chunk,
            sampling_rate=16000,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(device)
        predicted_ids = model.generate(
            input_features=input_features,
            forced_decoder_ids=forced_decoder_ids,
            no_repeat_ngram_size=4,
        )
        chunk_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        chunk_text = _normalize_spaces(chunk_text)
        if chunk_text:
            texts.append(chunk_text)
    text = _normalize_spaces(" ".join(texts))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    try:
        print(json.dumps({"audio": args.audio, "text": text}, ensure_ascii=False, indent=2))
    except UnicodeEncodeError:
        print(json.dumps({"audio": args.audio, "text": text}, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
