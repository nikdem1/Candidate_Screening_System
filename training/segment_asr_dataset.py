#!/usr/bin/env python
"""Segment long interview audio into short ASR training chunks."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rapidfuzz import fuzz


@dataclass
class SegmentItem:
    start: float
    end: float
    predicted_text: str
    target_text: str
    score: float


def normalize_text(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^0-9a-zа-я\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_sentences(text: str) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    merged = " ".join(lines)
    parts = re.split(r"(?<=[.!?])\s+", merged)
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences


def similarity(a: str, b: str) -> float:
    na = normalize_text(a)
    nb = normalize_text(b)
    if not na or not nb:
        return 0.0
    return float(fuzz.ratio(na, nb)) / 100.0


def ensure_ffmpeg_on_path() -> str:
    if shutil.which("ffmpeg"):
        return shutil.which("ffmpeg") or "ffmpeg"
    import imageio_ffmpeg  # type: ignore

    ffmpeg_exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
    ffmpeg_dir = str(ffmpeg_exe.parent)
    current = os.environ.get("PATH", "")
    if ffmpeg_dir not in current:
        os.environ["PATH"] = f"{ffmpeg_dir}{os.pathsep}{current}"

    ffmpeg_alias = ffmpeg_exe.parent / "ffmpeg.exe"
    if not ffmpeg_alias.exists():
        try:
            shutil.copyfile(ffmpeg_exe, ffmpeg_alias)
        except Exception:
            pass
    return str(ffmpeg_alias if ffmpeg_alias.exists() else ffmpeg_exe)


def transcribe_segments(
    whisper_model: Any,
    audio_path: Path,
    language: str,
) -> List[Dict[str, Any]]:
    result = whisper_model.transcribe(
        str(audio_path),
        language=language,
        temperature=0.0,
        condition_on_previous_text=True,
    )
    return list(result.get("segments", []))


def align_segments_to_reference(
    predicted_segments: List[Dict[str, Any]],
    sentences: List[str],
    max_sentences_per_segment: int,
    min_score: float,
) -> List[SegmentItem]:
    aligned: List[SegmentItem] = []
    sent_idx = 0
    total_sent = len(sentences)

    for seg in predicted_segments:
        pred = str(seg.get("text", "")).strip()
        if not pred:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        if end <= start:
            continue

        if sent_idx >= total_sent:
            break

        best_score = -1.0
        best_k = 1
        best_text = sentences[sent_idx]

        for k in range(1, max_sentences_per_segment + 1):
            if sent_idx + k > total_sent:
                break
            candidate = " ".join(sentences[sent_idx:sent_idx + k]).strip()
            score = similarity(pred, candidate)
            if score > best_score:
                best_score = score
                best_k = k
                best_text = candidate

        # If local match is weak, try small forward search window to recover alignment drift.
        if best_score < min_score:
            window_end = min(total_sent, sent_idx + 6)
            for probe_idx in range(sent_idx + 1, window_end):
                candidate = sentences[probe_idx]
                score = similarity(pred, candidate)
                if score > best_score:
                    best_score = score
                    best_k = 1
                    best_text = candidate
                    sent_idx = probe_idx

        aligned.append(
            SegmentItem(
                start=start,
                end=end,
                predicted_text=pred,
                target_text=best_text,
                score=best_score,
            )
        )
        sent_idx += best_k

    return aligned


def cut_audio_segment(
    ffmpeg_bin: str,
    src_audio: Path,
    dst_audio: Path,
    start: float,
    end: float,
) -> None:
    dst_audio.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_bin,
        "-y",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        str(src_audio),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(dst_audio),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Segment interview audio and align with reference transcript."
    )
    parser.add_argument("--audio-dir", default="training/asr_raw/audio", help="Input audio dir.")
    parser.add_argument("--text-dir", default="training/asr_raw/text_clean", help="Reference text dir.")
    parser.add_argument("--audio-glob", default="int_*.wav", help="Audio glob pattern.")
    parser.add_argument("--language", default="ru", help="Whisper language.")
    parser.add_argument("--model-size", default="tiny", help="Whisper model size for segmentation.")
    parser.add_argument("--output-dir", default="training/asr_segmented", help="Output dir.")
    parser.add_argument("--limit", type=int, help="Optional limit on number of source files.")
    parser.add_argument("--min-duration", type=float, default=2.0, help="Min segment duration seconds.")
    parser.add_argument("--max-duration", type=float, default=30.0, help="Max segment duration seconds.")
    parser.add_argument("--min-score", type=float, default=0.45, help="Min alignment score to keep.")
    parser.add_argument(
        "--max-sentences-per-segment",
        type=int,
        default=3,
        help="How many reference sentences can be merged for one predicted segment.",
    )
    parser.add_argument(
        "--keep-low-score",
        action="store_true",
        help="Keep low-score segments as well (not recommended).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    audio_dir = Path(args.audio_dir)
    text_dir = Path(args.text_dir)
    out_dir = Path(args.output_dir)
    out_audio_dir = out_dir / "audio"
    out_text_dir = out_dir / "text"
    out_audio_dir.mkdir(parents=True, exist_ok=True)
    out_text_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_bin = ensure_ffmpeg_on_path()

    import whisper  # type: ignore

    model = whisper.load_model(args.model_size)

    src_files = sorted(audio_dir.glob(args.audio_glob))
    if args.limit is not None:
        src_files = src_files[: args.limit]

    manifest_rows: List[Dict[str, Any]] = []
    total_kept = 0
    total_dropped = 0
    processed_files = 0

    for audio_path in src_files:
        txt_path = text_dir / f"{audio_path.stem}.txt"
        if not txt_path.exists():
            continue
        reference_text = txt_path.read_text(encoding="utf-8", errors="replace").strip()
        if not reference_text:
            continue
        sentences = split_sentences(reference_text)
        if not sentences:
            continue

        predicted_segments = transcribe_segments(model, audio_path, args.language)
        aligned = align_segments_to_reference(
            predicted_segments=predicted_segments,
            sentences=sentences,
            max_sentences_per_segment=args.max_sentences_per_segment,
            min_score=args.min_score,
        )

        local_idx = 0
        for item in aligned:
            duration = item.end - item.start
            if duration < args.min_duration or duration > args.max_duration:
                total_dropped += 1
                continue
            if item.score < args.min_score and not args.keep_low_score:
                total_dropped += 1
                continue

            seg_id = f"{audio_path.stem}_seg_{local_idx:04d}"
            seg_audio = out_audio_dir / f"{seg_id}.wav"
            seg_text = out_text_dir / f"{seg_id}.txt"
            cut_audio_segment(ffmpeg_bin, audio_path, seg_audio, item.start, item.end)
            seg_text.write_text(item.target_text, encoding="utf-8")

            manifest_rows.append(
                {
                    "audio": str(seg_audio.resolve()),
                    "text": item.target_text,
                    "language": args.language,
                    "meta": {
                        "source_audio": str(audio_path.resolve()),
                        "start": item.start,
                        "end": item.end,
                        "duration": duration,
                        "alignment_score": item.score,
                        "predicted_text": item.predicted_text,
                    },
                }
            )
            local_idx += 1
            total_kept += 1

        processed_files += 1

    manifest_path = out_dir / "segments.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "processed_source_files": processed_files,
                "total_segments_kept": total_kept,
                "total_segments_dropped": total_dropped,
                "segments_manifest": str(manifest_path),
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
