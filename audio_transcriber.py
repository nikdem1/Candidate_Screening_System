#!/usr/bin/env python
"""Whisper-based audio transcription helpers for interview ingestion."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional


AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
    ".flac",
    ".wma",
    ".amr",
    ".aiff",
    ".aif",
    ".caf",
}
VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
    ".mpeg",
    ".mpg",
    ".m4v",
    ".3gp",
    ".ts",
    ".mts",
}
MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


@dataclass
class InterviewTurn:
    question: str
    answer: str


class AudioTranscriber:
    def __init__(self, model_size: str = "small") -> None:
        self._ensure_ffmpeg_on_path()
        try:
            import whisper  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Failed to import Whisper runtime dependencies. "
                "Install required packages (openai-whisper, torch, dill) "
                f"or rebuild executable with these modules included. Original error: {exc}"
            ) from exc

        self._whisper = whisper
        self.model = whisper.load_model(model_size)

    @staticmethod
    def _ensure_ffmpeg_on_path() -> None:
        if shutil.which("ffmpeg"):
            return
        try:
            import imageio_ffmpeg  # type: ignore
        except ImportError:
            return

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

    def transcribe_file(self, audio_path: str | Path, language: str = "ru") -> str:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Media file not found: {path}")

        prepared_path, should_cleanup = self._prepare_media_for_whisper(path)
        try:
            result = self.model.transcribe(
                str(prepared_path),
                language=language,
                task="transcribe",
                condition_on_previous_text=False,
                temperature=(0.0, 0.2, 0.4, 0.6),
                beam_size=5,
                best_of=5,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
            )
            return str(result.get("text", "")).strip()
        finally:
            if should_cleanup:
                prepared_path.unlink(missing_ok=True)

    def _prepare_media_for_whisper(self, media_path: Path) -> tuple[Path, bool]:
        suffix = media_path.suffix.lower()
        if suffix not in MEDIA_EXTENSIONS:
            allowed = ", ".join(sorted(MEDIA_EXTENSIONS))
            raise ValueError(f"Unsupported media format: {media_path.suffix}. Allowed: {allowed}")
        if self._is_whisper_ready_wav(media_path):
            return media_path, False
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError(
                "ffmpeg is required for video transcription. "
                "Install ffmpeg or imageio-ffmpeg and retry."
            )
        with tempfile.NamedTemporaryFile(prefix="whisper_video_", suffix=".wav", delete=False) as tmp:
            tmp_wav = Path(tmp.name)
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(media_path),
            "-vn",
            "-sn",
            "-dn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-sample_fmt",
            "s16",
            "-acodec",
            "pcm_s16le",
            "-threads",
            "0",
            "-loglevel",
            "error",
            "-nostdin",
            str(tmp_wav),
        ]
        run = subprocess.run(cmd, capture_output=True, text=True)
        if run.returncode != 0:
            tmp_wav.unlink(missing_ok=True)
            err = (run.stderr or run.stdout or "unknown ffmpeg error").strip()
            raise RuntimeError(f"Failed to extract audio from video: {err}")
        return tmp_wav, True

    @staticmethod
    def _is_whisper_ready_wav(path: Path) -> bool:
        if path.suffix.lower() != ".wav":
            return False
        try:
            with wave.open(str(path), "rb") as wav:
                return (
                    wav.getnchannels() == 1
                    and wav.getframerate() == 16000
                    and wav.getsampwidth() == 2
                )
        except Exception:
            return False

    def transcribe_question_answer_dir(
        self,
        audio_dir: str | Path,
        question_prefix: str = "question_",
        answer_prefix: str = "answer_",
        language: str = "ru",
    ) -> List[InterviewTurn]:
        base = Path(audio_dir)
        if not base.exists():
            raise FileNotFoundError(f"Directory not found: {base}")

        questions = self._index_files(base, question_prefix)
        answers = self._index_files(base, answer_prefix)
        indexes = sorted(set(questions.keys()) & set(answers.keys()))
        if not indexes:
            raise ValueError(
                "No question/answer audio pairs found. "
                "Expected files like question_1.mp3 and answer_1.mp3."
            )

        turns: List[InterviewTurn] = []
        for idx in indexes:
            question = self.transcribe_file(questions[idx], language=language)
            answer = self.transcribe_file(answers[idx], language=language)
            turns.append(InterviewTurn(question=question, answer=answer))
        return turns

    @staticmethod
    def _index_files(base: Path, prefix: str) -> dict[int, Path]:
        files: dict[int, Path] = {}
        for path in base.iterdir():
            if not path.is_file():
                continue
            if path.suffix.lower() not in MEDIA_EXTENSIONS:
                continue
            if not path.stem.lower().startswith(prefix.lower()):
                continue
            match = re.search(r"(\d+)$", path.stem)
            if not match:
                continue
            files[int(match.group(1))] = path
        return files


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # UTF-8 BOM improves Russian text readability in default Windows editors.
    path.write_text(text, encoding="utf-8-sig")


def _turns_to_transcript(turns: Iterable[InterviewTurn]) -> str:
    blocks: List[str] = []
    for idx, turn in enumerate(turns, start=1):
        blocks.append(f"Вопрос {idx}: {turn.question}")
        blocks.append(f"Ответ {idx}: {turn.answer}")
    return "\n".join(blocks).strip()


def build_interview_payload(
    *,
    candidate_id: str,
    vacancy_title: str,
    vacancy_description: str,
    position_level: str,
    turns: Optional[List[InterviewTurn]] = None,
    transcript: str = "",
    source: str = "whisper",
) -> dict:
    return {
        "candidate_id": candidate_id,
        "vacancy_title": vacancy_title,
        "vacancy_description": vacancy_description,
        "position_level": position_level,
        "interview_turns": [
            {"question": turn.question, "answer": turn.answer} for turn in (turns or [])
        ],
        "transcript": transcript,
        "metadata": {
            "source": source,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe interview audio with Whisper and export TXT/JSON."
    )
    parser.add_argument("--model-size", default="small", help="Whisper model size.")
    parser.add_argument("--language", default="ru", help="Audio language code.")
    parser.add_argument("--candidate-id", required=True, help="Candidate ID.")
    parser.add_argument("--vacancy-title", required=True, help="Vacancy title.")
    parser.add_argument("--vacancy-description", required=True, help="Vacancy description.")
    parser.add_argument("--position-level", default="junior", help="Position level.")
    parser.add_argument("--output-dir", default="output", help="Directory for outputs.")

    subparsers = parser.add_subparsers(dest="mode", required=True)

    single_parser = subparsers.add_parser(
        "file", help="Transcribe one full interview media file (audio/video)."
    )
    single_parser.add_argument("--audio", help="Path to interview audio/video.")
    single_parser.add_argument("--video", help="Path to interview video (auto-converted to audio).")

    pairs_parser = subparsers.add_parser(
        "pairs",
        help=(
            "Transcribe folder with question/answer files: "
            "question_1.mp3, answer_1.mp3, ..."
        ),
    )
    pairs_parser.add_argument("--audio-dir", required=True, help="Folder with pair audio files.")
    pairs_parser.add_argument("--question-prefix", default="question_", help="Question filename prefix.")
    pairs_parser.add_argument("--answer-prefix", default="answer_", help="Answer filename prefix.")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    transcriber = AudioTranscriber(model_size=args.model_size)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "file":
        if bool(args.audio) == bool(args.video):
            parser.error("For mode=file provide exactly one of --audio or --video")
        media_path = args.audio or args.video
        transcript = transcriber.transcribe_file(media_path, language=args.language)
        payload = build_interview_payload(
            candidate_id=args.candidate_id,
            vacancy_title=args.vacancy_title,
            vacancy_description=args.vacancy_description,
            position_level=args.position_level,
            transcript=transcript,
        )
        txt_path = output_dir / f"{args.candidate_id}_transcript.txt"
        json_path = output_dir / f"{args.candidate_id}.json"
        _save_text(txt_path, transcript)
        _save_json(json_path, payload)
        print(f"Transcript TXT: {txt_path}")
        print(f"Interview JSON: {json_path}")
        return 0

    turns = transcriber.transcribe_question_answer_dir(
        args.audio_dir,
        question_prefix=args.question_prefix,
        answer_prefix=args.answer_prefix,
        language=args.language,
    )
    transcript = _turns_to_transcript(turns)
    payload = build_interview_payload(
        candidate_id=args.candidate_id,
        vacancy_title=args.vacancy_title,
        vacancy_description=args.vacancy_description,
        position_level=args.position_level,
        turns=turns,
        transcript=transcript,
    )
    txt_path = output_dir / f"{args.candidate_id}_transcript.txt"
    json_path = output_dir / f"{args.candidate_id}.json"
    _save_text(txt_path, transcript)
    _save_json(json_path, payload)
    print(f"Transcript TXT: {txt_path}")
    print(f"Interview JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
