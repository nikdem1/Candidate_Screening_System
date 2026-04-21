#!/usr/bin/env python
"""Clean transcript text for ASR training (remove non-spoken markup)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


HEADER_RE = re.compile(
    r"^\s*Кандидат\s+\d+\s*:\s*.+$",
    flags=re.IGNORECASE | re.MULTILINE,
)
HEADER_LINE_RE = re.compile(r"^\s*Кандидат\s+\d+\s*:", flags=re.IGNORECASE)
LABEL_LINE_RE = re.compile(
    r"^\s*(Интервьюер|Кандидат|[А-ЯЁA-Z][а-яёa-zA-Z-]{1,30})\s*:\s*",
    flags=re.MULTILINE,
)
QUALITY_TAG_RE = re.compile(r"-\((?:[^)]*)\)-", flags=re.IGNORECASE)
MULTISPACE_RE = re.compile(r"[ \t]+")
MULTINEW_RE = re.compile(r"\n{2,}")


def clean_text(raw: str) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = HEADER_RE.sub("", text)
    text = QUALITY_TAG_RE.sub("", text)

    cleaned_lines = []
    for line in text.splitlines():
        if HEADER_LINE_RE.match(line):
            continue
        line = LABEL_LINE_RE.sub("", line).strip()
        line = MULTISPACE_RE.sub(" ", line)
        if line:
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines).strip()
    text = MULTINEW_RE.sub("\n", text)
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean ASR transcript files from speaker labels/metadata."
    )
    parser.add_argument(
        "--input-dir",
        default="training/asr_raw/text",
        help="Directory with raw transcript .txt files.",
    )
    parser.add_argument(
        "--output-dir",
        default="training/asr_raw/text_clean",
        help="Directory for cleaned transcript files.",
    )
    parser.add_argument(
        "--glob",
        default="*.txt",
        help="Glob pattern for transcript files.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob(args.glob))
    written = 0
    skipped_empty = 0
    for path in files:
        raw = path.read_text(encoding="utf-8", errors="replace")
        cleaned = clean_text(raw)
        if not cleaned:
            skipped_empty += 1
            continue
        target = output_dir / path.name
        target.write_text(cleaned, encoding="utf-8")
        written += 1

    print(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "total_files": len(files),
                "written_files": written,
                "skipped_empty": skipped_empty,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
