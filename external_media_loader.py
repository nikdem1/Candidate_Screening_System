#!/usr/bin/env python
"""Download and validate external interview media files."""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from urllib import error, request

from audio_transcriber import AUDIO_EXTENSIONS, MEDIA_EXTENSIONS, VIDEO_EXTENSIONS


USER_AGENT = "CandidateScreeningSystem/1.0"
CHUNK_SIZE = 1024 * 1024
DEFAULT_TIMEOUT = 15


@dataclass
class SourceValidation:
    url: str
    reachable: bool
    http_status: Optional[int]
    format_supported: bool
    convertible: bool
    extension: str
    filename: str
    reason: str


@dataclass
class DownloadResult:
    validation: SourceValidation
    downloaded_path: Optional[Path]


def validate_external_source(url: str, timeout: int = DEFAULT_TIMEOUT) -> SourceValidation:
    resolved_url = _resolve_download_url(url)
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"}:
        return SourceValidation(
            url=url,
            reachable=False,
            http_status=None,
            format_supported=False,
            convertible=False,
            extension="",
            filename="",
            reason="Ссылка должна начинаться с http:// или https://",
        )

    reachable, status, headers = _check_reachability(resolved_url, timeout=timeout)
    filename = _filename_from_url_or_headers(resolved_url, headers)
    extension = Path(filename).suffix.lower()
    if not extension:
        extension = _extension_from_content_type(headers.get("Content-Type", ""))
        if extension:
            filename = f"{filename}{extension}"
    format_supported = extension in MEDIA_EXTENSIONS
    convertible = extension in VIDEO_EXTENSIONS or extension in AUDIO_EXTENSIONS

    if not reachable:
        reason = "Ссылка недоступна или не отвечает"
    elif not extension:
        reason = "Не удалось определить расширение файла"
    elif not format_supported:
        allowed = ", ".join(sorted(MEDIA_EXTENSIONS))
        reason = f"Формат {extension} не поддерживается. Разрешены: {allowed}"
    elif extension in VIDEO_EXTENSIONS:
        reason = "Формат поддерживается: видео будет сконвертировано во временный WAV"
    elif extension == ".wav":
        reason = "Формат поддерживается: WAV (конвертация может не потребоваться)"
    else:
        reason = "Формат поддерживается: аудио будет подготовлено для транскрибации"

    return SourceValidation(
        url=url,
        reachable=reachable,
        http_status=status,
        format_supported=format_supported,
        convertible=convertible,
        extension=extension,
        filename=filename,
        reason=reason,
    )


def download_external_media(
    *,
    url: str,
    downloads_dir: str | Path,
    timeout: int = DEFAULT_TIMEOUT,
    progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
) -> DownloadResult:
    validation = validate_external_source(url, timeout=timeout)
    if not validation.reachable or not validation.format_supported:
        return DownloadResult(validation=validation, downloaded_path=None)

    target_dir = Path(downloads_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = _build_unique_path(target_dir, validation.filename)

    resolved_url = _resolve_download_url(url)
    req = request.Request(resolved_url, headers={"User-Agent": USER_AGENT})
    with request.urlopen(req, timeout=timeout) as resp, out_path.open("wb") as f:
        total_header = resp.headers.get("Content-Length")
        total_bytes = int(total_header) if total_header and total_header.isdigit() else None
        downloaded = 0
        if progress_callback:
            progress_callback(0, total_bytes)
        while True:
            chunk = resp.read(CHUNK_SIZE)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if progress_callback:
                progress_callback(downloaded, total_bytes)
    return DownloadResult(validation=validation, downloaded_path=out_path)


def _check_reachability(url: str, timeout: int) -> tuple[bool, Optional[int], dict[str, str]]:
    headers: dict[str, str] = {}

    head_req = request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with request.urlopen(head_req, timeout=timeout) as resp:
            headers = dict(resp.headers.items())
            return resp.status < 400, int(resp.status), headers
    except error.HTTPError as exc:
        if exc.code < 400:
            return True, int(exc.code), dict(exc.headers.items()) if exc.headers else {}
        return False, int(exc.code), dict(exc.headers.items()) if exc.headers else {}
    except Exception:
        pass

    # Fallback "ping": minimal GET with byte range.
    get_req = request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Range": "bytes=0-0"},
    )
    try:
        with request.urlopen(get_req, timeout=timeout) as resp:
            headers = dict(resp.headers.items())
            status = int(resp.status)
            return status < 400, status, headers
    except error.HTTPError as exc:
        if exc.code == 416:
            return True, int(exc.code), dict(exc.headers.items()) if exc.headers else {}
        return False, int(exc.code), dict(exc.headers.items()) if exc.headers else {}
    except Exception:
        return False, None, {}


def _resolve_download_url(url: str) -> str:
    """Convert known share URLs (e.g. Google Drive view links) to direct download URLs."""
    parsed = urllib.parse.urlparse(url)
    if "drive.google.com" not in parsed.netloc.lower():
        return url

    file_id = _extract_gdrive_file_id(url)
    if not file_id:
        return url
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def _extract_gdrive_file_id(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    if "id" in query and query["id"]:
        return query["id"][0]

    match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", parsed.path)
    if match:
        return match.group(1)
    return ""


def _filename_from_url_or_headers(url: str, headers: dict[str, str]) -> str:
    disposition = headers.get("Content-Disposition", "")
    match = re.search(r'filename\*=UTF-8\'\'([^;]+)', disposition, flags=re.IGNORECASE)
    if match:
        return _safe_filename(urllib.parse.unquote(match.group(1)))
    match = re.search(r'filename="?([^";]+)"?', disposition, flags=re.IGNORECASE)
    if match:
        return _safe_filename(match.group(1))

    parsed = urllib.parse.urlparse(url)
    name = Path(urllib.parse.unquote(parsed.path)).name or "downloaded_media"
    return _safe_filename(name)


def _extension_from_content_type(content_type: str) -> str:
    ct = content_type.lower().split(";")[0].strip()
    mapping = {
        "audio/mpeg": ".mp3",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/flac": ".flac",
        "audio/ogg": ".ogg",
        "audio/aac": ".aac",
        "audio/mp4": ".m4a",
        "video/mp4": ".mp4",
        "video/quicktime": ".mov",
        "video/x-matroska": ".mkv",
        "video/webm": ".webm",
        "video/x-msvideo": ".avi",
    }
    return mapping.get(ct, "")


def _build_unique_path(base_dir: Path, filename: str) -> Path:
    candidate = base_dir / filename
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    idx = 1
    while True:
        next_candidate = base_dir / f"{stem}_{idx}{suffix}"
        if not next_candidate.exists():
            return next_candidate
        idx += 1


def _safe_filename(name: str) -> str:
    clean = re.sub(r'[<>:"/\\|?*]+', "_", name).strip(" .")
    return clean or "downloaded_media"
