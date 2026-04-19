#!/usr/bin/env python
"""CandidateSS desktop GUI."""

from __future__ import annotations

import os
import queue
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from audio_transcriber import MEDIA_EXTENSIONS, AudioTranscriber
from candidate_screening_system import (
    CandidateScreeningSystem,
    InterviewInput,
    NeuralEvaluator,
    RuleBasedEvaluator,
    safe_write_json,
    safe_write_text,
)
from external_media_loader import download_external_media, validate_external_source


DEFAULT_OUTPUT_DIR = "output"
DEFAULT_DOWNLOADS_DIR = "incoming_media"


@dataclass
class RunConfig:
    source_type: str
    source_value: str
    candidate_id: str
    vacancy_title: str
    vacancy_description: str
    position_level: str
    language: str
    model_size: str
    output_dir: str
    downloads_dir: str
    engine: str
    neural_model_path: str


class CandidateSSApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("CandidateSS")
        self.root.geometry("1080x760")
        self.root.minsize(980, 680)
        self.root.configure(bg="#f4f7fb")

        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.worker: Optional[threading.Thread] = None
        self.last_files: list[Path] = []

        self._build_style()
        self._build_ui()
        self._install_edit_shortcuts()
        self.root.after(120, self._poll_events)

    def _build_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background="#f4f7fb")
        style.configure("Card.TFrame", background="#ffffff")
        style.configure("TLabel", background="#f4f7fb", foreground="#1e2a3a", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 19), foreground="#0b4a8b")
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10), foreground="#4b5a6b")
        style.configure("CardTitle.TLabel", background="#ffffff", font=("Segoe UI Semibold", 11))
        style.configure("TButton", font=("Segoe UI Semibold", 10), padding=7)
        style.configure("Accent.TButton", background="#0b63b6", foreground="white")
        style.map("Accent.TButton", background=[("active", "#0a4f92")], foreground=[("active", "white")])
        style.configure("TEntry", fieldbackground="#ffffff")
        style.configure("TCombobox", fieldbackground="#ffffff")
        style.configure(
            "Horizontal.TProgressbar",
            troughcolor="#dbe5f0",
            background="#0b63b6",
            bordercolor="#dbe5f0",
            lightcolor="#0b63b6",
            darkcolor="#0b63b6",
        )

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=16)
        main.pack(fill="both", expand=True)

        header = ttk.Frame(main)
        header.pack(fill="x", pady=(0, 10))
        ttk.Label(header, text="CandidateSS", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Загрузка интервью (файл/ссылка), транскрибация, оценка и готовый HR-отчет в одном окне.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        ttk.Label(
            main,
            text="Подсказка: перетаскивайте разделители между блоками, чтобы изменить их размер.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(0, 6))

        vertical_pane = ttk.Panedwindow(main, orient="vertical")
        vertical_pane.pack(fill="both", expand=True)

        top_container = ttk.Frame(vertical_pane)
        bottom_container = ttk.Panedwindow(vertical_pane, orient="horizontal")
        vertical_pane.add(top_container, weight=3)
        vertical_pane.add(bottom_container, weight=4)

        left_container = ttk.Frame(bottom_container)
        right_container = ttk.Frame(bottom_container)
        bottom_container.add(left_container, weight=1)
        bottom_container.add(right_container, weight=1)

        self._build_input_card(top_container)
        self._build_progress_card(left_container)
        self._build_report_card(right_container)

        def _set_initial_sashes() -> None:
            try:
                total_h = vertical_pane.winfo_height()
                if total_h > 200:
                    vertical_pane.sashpos(0, int(total_h * 0.48))
                total_w = bottom_container.winfo_width()
                if total_w > 300:
                    bottom_container.sashpos(0, int(total_w * 0.50))
            except Exception:
                pass

        self.root.after(200, _set_initial_sashes)

    def _build_input_card(self, parent: ttk.Frame) -> None:
        card = ttk.Frame(parent, style="Card.TFrame", padding=14)
        card.pack(fill="both", expand=True)
        card.columnconfigure(1, weight=1)
        ttk.Label(card, text="Параметры запуска", style="CardTitle.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 8)
        )

        self.source_type = tk.StringVar(value="local")
        self.source_value = tk.StringVar()
        self.candidate_id = tk.StringVar(value=f"candidate_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.vacancy_title = tk.StringVar(value="Менеджер по работе с клиентами")
        self.position_level = tk.StringVar(value="junior")
        self.language = tk.StringVar(value="ru")
        self.model_size = tk.StringVar(value="small")
        self.output_dir = tk.StringVar(value=DEFAULT_OUTPUT_DIR)
        self.downloads_dir = tk.StringVar(value=DEFAULT_DOWNLOADS_DIR)
        self.engine = tk.StringVar(value="rule")
        self.neural_model_path = tk.StringVar(value="training/model_bundle.joblib")

        source_row = ttk.Frame(card, style="Card.TFrame")
        source_row.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(2, 6))
        ttk.Radiobutton(source_row, text="Локальный файл", variable=self.source_type, value="local").pack(
            side="left", padx=(0, 10)
        )
        ttk.Radiobutton(source_row, text="Внешняя ссылка", variable=self.source_type, value="url").pack(side="left")

        ttk.Label(card, text="Файл или ссылка").grid(row=2, column=0, sticky="w", pady=3)
        ttk.Entry(card, textvariable=self.source_value).grid(row=2, column=1, sticky="ew", padx=6, pady=3)
        ttk.Button(card, text="Обзор...", command=self._pick_file).grid(row=2, column=2, sticky="ew", pady=3)

        ttk.Label(card, text="Candidate ID").grid(row=3, column=0, sticky="w", pady=3)
        ttk.Entry(card, textvariable=self.candidate_id).grid(row=3, column=1, sticky="ew", padx=6, pady=3)
        ttk.Label(card, text="Уровень").grid(row=3, column=2, sticky="w", pady=3)

        ttk.Label(card, text="Вакансия").grid(row=4, column=0, sticky="w", pady=3)
        ttk.Entry(card, textvariable=self.vacancy_title).grid(row=4, column=1, sticky="ew", padx=6, pady=3)
        ttk.Combobox(card, textvariable=self.position_level, values=["junior", "middle", "senior"], state="readonly").grid(
            row=4, column=2, sticky="ew", pady=3
        )

        ttk.Label(card, text="Описание вакансии").grid(row=5, column=0, sticky="nw", pady=3)
        self.vacancy_desc_text = tk.Text(card, height=4, wrap="word", font=("Segoe UI", 10))
        self.vacancy_desc_text.insert(
            "1.0",
            "Работа с клиентами, обработка обращений, CRM, соблюдение KPI и SLA.",
        )
        self.vacancy_desc_text.grid(row=5, column=1, columnspan=2, sticky="ew", padx=6, pady=3)

        opts = ttk.Frame(card, style="Card.TFrame")
        opts.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        for col in range(8):
            opts.columnconfigure(col, weight=1)

        ttk.Label(opts, text="Язык").grid(row=0, column=0, sticky="w")
        ttk.Combobox(opts, textvariable=self.language, values=["ru", "en", "auto"], state="readonly").grid(
            row=1, column=0, sticky="ew", padx=(0, 6)
        )
        ttk.Label(opts, text="Whisper").grid(row=0, column=1, sticky="w")
        ttk.Combobox(
            opts,
            textvariable=self.model_size,
            values=["tiny", "base", "small", "medium", "large"],
            state="readonly",
        ).grid(row=1, column=1, sticky="ew", padx=(0, 6))

        ttk.Label(opts, text="Оценщик").grid(row=0, column=2, sticky="w")
        ttk.Combobox(opts, textvariable=self.engine, values=["rule", "neural"], state="readonly").grid(
            row=1, column=2, sticky="ew", padx=(0, 6)
        )
        ttk.Label(opts, text="Neural model path").grid(row=0, column=3, sticky="w")
        ttk.Entry(opts, textvariable=self.neural_model_path).grid(row=1, column=3, sticky="ew", padx=(0, 6))

        ttk.Label(opts, text="Папка загрузок").grid(row=0, column=4, sticky="w")
        ttk.Entry(opts, textvariable=self.downloads_dir).grid(row=1, column=4, sticky="ew", padx=(0, 6))
        ttk.Label(opts, text="Папка отчетов").grid(row=0, column=5, sticky="w")
        ttk.Entry(opts, textvariable=self.output_dir).grid(row=1, column=5, sticky="ew", padx=(0, 6))

        actions = ttk.Frame(card, style="Card.TFrame")
        actions.grid(row=7, column=0, columnspan=3, sticky="e", pady=(10, 0))
        self.validate_btn = ttk.Button(actions, text="Проверить источник", command=self._validate_only)
        self.validate_btn.pack(side="left", padx=(0, 8))
        self.run_btn = ttk.Button(actions, text="Запустить", style="Accent.TButton", command=self._run_pipeline)
        self.run_btn.pack(side="left")

    def _build_progress_card(self, parent: ttk.Frame) -> None:
        card = ttk.Frame(parent, style="Card.TFrame", padding=14)
        card.pack(fill="both", expand=True)
        card.rowconfigure(4, weight=1)
        card.columnconfigure(0, weight=1)
        ttk.Label(card, text="Процесс выполнения", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")

        self.stage_label = ttk.Label(card, text="Ожидание запуска", style="Subtitle.TLabel")
        self.stage_label.grid(row=1, column=0, sticky="w", pady=(4, 2))

        ttk.Label(card, text="Общий прогресс").grid(row=2, column=0, sticky="w")
        self.overall_progress = ttk.Progressbar(card, orient="horizontal", mode="determinate", maximum=100)
        self.overall_progress.grid(row=3, column=0, sticky="ew", pady=(2, 10))

        ttk.Label(card, text="Прогресс текущего этапа").grid(row=4, column=0, sticky="nw")
        self.detail_progress = ttk.Progressbar(card, orient="horizontal", mode="determinate", maximum=100)
        self.detail_progress.grid(row=5, column=0, sticky="ew", pady=(2, 10))

        ttk.Label(card, text="Журнал").grid(row=6, column=0, sticky="w")
        self.log_text = tk.Text(card, height=18, wrap="word", font=("Consolas", 10))
        self.log_text.grid(row=7, column=0, sticky="nsew", pady=(3, 0))
        self.log_text.configure(state="disabled")

    def _build_report_card(self, parent: ttk.Frame) -> None:
        card = ttk.Frame(parent, style="Card.TFrame", padding=14)
        card.pack(fill="both", expand=True)
        card.rowconfigure(4, weight=1)
        card.columnconfigure(0, weight=1)
        ttk.Label(card, text="Результат и быстрые ссылки", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.result_summary = ttk.Label(card, text="Отчет пока не сгенерирован", style="Subtitle.TLabel")
        self.result_summary.grid(row=1, column=0, sticky="w", pady=(4, 6))

        links = ttk.Frame(card, style="Card.TFrame")
        links.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        links.columnconfigure(0, weight=1)
        links.columnconfigure(1, weight=1)
        self.open_output_btn = ttk.Button(links, text="Открыть папку отчетов", command=self._open_output_folder)
        self.open_output_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.open_downloads_btn = ttk.Button(links, text="Открыть папку загрузок", command=self._open_downloads_folder)
        self.open_downloads_btn.grid(row=0, column=1, sticky="ew")

        self.file_links_frame = ttk.Frame(card, style="Card.TFrame")
        self.file_links_frame.grid(row=3, column=0, sticky="ew", pady=(0, 8))

        ttk.Label(card, text="Предпросмотр HR-отчета").grid(row=4, column=0, sticky="w")
        self.report_text = tk.Text(card, height=22, wrap="word", font=("Segoe UI", 10))
        self.report_text.grid(row=5, column=0, sticky="nsew", pady=(3, 0))
        self.report_text.configure(state="disabled")

    def _pick_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Выберите аудио/видео интервью",
            filetypes=[("Media files", "*.*"), ("All files", "*.*")],
        )
        if path:
            self.source_type.set("local")
            self.source_value.set(path)

    def _install_edit_shortcuts(self) -> None:
        # Ensure clipboard shortcuts work consistently in packaged builds.
        self.root.bind_class("TEntry", "<Control-c>", lambda e: e.widget.event_generate("<<Copy>>") or "break")
        self.root.bind_class("TEntry", "<Control-v>", lambda e: e.widget.event_generate("<<Paste>>") or "break")
        self.root.bind_class("TEntry", "<Control-x>", lambda e: e.widget.event_generate("<<Cut>>") or "break")
        self.root.bind_class("TEntry", "<Control-a>", self._select_all_entry)
        self.root.bind_class("Entry", "<Control-c>", lambda e: e.widget.event_generate("<<Copy>>") or "break")
        self.root.bind_class("Entry", "<Control-v>", lambda e: e.widget.event_generate("<<Paste>>") or "break")
        self.root.bind_class("Entry", "<Control-x>", lambda e: e.widget.event_generate("<<Cut>>") or "break")
        self.root.bind_class("Entry", "<Control-a>", self._select_all_entry)
        self.root.bind_class("Text", "<Control-c>", lambda e: e.widget.event_generate("<<Copy>>") or "break")
        self.root.bind_class("Text", "<Control-v>", lambda e: e.widget.event_generate("<<Paste>>") or "break")
        self.root.bind_class("Text", "<Control-x>", lambda e: e.widget.event_generate("<<Cut>>") or "break")
        self.root.bind_class("Text", "<Control-a>", self._select_all_text)
        self.root.bind_all("<Control-Insert>", lambda e: e.widget.event_generate("<<Copy>>") or "break")
        self.root.bind_all("<Shift-Insert>", lambda e: e.widget.event_generate("<<Paste>>") or "break")
        self.root.bind_all("<Shift-Delete>", lambda e: e.widget.event_generate("<<Cut>>") or "break")

    @staticmethod
    def _select_all_entry(event: tk.Event) -> str:
        try:
            event.widget.selection_range(0, "end")
            event.widget.icursor("end")
        except Exception:
            pass
        return "break"

    @staticmethod
    def _select_all_text(event: tk.Event) -> str:
        try:
            event.widget.tag_add("sel", "1.0", "end-1c")
            event.widget.mark_set("insert", "end-1c")
            event.widget.see("insert")
        except Exception:
            pass
        return "break"

    def _open_output_folder(self) -> None:
        folder = Path(self.output_dir.get().strip() or DEFAULT_OUTPUT_DIR)
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(str(folder))

    def _open_downloads_folder(self) -> None:
        folder = Path(self.downloads_dir.get().strip() or DEFAULT_DOWNLOADS_DIR)
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(str(folder))

    def _append_log(self, line: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{line}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.validate_btn.configure(state=state)
        self.run_btn.configure(state=state)

    def _collect_config(self) -> RunConfig:
        vacancy_description = self.vacancy_desc_text.get("1.0", "end").strip()
        source_value = self.source_value.get().strip()
        candidate_id = self.candidate_id.get().strip() or f"candidate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return RunConfig(
            source_type=self.source_type.get().strip(),
            source_value=source_value,
            candidate_id=candidate_id,
            vacancy_title=self.vacancy_title.get().strip(),
            vacancy_description=vacancy_description,
            position_level=self.position_level.get().strip() or "junior",
            language=self.language.get().strip() or "ru",
            model_size=self.model_size.get().strip() or "small",
            output_dir=self.output_dir.get().strip() or DEFAULT_OUTPUT_DIR,
            downloads_dir=self.downloads_dir.get().strip() or DEFAULT_DOWNLOADS_DIR,
            engine=self.engine.get().strip() or "rule",
            neural_model_path=self.neural_model_path.get().strip(),
        )

    def _validate_only(self) -> None:
        cfg = self._collect_config()
        if not cfg.source_value:
            messagebox.showwarning("Источник не указан", "Укажите локальный файл или ссылку.")
            return
        self._start_worker(cfg, validate_only=True)

    def _run_pipeline(self) -> None:
        cfg = self._collect_config()
        if not cfg.source_value:
            messagebox.showwarning("Источник не указан", "Укажите локальный файл или ссылку.")
            return
        if not cfg.vacancy_title or not cfg.vacancy_description:
            messagebox.showwarning("Данные вакансии", "Заполните название и описание вакансии.")
            return
        self._start_worker(cfg, validate_only=False)

    def _start_worker(self, cfg: RunConfig, validate_only: bool) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Выполняется задача", "Дождитесь завершения текущей операции.")
            return
        self._set_controls_enabled(False)
        self.overall_progress["value"] = 0
        self.detail_progress.stop()
        self.detail_progress.configure(mode="determinate")
        self.detail_progress["value"] = 0
        self.result_summary.configure(text="Выполнение...")
        self._clear_file_links()
        if not validate_only:
            self._set_report_text("")
        self.worker = threading.Thread(
            target=self._worker_run,
            args=(cfg, validate_only),
            daemon=True,
        )
        self.worker.start()

    def _worker_run(self, cfg: RunConfig, validate_only: bool) -> None:
        try:
            self._emit("stage", ("Проверка источника", 8))
            media_path = self._validate_and_prepare_source(cfg)
            if validate_only:
                self._emit("done_validate", None)
                return

            self._emit("stage", ("Транскрибация", 35))
            self._emit("detail_busy", True)
            transcriber = AudioTranscriber(model_size=cfg.model_size)
            transcript = transcriber.transcribe_file(media_path, language=cfg.language)
            self._emit("detail_busy", False)

            out_dir = Path(cfg.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            self._emit("log", f"Папка отчетов: {out_dir.resolve()}")
            interview_json_path = out_dir / f"{cfg.candidate_id}.json"
            transcript_txt_path = out_dir / f"{cfg.candidate_id}_transcript.txt"
            payload = {
                "candidate_id": cfg.candidate_id,
                "vacancy_title": cfg.vacancy_title,
                "vacancy_description": cfg.vacancy_description,
                "position_level": cfg.position_level,
                "interview_turns": [],
                "transcript": transcript,
                "metadata": {"source": "candidate_ss_gui", "created_at": datetime.now().isoformat(timespec="seconds")},
            }
            safe_write_text(transcript_txt_path, transcript)
            safe_write_json(interview_json_path, payload)
            self._emit("log", f"Сохранен транскрипт: {transcript_txt_path}")
            self._emit("log", f"Сохранен JSON интервью: {interview_json_path}")
            self._emit("stage", ("Оценка кандидата", 70))
            self._emit("detail_busy", True)

            interview = InterviewInput(
                candidate_id=cfg.candidate_id,
                vacancy_title=cfg.vacancy_title,
                vacancy_description=cfg.vacancy_description,
                position_level=cfg.position_level,
                interview_turns=[],
                transcript=transcript,
                metadata={"source": "candidate_ss_gui"},
            )
            if cfg.engine == "neural":
                evaluator = NeuralEvaluator(model_path=cfg.neural_model_path)
            else:
                evaluator = RuleBasedEvaluator()
            system = CandidateScreeningSystem(evaluator=evaluator)
            evaluation = system.run(interview)
            self._emit("detail_busy", False)
            self._emit("stage", ("Генерация отчета", 90))
            report_json_path, report_txt_path = system.save_outputs(evaluation, out_dir)
            report_text = report_txt_path.read_text(encoding="utf-8-sig", errors="replace")
            self._emit(
                "done",
                {
                    "summary": f"Готово: score={evaluation.score}, решение={evaluation.decision}",
                    "files": [transcript_txt_path, interview_json_path, report_json_path, report_txt_path],
                    "report_text": report_text,
                },
            )
        except Exception as exc:
            self._emit("error", f"{exc}\n\n{traceback.format_exc()}")

    def _validate_and_prepare_source(self, cfg: RunConfig) -> str:
        if cfg.source_type == "url":
            validation = validate_external_source(cfg.source_value)
            self._emit("log", f"URL: {validation.url}")
            self._emit(
                "log",
                "Проверка доступности: "
                + ("OK" if validation.reachable else "FAILED")
                + (f" (HTTP {validation.http_status})" if validation.http_status else ""),
            )
            self._emit("log", f"Определенный формат: {validation.extension or 'unknown'}")
            self._emit("log", f"Статус: {validation.reason}")
            if not validation.reachable or not validation.format_supported:
                raise RuntimeError("Источник не прошел валидацию и не может быть обработан.")

            self._emit("stage", ("Скачивание файла", 18))
            downloads_dir = Path(cfg.downloads_dir)
            downloads_dir.mkdir(parents=True, exist_ok=True)
            self._emit("log", f"Папка загрузок: {downloads_dir.resolve()}")

            def _dl_progress(downloaded: int, total: Optional[int]) -> None:
                if total and total > 0:
                    percent = int(downloaded * 100 / total)
                    self._emit("detail", percent)
                    self._emit(
                        "log_throttle",
                        f"Загрузка: {percent}% ({downloaded / (1024 * 1024):.1f}/{total / (1024 * 1024):.1f} MB)",
                    )
                else:
                    self._emit("log_throttle", f"Загрузка: {downloaded / (1024 * 1024):.1f} MB")

            result = download_external_media(
                url=cfg.source_value,
                downloads_dir=downloads_dir,
                progress_callback=_dl_progress,
            )
            if not result.downloaded_path:
                raise RuntimeError("Ошибка скачивания: файл не загружен.")
            self._emit("log", f"Файл загружен: {result.downloaded_path}")
            return str(result.downloaded_path)

        local = Path(cfg.source_value)
        if not local.exists():
            raise FileNotFoundError(f"Файл не найден: {local}")
        if local.suffix.lower() not in MEDIA_EXTENSIONS:
            allowed = ", ".join(sorted(MEDIA_EXTENSIONS))
            raise ValueError(f"Неподдерживаемый формат: {local.suffix}. Разрешены: {allowed}")
        self._emit("log", f"Локальный файл: {local}")
        self._emit("log", "Валидация формата: OK")
        return str(local)

    def _emit(self, event: str, payload: Any) -> None:
        self.events.put((event, payload))

    def _poll_events(self) -> None:
        try:
            while True:
                event, payload = self.events.get_nowait()
                self._handle_event(event, payload)
        except queue.Empty:
            pass
        self.root.after(120, self._poll_events)

    def _handle_event(self, event: str, payload: Any) -> None:
        if event == "log":
            self._append_log(str(payload))
            return
        if event == "log_throttle":
            # Reduces terminal-like spam while preserving real-time feedback.
            now = datetime.now().strftime("%H:%M:%S")
            if not hasattr(self, "_last_throttled"):
                self._last_throttled = ""
            if payload != self._last_throttled:
                self._append_log(f"[{now}] {payload}")
                self._last_throttled = str(payload)
            return
        if event == "stage":
            stage_text, progress = payload
            self.stage_label.configure(text=stage_text)
            self.overall_progress["value"] = progress
            self.detail_progress["value"] = 0
            return
        if event == "detail":
            self.detail_progress.configure(mode="determinate")
            self.detail_progress["value"] = int(payload)
            return
        if event == "detail_busy":
            busy = bool(payload)
            if busy:
                self.detail_progress.configure(mode="indeterminate")
                self.detail_progress.start(10)
            else:
                self.detail_progress.stop()
                self.detail_progress.configure(mode="determinate")
                self.detail_progress["value"] = 100
            return
        if event == "done_validate":
            self._append_log("Проверка источника завершена успешно.")
            self.stage_label.configure(text="Проверка завершена")
            self.overall_progress["value"] = 100
            self.detail_progress["value"] = 100
            self.result_summary.configure(text="Источник валиден и готов к запуску.")
            self._set_controls_enabled(True)
            return
        if event == "done":
            self.overall_progress["value"] = 100
            self.detail_progress.stop()
            self.detail_progress.configure(mode="determinate")
            self.detail_progress["value"] = 100
            self.stage_label.configure(text="Готово")
            summary = payload["summary"]
            files = payload["files"]
            report_text = payload["report_text"]
            self._append_log(summary)
            self.result_summary.configure(text=summary)
            self._set_report_text(report_text)
            self._set_file_links(files)
            self._set_controls_enabled(True)
            return
        if event == "error":
            self.detail_progress.stop()
            self.stage_label.configure(text="Ошибка")
            self._append_log("ОШИБКА:\n" + str(payload))
            self.result_summary.configure(text="Ошибка выполнения. См. журнал.")
            self._set_controls_enabled(True)
            messagebox.showerror("Ошибка", "Операция завершилась с ошибкой. Подробности в журнале.")

    def _set_report_text(self, content: str) -> None:
        self.report_text.configure(state="normal")
        self.report_text.delete("1.0", "end")
        self.report_text.insert("1.0", content)
        self.report_text.configure(state="disabled")

    def _clear_file_links(self) -> None:
        for child in self.file_links_frame.winfo_children():
            child.destroy()
        self.last_files = []

    def _set_file_links(self, files: list[Path]) -> None:
        self._clear_file_links()
        self.last_files = files
        for i, path in enumerate(files):
            label = ttk.Label(self.file_links_frame, text=path.name, style="Subtitle.TLabel")
            label.grid(row=i, column=0, sticky="w", pady=2)
            btn = ttk.Button(self.file_links_frame, text="Открыть", command=lambda p=path: os.startfile(str(p)))
            btn.grid(row=i, column=1, sticky="e", pady=2, padx=(8, 0))


def main() -> int:
    root = tk.Tk()
    CandidateSSApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
