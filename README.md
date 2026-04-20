# Candidate Screening System

Система первичного скрининга кандидатов:
1. принимает данные интервью в аудио, видео;
2. проводит транскрибацию аудио или аудио из видео через Whisper;
3. передаёт результат в neural- или rule-base-оценщик;
4. сохраняет готовый отчёт в `.json` и `.txt`.

## Установка с помощью installer (Windows)

1. скачиваем installer с Google Drive по ссылке -> https://clck.ru/3TBpLv ;
2. следуем инструкциям в установщике;
3. пользуемся CandidateSS :)

## Установка через терминал (Windows, Linux, macOS)
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

## Использование программы без GUI (через терминал на Linux и MacOS)

(!) Ведется подготовка быстрых и интуитивно внятных скриптов (v.1.1.0)

## Полный конвейер (аудио / аудио из видео -> оценка)
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

## Быстрый запуск (JSON -> оценка)
```bash
python candidate_screening_system.py --demo --output-dir output
python candidate_screening_system.py --input output/sample_interview.json --output-dir output
```

## Аудио -> TXT/JSON
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

## Информация для разработчиков (!)

## Обучение neural-оценщика (текст)
```bash
pip install -r requirements-ml.txt
python training/train_evaluator.py \
  --data training/training_data.json \
  --model-out training/model_bundle.joblib \
  --metrics-out training/metrics.json
```

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

## Авто-тесты
```bash
pytest -q
```

## Примечание по репозиторию
В `.gitignore` исключены:
- кэши и временные артефакты;
- `output/`;
- тяжёлые чекпоинты Whisper;
- сырые локальные обучающие аудиоданные;
- примеры данных.
