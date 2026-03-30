# Candidate Screening System

Полноценный Python-прототип системы первичного отбора кандидатов.

## Что умеет
- Принимает интервью в формате JSON (вопросы-ответы или единый транскрипт).
- Сопоставляет ответы с описанием вакансии.
- Выдаёт итоговый балл (0–100), решение (рекомендован / условно рекомендован / не рекомендован) и подробный HR-отчёт.
- Сохраняет результат в `.json` и `.txt`.
- Работает полностью офлайн на основе правил (rule‑based).
- Может обращаться к внешнему Yandex‑совместимому LLM endpoint (опционально) с автоматическим fallback на rule‑based.

## Установка

```bash
git clone https://github.com/username/Candidate_Screening_System.git
cd Candidate_Screening_System
# (опционально) python -m venv venv && source venv/bin/activate
pip install -r requirements.txt   # (файл requirements.txt добавлен, содержит pytest, flake8, mypy)

## Запуск

### 1. Создать demo-файлы
```bash
python candidate_screening_system.py --demo --output-dir ./output
```

### 2. Запустить анализ
```bash
python candidate_screening_system.py --input ./output/sample_interview.json --output-dir ./output
```

### 3. Запуск через внешний LLM
```bash
export YANDEX_LLM_URL="https://..."
export YANDEX_LLM_API_KEY="..."
export YANDEX_FOLDER_ID="..."
python candidate_screening_system.py --engine yandex --input ./output/sample_interview.json --output-dir ./output
```

## Формат входного JSON
```json
{
  "candidate_id": "candidate_001",
  "vacancy_title": "Менеджер по работе с клиентами",
  "vacancy_description": "Описание вакансии...",
  "position_level": "junior",
  "interview_turns": [
    {
      "question": "Расскажите о себе",
      "answer": "Ответ кандидата"
    }
  ],
  "metadata": {
    "source": "demo"
  }
}
```
