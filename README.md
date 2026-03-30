# Candidate Screening System

Полноценный Python-прототип системы первичного отбора кандидатов.

## Что умеет
- принимает интервью в JSON;
- анализирует ответы кандидата;
- сопоставляет ответы с описанием вакансии;
- выдает итоговый балл, решение и HR-отчет;
- сохраняет результат в `.json` и `.txt`;
- может работать полностью офлайн;
- при наличии API-параметров может обращаться к Yandex-совместимому LLM endpoint.

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
