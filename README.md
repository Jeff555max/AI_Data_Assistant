# Data Assistant

AI-чат на FastAPI + Jinja2 + HTMX, который использует OpenAI API для ответов по данным и анализа изображений, а локально строит графики и сохраняет артефакты в файлы.

## Что умеет

- принимать обычные текстовые сообщения;
- загружать CSV, Excel, JSON и изображения;
- использовать OpenAI для анализа контекста, данных и изображений;
- строить `line`, `bar` и `histogram` графики в PNG;
- собирать DOCX-отчёты;
- сохранять markdown-summary;
- показывать и скачивать артефакты прямо из чата.

## Настройка OpenAI

Добавьте в `.env`:

```env
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-5-mini
OPENAI_MAX_HISTORY_MESSAGES=8
```

## Локальный запуск

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Откройте `http://localhost:8000`.

## Docker

```bash
docker-compose up --build -d
```

## Пример данных

Используйте файл `examples/sample_sales.csv`.
# AI_Data_Assistant
