# Чат-бот для абитуриентов НИУ ВШЭ

RAG-система для ответов на вопросы абитуриентов бакалавриата НИУ ВШЭ (кампус Москва) на основе официальных документов и страниц приёмной комиссии.

Репозиторий: <https://github.com/slanalan1203/ChatBotForApplicants>

## Архитектура

- **Источники**: правила приёма, приложения, перечни олимпиад (PDF), страницы сайта `ba.hse.ru`, словарь абитуриента, FAQ.
- **Ингест**: парсинг → очистка → чанкинг (RecursiveCharacterTextSplitter, 2000 символов с перекрытием 200) → индексация в ChromaDB.
- **Эмбеддер**: `BAAI/bge-m3`.
- **Генератор**: `mistral:latest` через Ollama, температура 0.0, simple-промпт.
- **Интерфейс**: Telegram-бот на `aiogram` (long polling), напрямую вызывающий RAG-пайплайн.
- **Опциональный режим**: ReAct-агент с инструментом `search_kb` (`src/rag/agent.py`).

## Структура репозитория

```
src/
  ingestion/   парсеры PDF и HTML, очистка, чанкинг, индексатор
  rag/         retriever, generator, pipeline, agent
  bot/         aiogram-обвязка
  eval/        retrieval-метрики и runner
scripts/       сборка KB, эвал, разметка golden set
data/
  golden_set.jsonl          120 вопросов с разметкой relevant_chunk_ids
  golden_dev.jsonl          85 вопросов (стратифицированный сплит)
  golden_test.jsonl         35 вопросов
  eval/                     результаты экспериментов 0–6
cloud/
  setup.sh                  установка окружения на GPU-VM
```

## Запуск

Требуется Python 3.11+, локально установленный Ollama, ChromaDB.

```bash
pip install -e .

ollama pull mistral:latest

python scripts/download_docs.py
python scripts/ingest_all.py --reset

python -m src.rag.pipeline "Какие документы нужны для поступления?"
python -m src.bot.main
```

Переменные окружения — в `.env.example`.

## Эксперименты

Результаты — в `data/eval/`:

- `exp0/` — расширение базы знаний (5 конфигураций).
- `exp1/` — предобработка (3 конфигурации).
- `exp2/` — чанкинг (3 конфигурации).
- `exp3/` — сравнение эмбеддеров (4 модели).
- `exp3_5/` — гибридный поиск и переписывание запроса.
- `exp4/` — реранкер.
- `exp5/` — генерация (модель × промпт × температура).
- `exp6/` — AgenticRAG на multi-hop вопросах.
- `exp_final_test.json` — прогон лучшей конфигурации на запечатанном тестовом наборе.
