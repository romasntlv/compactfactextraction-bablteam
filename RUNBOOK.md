# CompactFactExtraction — Qwen, команда 3

Одна Qwen2.5-0.5B-Instruct с объединённым LoRA checkpoint-2142. Без date_extractor и без каскада моделей. Вход — полный звонок с метаданными, выход — факты официальной схемы. Результаты до упаковки: temporal **51,25% (328/640)**, state **56,82% (354/623)** на 383 звонках val. Повторная генерация GPU-релиза проверяется отдельным Colab notebook; сохранённые предсказания не подменяют этот тест.

## Установка и веса

Python 3.11. Все команды ниже выполняются из этой папки. Для чистого окружения:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.lock
python tools/fetch_base.py
python tools/merge_model.py
```

Для CUDA используется Torch 2.8.0 из `https://download.pytorch.org/whl/cu128`. `requirements.lock` фиксирует прямые зависимости; фактические транзитивные версии проверенного окружения сохраняются отдельно в reports. Это не универсальный hash-lock для всех ОС. Точный граф зависимостей проверенного Linux CPU-образа (без платформенного Torch) также сохранён в requirements-linux.lock; его можно установить вместо requirements.lock для повторения этого окружения.

Готовые merged-веса и адаптер доступны отдельными архивами в GitHub Release; распакуйте их из корня репозитория. В небольшом Colab ZIP включён адаптер, а база скачивается закреплённой ревизии `7ae557604adf67be50417f59c2c2f167def9a775`. Merge производится в FP16 на CPU, затем один merged-файл может загружаться в FP32 на CPU либо FP16 на GPU. На инференсе сеть не нужна. Проверка повторной GPU-генерации нужна в том числе из-за переноса Torch и устройства merge.

Весов после merge: **494 032 768 параметров**, исходный адаптер — **17 596 416 обучаемых параметров**. Архитектура, генерация и контекст закреплены в configs/inference.json. Лицензия базы скопирована в models/LICENSE.Qwen.

## CLI

Вход JSONL не должен содержать facts — даже пустое поле отклоняется, чтобы исключить случайную передачу эталона. Входные файлы без gold находятся в data/*_input.jsonl.

```bash
python -m src.cli --input data/curve24_input.jsonl --output benchmarks/curve24_cpu.jsonl --device cpu --dtype float32 --threads 4 --warmup
python tools/score.py data/curve24_gold.jsonl benchmarks/curve24_cpu.jsonl reports/curve24_cpu.json
```

Для GPU: `--device cuda --dtype float16`. Выходной файл не перезаписывается без явного выбора нового имени. `--resume` продолжает совместимый частичный прогон; конфигурация и целостность записей проверяются. Каждый звонок сразу записывается вместе с сырым ответом и временем. Лимит отдельных тестов: `--limit N`; полноценная оценка требует соответствующего gold-поднабора.

Файлы результата:

- `*.jsonl` — call_id и facts;
- `*.timings.jsonl` — время каждого звонка, окна, токены, ошибки, partial;
- `*.raw.jsonl` — модельные JSON-ответы до парсинга;
- `*.segments.jsonl` — оборудование, версии, время загрузки/прогрева и пиковая память каждого сегмента;
- `*.run_config.json` — идентификатор совместимости возобновления.

Таймер звонка включает токенизацию, генерацию, парсинг и проверку фактов; CUDA синхронизируется. Загрузка модели, прогрев, файловый ввод/вывод и HTTP измеряются отдельно. RSS семплируется каждые 20 мс и может пропустить краткий пик. GPU allocated/reserved — отдельные показатели PyTorch, не RAM процесса.

## HTTP API

```bash
DEVICE=cpu CPU_THREADS=4 MODEL_DIR=./models/merged python -m uvicorn src.api:app --host 127.0.0.1 --port 8000 --workers 1
```

```bash
curl http://127.0.0.1:8000/health
curl -H 'Content-Type: application/json' --data-binary @data/example_request.json http://127.0.0.1:8000/extract
```

`POST /extract` принимает объект `call_id`, `started_at`, `turns` (каждая реплика: `idx`, `speaker=client|operator`, `text`). Остальные метаданные допустимы. Возвращает `{"call_id":"...","facts":[...]}`. OpenAPI: `/docs`.

- Модель загружается один раз при старте; один worker, генерация запросов сериализована блокировкой. Не запускайте несколько workers без отдельного замера памяти.
- Ссылка на реплику строится по idx, а не позиции массива. quote и speaker берутся из исходной реплики. span=null — разрешён схемой, поэтому grounding_span=N/A.
- Относительные смещения дат преобразуются в ISO относительно started_at. Сохраняется исходная prompt-аннотация дат из оценщика 2.1; gold при этом не используется. Новых правил date_extractor нет.
- `facts: []` — корректный ответ при отсутствии фактов. Невалидный вход → 422; полностью неразбираемая генерация → 502; техническая ошибка → 500.
- Если часть строк/окон не разобрана или достигнут лимит генерации, валидные факты возвращаются с `X-CFE-Partial: true`. Заголовки `X-CFE-Invalid-JSON`, `X-CFE-Invalid-Rows`, `X-CFE-Limit-Hits` показывают причины; `X-CFE-Model-Seconds` — время извлечения. Клиенту следует учитывать partial, а не воспринимать такой ответ как гарантированно полный.

Настоящий HTTP-бенчмарк уже запущенного сервиса:

```bash
python tools/benchmark_http.py --input data/curve24_input.jsonl --output benchmarks/http_cpu.jsonl
```

## Docker

Веса монтируются read-only отдельно. В образ не попадают train/val, эталоны, адаптер и экспериментальные отчёты.

```bash
docker build --target cpu -t cfe-qwen:cpu .
docker run --rm -p 127.0.0.1:8000:8000 -v "$PWD/models/merged:/models:ro" cfe-qwen:cpu
```

```bash
docker build --target gpu -t cfe-qwen:gpu .
docker run --rm --gpus all -p 127.0.0.1:8000:8000 -v "$PWD/models/merged:/models:ro" cfe-qwen:gpu
```

Для стенда Linux x86_64 собирать с `--platform linux/amd64` на соответствующем стенде; нативная ARM64-сборка на Mac не является проверкой x86_64/CUDA. Образ GPU требует NVIDIA Container Toolkit. Статус фактически выполненных проверок — reports/READINESS.md. Проверенный ARM64 CPU-образ и merged-веса поставляются в GitHub Release v1.0.0; команды загрузки — в корневом README.

## Проверки и GPU-эксперименты

```bash
python -m unittest discover -s tests -v
python tools/verify_raw_codec.py
python tools/prepare_checks.py
python tools/summarize_run.py benchmarks
```

Проверка codec воспроизводит 385/385 dev и 383/383 val из сохранённых сырых ответов модели; это проверка переноса сборщика, не новая генерация. Тесты HTTP-контракта с тестовой моделью также не доказывают качество настоящего Qwen.

Для повторных GPU-проверок используйте `notebooks/release_checks_colab.ipynb` и архив `Qwen_release_Colab.zip` из GitHub Release. Все вычислительные вызовы идут в видимых ячейках, без фонового subprocess. Прогоны: полный dev, сокращённый контекст на заранее выбранных 24 звонках, два уровня шума на 96 звонках, 5 ловушек, полный val, настоящий HTTP-тест. Сохраните `Qwen_release_results.zip`. Уже завершённые проверки перечислены в reports/READINESS.md; исходный notebook содержит и дополнительные незавершённые проверки.

Три точки кривой используют одни и те же 24 dev-звонка, отобранные по длине текста без gold. CPU-точка измеряется локально, GPU-точки — на одной T4. Ограниченный набор означает высокую неопределённость оценки качества: это ресурсное сравнение, не замена официального val. Пиковая память GPU-поднабора наследуется от полного прогона и так обозначается.

## Воспроизводимое обучение

Исходный train/dev split по аккаунтам, seed 42. Базовый корпус организаторов синтетический; в коде историческое название «real» означает исходные окна корпуса, а не реальные клиентские звонки. Добавлены 1600 окон синтетических ловушек.

```bash
python train/build_training.py
python train/train.py --epochs 2 --train-data train/prepared/train_augmented.jsonl --output train/run
python train/predict.py --checkpoint train/run/checkpoint-2142 --manifest train/run/model_manifest.json --input train/prepared/dev_input.jsonl --output train/run/dev.jsonl
python train/evaluate.py train/prepared/dev_gold.jsonl train/run/dev.jsonl
```

Подготовка проверяет точный SHA256 всех 17 133 обучающих окон: `c784e302cab2604ce664334093d9061d9ecc661e7541d314bbd31ddf72445087`. train/train.py закреплён на исходной ревизии базы, выводит эпохи, шаги и loss непосредственно в консоль. Конфигурация: LoRA r32/alpha64/dropout0.05, lr2e-4, microbatch1, accumulation16, FP16, gradient checkpointing, max_length2048.

Для нового обучения оценивать каждый checkpoint на dev по train/evaluate.py:key; имя checkpoint-2142 соответствует исходным данным и batch-настройкам, не является универсальным выбором для изменённого обучения. Официальный val запускать после фиксации выбора. Сохранены manifest, исходный dev_selection и архивный notebook как свидетельства обученного запуска. Полное переобучение при упаковке не повторялось; seed не гарантирует побитовую идентичность между версиями CUDA/Torch.

## Отчёты

- reports/REPORT.md — архитектура, качество, ошибки, ограничения и ресурсы;
- reports/READINESS.md — что проверено и что ещё ждёт выполнения;
- reports/*_reference_score.json — воспроизведённая официальная оценка;
- benchmarks/reports/RESULTS.md — новые измерения, формируется из завершённых прогонов;
- reports/checks_manifest.json — предопределённые поднаборы и хеши.

### Исправление Colab: torchvision::nms

При замене предустановленного Torch его torchvision/torchaudio могут остаться несовместимыми. В обновлённом notebook удаляются необязательные torchvision, torchaudio и librosa; после установки требуется Runtime → Restart session. Следующая ячейка восстанавливает пути и явно проверяет импорт PreTrainedModel/Qwen2ForCausalLM. Это изменение окружения, веса и алгоритм инференса не меняются.
