# MLOps Toolkit

Инструментарий для развёртывания ML-моделей в продакшене, демонстрирующий обслуживание моделей через Docker, мониторинг с помощью Prometheus/Grafana, версионирование моделей и инфраструктуру A/B-тестирования.

## Архитектура

```
┌─────────────────────────────────────────────────────────────────┐
│                        Docker Compose                           │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │              │    │              │    │              │      │
│  │  Сервер      │───>│  Prometheus  │───>│   Grafana    │      │
│  │  моделей     │    │              │    │              │      │
│  │  (FastAPI)   │    │  :9090       │    │  :3000       │      │
│  │  :8000       │    │              │    │              │      │
│  │              │    └──────────────┘    └──────────────┘      │
│  │  ┌────────┐  │                                              │
│  │  │Реестр  │  │    ┌──────────────┐                          │
│  │  │A/B-тест│  │    │              │                          │
│  │  │Метрики │  │───>│    Redis     │                          │
│  │  │Дрифт   │  │    │    :6379     │                          │
│  │  └────────┘  │    │              │                          │
│  └──────────────┘    └──────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

**Компоненты:**

| Сервис | Порт | Описание |
|--------|------|----------|
| **Сервер моделей** | 8000 | FastAPI-сервер с эндпоинтами для предсказаний, управления моделями, A/B-тестирования и мониторинга |
| **Prometheus** | 9090 | Сбор метрик и хранение временных рядов, опрашивает сервер моделей каждые 15 секунд |
| **Grafana** | 3000 | Предварительно настроенный дашборд с мониторингом частоты запросов, задержки, уверенности и дрифта данных |
| **Redis** | 6379 | Слой кэширования для сервера моделей |

## Быстрый старт

### 1. Обучение примеров моделей

```bash
pip install scikit-learn joblib numpy
python scripts/train_example_model.py
```

Скрипт обучает две модели sklearn (LogisticRegression v1, RandomForest v2) на датасете Iris и сохраняет их в каталог `models/`.

### 2. Запуск стека

```bash
cp .env.example .env
docker compose up --build
```

### 3. Проверка

- **Сервер моделей**: http://localhost:8000/health
- **Документация API**: http://localhost:8000/docs
- **Prometheus**: http://localhost:9090
- **Дашборд Grafana**: http://localhost:3000 (настроен автоматически, вход не требуется)

### 4. Генерация трафика

```bash
# Запуск примера рабочего процесса
python examples/deploy_sklearn_model.py

# Или нагрузочное тестирование для визуализации в Grafana
bash scripts/load_test.sh
```

## Документация API

### Предсказания

#### `POST /predict`

Запуск инференса на входных признаках. При настроенном A/B-тестировании запросы маршрутизируются автоматически.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

Ответ:
```json
{
  "prediction": 0,
  "confidence": 0.9714,
  "model_version": "v1",
  "latency_ms": 1.23,
  "timestamp": "2024-01-01T00:00:00+00:00"
}
```

Опционально: укажите `"model_version": "v2"`, чтобы обойти A/B-маршрутизацию и использовать конкретную версию.

#### `GET /health`

```json
{"status": "healthy", "active_model": "v1", "registered_models": 2}
```

#### `GET /metrics`

Эндпоинт метрик в формате Prometheus (автоматически опрашивается Prometheus).

#### `GET /model/info`

Список всех зарегистрированных версий моделей с метаданными и метриками.

### Управление моделями

#### `POST /model/register`

Регистрация новой версии модели. Файл модели должен существовать по указанному пути.

```bash
curl -X POST http://localhost:8000/model/register \
  -H "Content-Type: application/json" \
  -d '{
    "version": "v3",
    "model_path": "/app/models/v3/model.pkl",
    "framework": "sklearn",
    "metrics": {"accuracy": 0.96},
    "description": "Improved model with feature engineering"
  }'
```

#### `POST /model/activate`

Переключение активной версии модели (без простоя).

```bash
curl -X POST http://localhost:8000/model/activate \
  -H "Content-Type: application/json" \
  -d '{"version": "v2"}'
```

#### `POST /model/rollback`

Откат к ранее активной версии модели.

```bash
curl -X POST http://localhost:8000/model/rollback
```

### A/B-тестирование

#### `POST /ab-test/configure`

Настройка распределения трафика между двумя версиями моделей.

```bash
curl -X POST http://localhost:8000/ab-test/configure \
  -H "Content-Type: application/json" \
  -d '{
    "model_a_version": "v1",
    "model_b_version": "v2",
    "traffic_split": 0.8,
    "enabled": true
  }'
```

- `traffic_split: 0.8` означает, что 80% запросов направляются к модели A, 20% — к модели B
- Установите `enabled: false`, чтобы направлять весь трафик на модель A без удаления конфигурации

#### `GET /ab-test/results`

Получение агрегированной статистики A/B-теста.

```json
{
  "total_requests": 100,
  "model_a_count": 79,
  "model_b_count": 21,
  "model_a_avg_latency": 1.45,
  "model_b_avg_latency": 2.31,
  "model_a_avg_confidence": 0.9234,
  "model_b_avg_confidence": 0.9567
}
```

#### `POST /ab-test/reset`

Сброс накопленных результатов A/B-теста.

### Дрифт данных

#### `GET /drift/status`

Получение текущего состояния обнаружения дрифта данных (KL-дивергенция).

```json
{
  "kl_divergence": 0.034,
  "is_drifting": false,
  "feature_scores": [0.012, 0.045, 0.028, 0.051],
  "threshold": 0.1,
  "window_size": 150
}
```

## Руководство по регистрации моделей

### Структура каталогов

Модели хранятся в версионированных каталогах:

```
models/
├── v1/
│   ├── model.pkl          # модель sklearn (joblib)
│   └── metadata.json
├── v2/
│   ├── model.pkl
│   └── metadata.json
├── v3/
│   ├── model.pt           # модель PyTorch
│   └── metadata.json
└── active_version.txt     # Текущая активная версия
```

### Поддерживаемые фреймворки

| Фреймворк | Формат файла | Расширение |
|-----------|-------------|-----------|
| scikit-learn | joblib | `.pkl` |
| PyTorch | torch.save | `.pt` |

### Регистрация через скрипт

```python
from sklearn.linear_model import LogisticRegression
import joblib

# Обучение
model = LogisticRegression()
model.fit(X_train, y_train)

# Сохранение
joblib.dump(model, "models/v3/model.pkl")

# Регистрация через API
import requests
requests.post("http://localhost:8000/model/register", json={
    "version": "v3",
    "model_path": "/app/models/v3/model.pkl",
    "framework": "sklearn",
    "metrics": {"accuracy": 0.96},
    "description": "My new model"
})
```

### Регистрация через предварительно заполненный каталог

1. Сохраните модель и `metadata.json` в `models/v3/`
2. Перезапустите сервер — он автоматически обнаруживает модели при запуске

Формат `metadata.json`:
```json
{
  "version": "v3",
  "framework": "sklearn",
  "created_at": "2024-01-01T00:00:00+00:00",
  "metrics": {"accuracy": 0.96, "f1_score": 0.95},
  "description": "Model description"
}
```

## Руководство по A/B-тестированию

### Настройка

1. Зарегистрируйте как минимум две версии моделей
2. Настройте A/B-тест:

```bash
curl -X POST http://localhost:8000/ab-test/configure \
  -d '{"model_a_version":"v1","model_b_version":"v2","traffic_split":0.8,"enabled":true}'
```

3. Отправляйте запросы на предсказание как обычно — маршрутизация выполняется автоматически:

```bash
curl -X POST http://localhost:8000/predict \
  -d '{"features":[5.1,3.5,1.4,0.2]}'
```

### Мониторинг результатов

```bash
# Проверка статистики A/B-теста
curl http://localhost:8000/ab-test/results

# Просмотр в Grafana — метрики разделены по model_version
open http://localhost:3000
```

### Продвижение победителя

```bash
# После сбора достаточного количества данных продвиньте лучшую модель
curl -X POST http://localhost:8000/model/activate -d '{"version":"v2"}'

# Отключение A/B-теста
curl -X POST http://localhost:8000/ab-test/configure \
  -d '{"model_a_version":"v2","model_b_version":"v2","traffic_split":1.0,"enabled":false}'
```

## Дашборд Grafana

Дашборд предварительно настроен и доступен сразу по адресу http://localhost:3000.

**Панели:**

| Панель | Описание |
|--------|----------|
| Частота запросов | Количество запросов в секунду по версии модели |
| Перцентили задержки | Гистограммы задержки p50, p95, p99 |
| Частота ошибок | Количество ошибок в секунду по типу ошибки |
| Уверенность предсказаний | Распределение оценок уверенности |
| Активная версия модели | Текущая обслуживающая модель |
| Оценка дрифта данных | Индикатор KL-дивергенции с пороговыми оповещениями |
| Всего запросов | Накопительный счётчик запросов |
| Средняя задержка | Среднее время обработки запроса |

## Метрики Prometheus

| Метрика | Тип | Метки | Описание |
|---------|-----|-------|----------|
| `model_request_total` | Counter | model_version, status | Общее количество запросов на предсказание |
| `model_request_latency_seconds` | Histogram | model_version | Задержка запроса в секундах |
| `model_prediction_confidence` | Histogram | model_version | Оценки уверенности предсказаний |
| `model_errors_total` | Counter | error_type | Общее количество ошибок по категориям |
| `model_active_version` | Gauge | version | Индикатор активной версии модели |
| `model_data_drift_score` | Gauge | — | Оценка дрифта по KL-дивергенции |
| `model_server_info` | Info | — | Информация о версии сервера и фреймворке |

## Запуск тестов

```bash
cd model_server
pip install -r requirements.txt pytest pytest-asyncio
pytest tests/ -v
```

## Структура проекта

```
mlops-toolkit/
├── docker-compose.yml           # Оркестрация полного стека
├── requirements.txt             # Зависимости для разработки
├── .env.example                 # Шаблон переменных окружения
├── model_server/
│   ├── Dockerfile
│   ├── requirements.txt         # Продакшен-зависимости
│   ├── app/
│   │   ├── main.py              # Приложение FastAPI
│   │   ├── config.py            # Настройки Pydantic
│   │   ├── serving/
│   │   │   ├── model_registry.py   # Версионированное управление моделями
│   │   │   ├── predictor.py        # Фреймворк-независимый инференс
│   │   │   └── ab_testing.py       # Распределение трафика и сравнение
│   │   ├── monitoring/
│   │   │   ├── metrics.py          # Определения метрик Prometheus
│   │   │   └── data_drift.py       # Обнаружение дрифта по KL-дивергенции
│   │   └── middleware/
│   │       └── logging_middleware.py  # Структурированное логирование в JSON
│   └── tests/
├── monitoring/
│   ├── prometheus/prometheus.yml
│   └── grafana/                 # Предварительно настроенные дашборды
├── examples/                    # Примеры развёртывания
├── scripts/                     # Обучение и нагрузочное тестирование
├── notebooks/demo.ipynb         # Интерактивное пошаговое руководство
└── models/                      # Артефакты моделей (исключены из git)
```

## Технологический стек

- **Python 3.11** / **FastAPI** — API для обслуживания моделей
- **prometheus-client** — Инструментирование метрик
- **Prometheus** — Сбор и хранение метрик
- **Grafana** — Дашборды мониторинга
- **Redis** — Слой кэширования
- **Docker / Docker Compose** — Оркестрация контейнеров
- **scikit-learn / PyTorch** — ML-фреймворки
- **scipy** — Статистическое обнаружение дрифта
