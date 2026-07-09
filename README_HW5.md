# HW-5 — CI/CD

## Мета роботи

У межах домашньої роботи 5 було налаштовано CI/CD-процес для MLOps-проєкту.

Мета роботи — автоматизувати процес доставки ML-моделі у production за допомогою GitHub Actions. Було побудовано pipeline, який охоплює два основні етапи: тренування моделі та тестовий деплой сервісу для інференсу.

Проєкт спирається на попередні домашні роботи, де вже були реалізовані тренування моделі, трекінг експериментів у MLflow та FastAPI-сервіс для інференсу.

## Що було реалізовано

У проєкті реалізовано:

- налаштування CI/CD pipeline за допомогою GitHub Actions;
- автоматичний запуск pipeline після push у GitHub;
- ручний запуск pipeline через workflow_dispatch;
- запуск pipeline після публікації release;
- запуск pipeline за розкладом;
- автоматичну перевірку датасету;
- автоматичне розділення даних на train/test;
- автоматичне тренування моделей;
- логування експериментів у MLflow;
- реєстрацію найкращої моделі в MLflow Model Registry;
- встановлення production alias для актуальної моделі;
- тестовий деплой FastAPI inference API;
- перевірку endpoint-ів `/health` та `/predict`;
- генерацію Evidently data drift report;
- збереження артефактів виконання pipeline.

## CI/CD інструмент

Для реалізації CI/CD використано **GitHub Actions**.

Файл pipeline знаходиться у репозиторії за шляхом:

```
.github/workflows/mlops-ci-cd.yml
```

Назва workflow у GitHub Actions:

```
MLOps CI/CD Pipeline
```

## Змінні середовища

Для роботи pipeline використовується змінна середовища MLFLOW_TRACKING_URI.

Вона задана безпосередньо у файлі GitHub Actions workflow:

MLFLOW_TRACKING_URI=http://127.0.0.1:5000

Ця змінна вказує адресу MLflow tracking server, який запускається всередині CI-середовища GitHub Actions під час виконання pipeline.

Окремі GitHub Secrets у цьому pipeline не використовуються, оскільки MLflow server запускається локально всередині workflow.

Назва зареєстрованої моделі:

sentiment-classifier

використовується у коді проєкту для реєстрації найкращої моделі в MLflow Model Registry.

## Тригери запуску pipeline

Pipeline налаштовано на кілька типів запуску.

### Push у гілку

Pipeline автоматично запускається після push у гілки:

- `main`
- `final-project`

### Release

Pipeline запускається після публікації нового GitHub Release.

### Ручний запуск (workflow_dispatch)

Pipeline можна запустити вручну через вкладку GitHub Actions. Кроки:

1. відкрити репозиторій на GitHub;
2. перейти у вкладку **Actions**;
3. вибрати workflow **MLOps CI/CD Pipeline**;
4. натиснути **Run workflow**;
5. вибрати потрібну гілку (наприклад, `main`);
6. натиснути **Run workflow** для запуску та дочекатись завершення job `train-and-deploy`.

### Запуск за розкладом (schedule)

Pipeline також має scheduled trigger.

Розклад запуску:

```
щопонеділка о 06:00 UTC
```

## Структура CI/CD pipeline

Pipeline складається з таких етапів:

1. Checkout repository.
2. Set up Python 3.11.
3. Install dependencies.
4. Validate dataset.
5. Split dataset.
6. Start MLflow tracking server.
7. Train models and track experiments.
8. Register best model.
9. Generate Evidently drift report.
10. Deploy FastAPI service in CI environment.
11. Smoke test prediction endpoint.
12. Upload MLflow and report artifacts.

## Етап підготовки даних

На початку pipeline виконується перевірка датасету:

```bash
python scripts/validate_data.py
```

Скрипт перевіряє, що датасет існує, має правильну структуру та містить необхідні поля для навчання моделі.

Після цього виконується розділення датасету на train/test:

```bash
python scripts/split_data.py
```

У результаті створюються файли:

- `data/processed/train.csv`
- `data/processed/test.csv`

## Етап тренування моделі

Для тренування моделей у pipeline запускається команда:

```bash
python scripts/train_experiments.py
```

На цьому етапі виконується тренування моделей для задачі класифікації тональності українських текстових відгуків.

Модель класифікує відгуки на три класи:

- `positive`
- `negative`
- `neutral`

Під час тренування використовуються TF-IDF-ознаки тексту та кілька варіантів моделей.

У MLflow логуються:

- параметри моделей;
- метрики;
- артефакти;
- classification report;
- збережені моделі.

Основні метрики:

- Accuracy
- F1 macro

Найкращий результат показала модель:

```
logreg_word_bigram_tfidf
```

Метрики найкращої моделі:

- **Accuracy:** 0.6897
- **F1 macro:** 0.6867

## Етап реєстрації моделі

Після тренування pipeline автоматично запускає скрипт:

```bash
python scripts/register_best_model.py
```

Цей скрипт знаходить найкращий MLflow run, реєструє модель у MLflow Model Registry та встановлює для неї production alias.

Зареєстрована модель:

```
sentiment-classifier
```

Production URI моделі:

```
models:/sentiment-classifier@production
```

Саме ця production-модель використовується FastAPI-сервісом для інференсу.

## Етап деплою

Після тренування та реєстрації моделі pipeline виконує тестовий деплой FastAPI-сервісу в CI-середовищі.

Для запуску API використовується команда:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Після запуску сервісу pipeline перевіряє endpoint:

```
GET /health
```

Успішна відповідь має такий вигляд:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_uri": "models:/sentiment-classifier@production"
}
```

Це підтверджує, що FastAPI-сервіс запустився, а production-модель успішно завантажена.

## Smoke test endpoint-а /predict

Після перевірки `/health` pipeline виконує smoke test endpoint-а:

```
POST /predict
```

Тестовий запит:

```json
{
  "texts": [
    "Сервіс працює добре",
    "Мені не сподобалось обслуговування",
    "Замовлення оформлено"
  ]
}
```

Успішна відповідь API містить передбачення:

```json
{
  "predictions": ["positive", "negative", "neutral"],
  "latency_seconds": 0.001,
  "model_uri": "models:/sentiment-classifier@production"
}
```

Це підтверджує, що після тестового деплою inference API реально працює та класифікує текстові відгуки.

## Evidently report

Pipeline також автоматично генерує Evidently data drift report.

Команда:

```bash
python scripts/evidently_report.py
```

Звіт зберігається у файлі:

```
reports/data_drift_report.html
```

Для порівняння використовуються:

- reference dataset: `data/processed/train.csv`
- current dataset: `data/processed/test.csv`

Для drift-аналізу використовуються ознаки:

- `text_length`;
- `word_count`;
- `avg_word_length`;
- `label`.

Цей крок дозволяє перевіряти, чи не змінився розподіл даних між базовою та поточною вибіркою.

## Artifacts

Після завершення pipeline GitHub Actions зберігає артефакти виконання.

Назва artifact:

```
mlops-artifacts
```

До артефактів входять:

- MLflow artifacts;
- `mlflow.db`;
- model files;
- Evidently reports;
- `mlflow.log`;
- `api.log`.

Це дозволяє після завершення pipeline переглянути результати тренування, модельні файли, звіти та логи.

## Результат виконання

Pipeline було успішно запущено в GitHub Actions.

Статус запуску: **Success**

Успішно виконано job: `train-and-deploy`

> Посилання на успішний прогін: `https://github.com/OwLIRV/mlops-final-project-sentiment/actions/runs/29015287549/job/86108644568`

Це підтверджує, що CI/CD pipeline реально працює та автоматизує основні етапи доставки ML-моделі:

- підготовку даних;
- тренування моделі;
- реєстрацію найкращої моделі;
- тестовий деплой FastAPI-сервісу;
- перевірку endpoint-ів `/health` та `/predict`;
- збереження артефактів виконання.

## Як запустити pipeline вручну

Щоб вручну запустити pipeline:

1. відкрити GitHub repository;
2. перейти у вкладку **Actions**;
3. вибрати **MLOps CI/CD Pipeline**;
4. натиснути **Run workflow**;
5. вибрати гілку `main`;
6. натиснути **Run workflow**;
7. дочекатися завершення job `train-and-deploy`.

## Як запустити основні частини локально

**Встановлення залежностей**

```bash
pip install -r requirements.txt
```

**Валідація датасету**

```bash
python scripts/validate_data.py
```

**Розділення даних на train/test**

```bash
python scripts/split_data.py
```

**Запуск MLflow**

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

**Тренування моделей**

```bash
python scripts/train_experiments.py
```

**Реєстрація найкращої моделі**

```bash
python scripts/register_best_model.py
```

**Запуск FastAPI**

```bash
set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

**Swagger UI**

```
http://127.0.0.1:8000/docs
```
