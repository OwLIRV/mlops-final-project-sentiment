# Final MLOps Project — Ukrainian Review Sentiment Classification (https://github.com/OwLIRV/mlops-final-project-sentiment/actions/runs/29015287549)

## Опис проєкту

Цей проєкт реалізує повний MLOps-цикл для задачі класифікації тональності українських текстових відгуків.

Модель отримує текстовий відгук користувача і визначає один із трьох класів:

- `positive` — позитивний відгук;
- `negative` — негативний відгук;
- `neutral` — нейтральний відгук.

Таке рішення може бути корисним для онлайн-магазинів, сервісних компаній, маркетплейсів, служб підтримки або будь-яких систем, де потрібно швидко аналізувати велику кількість текстових відгуків користувачів.

У межах проєкту реалізовано повний шлях ML-рішення:

дані → розмітка → версіонування → тренування → трекінг експериментів → реєстр моделей → інференс → моніторинг → CI/CD

---

## Мета проєкту

Мета фінального проєкту — пройти повний життєвий цикл MLOps-рішення та показати, як ML-модель може бути підготовлена, навчена, зареєстрована, розгорнута як сервіс, промоніторена та автоматизована через CI/CD.

Проєкт охоплює всі ключові етапи курсу:

- збір та розмітку даних;
- версіонування даних;
- тренування моделі;
- трекінг експериментів;
- реєстр моделей;
- інференс / сервінг;
- моніторинг та observability;
- CI/CD pipeline.

---

## Архітектура рішення

Архітектура проєкту побудована як end-to-end MLOps pipeline.

1. Дані створюються у вигляді коротких українських текстових відгуків.
2. Відгуки розмічаються вручну у Label Studio.
3. Розмічений датасет експортується у JSON.
4. Дані версіонуються за допомогою DVC.
5. DVC remote використовується для збереження версій датасету, зокрема у S3-сумісному сховищі MinIO або локальному DVC-сховищі.
6. Дані проходять валідацію та розділяються на train/test.
7. Моделі тренуються на train.csv і перевіряються на test.csv.
8. Експерименти, метрики та артефакти логуються в MLflow.
9. Найкраща модель реєструється в MLflow Model Registry.
10. FastAPI завантажує production-модель і надає REST API для інференсу.
11. Prometheus збирає метрики з FastAPI endpoint `/metrics`.
12. Grafana візуалізує метрики сервісу.
13. Evidently генерує data drift report.
14. GitHub Actions автоматизує тренування, реєстрацію моделі та тестовий деплой.

Схематично:

`Label Studio → DVC / MinIO → train.csv / test.csv → MLflow → Model Registry → FastAPI → Prometheus / Grafana / Evidently → GitHub Actions`

---

## Структура проєкту

```text
mlops-final-project-sentiment/
│
├── app/
│   └── main.py
│
├── data/
│   ├── labeled/
│   │   └── reviews_labeled_v3.json
│   └── processed/
│       ├── train.csv
│       └── test.csv
│
├── monitoring/
│   └── prometheus.yml
│
├── reports/
│   └── data_drift_report.html
│
├── scripts/
│   ├── validate_data.py
│   ├── split_data.py
│   ├── train.py
│   ├── train_experiments.py
│   ├── register_best_model.py
│   ├── evidently_report.py
│   └── load_test.py
│
├── .github/
│   └── workflows/
│       └── mlops-ci-cd.yml
│
├── docker-compose.yml
├── requirements.txt
├── README.md
└── README_HW5.md
```

---

## 1. Створення датасету

На першому етапі було створено датасет коротких українських текстових відгуків.

Приклади відгуків:

| text | label |
|---|---|
| Замовлення доставили швидко і без пошкоджень | positive |
| Доставка тривала занадто довго | negative |
| Покупка пройшла нормально | neutral |

Класи:

- `positive`;
- `negative`;
- `neutral`.

Початковий датасет використовувався як основа для ручної розмітки та подальшого тренування моделі.

---

## 2. Розмітка даних у Label Studio

Для ручної розмітки було використано Label Studio.

Запуск Label Studio локально:

```bash
label-studio start
```

Після запуску інтерфейс доступний у браузері:

```text
http://localhost:8080
```

У Label Studio було створено проєкт для задачі sentiment classification. Кожен текстовий відгук розмічався одним із трьох класів:

- positive;
- negative;
- neutral.

Після завершення розмітки результати було експортовано у JSON-формат і збережено у папці:

```text
data/labeled/
```

Початкова версія датасету містила 70 прикладів:

- positive: 24;
- negative: 24;
- neutral: 22.

Після тестування baseline-моделі було виявлено, що модель не завжди правильно розпізнає негативні відгуки, зокрема фрази типу:

- “не сподобався”;
- “довге очікування”;
- “поганий сервіс”;
- “розчарував”;
- “незадоволена”.

Тому датасет було розширено. Було додано 45 нових вручну розмічених прикладів.

Фінальна версія датасету:

```text
data/labeled/reviews_labeled_v3.json
```

Фінальний розподіл класів:

- positive: 39;
- negative: 39;
- neutral: 37.

Загальна кількість прикладів:

```text
115
```

---

## 3. Зберігання даних у MinIO

Для зберігання версій датасету використовувалось S3-сумісне сховище MinIO.

MinIO можна запустити локально через Docker:

```bash
docker run -p 9000:9000 -p 9001:9001 --name minio -e "MINIO_ROOT_USER=minioadmin" -e "MINIO_ROOT_PASSWORD=minioadmin" quay.io/minio/minio server /data --console-address ":9001"
```

Після запуску MinIO доступний у браузері:

```text
http://localhost:9001
```

У MinIO створюється bucket для збереження DVC-об’єктів, наприклад:

```text
mlops-dataset
```

MinIO використовується як S3-сумісне сховище, куди DVC може завантажувати версії датасету.

---

## 4. Версіонування даних через DVC

Для версіонування даних було використано DVC.

DVC дозволяє не зберігати великі файли датасету безпосередньо в Git, а зберігати у Git лише метадані, необхідні для відтворення конкретної версії даних.

Ініціалізація DVC:

```bash
dvc init
```

Додавання датасету до DVC:

```bash
dvc add data/labeled/reviews_labeled_v3.json
```

Додавання train/test файлів:

```bash
dvc add data/processed/train.csv
dvc add data/processed/test.csv
```

Налаштування DVC remote для MinIO:

```bash
dvc remote add -d minio s3://mlops-dataset
dvc remote modify minio endpointurl http://localhost:9000
dvc remote modify minio access_key_id minioadmin
dvc remote modify minio secret_access_key minioadmin
```

Завантаження даних у remote storage:

```bash
dvc push
```

Перевірка стану DVC:

```bash
dvc status
```

Очікуваний результат:

```text
Data and pipelines are up to date.
```

У фінальній версії проєкту дані також можна відтворити з локального DVC-сховища або іншого налаштованого DVC remote.

---

## 5. Валідація датасету

Перед тренуванням було реалізовано перевірку якості даних.

Команда:

```bash
python scripts/validate_data.py
```

Скрипт перевіряє:

- чи існує файл датасету;
- чи є текст у кожному прикладі;
- чи є label у кожному прикладі;
- чи немає порожніх текстів;
- чи немає порожніх міток;
- який розподіл класів у датасеті.

Очікуваний результат для фінального датасету:

```text
Dataset validation completed successfully
Total examples: 115
Empty texts: 0
Empty labels: 0
Label distribution:
positive: 39
negative: 39
neutral: 37
All rows contain text and label
```

---

## 6. Розділення даних на train/test

Після валідації датасет розділяється на train/test.

Команда:

```bash
python scripts/split_data.py
```

Результат:

```text
Total examples: 115
Train size: 86
Test size: 29
```

Файли після розділення:

```text
data/processed/train.csv
data/processed/test.csv
```

Розподіл класів у train/test залишився збалансованим, що важливо для коректного тренування та оцінки моделі.

---

## 7. Тренування моделі

Для тренування моделі було використано TF-IDF-векторизацію тексту та класичні ML-алгоритми.

Під час експериментів тестувалися:

- Logistic Regression;
- Linear SVM;
- Naive Bayes;
- word-level TF-IDF;
- char-level TF-IDF;
- bigram TF-IDF.

Основний скрипт для запуску експериментів:

```bash
python scripts/train_experiments.py
```

Модель навчається на:

```text
data/processed/train.csv
```

і тестується на:

```text
data/processed/test.csv
```

Основні метрики:

- Accuracy;
- F1 macro.

Найкращий результат показала модель:

```text
logreg_word_bigram_tfidf
```

Метрики найкращої моделі:

```text
Accuracy: 0.6897
F1 macro: 0.6867
```

---

## 8. Трекінг експериментів у MLflow

Для логування експериментів було використано MLflow.

Запуск MLflow tracking server:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

MLflow UI доступний за адресою:

```text
http://127.0.0.1:5000
```

У MLflow логуються:

- назва експерименту;
- параметри моделі;
- метрики Accuracy і F1 macro;
- classification report;
- артефакти;
- збережені моделі.

Назва експерименту:

```text
sentiment-classification-final-project
```

MLflow дозволяє порівнювати різні експерименти та обирати найкращу модель за метрикою F1 macro.

---

## 9. Реєстр моделей MLflow Model Registry

Після тренування найкраща модель реєструється у MLflow Model Registry.

Команда:

```bash
python scripts/register_best_model.py
```

Назва зареєстрованої моделі:

```text
sentiment-classifier
```

Production URI:

```text
models:/sentiment-classifier@production
```

Для найкращої моделі встановлюється alias:

```text
production
```

Це означає, що FastAPI-сервіс завантажує саме актуальну production-модель із MLflow Model Registry.

---

## 10. Інференс та сервінг через FastAPI

Для розгортання моделі як сервісу було використано FastAPI.

Основний файл сервісу:

```text
app/main.py
```

FastAPI завантажує модель із MLflow Model Registry:

```text
models:/sentiment-classifier@production
```

Запуск FastAPI локально:

```bash
set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

FastAPI доступний за адресою:

```text
http://127.0.0.1:8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Основні endpoint-и:

- `GET /health`;
- `POST /predict`;
- `GET /metrics`.

### Health check

```text
GET /health
```

Приклад відповіді:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_uri": "models:/sentiment-classifier@production"
}
```

### Prediction endpoint

```text
POST /predict
```

Приклад запиту:

```json
{
  "texts": [
    "Сервіс працює добре",
    "Мені не сподобалось обслуговування",
    "Замовлення оформлено"
  ]
}
```

Приклад відповіді:

```json
{
  "predictions": [
    "positive",
    "negative",
    "neutral"
  ],
  "latency_seconds": 0.001,
  "model_uri": "models:/sentiment-classifier@production"
}
```

---

## 11. Моніторинг FastAPI-сервісу

FastAPI експортує метрики у форматі Prometheus через endpoint:

```text
/metrics
```

Основні кастомні метрики:

```text
sentiment_predictions_total
sentiment_prediction_latency_seconds
```

Метрика `sentiment_predictions_total` рахує кількість передбачень за кожним класом:

- positive;
- negative;
- neutral.

Метрика `sentiment_prediction_latency_seconds` показує час виконання інференсу.

---

## 12. Prometheus і Grafana

Для збору і візуалізації метрик використовуються Prometheus і Grafana.

Конфігурація Prometheus:

```text
monitoring/prometheus.yml
```

Запуск Prometheus і Grafana:

```bash
docker compose up -d
```

Prometheus:

```text
http://127.0.0.1:9090
```

Grafana:

```text
http://127.0.0.1:3000
```

FastAPI metrics:

```text
http://127.0.0.1:8000/metrics
```

Prometheus збирає метрики з FastAPI, а Grafana використовується для їх візуалізації.

---

## 13. Evidently Data Drift Report

Для аналізу data drift використано Evidently.

Команда:

```bash
python scripts/evidently_report.py
```

Звіт зберігається у файлі:

```text
reports/data_drift_report.html
```

Для порівняння використовуються:

- reference dataset: `data/processed/train.csv`;
- current dataset: `data/processed/test.csv`.

Ознаки для drift-аналізу:

- `text_length`;
- `word_count`;
- `avg_word_length`;
- `label`.

За результатами перевірки dataset drift не виявлено: 0 із 4 ознак показали суттєву зміну розподілу.

---

## 14. CI/CD через GitHub Actions

Для автоматизації тренування та тестового деплою було налаштовано GitHub Actions pipeline.

Файл workflow:

```text
.github/workflows/mlops-ci-cd.yml
```

Назва workflow:

```text
MLOps CI/CD Pipeline
```

Pipeline запускається за такими тригерами:

- push у гілки `main` або `final-project`;
- release published;
- ручний запуск через `workflow_dispatch`;
- запуск за розкладом через `schedule`.

Основні кроки pipeline:

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

Успішний прогін pipeline підтверджує, що CI/CD процес реально працює.

Посилання на успішний GitHub Actions run:

```text
https://github.com/OwLIRV/mlops-final-project-sentiment/actions/runs/29015287549
```

Окремий README для домашньої роботи 5:

```text
README_HW5.md
```

---

## Як відтворити проєкт локально

### 1. Клонувати репозиторій

```bash
git clone https://github.com/OwLIRV/mlops-final-project-sentiment.git
cd mlops-final-project-sentiment
```

### 2. Створити віртуальне середовище

```bash
python -m venv .venv
```

### 3. Активувати віртуальне середовище

Для Windows CMD:

```bash
.venv\Scripts\activate
```

### 4. Встановити залежності

```bash
pip install -r requirements.txt
```

### 5. Перевірити датасет

```bash
python scripts/validate_data.py
```

### 6. Розділити дані на train/test

```bash
python scripts/split_data.py
```

### 7. Запустити MLflow

В окремому терміналі:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

MLflow UI:

```text
http://127.0.0.1:5000
```

### 8. Натренувати моделі

В іншому терміналі:

```bash
set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python scripts/train_experiments.py
```

### 9. Зареєструвати найкращу модель

```bash
python scripts/register_best_model.py
```

### 10. Запустити FastAPI

```bash
set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```

Metrics:

```text
http://127.0.0.1:8000/metrics
```

### 11. Запустити Prometheus і Grafana

```bash
docker compose up -d
```

Prometheus:

```text
http://127.0.0.1:9090
```

Grafana:

```text
http://127.0.0.1:3000
```

### 12. Згенерувати Evidently report

```bash
python scripts/evidently_report.py
```

Відкрити файл:

```text
reports/data_drift_report.html
```

### 13. Запустити CI/CD pipeline вручну

1. Відкрити GitHub repository.
2. Перейти у вкладку `Actions`.
3. Вибрати `MLOps CI/CD Pipeline`.
4. Натиснути `Run workflow`.
5. Вибрати гілку `main`.
6. Натиснути `Run workflow`.
7. Дочекатися завершення job `train-and-deploy`.

---

## 3 найцікавіші моменти реалізації

### 1. Покращення моделі через покращення датасету

Початкова baseline-модель працювала неточно на деяких негативних відгуках. Після аналізу помилок було прийнято рішення розширити датасет і додати приклади з проблемними формулюваннями.

Після додавання 45 нових прикладів і повторного тренування якість моделі покращилась до:

- Accuracy: 0.6897;
- F1 macro: 0.6867.

Це показує важливість ітеративного MLOps-підходу:

аналіз помилок → покращення даних → повторне тренування → вибір кращої моделі.

### 2. Використання MLflow Model Registry

FastAPI-сервіс не використовує випадковий локальний файл моделі, а завантажує production-модель із MLflow Model Registry.

Production URI:

```text
models:/sentiment-classifier@production
```

Це дозволяє оновлювати production-модель через MLflow alias без зміни коду FastAPI.

### 3. Повний CI/CD pipeline для ML-рішення

GitHub Actions pipeline автоматизує не тільки технічну перевірку, а й ML-етапи:

- валідацію датасету;
- train/test split;
- запуск MLflow;
- тренування моделей;
- реєстрацію найкращої моделі;
- генерацію Evidently-звіту;
- тестовий деплой FastAPI;
- перевірку endpoint-ів `/health` і `/predict`.

Це демонструє повний цикл доставки ML-моделі від даних до працюючого API.

---

## Висновок

У фінальному проєкті було реалізовано повний MLOps-цикл для задачі класифікації тональності українських текстових відгуків.

Проєкт охоплює всі ключові етапи:

- створення та розмітку датасету;
- версіонування даних через DVC;
- збереження даних у MinIO / DVC remote;
- тренування моделей;
- трекінг експериментів у MLflow;
- реєстр моделей MLflow Model Registry;
- сервінг через FastAPI;
- моніторинг через Prometheus і Grafana;
- drift analysis через Evidently;
- CI/CD через GitHub Actions.
