import random
import time
import requests


API_URL = "http://127.0.0.1:8000/predict"

EXAMPLES = [
    "Дуже хороший сервіс, мені все сподобалось",
    "Поганий товар, більше не буду купувати",
    "Замовлення отримала, все нормально",
    "Я задоволена якістю обслуговування",
    "Це було жахливо, дуже розчарована",
    "Нейтральний відгук без сильних емоцій",
    "Все швидко, якісно і зручно",
    "Не рекомендую, поганий досвід",
    "Загалом нормально, але є недоліки"
]


def main():
    total_requests = 120

    for i in range(total_requests):
        payload = {
            "texts": [
                random.choice(EXAMPLES)
            ]
        }

        try:
            response = requests.post(API_URL, json=payload, timeout=10)
            print(
                f"Request {i + 1}/{total_requests}: "
                f"status={response.status_code}, "
                f"response={response.json()}"
            )
        except Exception as error:
            print(f"Request {i + 1}/{total_requests}: error={error}")

        time.sleep(0.3)


if __name__ == "__main__":
    main()