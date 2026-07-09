import json
from pathlib import Path


INPUT_PATH = Path("data/labeled/reviews_labeled_v2.json")
OUTPUT_PATH = Path("data/labeled/reviews_labeled_v3.json")


extra_examples = [
    # negative
    {"text": "Мені не сподобався сервіс і довге очікування", "label": "negative"},
    {"text": "Сервіс був поганий, я залишилась незадоволена", "label": "negative"},
    {"text": "Доставка тривала дуже довго, це розчарувало", "label": "negative"},
    {"text": "Товар прийшов пошкоджений і не відповідав опису", "label": "negative"},
    {"text": "Якість обслуговування погана, не рекомендую", "label": "negative"},
    {"text": "Мені не сподобалась якість товару", "label": "negative"},
    {"text": "Замовлення затримали, сервіс розчарував", "label": "negative"},
    {"text": "Підтримка не допомогла вирішити проблему", "label": "negative"},
    {"text": "Дуже поганий досвід, більше не замовлятиму", "label": "negative"},
    {"text": "Очікування було занадто довгим, результат поганий", "label": "negative"},
    {"text": "Клієнтський сервіс працює повільно і неякісно", "label": "negative"},
    {"text": "Я незадоволена покупкою і хочу повернення коштів", "label": "negative"},
    {"text": "Продукт не працює так, як було заявлено", "label": "negative"},
    {"text": "Враження негативне, сервіс не виправдав очікувань", "label": "negative"},
    {"text": "Мені не сподобалося обслуговування", "label": "negative"},

    # positive
    {"text": "Сервіс працює добре, мені все сподобалось", "label": "positive"},
    {"text": "Дуже швидка доставка і приємне обслуговування", "label": "positive"},
    {"text": "Все було чудово, буду замовляти ще", "label": "positive"},
    {"text": "Продукт якісний, сервісом задоволена", "label": "positive"},
    {"text": "Замовлення виконали швидко і без проблем", "label": "positive"},
    {"text": "Мені дуже сподобалась якість товару", "label": "positive"},
    {"text": "Підтримка швидко допомогла вирішити питання", "label": "positive"},
    {"text": "Доставка була швидкою, товар чудовий", "label": "positive"},
    {"text": "Дуже хороший сервіс і приємні працівники", "label": "positive"},
    {"text": "Покупкою повністю задоволена", "label": "positive"},
    {"text": "Все пройшло добре, рекомендую цей сервіс", "label": "positive"},
    {"text": "Я отримала якісний товар і гарне обслуговування", "label": "positive"},
    {"text": "Сервіс перевершив мої очікування", "label": "positive"},
    {"text": "Дуже приємний досвід користування", "label": "positive"},
    {"text": "Результат чудовий, я задоволена", "label": "positive"},

    # neutral
    {"text": "Замовлення було доставлено сьогодні", "label": "neutral"},
    {"text": "Оплата пройшла успішно, замовлення оформлено", "label": "neutral"},
    {"text": "Консультант відповів на питання без додаткових деталей", "label": "neutral"},
    {"text": "Товар отримано згідно з описом", "label": "neutral"},
    {"text": "Замовлення знаходиться в обробці", "label": "neutral"},
    {"text": "Доставка запланована на завтра", "label": "neutral"},
    {"text": "Користувач отримав повідомлення про статус замовлення", "label": "neutral"},
    {"text": "Система підтвердила оплату", "label": "neutral"},
    {"text": "Посилка передана до служби доставки", "label": "neutral"},
    {"text": "Менеджер уточнив деталі замовлення", "label": "neutral"},
    {"text": "Замовлення було створено в особистому кабінеті", "label": "neutral"},
    {"text": "Клієнт залишив запит на консультацію", "label": "neutral"},
    {"text": "Товар є в наявності на складі", "label": "neutral"},
    {"text": "Інформацію про доставку оновлено", "label": "neutral"},
    {"text": "Покупець отримав номер накладної", "label": "neutral"},
]


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        original_data = json.load(f)

    combined_data = original_data + extra_examples

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)

    print("New dataset version created")
    print(f"Original examples: {len(original_data)}")
    print(f"Added examples: {len(extra_examples)}")
    print(f"Total examples: {len(combined_data)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()