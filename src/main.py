from pathlib import Path
import yaml
from src.utils.helpers import load_dataset
from src.models.recommender import PretrainedRecommender
import sys

# Установка кодировки для консоли (Windows)
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

# Получение пути к pretrained_config.yaml
CONFIG_PATH = Path(__file__).parent / "config" / "pretrained_config.yaml"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Config file not found at: {CONFIG_PATH}")

def main():
    with open(CONFIG_PATH, "r") as file:
        config = yaml.safe_load(file)

    dataset_path = Path(__file__).parent.parent / config["paths"]["dataset"]
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")
    documents = load_dataset(dataset_path)

    recommender = PretrainedRecommender(documents, config)

    print("Введите запрос для рекомендаций фильмов/сериалов (например, 'Recommend me superheroes film like Captain America').")
    print("Для выхода введите 'exit'.")
    while True:
        query = input("Ваш запрос: ").strip()
        if query.lower() == "exit":
            print("Выход из программы.")
            break
        if not query:
            print("Ошибка: запрос не может быть пустым. Попробуйте снова.")
            continue

        try:
            result = recommender.recommend(query)
            print("\nОтвет модели:")
            print(result["answer"].strip())

            print("\nИсточники:")
            for doc in result["sources"]:
                print(f"- {doc.metadata['title']} (Rating: {doc.metadata['weighted_rate']})")
            print()
        except Exception as e:
            print(f"Ошибка при обработке запроса: {e}")

if __name__ == "__main__":
    main()