from src.models.finetuned import FinetunedRecommender


def main():
    recommender = FinetunedRecommender(config_path=r"C:\PycharmProjects\PythonProject\recommendation_for_film_and_tv_shows\src\config\finetuned_config.yaml")

    prompt1 = "Recommend me a good sci-fi action movie."
    print("Generated text (first prompt):")
    print(recommender.generate(prompt1))

    prompt2 = "Порекомендуй комедії для перегляду?"
    print("\n--- Второй тест ---")
    print("Generated text (second prompt):")
    print(recommender.generate(prompt2, use_second_params=True))


if __name__ == "__main__":
    main()