from src.models.finetuned import FinetunedRecommender

def main():
    config_path = r"C:\PycharmProjects\PythonProject\recommendation_for_film_and_tv_shows\src\config\finetuned_config.yaml"
    print(f"Инициализация рекомендательной системы с конфигом: {config_path}")
    recommender = FinetunedRecommender(config_path=config_path)

    print("\nДобро пожаловать в чат с рекомендательной системой!")
    print("Вводите запросы для рекомендаций (например, 'Порекомендуй фантастику').")
    print("Для выхода введите 'exit' или 'quit'.")
    print("Чтобы использовать другие параметры генерации, добавьте ':second' в конец запроса (например, 'Порекомендуй комедии:second').")
    print("Чтобы проанализировать вероятности следующих токенов, введите ':analyze <ваш_запрос>' (например, ':analyze Порекомендуй').")

    while True:
        user_input = input("\nВаш запрос: ").strip()

        if user_input.lower() in ['exit', 'quit']:
            print("Выход из чата. До свидания!")
            break

        if not user_input:
            print("Пожалуйста, введите непустой запрос.")
            continue

        # Обработка команды :analyze
        if user_input.lower().startswith(':analyze'):
            prompt_to_analyze = user_input[len(':analyze'):].strip()
            if prompt_to_analyze:
                print(f"\nАнализ вероятностей токенов для запроса: '{prompt_to_analyze}'")
                try:
                    generated_text, _, _ = recommender.analyze_token_probabilities(prompt_to_analyze, max_tokens=50, top_k=5)
                    print(f"\nПолный сгенерированный текст (greedy): {generated_text}")
                except Exception as e:
                    print(f"Ошибка при анализе: {e}")
            else:
                print("Пожалуйста, введите текст для анализа после ':analyze'.")
            continue

        # Обработка флага :second для generate
        use_second_params = False
        if user_input.endswith(':second'):
            use_second_params = True
            user_input = user_input[:-7].strip()

        # Стандартная генерация
        try:
            print("\nОтвет модели:")
            response = recommender.generate(user_input, use_second_params=use_second_params)
            if response:
                print(response)
            else:
                print("(Модель не вернула обработанных рекомендаций. Попробуйте другой запрос или посмотрите анализ токенов.)")
        except Exception as e:
            print(f"Ошибка при генерации: {e}")

if __name__ == "__main__":
    main()