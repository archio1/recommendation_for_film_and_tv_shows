GENRE_MAPPING = {
    # --- TMDB TV Specific ---
    "Action & Adventure": "Action",  # В MovieLens это Action или Adventure
    "Sci-Fi & Fantasy": "Science Fiction",  # Объединяем в Sci-Fi
    "War & Politics": "War",  # Ближе всего к военным
    "Soap": "Drama",  # Мыльные оперы — это всегда драма
    "Reality": "Documentary",  # Реалити-шоу ближе к документалистике
    "Talk": "Comedy",  # Ток-шоу чаще комедийные (или можно в Drama)
    "Kids": "Children",  # Специфика сериалов
    "News": "Documentary",

    # --- TMDB Movies to MovieLens Style ---
    "Science Fiction": "Sci-Fi",  # MovieLens использует сокращение 'Sci-Fi'
    "TV Movie": "Drama",  # Обычно это драмы
    "History": "Drama",  # В MovieLens часто исторические фильмы идут как драмы
    "Family": "Children",  # Стандарт MovieLens
    "Music": "Musical",  # Стандарт MovieLens

    # --- Прямые соответствия (для надежности) ---
    "Action": "Action",
    "Adventure": "Adventure",
    "Animation": "Animation",
    "Comedy": "Comedy",
    "Crime": "Crime",
    "Documentary": "Documentary",
    "Drama": "Drama",
    "Fantasy": "Fantasy",
    "Horror": "Horror",
    "Mystery": "Mystery",
    "Romance": "Romance",
    "Thriller": "Thriller",
    "War": "War",
    "Western": "Western"
}

MOVIE_COLUMNS = ['id', 'title', 'release_date', 'genres', 'popularity', 'vote_average']
TV_COLUMNS = ['id', 'name', 'first_air_date', 'genres', 'popularity', 'vote_average']