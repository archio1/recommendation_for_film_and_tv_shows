import os
import asyncio
import random
import logging
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import torch
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder

from trainer_gui import  InferenceEngine
from bilingual_utils import ScalableBilingualEngine

load_dotenv(find_dotenv())

# --- НАСТРОЙКИ ---
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[4]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "lightgcn_best_v3.pt"

API_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

if not API_TOKEN:
    print("❌ Ошибка: TELEGRAM_BOT_TOKEN не найден в .env")

# Глобальное хранилище выбранных фильмов {user_id: [item_ids]}
user_sessions = {}

# Инициализация движка
engine = InferenceEngine(DATA_DIR, MODEL_PATH, device='cpu')
success, msg = engine.load_resources()
print(msg)

bot = Bot(token=API_TOKEN)
dp = Dispatcher()

bilingual_engine = ScalableBilingualEngine(
    base_engine=engine,
    cache_dir=DATA_DIR / "cache",
    tmdb_api_key='62814d4a01feef50a1344193e60b49be'
)


# --- ЛОГИКА БОТА ---

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    user_sessions[message.from_user.id] = []
    await message.answer(
        "🎬 **Привет! Я бот-киноман.**\n\n"
        "Напиши название фильма (на английском), который тебе нравится, "
        "и я добавлю его в список для анализа."
    )

@dp.message(Command("list"))
async def show_my_list(message: types.Message):
    await send_list_message(message.from_user.id, message)


async def send_list_message(user_id, message_or_callback):
    liked_ids = user_sessions.get(user_id, [])

    if not liked_ids:
        text = "📭 **Твой список пуст.**"
        if isinstance(message_or_callback, types.Message):
            await message_or_callback.answer(text, parse_mode="Markdown")
        else:
            await message_or_callback.message.edit_text(text, parse_mode="Markdown")
        return

    response = f"📝 **Твой список ({len(liked_ids)} фильмов):**\n\n"
    builder = InlineKeyboardBuilder()

    # Кнопки управления вверху
    builder.row(types.InlineKeyboardButton(text="🚀 ПОЛУЧИТЬ РЕКОМЕНДАЦИИ", callback_data="get_recs"))
    builder.row(types.InlineKeyboardButton(text="🗑 ОЧИСТИТЬ ВЕСЬ СПИСОК", callback_data="clear"))

    display_limit = 15
    show_ids = liked_ids[-display_limit:]

    response += "_Список:_\n"

    # Итерируемся по списку ID пользователя
    for item_id in liked_ids:
        # Получаем полную информацию (включая перевод) через интеллект-слой
        info = bilingual_engine.intelligence.get_movie_info(item_id)
        if not info:
            continue

        # Формируем красивое название: "Русское (English)"
        if info.title_ru and info.title_ru != info.title_en:
            display_name = f"{info.title_ru} ({info.title_en})"
            short_name = info.title_ru  # Для текста на кнопке удаления
        else:
            display_name = info.title_en
            short_name = info.title_en

        response += f"• {display_name}\n"

        # Добавляем кнопку удаления, если фильм в числе последних 15
        if item_id in show_ids:
            builder.row(types.InlineKeyboardButton(
                text=f"❌ Удалить: {short_name[:25]}",
                callback_data=f"remove_{item_id}")
            )

    if len(liked_ids) > display_limit:
        response += f"\n⚠️ _Показаны кнопки удаления только для последних {display_limit} фильмов._"

    if isinstance(message_or_callback, types.Message):
        await message_or_callback.answer(response, reply_markup=builder.as_markup(), parse_mode="Markdown")
    else:
        await message_or_callback.message.edit_text(response, reply_markup=builder.as_markup(), parse_mode="Markdown")

@dp.message(Command("stress_test"))
async def cmd_stress_test(message: types.Message):
    user_id = message.from_user.id

    # 1. Получаем список всех доступных item_id из метаданных
    all_ids = engine.metadata['item_id'].tolist()

    # 2. Выбираем 100 случайных фильмов
    # (если в базе меньше 100, выберем все)
    count_to_sample = min(100, len(all_ids))
    random_ids = random.sample(all_ids, count_to_sample)

    # 3. Записываем их в сессию пользователя
    user_sessions[user_id] = random_ids

    await message.answer(
        f"🧪 **Стресс-тест запущен!**\n\n"
        f"Я только что добавил в твой список {count_to_sample} случайных фильмов.\n"
        f"Теперь попробуй:\n"
        f"1. Отправить команду /list\n"
        f"2. Нажать кнопку '🚀 Получить рекомендации'"
    )

@dp.message(F.text)
async def handle_search(message: types.Message):
    query = message.text
    if len(query) < 2: return

    results = bilingual_engine.search_movies_bilingual(query, limit=8)

    if not results:
        await message.answer("❌ Ничего не нашел. Попробуй другое название.")
        return

    # Строим кнопки с результатами поиска
    builder = InlineKeyboardBuilder()
    for m in results:
        # Проверяем названия (если перевода нет, будет только английское)
        if m['title_ru'] and m['title_ru'] != m['title_en']:
            display_name = f"{m['title_ru']} / {m['title_en']}"
        else:
            display_name = m['title_en']

        builder.row(types.InlineKeyboardButton(
            text=f"➕ {display_name} ({int(m['year'])})",
            callback_data=f"add_{m['item_id']}")
        )

    await message.answer(f"Вот что я нашел по запросу '{query}':", reply_markup=builder.as_markup())


@dp.callback_query(F.data.startswith("add_"))
async def add_movie(callback: types.CallbackQuery):
    item_id = int(callback.data.split("_")[1])
    user_id = callback.from_user.id

    if user_id not in user_sessions:
        user_sessions[user_id] = []

    if item_id not in user_sessions[user_id]:
        user_sessions[user_id].append(item_id)

    count = len(user_sessions[user_id])

    # Кнопки управления списком
    builder = InlineKeyboardBuilder()
    if count >= 1:
        builder.row(types.InlineKeyboardButton(text="🚀 Рекомендации", callback_data="get_recs"))
        builder.row(types.InlineKeyboardButton(text="📋 Посмотреть мой список", callback_data="view_list"))

    await callback.message.answer(
        f"✅ Добавлено! В твоем списке: {count} фильм(ов).\n"
        f"Можешь добавить еще или нажать кнопку ниже.",
        reply_markup=builder.as_markup()
    )
    await callback.answer()


@dp.callback_query(F.data == "clear")
async def clear_list(callback: types.CallbackQuery):
    user_sessions[callback.from_user.id] = []
    await callback.message.answer("Список очищен. Напиши название фильма для поиска.")
    await callback.answer()

@dp.callback_query(F.data == "view_list")
async def view_list_callback(callback: types.CallbackQuery):
    await send_list_message(callback.from_user.id, callback)
    await callback.answer()

@dp.callback_query(F.data.startswith("remove_"))
async def remove_single_movie(callback: types.CallbackQuery):
    item_id = int(callback.data.split("_")[1])
    user_id = callback.from_user.id

    if user_id in user_sessions and item_id in user_sessions[user_id]:
        user_sessions[user_id].remove(item_id)
        await callback.answer("Фильм удален из списка")
        # Обновляем сообщение со списком
        await send_list_message(user_id, callback)
    else:
        await callback.answer("Ошибка: фильм не найден")


@dp.callback_query(F.data == "get_recs")
async def send_recs(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    liked_ids = user_sessions.get(user_id, [])

    if not liked_ids:
        await callback.answer("Сначала выбери фильмы!", show_alert=True)
        return

    # Отправляем временное сообщение, чтобы пользователь видел, что бот "думает"
    status_msg = await callback.message.answer("⚙️ Анализирую твои вкусы... Пожалуйста, подожди.")

    # Получаем рекомендации (теперь с поддержкой RU и фильтром Saw)
    recs = bilingual_engine.get_recommendations_bilingual(liked_ids, top_k=8)

    if not recs:
        await status_msg.edit_text(
            "Хм... не удалось подобрать уникальные фильмы. Попробуй добавить больше разных жанров в список.")
        return

    response = "🍿 **Твои персональные рекомендации:**\n\n"

    for i, r in enumerate(recs):
        # 1. Формируем название: "Русское (English)"
        if r['title_ru'] and r['title_ru'] != r['title_en']:
            title_display = f"{r['title_ru']} / {r['title_en']}"
        else:
            title_display = r['title_en']

        # 2. Проверяем жанры (если там пусто или "—", ставим заглушку)
        genres_str = r['genres']
        if not genres_str or genres_str.strip() == "—":
            genres_str = "Интригующий сюжет"  # Запасной вариант

        # 3. Собираем блок фильма
        response += f"{i + 1}. [{title_display} ({int(r['year'])})]({r['imdb_url']})\n"
        response += f"{r.get('emoji_icons', '🎬')} _{genres_str}_\n"

        # ДОБАВЛЯЕМ ПРИЧИНУ (Reason), которую генерирует наш ScalableBilingualEngine
        if 'reason_ru' in r:
            response += f"💡 {r['reason_ru']}\n"

        response += "\n"

    # Удаляем временное сообщение "Анализирую..."
    await status_msg.delete()

    # Отправляем финальный список
    await callback.message.answer(
        response,
        parse_mode="Markdown",
        disable_web_page_preview=True  # Это убирает огромные картинки
    )
    await callback.answer()


async def main():
    print("Бот запущен...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())