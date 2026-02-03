import asyncio
import logging
from pathlib import Path
import torch
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder

from trainer_gui import  InferenceEngine

# --- НАСТРОЙКИ ---
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[4]
API_TOKEN = '8549170820:AAG8cMAm1OLATni3t5JolFw121e2TAfVoo0'
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "lightgcn_best_v3.pt"

# Глобальное хранилище выбранных фильмов {user_id: [item_ids]}
user_sessions = {}

# Инициализация движка
engine = InferenceEngine(DATA_DIR, MODEL_PATH, device='cpu')
success, msg = engine.load_resources()
print(msg)

bot = Bot(token=API_TOKEN)
dp = Dispatcher()


# --- ЛОГИКА БОТА ---

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    user_sessions[message.from_user.id] = []
    await message.answer(
        "🎬 **Привет! Я бот-киноман.**\n\n"
        "Напиши название фильма (на английском), который тебе нравится, "
        "и я добавлю его в список для анализа."
    )


@dp.message(F.text)
async def handle_search(message: types.Message):
    query = message.text
    if len(query) < 2: return

    results = engine.search_movies(query, limit=5)

    if not results:
        await message.answer("❌ Ничего не нашел. Попробуй другое название.")
        return

    # Строим кнопки с результатами поиска
    builder = InlineKeyboardBuilder()
    for m in results:
        builder.row(types.InlineKeyboardButton(
            text=f"➕ {m['title']} ({int(m['year'])})",
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
        builder.row(types.InlineKeyboardButton(text="🚀 Получить рекомендации", callback_data="get_recs"))
    builder.row(types.InlineKeyboardButton(text="🗑 Очистить список", callback_data="clear"))

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


@dp.callback_query(F.data == "get_recs")
async def send_recs(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    liked_ids = user_sessions.get(user_id, [])

    if not liked_ids:
        await callback.answer("Сначала выбери фильмы!", show_alert=True)
        return

    await callback.message.answer("⚙️ Анализирую твои вкусы... Пожалуйста, подожди.")

    # Твой модифицированный метод с фильтрацией сиквелов и ссылками
    recs = engine.get_recommendations(liked_ids, top_k=8)

    if not recs:
        await callback.message.answer("Хм... не удалось подобрать. Попробуй выбрать другие фильмы.")
        return

    # Формируем ответ
    response = "🍿 **Твои персональные рекомендации:**\n\n"
    for i, r in enumerate(recs):
        response += f"{i + 1}. [{r['title']} ({int(r['year'])})]({r['imdb_url']})\n"
        response += f"🎭 {', '.join(r['genres'][:3])}\n\n"

    await callback.message.answer(response, parse_mode="Markdown", disable_web_page_preview=False)
    await callback.answer()


async def main():
    print("Бот запущен...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())