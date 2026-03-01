import asyncio
import logging
import os
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, filters
from agent import run_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")


async def start(update: Update, context):
    await update.message.reply_text(
        "Привет! Я coding-агент.\n\n"
        "Напиши задачу — я напишу код и запушу в GitHub.\n\n"
        "Пример:\n"
        "• Добавь эндпоинт GET /users с пагинацией\n"
        "• Создай модель User с полями name, email, created_at\n"
        "• Добавь rate limiting к роутеру agents"
    )


async def handle_message(update: Update, context):
    prompt = update.message.text
    user = update.message.from_user.username or update.message.from_user.first_name

    logger.info(f"Запрос от {user}: {prompt}")

    # Сообщаем что работаем
    status_msg = await update.message.reply_text("⏳ Работаю над задачей...")

    try:
        result = await run_agent(prompt)

        await status_msg.edit_text(
            f"✅ Готово!\n\n"
            f"📝 {result['commit_message']}\n\n"
            f"📁 Изменено файлов: {result['files_changed']}\n"
            f"🔗 {result['repo_url']}"
        )

    except Exception as e:
        logger.error(f"Ошибка агента: {e}")
        await status_msg.edit_text(f"❌ Ошибка: {str(e)}")


def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Бот запущен")
    app.run_polling()


if __name__ == "__main__":
    main()
