import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters
from agent import run_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

bot_app = Application.builder().token(TELEGRAM_TOKEN).build()


async def start(update: Update, context):
    await update.message.reply_text(
        "Привет! Я coding-агент.\n\n"
        "Напиши задачу — напишу код и запушу в GitHub.\n\n"
        "Пример:\n"
        "• Добавь эндпоинт GET /users с пагинацией\n"
        "• Создай модель User с полями name, email, created_at\n"
        "• Добавь rate limiting к роутеру agents"
    )


async def handle_message(update: Update, context):
    prompt = update.message.text
    user = update.message.from_user.username or update.message.from_user.first_name
    logger.info(f"Запрос от {user}: {prompt}")

    status_msg = await update.message.reply_text("⏳ Работаю над задачей...")

    try:
        result = await run_agent(prompt)
        if result['files_changed'] == 0:
            await status_msg.edit_text(
                "⚠️ Агент не внёс изменений.\n\n"
                "Попробуй переформулировать задачу точнее, например:\n"
                "• «Добавь в файл X функцию Y»\n"
                "• «Измени в файле X строку Y на Z»"
            )
        else:
            await status_msg.edit_text(
                f"✅ Готово!\n\n"
                f"📝 {result['commit_message']}\n\n"
                f"📁 Изменено файлов: {result['files_changed']}\n"
                f"🔗 {result['repo_url']}"
            )
    except Exception as e:
        logger.error(f"Ошибка агента: {e}")
        await status_msg.edit_text(f"❌ Ошибка: {str(e)}")


bot_app.add_handler(CommandHandler("start", start))
bot_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup — устанавливаем webhook
    await bot_app.initialize()
    await bot_app.bot.set_webhook(f"{WEBHOOK_URL}/webhook")
    logger.info(f"Webhook установлен: {WEBHOOK_URL}/webhook")
    yield
    # Shutdown — НЕ удаляем webhook, чтобы он восстанавливался после sleep/redeploy
    await bot_app.shutdown()


app = FastAPI(lifespan=lifespan)


@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, bot_app.bot)
    await bot_app.process_update(update)
    return {"ok": True}


@app.get("/")
async def health():
    return {"status": "ok"}
