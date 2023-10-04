from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

from mirage.config import TELEGRAM_BOT_TOKEN


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_message is None:
        return
    await update.effective_message.reply_text(
        f"{update.effective_message.chat_id}",
    )

application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
application.add_handler(CommandHandler("start", start))
