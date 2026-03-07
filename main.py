import asyncio
import io
import logging

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import CommandStart
from aiogram.types import (
    Message, CallbackQuery,
    InlineKeyboardMarkup, InlineKeyboardButton,
    ChatAction, BufferedInputFile,
)
from aiogram.client.default import DefaultBotProperties
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from config import TELEGRAM_BOT_TOKEN, ADMIN_USER_ID, MSK
import brain
import memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

router = Router()
_bot: Bot | None = None


# ── helpers ─────────────────────────────────────────────────────────

def _tts_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🔊 Озвучить", callback_data="tts")]
    ])


async def _typing(message: Message):
    try:
        await message.bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)
    except Exception:
        pass


# ── /start ──────────────────────────────────────────────────────────

@router.message(CommandStart())
async def cmd_start(message: Message):
    await _typing(message)
    await message.answer(
        "Привет! Я Коровик 🐮\n\n"
        "Спрашивай что хочешь — помогу придумать, объясню, расскажу историю "
        "или просто поболтаем! Можешь писать текстом или отправить голосовое 🎤",
        reply_markup=_tts_kb(),
    )


# ── text messages ───────────────────────────────────────────────────

@router.message(F.text)
async def handle_text(message: Message):
    user_id = message.from_user.id
    text = message.text or ""

    if text.strip().lower() == "##сброс" and user_id == ADMIN_USER_ID:
        await memory.delete_user_data(user_id)
        await message.answer("Данные сброшены 🔄")
        return

    await _typing(message)

    prev_msgs = await memory.get_recent_messages(user_id, 1)
    prev_bot_msg = prev_msgs[-1]["text"] if prev_msgs and prev_msgs[-1]["role"] == "assistant" else ""

    answer = await brain.generate_response(user_id, text)
    await message.answer(answer, reply_markup=_tts_kb())

    if brain.is_complaint(text):
        asyncio.create_task(brain.log_complaint(user_id, text, prev_bot_msg))


# ── voice messages ──────────────────────────────────────────────────

@router.message(F.voice)
async def handle_voice(message: Message):
    user_id = message.from_user.id
    await _typing(message)

    try:
        bot = message.bot
        file = await bot.get_file(message.voice.file_id)
        buf = io.BytesIO()
        await bot.download_file(file.file_path, buf)
        audio_bytes = buf.getvalue()
    except Exception:
        await message.answer("Не получилось скачать голосовое 😕 Попробуй ещё раз!", reply_markup=_tts_kb())
        return

    text = await brain.transcribe_voice(audio_bytes)
    if not text:
        await message.answer("Не удалось распознать 😕 Попробуй написать текстом!", reply_markup=_tts_kb())
        return

    await message.answer(f"🎤 Услышал: «{text}»")
    await _typing(message)

    prev_msgs = await memory.get_recent_messages(user_id, 1)
    prev_bot_msg = prev_msgs[-1]["text"] if prev_msgs and prev_msgs[-1]["role"] == "assistant" else ""

    answer = await brain.generate_response(user_id, text)
    await message.answer(answer, reply_markup=_tts_kb())

    if brain.is_complaint(text):
        asyncio.create_task(brain.log_complaint(user_id, text, prev_bot_msg))


# ── TTS button ──────────────────────────────────────────────────────

@router.callback_query(F.data == "tts")
async def handle_tts(callback: CallbackQuery):
    text = callback.message.text
    if not text:
        await callback.answer("Нечего озвучивать 🤷")
        return

    await callback.answer("Генерирую озвучку... 🎙️")

    try:
        await callback.message.bot.send_chat_action(
            chat_id=callback.message.chat.id, action=ChatAction.TYPING
        )
    except Exception:
        pass

    ogg = await brain.text_to_speech(text)
    if ogg:
        voice_file = BufferedInputFile(ogg, filename="voice.ogg")
        await callback.message.reply_voice(voice_file)
    else:
        await callback.message.answer("Не получилось озвучить 😕 Попробуй позже!")


# ── daily report job ────────────────────────────────────────────────

async def _send_daily_report():
    if not _bot or not ADMIN_USER_ID:
        return
    try:
        report = await brain.generate_daily_report()
        if report:
            await _bot.send_message(ADMIN_USER_ID, report)
            await memory.clear_today_complaints()
            log.info("Daily report sent to admin")
    except Exception:
        log.exception("Failed to send daily report")


# ── main ────────────────────────────────────────────────────────────

async def main():
    global _bot
    _bot = Bot(token=TELEGRAM_BOT_TOKEN, default=DefaultBotProperties())
    dp = Dispatcher()
    dp.include_router(router)

    scheduler = AsyncIOScheduler(timezone=MSK)
    scheduler.add_job(_send_daily_report, "cron", hour=22, minute=0)
    scheduler.start()
    log.info("Scheduler started — daily report at 22:00 MSK")

    log.info("Коровик запущен 🐮")
    await dp.start_polling(_bot)


if __name__ == "__main__":
    asyncio.run(main())
