import asyncio
import io
import logging
import time as _time

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import CommandStart
from aiogram.types import (
    Message, CallbackQuery,
    InlineKeyboardMarkup, InlineKeyboardButton,
    BufferedInputFile, InputMediaPhoto,
)
from aiogram.enums import ChatAction, ContentType
from aiogram.client.default import DefaultBotProperties
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from config import TELEGRAM_BOT_TOKEN, ADMIN_USER_ID, CHILD_USER_ID, MSK
import brain
import memory

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

router = Router()
_bot: Bot | None = None

TG_TEXT_LIMIT = 4096
TG_CAPTION_LIMIT = 1024
PHOTO_DEBOUNCE_SEC = 3.5


# ── pending photo buffer ───────────────────────────────────────────

class _PendingPhoto:
    __slots__ = ("image_bytes", "caption", "message", "timer", "ts")

    def __init__(self, image_bytes: bytes, caption: str, message: Message):
        self.image_bytes = image_bytes
        self.caption = caption
        self.message = message
        self.timer: asyncio.Task | None = None
        self.ts = _time.monotonic()


_pending: dict[int, _PendingPhoto] = {}


async def _flush_photo(user_id: int):
    """Called after debounce timeout — process the photo alone."""
    pp = _pending.pop(user_id, None)
    if not pp:
        return
    await _process_photo_with_text(pp, extra_text=None, voice_bytes=None, reply_msg=pp.message)


async def _process_photo_with_text(
    pp: _PendingPhoto,
    extra_text: str | None,
    voice_bytes: bytes | None,
    reply_msg: Message,
):
    """Unified handler: photo + optional text/voice."""
    user_id = reply_msg.from_user.id
    await _typing(reply_msg)

    caption = pp.caption
    voice_text = ""

    if voice_bytes:
        voice_text = await brain.transcribe_voice(voice_bytes)
        if voice_text:
            extra_text = f"[голосовое сообщение] {voice_text}"

    combined_text = " ".join(filter(None, [caption, extra_text or voice_text])).strip()

    if brain._wants_image(combined_text):
        answer, img_bytes, mime = await brain.generate_image(
            user_id, combined_text, ref_image=pp.image_bytes, ref_mime="image/jpeg",
        )
        if img_bytes:
            photo = BufferedInputFile(img_bytes, filename="image.png")
            await reply_msg.answer_photo(photo, caption=(answer or "")[:TG_CAPTION_LIMIT] or None)
        else:
            await _safe_answer(reply_msg, answer, reply_markup=_tts_kb())
    else:
        answer = await brain.analyze_image(
            user_id, pp.image_bytes, "image/jpeg", combined_text,
        )
        await _safe_answer(reply_msg, answer, reply_markup=_tts_kb())

    if combined_text and brain.is_complaint(combined_text):
        prev_msgs = await memory.get_recent_messages(user_id, 1)
        prev_bot = prev_msgs[-1]["text"] if prev_msgs and prev_msgs[-1]["role"] == "assistant" else ""
        asyncio.create_task(brain.log_complaint(user_id, combined_text, prev_bot))


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


async def _safe_answer(message: Message, text: str, **kwargs):
    if len(text) > TG_TEXT_LIMIT:
        text = text[:TG_TEXT_LIMIT - 20] + "\n\n(продолжение...)"
    await message.answer(text, **kwargs)


# ── /start ──────────────────────────────────────────────────────────

@router.message(CommandStart())
async def cmd_start(message: Message):
    await _typing(message)
    user_id = message.from_user.id
    if user_id == CHILD_USER_ID:
        name = "Злата"
    else:
        name = message.from_user.first_name or "друг"
    await message.answer(
        f"Муу! Привет, {name}! Я Коровик — маленький телёнок с Облачной фермы 🐮\n\n"
        "Спрашивай что хочешь — могу придумать идею, рассказать историю "
        "или просто поболтаем! Можешь писать текстом или отправить голосовое 🎤",
        reply_markup=_tts_kb(),
    )


# ── text messages ───────────────────────────────────────────────────

@router.message(F.text)
async def handle_text(message: Message):
    user_id = message.from_user.id
    text = message.text or ""

    if text.strip().lower() == "##сброс" and user_id == ADMIN_USER_ID:
        await memory.delete_user_data(CHILD_USER_ID)
        await memory.delete_user_data(user_id)
        await message.answer("Данные сброшены 🔄")
        return

    pp = _pending.pop(user_id, None)
    if pp:
        if pp.timer:
            pp.timer.cancel()
        await _process_photo_with_text(pp, extra_text=text, voice_bytes=None, reply_msg=message)
        return

    await _typing(message)

    prev_msgs = await memory.get_recent_messages(user_id, 1)
    prev_bot_msg = prev_msgs[-1]["text"] if prev_msgs and prev_msgs[-1]["role"] == "assistant" else ""

    if brain._wants_image(text):
        answer, img_bytes, mime = await brain.generate_image(user_id, text)
        if img_bytes:
            photo = BufferedInputFile(img_bytes, filename="image.png")
            await message.answer_photo(photo, caption=(answer or "")[:TG_CAPTION_LIMIT] or None)
        else:
            await _safe_answer(message, answer, reply_markup=_tts_kb())
    else:
        answer = await brain.generate_response(user_id, text)
        await _safe_answer(message, answer, reply_markup=_tts_kb())

    if brain.is_complaint(text):
        asyncio.create_task(brain.log_complaint(user_id, text, prev_bot_msg))


# ── photo messages ─────────────────────────────────────────────────

@router.message(F.photo)
async def handle_photo(message: Message):
    user_id = message.from_user.id

    try:
        photo = message.photo[-1]
        file = await message.bot.get_file(photo.file_id)
        buf = io.BytesIO()
        await message.bot.download_file(file.file_path, buf)
        image_bytes = buf.getvalue()
    except Exception:
        await message.answer("Не получилось загрузить фото 😕 Попробуй ещё раз!", reply_markup=_tts_kb())
        return

    caption = message.caption or ""

    if caption and (brain._wants_image(caption) or len(caption) > 20):
        await _typing(message)
        if brain._wants_image(caption):
            answer, img_bytes, mime = await brain.generate_image(
                user_id, caption, ref_image=image_bytes, ref_mime="image/jpeg",
            )
            if img_bytes:
                gen_photo = BufferedInputFile(img_bytes, filename="image.png")
                await message.answer_photo(gen_photo, caption=(answer or "")[:TG_CAPTION_LIMIT] or None)
            else:
                await _safe_answer(message, answer, reply_markup=_tts_kb())
        else:
            answer = await brain.analyze_image(user_id, image_bytes, "image/jpeg", caption)
            await _safe_answer(message, answer, reply_markup=_tts_kb())
        return

    old = _pending.pop(user_id, None)
    if old and old.timer:
        old.timer.cancel()

    pp = _PendingPhoto(image_bytes, caption, message)
    _pending[user_id] = pp
    await _typing(message)

    async def _debounce_fire():
        await asyncio.sleep(PHOTO_DEBOUNCE_SEC)
        await _flush_photo(user_id)

    pp.timer = asyncio.create_task(_debounce_fire())


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

    pp = _pending.pop(user_id, None)
    if pp:
        if pp.timer:
            pp.timer.cancel()
        await _process_photo_with_text(pp, extra_text=None, voice_bytes=audio_bytes, reply_msg=message)
        return

    text = await brain.transcribe_voice(audio_bytes)
    if not text:
        await message.answer("Не удалось распознать 😕 Попробуй ещё раз или напиши текстом!", reply_markup=_tts_kb())
        return

    prev_msgs = await memory.get_recent_messages(user_id, 1)
    prev_bot_msg = prev_msgs[-1]["text"] if prev_msgs and prev_msgs[-1]["role"] == "assistant" else ""

    voice_text = f"[голосовое сообщение] {text}"

    if brain._wants_image(text):
        answer, img_bytes, mime = await brain.generate_image(user_id, text)
        if img_bytes:
            photo = BufferedInputFile(img_bytes, filename="image.png")
            await message.answer_photo(photo, caption=(answer or "")[:TG_CAPTION_LIMIT] or None)
        else:
            await _safe_answer(message, answer, reply_markup=_tts_kb())
    else:
        answer = await brain.generate_response(user_id, voice_text)
        await _safe_answer(message, answer, reply_markup=_tts_kb())

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
            await _bot.send_message(ADMIN_USER_ID, report[:TG_TEXT_LIMIT])
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
