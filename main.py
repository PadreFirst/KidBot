import asyncio
import html
import io
import logging
import random
import re
from collections import OrderedDict
from datetime import datetime

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import CommandStart
from aiogram.types import (
    Message, CallbackQuery,
    InlineKeyboardMarkup, InlineKeyboardButton,
    BufferedInputFile,
)
from aiogram.enums import ChatAction
from aiogram.client.default import DefaultBotProperties
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from config import TELEGRAM_BOT_TOKEN, ADMIN_USER_ID, CHILD_USER_ID, CHILD_NAME, MSK
import brain
import memory
import router as route_rules

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

router = Router()
_bot: Bot | None = None

TG_TEXT_LIMIT = 4096
TG_CAPTION_LIMIT = 1024
CHUNK_LIMIT = 3800          # с запасом под разметку
DEBOUNCE_SEC = 3.0          # склейка подряд идущих сообщений в один запрос
AUTO_VOICE_MAX_CHARS = 700  # короткий ответ на голосовое озвучиваем сразу


# ── per-user serialization ──────────────────────────────────────────

_locks: dict[int, asyncio.Lock] = {}


def _lock(user_id: int) -> asyncio.Lock:
    lk = _locks.get(user_id)
    if lk is None:
        lk = _locks[user_id] = asyncio.Lock()
    return lk


# ── answer cache for the TTS button ─────────────────────────────────

_tts_cache: "OrderedDict[tuple[int, int], str]" = OrderedDict()


def _cache_answer(chat_id: int, message_id: int, text: str):
    _tts_cache[(chat_id, message_id)] = text
    while len(_tts_cache) > 200:
        _tts_cache.popitem(last=False)


# ── input buffer (photo + text + voice → один запрос) ───────────────

class _Buffer:
    __slots__ = ("texts", "image", "image_mime", "was_voice", "message", "timer")

    def __init__(self, message: Message):
        self.texts: list[str] = []
        self.image: bytes | None = None
        self.image_mime = "image/jpeg"
        self.was_voice = False
        self.message = message
        self.timer: asyncio.Task | None = None


_buffers: dict[int, _Buffer] = {}


def _buffer(message: Message) -> _Buffer:
    user_id = message.from_user.id
    buf = _buffers.get(user_id)
    if buf is None:
        buf = _buffers[user_id] = _Buffer(message)
    buf.message = message
    return buf


def _arm(user_id: int, buf: _Buffer):
    if buf.timer:
        buf.timer.cancel()

    async def _fire():
        try:
            await asyncio.sleep(DEBOUNCE_SEC)
        except asyncio.CancelledError:
            return
        await _flush(user_id)

    buf.timer = asyncio.create_task(_fire())


async def _flush(user_id: int):
    buf = _buffers.pop(user_id, None)
    if not buf:
        return
    async with _lock(user_id):
        try:
            await _handle(buf)
        except Exception:
            log.exception("Handling failed for user %s", user_id)
            await _safe_answer(buf.message, "Ой, я запутался в облаках 🐮 Попробуй ещё разок!")


# ── typing indicator ────────────────────────────────────────────────

class _Typing:
    """Держит «печатает…» всё время генерации — PRO думает дольше."""

    def __init__(self, message: Message, action: str = ChatAction.TYPING):
        self.message = message
        self.action = action
        self.task: asyncio.Task | None = None

    async def _loop(self):
        while True:
            try:
                await self.message.bot.send_chat_action(
                    chat_id=self.message.chat.id, action=self.action,
                )
            except Exception:
                pass
            await asyncio.sleep(4.5)

    async def __aenter__(self):
        self.task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, *exc):
        if self.task:
            self.task.cancel()


# ── sending ─────────────────────────────────────────────────────────

def _tts_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🔊 Озвучить", callback_data="tts")]
    ])


# Быстрые действия под списком идей.
#
# Печатать ребёнку долго, а голосовое каждый раз ради «дай ещё» — тем более.
# Три кнопки закрывают три самых частых продолжения разговора, которые видно
# в истории переписки: «ещё», «а поподробнее вот эту» и «давай совсем другое».
QUICK_ACTIONS: dict[str, tuple[str, str]] = {
    "more": (
        "✨ Ещё идеи",
        "Дай ещё 5 идей, совершенно новых. Ничего из того, что уже предлагал.",
    ),
    "detail": (
        "🔍 Разбери первую",
        "Возьми первую идею из прошлого сообщения и распиши по шагам: "
        "что снимать, в каком порядке, что понадобится и как сделать первые "
        "три секунды цепляющими. Коротко и по делу.",
    ),
    "other": (
        "🎲 Другое направление",
        "Дай 5 идей совсем другого типа: другой жанр, другой формат, "
        "другая механика. Не похоже на прошлые.",
    ),
}


def _answer_kb(with_actions: bool) -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton(text="🔊 Озвучить", callback_data="tts")]]
    if with_actions:
        rows.append([
            InlineKeyboardButton(text=QUICK_ACTIONS["more"][0], callback_data="act:more"),
            InlineKeyboardButton(text=QUICK_ACTIONS["other"][0], callback_data="act:other"),
        ])
        rows.append([
            InlineKeyboardButton(text=QUICK_ACTIONS["detail"][0], callback_data="act:detail"),
        ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.S)
_BULLET_RE = re.compile(r"^\s*[*-]\s+", re.M)


def _format(text: str) -> str:
    """Markdown от модели → безопасный HTML для Telegram."""
    out = html.escape(text or "")
    out = _BOLD_RE.sub(r"<b>\1</b>", out)
    out = _BULLET_RE.sub("• ", out)
    return out


def _split(text: str, limit: int = CHUNK_LIMIT) -> list[str]:
    """Длинный ответ режем по абзацам, а не обрезаем."""
    text = (text or "").strip()
    if len(text) <= limit:
        return [text] if text else []

    chunks, cur = [], ""
    for para in text.split("\n"):
        if len(cur) + len(para) + 1 > limit and cur:
            chunks.append(cur.rstrip())
            cur = ""
        while len(para) > limit:               # один абзац длиннее лимита
            chunks.append(para[:limit])
            para = para[limit:]
        cur += para + "\n"
    if cur.strip():
        chunks.append(cur.rstrip())
    return chunks


async def _safe_answer(
    message: Message,
    text: str,
    with_tts: bool = False,
    with_actions: bool = False,
) -> Message | None:
    """Отправляет ответ целиком, разбивая на части. Кнопки — на последней."""
    chunks = _split(text)
    if not chunks:
        return None
    sent = None
    for i, chunk in enumerate(chunks):
        last = i == len(chunks) - 1
        sent = await message.answer(
            _format(chunk),
            reply_markup=_answer_kb(with_actions) if (last and with_tts) else None,
        )
        if last and with_tts and sent:
            _cache_answer(sent.chat.id, sent.message_id, text)
    return sent


# ── core handling ───────────────────────────────────────────────────

THINKING_LINES = [
    "Так, включаю облачную фантазию... 🌪️",
    "Секундочку, копаюсь в Библиотеке! 📚",
    "О, интересная задачка! Сейчас придумаю 🎬",
    "Дай мне пару секунд, придумываю что-то новенькое ✨",
]


async def _maybe_filler(message: Message, text: str):
    """Тяжёлый запрос считается дольше — сразу подаём голос, чтобы не ждать молча."""
    if (route_rules.requested_count(text) >= 3
            or route_rules.is_complaint(text)
            or route_rules._looks_like_ideas(text)):
        try:
            await message.answer(random.choice(THINKING_LINES))
        except Exception:
            pass


async def _handle(buf: _Buffer):
    message = buf.message
    user_id = message.from_user.id
    text = " ".join(t for t in buf.texts if t).strip()

    await _maybe_filler(message, text)

    async with _Typing(message):
        if brain._wants_image(text):
            answer, img_bytes, _ = await brain.generate_image(
                user_id, text, ref_image=buf.image, ref_mime=buf.image_mime,
            )
            if img_bytes:
                photo = BufferedInputFile(img_bytes, filename="image.png")
                caption = _format((answer or "")[:TG_CAPTION_LIMIT]) or None
                await message.answer_photo(photo, caption=caption)
            else:
                await _safe_answer(message, answer, with_tts=True)
        else:
            prev = await memory.get_recent_messages(user_id, 1)
            prev_bot = prev[-1]["text"] if prev and prev[-1]["role"] == "assistant" else ""

            reply = await brain.generate_response(
                user_id, text, image_bytes=buf.image, image_mime=buf.image_mime,
            )
            # Кнопки продолжения — только там, где им есть что продолжать.
            sent = await _safe_answer(
                message,
                reply.text,
                with_tts=True,
                with_actions=reply.task in ("ideas", "creative"),
            )

            if text and brain.is_complaint(text):
                asyncio.create_task(brain.log_complaint(user_id, text, prev_bot))

            # На голосовое отвечаем голосом, если ответ короткий.
            if buf.was_voice and sent and len(reply.text) <= AUTO_VOICE_MAX_CHARS:
                asyncio.create_task(_send_voice(message, reply.text))


async def _send_voice(message: Message, text: str):
    try:
        async with _Typing(message, ChatAction.RECORD_VOICE):
            ogg = await brain.text_to_speech(text)
        if ogg:
            await message.answer_voice(BufferedInputFile(ogg, filename="voice.ogg"))
    except Exception:
        log.exception("Auto voice reply failed")


# ── /start ──────────────────────────────────────────────────────────

@router.message(CommandStart())
async def cmd_start(message: Message):
    user_id = message.from_user.id
    name = CHILD_NAME if user_id == CHILD_USER_ID else (message.from_user.first_name or "друг")
    hour = datetime.now(MSK).hour
    if hour < 11:
        greeting = "Доброе утро"
    elif hour < 17:
        greeting = "Привет"
    elif hour < 22:
        greeting = "Добрый вечер"
    else:
        greeting = "Ого, ты ещё не спишь"

    await message.answer(
        f"Муу! {greeting}, {name}! Я Коровик — телёнок с Облачной фермы 🐮\n\n"
        "Я твой напарник по блогу. Могу:\n"
        "• придумать идеи для видео и разобрать любую по шагам\n"
        "• объяснить трюки со съёмкой и монтажом\n"
        "• нарисовать картинку\n"
        "• просто поболтать\n\n"
        "Пиши текстом, говори голосом или шли фотку — я всё вижу и слышу 🎤",
        reply_markup=_tts_kb(),
    )


# ── admin ───────────────────────────────────────────────────────────

async def _admin_command(message: Message, text: str) -> bool:
    cmd = text.strip().lower()
    if cmd == "##сброс":
        await memory.delete_user_data(CHILD_USER_ID)
        await memory.delete_user_data(message.from_user.id)
        await message.answer("Данные сброшены 🔄")
        return True
    if cmd == "##профиль":
        profile = await memory.get_profile(CHILD_USER_ID)
        ideas = await memory.get_recent_ideas(CHILD_USER_ID, 20)
        lines = [f"{k}: {'; '.join(v) if v else '—'}" for k, v in profile.items()]
        lines.append("\nПоследние идеи:\n" + ("\n".join(f"• {i}" for i in ideas) if ideas else "—"))
        await _safe_answer(message, "🧾 Профиль Златы\n\n" + "\n".join(lines))
        return True
    if cmd == "##отчет" or cmd == "##отчёт":
        report = await brain.generate_daily_report()
        await _safe_answer(message, report or "Жалоб за сегодня нет 👌")
        return True
    return False


# ── text ────────────────────────────────────────────────────────────

@router.message(F.text)
async def handle_text(message: Message):
    user_id = message.from_user.id
    text = message.text or ""

    if user_id == ADMIN_USER_ID and text.strip().startswith("##"):
        if await _admin_command(message, text):
            return

    buf = _buffer(message)
    buf.texts.append(text)
    await _typing_once(message)
    _arm(user_id, buf)


# ── photo ───────────────────────────────────────────────────────────

@router.message(F.photo)
async def handle_photo(message: Message):
    user_id = message.from_user.id
    try:
        photo = message.photo[-1]
        file = await message.bot.get_file(photo.file_id)
        sink = io.BytesIO()
        await message.bot.download_file(file.file_path, sink)
        image_bytes = sink.getvalue()
    except Exception:
        log.exception("Photo download failed")
        await message.answer("Не получилось загрузить фото 😕 Попробуй ещё раз!")
        return

    buf = _buffer(message)
    buf.image = image_bytes
    if message.caption:
        buf.texts.append(message.caption)
    await _typing_once(message)
    _arm(user_id, buf)


# ── voice ───────────────────────────────────────────────────────────

@router.message(F.voice)
async def handle_voice(message: Message):
    user_id = message.from_user.id
    await _typing_once(message)

    try:
        file = await message.bot.get_file(message.voice.file_id)
        sink = io.BytesIO()
        await message.bot.download_file(file.file_path, sink)
        audio_bytes = sink.getvalue()
    except Exception:
        log.exception("Voice download failed")
        await message.answer("Не получилось скачать голосовое 😕 Попробуй ещё раз!")
        return

    text = await brain.transcribe_voice(audio_bytes)
    if not text:
        await message.answer("Не расслышал 😕 Скажи ещё разок или напиши текстом!")
        return

    buf = _buffer(message)
    buf.was_voice = True
    buf.texts.append(f"[голосовое сообщение] {text}")
    _arm(user_id, buf)


async def _typing_once(message: Message):
    try:
        await message.bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)
    except Exception:
        pass


# ── TTS button ──────────────────────────────────────────────────────

@router.callback_query(F.data.startswith("act:"))
async def handle_quick_action(callback: CallbackQuery):
    """Кнопка продолжения: то же самое, что если бы она сама попросила словами."""
    key = (callback.data or "").split(":", 1)[-1]
    action = QUICK_ACTIONS.get(key)
    msg = callback.message
    if not action or not msg:
        await callback.answer()
        return

    label, prompt = action
    await callback.answer(label)

    user_id = callback.from_user.id
    async with _lock(user_id):
        await _maybe_filler(msg, prompt)
        async with _Typing(msg):
            reply = await brain.generate_response(user_id, prompt)
        await _safe_answer(
            msg,
            reply.text,
            with_tts=True,
            with_actions=reply.task in ("ideas", "creative"),
        )


@router.callback_query(F.data == "tts")
async def handle_tts(callback: CallbackQuery):
    msg = callback.message
    text = _tts_cache.get((msg.chat.id, msg.message_id)) or msg.text or msg.caption
    if not text:
        await callback.answer("Нечего озвучивать 🤷")
        return

    await callback.answer("Читаю вслух... 🎙️")
    async with _Typing(msg, ChatAction.RECORD_VOICE):
        ogg = await brain.text_to_speech(text)
    if ogg:
        await msg.reply_voice(BufferedInputFile(ogg, filename="voice.ogg"))
    else:
        await msg.answer("Не получилось озвучить 😕 Попробуй ещё раз!")


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
    await memory.ensure_indexes()

    _bot = Bot(token=TELEGRAM_BOT_TOKEN, default=DefaultBotProperties(parse_mode="HTML"))
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
