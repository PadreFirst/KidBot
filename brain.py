import asyncio
import io
import wave
import logging
from datetime import datetime

from google import genai
from google.genai import types

from config import (
    GEMINI_API_KEY, MODEL_CHAT, MODEL_SERVICE, MODEL_TTS, TTS_VOICE,
    SYSTEM_PROMPT, SUMMARIZE_PROMPT, REPORT_PROMPT,
    COMPLAINT_KEYWORDS, WINDOW_SIZE, SUMMARIZE_THRESHOLD, MSK,
)
import memory

log = logging.getLogger(__name__)

ai = genai.Client(api_key=GEMINI_API_KEY)


# ── chat ────────────────────────────────────────────────────────────

async def generate_response(user_id: int, text: str) -> str:
    """Full pipeline: save user msg, build context, generate, save assistant msg, background tasks."""
    total = await memory.save_message(user_id, "user", text)

    summary = await memory.get_summary(user_id)
    recent = await memory.get_recent_messages(user_id, WINDOW_SIZE)

    contents = []
    if summary:
        contents.append(types.Content(role="user", parts=[types.Part.from_text(
            f"[Из наших прошлых разговоров ты помнишь: {summary}]"
        )]))
        contents.append(types.Content(role="model", parts=[types.Part.from_text(
            "Помню! Продолжаем 🐮"
        )]))

    for msg in recent:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(msg["text"])]))

    resp = await ai.aio.models.generate_content(
        model=MODEL_CHAT,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.85,
            max_output_tokens=1024,
        ),
    )
    answer = (resp.text or "").strip() or "Ой, что-то я завис 🐮 Попробуй ещё раз!"

    await memory.save_message(user_id, "assistant", answer)

    asyncio.create_task(_maybe_summarize(user_id, total + 1))

    return answer


async def _maybe_summarize(user_id: int, total: int):
    """If message count exceeds threshold, summarize oldest chunk."""
    if total <= SUMMARIZE_THRESHOLD:
        return
    try:
        count = await memory.message_count(user_id)
        if count <= SUMMARIZE_THRESHOLD:
            return
        to_summarize = count - WINDOW_SIZE
        if to_summarize < 10:
            return

        old_msgs = await memory.pop_oldest_messages(user_id, to_summarize)
        if not old_msgs:
            return

        prev_summary = await memory.get_summary(user_id) or ""
        text_block = "\n".join(f"{m['role']}: {m['text']}" for m in old_msgs)

        prompt = f"Предыдущее резюме:\n{prev_summary}\n\nНовые сообщения:\n{text_block}" if prev_summary else text_block

        resp = await ai.aio.models.generate_content(
            model=MODEL_SERVICE,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SUMMARIZE_PROMPT,
                temperature=0.3,
                max_output_tokens=512,
            ),
        )
        new_summary = (resp.text or "").strip()
        if new_summary:
            await memory.upsert_summary(user_id, new_summary)
            log.info("Summarized %d messages for user %s", len(old_msgs), user_id)
    except Exception:
        log.exception("Summarization failed for user %s", user_id)


# ── complaint detection ─────────────────────────────────────────────

def is_complaint(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in COMPLAINT_KEYWORDS)


async def log_complaint(user_id: int, user_msg: str, bot_msg: str):
    try:
        await memory.save_complaint(user_id, user_msg, bot_msg)
    except Exception:
        log.exception("Failed to log complaint")


# ── daily report ────────────────────────────────────────────────────

async def generate_daily_report() -> str | None:
    complaints = await memory.get_today_complaints()
    if not complaints:
        return None

    lines = []
    for c in complaints:
        ts = c["ts"].strftime("%H:%M")
        lines.append(f"[{ts}] Ребёнок: {c['user_message']}\nБот: {c['bot_message']}")
    block = "\n\n".join(lines)

    resp = await ai.aio.models.generate_content(
        model=MODEL_SERVICE,
        contents=f"Жалобы за сегодня ({len(complaints)} шт.):\n\n{block}",
        config=types.GenerateContentConfig(
            system_instruction=REPORT_PROMPT,
            temperature=0.4,
            max_output_tokens=1024,
        ),
    )
    report = (resp.text or "").strip()
    if report:
        now = datetime.now(MSK)
        header = f"📋 Коровик — отчёт за {now.strftime('%d.%m.%Y')}\nЖалоб: {len(complaints)}\n\n"
        return header + report
    return None


# ── TTS ─────────────────────────────────────────────────────────────

async def text_to_speech(text: str) -> bytes | None:
    """Generate OGG/Opus voice from text via Gemini TTS. Returns bytes or None."""
    try:
        resp = await ai.aio.models.generate_content(
            model=MODEL_TTS,
            contents=f"Прочитай вслух дружелюбным, весёлым тоном: {text}",
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=TTS_VOICE,
                        )
                    )
                ),
            ),
        )

        pcm_data = resp.candidates[0].content.parts[0].inline_data.data
        if not pcm_data:
            return None

        return await _pcm_to_ogg(pcm_data)
    except Exception:
        log.exception("TTS generation failed")
        return None


async def _pcm_to_ogg(pcm_data: bytes, sample_rate: int = 24000) -> bytes | None:
    """Convert raw PCM to OGG/Opus using ffmpeg."""
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)

    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-i", "pipe:0",
        "-c:a", "libopus", "-b:a", "64k",
        "-f", "ogg", "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    ogg_data, err = await proc.communicate(wav_buf.getvalue())

    if proc.returncode != 0:
        log.error("ffmpeg failed: %s", err.decode(errors="replace")[:500])
        return None
    return ogg_data


# ── voice transcription ────────────────────────────────────────────

async def transcribe_voice(audio_bytes: bytes) -> str:
    try:
        part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/ogg")
        resp = await ai.aio.models.generate_content(
            model=MODEL_SERVICE,
            contents=[part, "Транскрибируй аудио. Верни ТОЛЬКО текст, без пояснений."],
            config=types.GenerateContentConfig(temperature=0.1),
        )
        return (resp.text or "").strip()
    except Exception:
        log.exception("Voice transcription failed")
        return ""
