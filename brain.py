import asyncio
import io
import json
import logging
import re
import wave
from datetime import datetime
from typing import NamedTuple

from google import genai
from google.genai import types

import memory
import router
from config import (
    GEMINI_API_KEY, TIER_MODELS, MAX_TOKENS, THINKING_LEVEL, TEMPERATURE,
    MODEL_SERVICE, MODEL_TTS, MODEL_TTS_FALLBACK, MODEL_IMAGE, MODEL_IMAGE_PRO,
    TTS_VOICE, SYSTEM_PROMPT, SUMMARIZE_PROMPT, REPORT_PROMPT,
    PROFILE_EXTRACT_PROMPT, IDEAS_EXTRACT_PROMPT,
    IMAGE_GEN_TRIGGERS, ESCALATABLE_TASKS, CRITIC_THRESHOLD,
    WINDOW_SIZE, SUMMARIZE_THRESHOLD, MSK,
)

log = logging.getLogger(__name__)

ai = genai.Client(api_key=GEMINI_API_KEY)

FALLBACK_TEXT = "Ой, что-то я замечтался и потерял мысль 🐮 Спроси ещё разок!"


class Reply(NamedTuple):
    """Ответ вместе с тем, что это было за запрос.

    Тип задачи нужен интерфейсу: под списком идей уместны кнопки «ещё» и
    «разбери первую», а под обычной болтовнёй они только мешают. Раньше это
    знание оставалось внутри мозга, и пульт про него не знал.
    """

    text: str
    task: str


# ── prompt building ─────────────────────────────────────────────────

def _profile_block(profile: dict, summary: str | None, ideas: list[str]) -> str:
    lines = []
    labels = {
        "likes": "Любит",
        "dislikes": "НЕ любит и уже забраковала — не предлагай такое",
        "projects": "Чем занята сейчас",
        "people": "Близкие",
        "style": "Как с ней лучше",
    }
    for key, label in labels.items():
        items = profile.get(key) or []
        if items:
            lines.append(f"- {label}: " + "; ".join(items))

    block = "Ты помнишь ваши предыдущие разговоры.\n"
    if lines:
        block += "Профиль:\n" + "\n".join(lines) + "\n"
    if summary:
        block += f"Из прошлых разговоров: {summary}\n"
    if ideas:
        block += (
            "Идеи, которые ты ей УЖЕ предлагал (повторять их нельзя, "
            "нужны новые):\n- " + "\n- ".join(ideas[:30]) + "\n"
        )
    return block.strip()


async def _system_prompt(user_id: int, extra: str = "") -> str:
    now = datetime.now(MSK)
    time_str = f"Сейчас {now.strftime('%H:%M')}, {now.strftime('%d.%m.%Y')}"

    profile, summary, ideas = await asyncio.gather(
        memory.get_profile(user_id),
        memory.get_summary(user_id),
        memory.get_recent_ideas(user_id),
    )
    prompt = SYSTEM_PROMPT.format(
        current_time=time_str,
        profile_block=_profile_block(profile, summary, ideas),
    )
    if extra:
        prompt += "\n\n# Задание на этот ответ\n" + extra
    return prompt


def _turn_directives(d: router.Decision) -> str:
    """Инструкции под конкретный запрос — идут в system_instruction, не в историю."""
    parts = []
    if d.wants_count:
        parts.append(
            f"Она попросила {d.wants_count} пунктов — дай РОВНО {d.wants_count}, "
            "пронумерованных, каждый доведи до конца."
        )
    if d.is_complaint:
        parts.append(
            "Прошлый ответ ей не понравился. Не оправдывайся — одна короткая фраза, "
            "и сразу набор идей ДРУГОГО типа: другой жанр, другая механика, "
            "другой уровень неожиданности. Ничего похожего на прошлый раз."
        )
    if d.wants_more:
        parts.append("Она просит ещё — все идеи должны быть новыми, без повторов прошлых.")
    if d.task in ("ideas", "creative"):
        parts.append(
            "Она просит что-то придумать. Тема — РОВНО та, о которой она "
            "спросила: скучно — занятия на сегодня, подарок — подарок, игра — "
            "игра. Не подменяй её тему съёмками для канала. Каждый пункт "
            "конкретный, из того что есть под рукой, с короткой фишкой. "
            "Никаких банальностей из запретного списка и ничего из её дизлайков."
        )
    if d.task == "emotional":
        parts.append(
            "Тема чувствительная. Сначала признай чувство, будь спокоен, "
            "без диагнозов и советов — и мягко направь к родителям. "
            "Никаких списков идей и предложений «давай снимем видео»: "
            "сейчас она пришла не за этим."
        )
    return "\n".join(f"- {p}" for p in parts)


async def _build_context(user_id: int) -> list:
    recent = await memory.get_recent_messages(user_id, WINDOW_SIZE)
    contents = []
    for msg in recent:
        role = "user" if msg["role"] == "user" else "model"
        text = (msg.get("text") or "").strip()
        if not text:
            continue
        contents.append(types.Content(role=role, parts=[types.Part(text=text)]))
    return contents


def _wants_image(text: str) -> bool:
    lower = (text or "").lower()
    return any(t in lower for t in IMAGE_GEN_TRIGGERS)


# ── generation core ─────────────────────────────────────────────────

def _config(tier: str, system_instruction: str, use_search: bool) -> types.GenerateContentConfig:
    kwargs = dict(
        system_instruction=system_instruction,
        temperature=TEMPERATURE[tier],
        max_output_tokens=MAX_TOKENS[tier],
        thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL[tier]),
    )
    if use_search:
        kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
    return types.GenerateContentConfig(**kwargs)


def _text_of(resp) -> str:
    try:
        if resp.text:
            return resp.text.strip()
    except Exception:
        pass
    out = []
    for cand in (resp.candidates or []):
        if cand.content and cand.content.parts:
            for p in cand.content.parts:
                if getattr(p, "text", None):
                    out.append(p.text)
    return "".join(out).strip()


def _truncated(resp) -> bool:
    try:
        return str(resp.candidates[0].finish_reason or "").upper().endswith("MAX_TOKENS")
    except Exception:
        return False


async def _generate(
    tier: str,
    contents: list,
    system_instruction: str,
    use_search: bool,
) -> str:
    """Один заход генерации с дописыванием обрыва и падением на уровень ниже."""
    model = TIER_MODELS[tier]
    resp = await ai.aio.models.generate_content(
        model=model,
        contents=contents,
        config=_config(tier, system_instruction, use_search),
    )
    answer = _text_of(resp)

    # Ответ упёрся в лимит токенов — дописываем продолжение, а не обрываем.
    tries = 0
    while _truncated(resp) and answer and tries < 2:
        tries += 1
        cont = list(contents) + [
            types.Content(role="model", parts=[types.Part(text=answer[-4000:])]),
            types.Content(role="user", parts=[types.Part(
                text="Продолжи ровно с места обрыва и доведи ответ до конца. "
                     "Не повторяй уже написанное, не здоровайся заново."
            )]),
        ]
        resp = await ai.aio.models.generate_content(
            model=model,
            contents=cont,
            config=_config(tier, system_instruction, use_search),
        )
        more = _text_of(resp)
        if not more:
            break
        answer = answer.rstrip() + ("\n" if not answer.endswith("\n") else "") + more
        log.info("Continued truncated answer (part %d)", tries + 1)

    return answer


async def _generate_resilient(
    tier: str,
    contents: list,
    system_instruction: str,
    use_search: bool,
) -> str:
    """Ретраи и деградация модели: pro → flash → lite."""
    chain = {"pro": ["pro", "flash", "lite"], "flash": ["flash", "lite"], "lite": ["lite", "flash"]}[tier]
    last_err = None
    for i, t in enumerate(chain):
        for attempt in range(2):
            try:
                answer = await _generate(t, contents, system_instruction, use_search)
                if answer:
                    if t != tier:
                        log.warning("Degraded %s → %s", tier, t)
                    return answer
            except Exception as e:
                last_err = e
                log.warning("Generation %s attempt %d failed: %s", t, attempt + 1, str(e)[:200])
                await asyncio.sleep(1.0 + attempt)
        # поиск иногда сам по себе валит запрос — пробуем без него
        if use_search and i == 0:
            use_search = False
    if last_err:
        log.error("All generation attempts failed: %s", str(last_err)[:300])
    return ""


# ── public: chat ────────────────────────────────────────────────────

async def generate_response(
    user_id: int,
    text: str,
    image_bytes: bytes | None = None,
    image_mime: str = "image/jpeg",
) -> Reply:
    """Полный цикл: роутинг → генерация → критик/эскалация → сохранение."""
    stored = text if not image_bytes else (text or "[прислала фото]")
    total = await memory.save_message(user_id, "user", stored)

    decision = await router.route(ai, text, has_image=image_bytes is not None)
    decision.wants_count = decision.wants_count or router.requested_count(text)
    log.info("Route: %r", decision)

    directives = _turn_directives(decision)
    system_instruction = await _system_prompt(user_id, directives)

    contents = await _build_context(user_id)
    if image_bytes:
        # последняя реплика уже в истории — заменяем её версией с картинкой
        if contents and contents[-1].role == "user":
            contents.pop()
        contents.append(types.Content(role="user", parts=[
            types.Part.from_bytes(data=image_bytes, mime_type=image_mime),
            types.Part(text=text or "Что ты видишь на этом фото?"),
        ]))

    answer = await _generate_resilient(
        decision.tier, contents, system_instruction, decision.needs_search,
    )

    # Эскалация: слабый ответ flash по «тяжёлой» задаче — переспрашиваем PRO.
    if answer and decision.tier != "pro" and decision.task in ESCALATABLE_TASKS:
        score, fix = await router.critique(ai, text, answer, directives or "нет особых требований")
        if score < CRITIC_THRESHOLD:
            log.info("Escalating to PRO (score=%s, %s)", score, fix[:120])
            esc = system_instruction + (
                "\n\n# Разбор предыдущей попытки\n"
                f"Черновик получился слабым: {fix}\n"
                "Сделай заметно лучше: конкретнее, свежее, точно по её просьбе."
            )
            better = await _generate_resilient("pro", contents, esc, decision.needs_search)
            if better:
                answer = better

    if not answer:
        answer = FALLBACK_TEXT

    await memory.save_message(user_id, "assistant", answer)
    asyncio.create_task(_post_turn(user_id, text, answer, decision, total + 1))
    return Reply(answer, decision.task)


async def analyze_image(user_id: int, image_bytes: bytes, mime_type: str, caption: str = "") -> Reply:
    """Фото со Златы: анализ идёт тем же оркестрованным путём."""
    return await generate_response(
        user_id, caption or "Что ты видишь на этом фото?",
        image_bytes=image_bytes, image_mime=mime_type,
    )


# ── background: profile, idea ledger, summary ───────────────────────

async def _post_turn(user_id: int, user_text: str, answer: str, decision: router.Decision, total: int):
    try:
        tasks = [_update_profile(user_id, user_text, answer), _maybe_summarize(user_id, total)]
        if decision.task in ("ideas", "creative"):
            tasks.append(_remember_ideas(user_id, answer))
        await asyncio.gather(*tasks, return_exceptions=True)
    except Exception:
        log.exception("Post-turn tasks failed")


async def _update_profile(user_id: int, user_text: str, answer: str):
    """Профиль предпочтений — главное лекарство от «я же просила не челленджи»."""
    try:
        current = await memory.get_profile(user_id)
        payload = (
            f"Текущий профиль:\n{json.dumps(current, ensure_ascii=False)}\n\n"
            f"Ребёнок написал:\n{user_text[:2000]}\n\n"
            f"Бот ответил:\n{answer[:2000]}"
        )
        resp = await ai.aio.models.generate_content(
            model=MODEL_SERVICE,
            contents=payload,
            config=types.GenerateContentConfig(
                system_instruction=PROFILE_EXTRACT_PROMPT,
                temperature=0.1,
                max_output_tokens=1024,
                response_mime_type="application/json",
                response_schema={
                    "type": "object",
                    "properties": {
                        k: {"type": "array", "items": {"type": "string"}}
                        for k in ("likes", "dislikes", "projects", "people", "style")
                    },
                    "required": ["likes", "dislikes", "projects", "people", "style"],
                },
                thinking_config=types.ThinkingConfig(thinking_level="low"),
            ),
        )
        data = json.loads(resp.text or "{}")
        if isinstance(data, dict):
            await memory.upsert_profile(user_id, data)
    except Exception:
        log.exception("Profile update failed for %s", user_id)


async def _remember_ideas(user_id: int, answer: str):
    try:
        resp = await ai.aio.models.generate_content(
            model=MODEL_SERVICE,
            contents=answer[:8000],
            config=types.GenerateContentConfig(
                system_instruction=IDEAS_EXTRACT_PROMPT,
                temperature=0.0,
                max_output_tokens=512,
                thinking_config=types.ThinkingConfig(thinking_level="low"),
            ),
        )
        titles = [ln.strip(" -*•\t") for ln in (resp.text or "").splitlines() if ln.strip()]
        await memory.save_ideas(user_id, titles[:12])
    except Exception:
        log.exception("Idea extraction failed for %s", user_id)


async def _maybe_summarize(user_id: int, total: int):
    """Если сообщений накопилось больше порога — сворачиваем самые старые."""
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
        prompt = (
            f"Предыдущее резюме:\n{prev_summary}\n\nНовые сообщения:\n{text_block}"
            if prev_summary else text_block
        )

        resp = await ai.aio.models.generate_content(
            model=MODEL_SERVICE,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SUMMARIZE_PROMPT,
                temperature=0.3,
                max_output_tokens=1024,
            ),
        )
        new_summary = (resp.text or "").strip()
        if new_summary:
            await memory.upsert_summary(user_id, new_summary)
            log.info("Summarized %d messages for user %s", len(old_msgs), user_id)
    except Exception:
        log.exception("Summarization failed for user %s", user_id)


# ── image generation ───────────────────────────────────────────────

async def generate_image(
    user_id: int,
    text: str,
    ref_image: bytes | None = None,
    ref_mime: str = "image/jpeg",
) -> tuple[str, bytes | None, str | None]:
    """Рисование. Nano Banana 2, при отказе — Nano Banana Pro."""
    total = await memory.save_message(user_id, "user", text or "[прислала фото с просьбой нарисовать]")

    contents = await _build_context(user_id)
    if ref_image:
        contents.append(types.Content(role="user", parts=[
            types.Part.from_bytes(data=ref_image, mime_type=ref_mime),
            types.Part(text=text or "Нарисуй это"),
        ]))

    for model in (MODEL_IMAGE, MODEL_IMAGE_PRO):
        try:
            resp = await ai.aio.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                    temperature=1.0,
                ),
            )
            result_text, result_image, result_mime = "", None, None
            if resp.candidates and resp.candidates[0].content:
                for part in resp.candidates[0].content.parts:
                    if part.text:
                        result_text += part.text
                    elif part.inline_data and part.inline_data.data:
                        result_image = part.inline_data.data
                        result_mime = part.inline_data.mime_type or "image/png"

            if result_image:
                result_text = (result_text or "Вот что получилось! 🎨").strip()
                await memory.save_message(user_id, "assistant", result_text)
                asyncio.create_task(_maybe_summarize(user_id, total + 1))
                return result_text, result_image, result_mime
            log.warning("Model %s returned no image", model)
        except Exception:
            log.exception("Image generation failed on %s", model)

    fallback = "Ой, не получилось нарисовать 😕 Попробуй описать по-другому!"
    await memory.save_message(user_id, "assistant", fallback)
    return fallback, None, None


# ── complaints ──────────────────────────────────────────────────────

def is_complaint(text: str) -> bool:
    return router.is_complaint(text)


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
        ts = c["ts"].astimezone(MSK).strftime("%H:%M")
        lines.append(f"[{ts}] Ребёнок: {c['user_message']}\nБот: {c['bot_message']}")
    block = "\n\n".join(lines)

    resp = await ai.aio.models.generate_content(
        model=MODEL_SERVICE,
        contents=f"Жалобы за сегодня ({len(complaints)} шт.):\n\n{block}",
        config=types.GenerateContentConfig(
            system_instruction=REPORT_PROMPT,
            temperature=0.4,
            max_output_tokens=2048,
        ),
    )
    report = (resp.text or "").strip()
    if report:
        now = datetime.now(MSK)
        header = f"📋 Коровик — отчёт за {now.strftime('%d.%m.%Y')}\nЖалоб: {len(complaints)}\n\n"
        return header + report
    return None


# ── TTS ─────────────────────────────────────────────────────────────

TTS_CHUNK_CHARS = 1800

_MD_RE = re.compile(r"[*_`#]+")


def _strip_md(text: str) -> str:
    """Убираем разметку — иначе диктор читает звёздочки."""
    return _MD_RE.sub("", text or "").strip()


_SENT_RE = re.compile(r"(?<=[.!?…])\s+")


def _tts_chunks(text: str) -> list[str]:
    """Режем длинный текст по абзацам и предложениям — TTS не любит гигантские входы."""
    text = (text or "").strip()
    if len(text) <= TTS_CHUNK_CHARS:
        return [text] if text else []

    pieces = []
    for para in text.split("\n"):
        if len(para) <= TTS_CHUNK_CHARS:
            pieces.append(para)
            continue
        for sent in _SENT_RE.split(para):
            while len(sent) > TTS_CHUNK_CHARS:
                pieces.append(sent[:TTS_CHUNK_CHARS])
                sent = sent[TTS_CHUNK_CHARS:]
            pieces.append(sent)

    chunks, cur = [], ""
    for piece in pieces:
        if len(cur) + len(piece) + 1 > TTS_CHUNK_CHARS and cur:
            chunks.append(cur.strip())
            cur = ""
        cur += piece + "\n"
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


async def _tts_once(model: str, text: str) -> bytes | None:
    resp = await ai.aio.models.generate_content(
        model=model,
        contents=f"Прочитай вслух дружелюбным, весёлым тоном, как добрый телёнок: {text}",
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=TTS_VOICE)
                )
            ),
        ),
    )
    data = resp.candidates[0].content.parts[0].inline_data.data
    return data or None


async def text_to_speech(text: str) -> bytes | None:
    """OGG/Opus из текста. Ретраи и запасная модель — TTS часто отдаёт 500."""
    pcm_total = b""
    for chunk in _tts_chunks(_strip_md(text)):
        pcm = None
        for model in (MODEL_TTS, MODEL_TTS_FALLBACK):
            for attempt in range(2):
                try:
                    pcm = await _tts_once(model, chunk)
                    if pcm:
                        break
                except Exception as e:
                    log.warning("TTS %s attempt %d failed: %s", model, attempt + 1, str(e)[:200])
                    await asyncio.sleep(1.0 + attempt)
            if pcm:
                break
        if not pcm:
            log.error("TTS failed for chunk of %d chars", len(chunk))
            return None
        pcm_total += pcm

    if not pcm_total:
        return None
    return await _pcm_to_ogg(pcm_total)


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
        log.error("ffmpeg failed: %s", err.decode(errors='replace')[:500])
        return None
    return ogg_data


# ── voice transcription ────────────────────────────────────────────

TRANSCRIBE_PROMPT = (
    "Транскрибируй аудио на русском. Говорит девочка 8 лет по имени Злата, "
    "она обращается к боту-телёнку по имени Коровик (может произносить как "
    "«Королик», «Коробка», «Коровник» — пиши «Коровик»). Темы: видеоблог, "
    "идеи для видео, челленджи, игрушки, школа, семья. "
    "Верни ТОЛЬКО текст сказанного, без пояснений и без кавычек."
)


async def transcribe_voice(audio_bytes: bytes, retries: int = 3) -> str:
    part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/ogg")
    for attempt in range(retries):
        try:
            resp = await ai.aio.models.generate_content(
                model=MODEL_SERVICE,
                contents=[part, TRANSCRIBE_PROMPT],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=2048,
                ),
            )
            result = (resp.text or "").strip()
            if result:
                return result
        except Exception:
            log.exception("Voice transcription attempt %d failed", attempt + 1)
        if attempt < retries - 1:
            await asyncio.sleep(1.0 + attempt)
    return ""
