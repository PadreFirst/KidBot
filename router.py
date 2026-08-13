"""Оркестрация моделей.

Каждый запрос сначала проходит дешёвый классификатор (flash-lite),
который выбирает уровень: lite / flash / pro, нужен ли веб-поиск и
какой это тип задачи. Поверх классификатора — жёсткие правила
(недовольство ребёнка, просьба «дай ещё», длинный список) и
пост-проверка качества с эскалацией flash → pro.
"""

import asyncio
import json
import logging
import re

from google.genai import types

from config import (
    MODEL_ROUTER, MODEL_SERVICE, ROUTER_PROMPT, CRITIC_PROMPT,
    ROUTER_TIMEOUT_SEC, CRITIC_TIMEOUT_SEC, COMPLAINT_KEYWORDS, MORE_KEYWORDS,
)

log = logging.getLogger(__name__)

TASKS = [
    "chat", "ideas", "creative", "explain", "howto",
    "plan", "factual", "emotional", "image_request", "other",
]

_ROUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "tier": {"type": "string", "enum": ["lite", "flash", "pro"]},
        "needs_search": {"type": "boolean"},
        "task": {"type": "string", "enum": TASKS},
    },
    "required": ["tier", "needs_search", "task"],
}

_CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer"},
        "reason": {"type": "string"},
        "fix": {"type": "string"},
    },
    "required": ["score", "reason", "fix"],
}

IDEA_WORDS = (
    "иде", "придума", "предлож", "посоветуй", "варианты", "что снять",
    "про что снять", "сценар", "челлендж", "видео", "канал", "блог",
)
_COUNT_UNITS = r"(?:идей|идеи|идею|штук|вариант\w*|пункт\w*|способ\w*|совет\w*)"
_COUNT_RE = re.compile(rf"\b(\d{{1,2}})\s*{_COUNT_UNITS}", re.I)
_WORD_NUMS = {
    "два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5, "шесть": 6,
    "семь": 7, "восемь": 8, "девять": 9, "десять": 10, "двенадцать": 12,
    "пятнадцать": 15, "двадцать": 20,
}
_WORD_COUNT_RE = re.compile(
    rf"\b({'|'.join(_WORD_NUMS)})\s*{_COUNT_UNITS}", re.I,
)


class Decision:
    __slots__ = ("tier", "needs_search", "task", "wants_count", "is_complaint", "wants_more", "source")

    def __init__(self, tier="flash", needs_search=False, task="chat",
                 wants_count=0, is_complaint=False, wants_more=False, source="rules"):
        self.tier = tier
        self.needs_search = needs_search
        self.task = task
        self.wants_count = wants_count
        self.is_complaint = is_complaint
        self.wants_more = wants_more
        self.source = source

    def __repr__(self):
        return (f"Decision(tier={self.tier}, task={self.task}, search={self.needs_search}, "
                f"count={self.wants_count}, complaint={self.is_complaint}, src={self.source})")


def _lower(text: str) -> str:
    return (text or "").lower().replace("ё", "е")


def is_complaint(text: str) -> bool:
    low = _lower(text)
    return any(_lower(kw) in low for kw in COMPLAINT_KEYWORDS)


def wants_more(text: str) -> bool:
    low = _lower(text)
    return any(_lower(kw) in low for kw in MORE_KEYWORDS)


def requested_count(text: str) -> int:
    """«дай 10 идей» / «дай пять идей» → число. Ноль, если не названо.

    Если названы два числа («пять или десять»), берём большее —
    лучше дать больше, чем недодать.
    """
    text = text or ""
    found = []
    for m in _COUNT_RE.finditer(text):
        try:
            found.append(int(m.group(1)))
        except ValueError:
            pass
    for m in _WORD_COUNT_RE.finditer(text.lower()):
        found.append(_WORD_NUMS[m.group(1).lower()])

    found = [n for n in found if 1 <= n <= 30]
    return max(found) if found else 0


def _looks_like_ideas(text: str) -> bool:
    low = _lower(text)
    return any(w in low for w in IDEA_WORDS)


def _heuristic(text: str) -> Decision:
    """Запасной вариант, если LLM-роутер недоступен."""
    low = _lower(text).strip()
    complaint = is_complaint(low)
    count = requested_count(low)
    more = wants_more(low)

    if complaint or count >= 3 or _looks_like_ideas(low):
        task = "ideas" if (_looks_like_ideas(low) or count) else "creative"
        return Decision("pro", _looks_like_ideas(low), task, count, complaint, more, "rules")

    if len(low) <= 25 and not low.endswith("?"):
        return Decision("lite", False, "chat", 0, False, more, "rules")

    return Decision("flash", False, "chat", count, False, more, "rules")


async def route(ai, text: str, has_image: bool = False) -> Decision:
    """Выбор модели. LLM-классификатор + жёсткие правила поверх него."""
    fallback = _heuristic(text)
    decision = fallback

    try:
        resp = await asyncio.wait_for(
            ai.aio.models.generate_content(
                model=MODEL_ROUTER,
                contents=f"Запрос ребёнка: {text[:1500]}",
                config=types.GenerateContentConfig(
                    system_instruction=ROUTER_PROMPT,
                    temperature=0.0,
                    max_output_tokens=512,
                    response_mime_type="application/json",
                    response_schema=_ROUTE_SCHEMA,
                    thinking_config=types.ThinkingConfig(thinking_level="low"),
                ),
            ),
            timeout=ROUTER_TIMEOUT_SEC,
        )
        data = json.loads(resp.text or "{}")
        decision = Decision(
            tier=data.get("tier", fallback.tier),
            needs_search=bool(data.get("needs_search", False)),
            task=data.get("task", fallback.task),
            wants_count=fallback.wants_count,
            is_complaint=fallback.is_complaint,
            wants_more=fallback.wants_more,
            source="llm",
        )
    except Exception as e:
        log.warning("Router fell back to heuristics: %s", str(e)[:200])

    # Жёсткие правила поверх классификатора.
    if decision.is_complaint or decision.wants_count >= 3:
        decision.tier = "pro"
    if decision.task in ("ideas", "creative", "plan"):
        decision.tier = "pro"
    if has_image and decision.tier == "lite":
        decision.tier = "flash"
    if decision.needs_search and decision.tier == "lite":
        decision.tier = "flash"

    return decision


async def critique(ai, request: str, answer: str, constraints: str) -> tuple[int, str]:
    """Оценка ответа 1-5 дешёвой моделью. Возвращает (score, что исправить)."""
    payload = (
        f"Запрос ребёнка:\n{request[:1500]}\n\n"
        f"Ограничения и её предпочтения:\n{constraints[:1500]}\n\n"
        f"Ответ бота:\n{answer[:6000]}"
    )
    try:
        resp = await asyncio.wait_for(
            ai.aio.models.generate_content(
                model=MODEL_SERVICE,
                contents=payload,
                config=types.GenerateContentConfig(
                    system_instruction=CRITIC_PROMPT,
                    temperature=0.0,
                    max_output_tokens=512,
                    response_mime_type="application/json",
                    response_schema=_CRITIC_SCHEMA,
                    thinking_config=types.ThinkingConfig(thinking_level="low"),
                ),
            ),
            timeout=CRITIC_TIMEOUT_SEC,
        )
        data = json.loads(resp.text or "{}")
        score = int(data.get("score", 5))
        fix = f"{data.get('reason', '')} {data.get('fix', '')}".strip()
        return score, fix
    except Exception as e:
        log.warning("Critic unavailable: %s", str(e)[:200])
        return 5, ""
