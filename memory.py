from datetime import datetime, timezone
from motor.motor_asyncio import AsyncIOMotorClient

from config import MONGODB_URI, MONGODB_DB_NAME, WINDOW_SIZE, MSK

_client: AsyncIOMotorClient | None = None
_indexes_ready = False


def _db():
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(MONGODB_URI)
    return _client[MONGODB_DB_NAME]


async def ensure_indexes():
    """Create indexes once at startup — history queries hit these on every message."""
    global _indexes_ready
    if _indexes_ready:
        return
    db = _db()
    await db["messages"].create_index([("user_id", 1), ("ts", -1)])
    await db["complaints"].create_index([("ts", 1)])
    await db["profiles"].create_index("user_id", unique=True)
    await db["ideas"].create_index([("user_id", 1), ("ts", -1)])
    _indexes_ready = True


# ── messages ────────────────────────────────────────────────────────

async def save_message(user_id: int, role: str, text: str) -> int:
    """Save a message and return total message count for this user."""
    col = _db()["messages"]
    await col.insert_one({
        "user_id": user_id,
        "role": role,
        "text": text,
        "ts": datetime.now(timezone.utc),
    })
    return await col.count_documents({"user_id": user_id})


async def get_recent_messages(user_id: int, limit: int = WINDOW_SIZE) -> list[dict]:
    """Return last `limit` messages sorted oldest-first."""
    col = _db()["messages"]
    cursor = col.find(
        {"user_id": user_id},
        {"_id": 0, "role": 1, "text": 1},
    ).sort("ts", -1).limit(limit)
    msgs = await cursor.to_list(length=limit)
    msgs.reverse()
    return msgs


async def pop_oldest_messages(user_id: int, count: int) -> list[dict]:
    """Remove and return the oldest `count` messages for summarization."""
    col = _db()["messages"]
    cursor = col.find({"user_id": user_id}).sort("ts", 1).limit(count)
    oldest = await cursor.to_list(length=count)
    if oldest:
        ids = [m["_id"] for m in oldest]
        await col.delete_many({"_id": {"$in": ids}})
    return [{"role": m["role"], "text": m["text"]} for m in oldest]


async def message_count(user_id: int) -> int:
    return await _db()["messages"].count_documents({"user_id": user_id})


# ── summaries ───────────────────────────────────────────────────────

async def get_summary(user_id: int) -> str | None:
    doc = await _db()["summaries"].find_one({"user_id": user_id})
    return doc["summary"] if doc else None


async def upsert_summary(user_id: int, summary: str):
    await _db()["summaries"].update_one(
        {"user_id": user_id},
        {"$set": {"summary": summary, "updated_at": datetime.now(timezone.utc)}},
        upsert=True,
    )


# ── profile (likes / dislikes / projects) ───────────────────────────

EMPTY_PROFILE = {
    "likes": [],
    "dislikes": [],
    "projects": [],
    "people": [],
    "style": [],
}


async def get_profile(user_id: int) -> dict:
    doc = await _db()["profiles"].find_one({"user_id": user_id}, {"_id": 0, "user_id": 0})
    if not doc:
        return dict(EMPTY_PROFILE)
    profile = dict(EMPTY_PROFILE)
    for key in EMPTY_PROFILE:
        val = doc.get(key)
        if isinstance(val, list):
            profile[key] = [str(v) for v in val if v]
    return profile


async def upsert_profile(user_id: int, profile: dict):
    clean = {k: [str(v) for v in profile.get(k, []) if v][:8] for k in EMPTY_PROFILE}
    clean["updated_at"] = datetime.now(timezone.utc)
    await _db()["profiles"].update_one(
        {"user_id": user_id}, {"$set": clean}, upsert=True,
    )


# ── idea ledger (anti-repeat) ───────────────────────────────────────

async def save_ideas(user_id: int, titles: list[str]):
    if not titles:
        return
    now = datetime.now(timezone.utc)
    await _db()["ideas"].insert_many([
        {"user_id": user_id, "title": t.strip()[:120], "ts": now}
        for t in titles if t.strip()
    ])


async def get_recent_ideas(user_id: int, limit: int = 40) -> list[str]:
    cursor = _db()["ideas"].find(
        {"user_id": user_id}, {"_id": 0, "title": 1},
    ).sort("ts", -1).limit(limit)
    docs = await cursor.to_list(length=limit)
    seen, out = set(), []
    for d in docs:
        t = d.get("title", "").strip()
        low = t.lower()
        if t and low not in seen:
            seen.add(low)
            out.append(t)
    return out


# ── complaints ──────────────────────────────────────────────────────

async def save_complaint(user_id: int, user_msg: str, bot_msg: str):
    await _db()["complaints"].insert_one({
        "user_id": user_id,
        "user_message": user_msg,
        "bot_message": bot_msg,
        "ts": datetime.now(timezone.utc),
    })


def _start_of_day_utc() -> datetime:
    now = datetime.now(MSK)
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return start_of_day.astimezone(timezone.utc)


async def get_today_complaints() -> list[dict]:
    col = _db()["complaints"]
    cursor = col.find(
        {"ts": {"$gte": _start_of_day_utc()}},
        {"_id": 0, "user_message": 1, "bot_message": 1, "ts": 1},
    ).sort("ts", 1)
    return await cursor.to_list(length=200)


async def clear_today_complaints():
    await _db()["complaints"].delete_many({"ts": {"$gte": _start_of_day_utc()}})


# ── full reset (admin) ──────────────────────────────────────────────

async def delete_user_data(user_id: int):
    db = _db()
    await db["messages"].delete_many({"user_id": user_id})
    await db["summaries"].delete_many({"user_id": user_id})
    await db["complaints"].delete_many({"user_id": user_id})
    await db["profiles"].delete_many({"user_id": user_id})
    await db["ideas"].delete_many({"user_id": user_id})
