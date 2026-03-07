from datetime import datetime, timezone, timedelta
from motor.motor_asyncio import AsyncIOMotorClient

from config import MONGODB_URI, MONGODB_DB_NAME, WINDOW_SIZE, SUMMARIZE_THRESHOLD, MSK

_client: AsyncIOMotorClient | None = None


def _db():
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(MONGODB_URI)
    return _client[MONGODB_DB_NAME]


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


# ── complaints ──────────────────────────────────────────────────────

async def save_complaint(user_id: int, user_msg: str, bot_msg: str):
    await _db()["complaints"].insert_one({
        "user_id": user_id,
        "user_message": user_msg,
        "bot_message": bot_msg,
        "ts": datetime.now(timezone.utc),
    })


async def get_today_complaints() -> list[dict]:
    now = datetime.now(MSK)
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start_utc = start_of_day - timedelta(hours=3)
    col = _db()["complaints"]
    cursor = col.find(
        {"ts": {"$gte": start_utc}},
        {"_id": 0, "user_message": 1, "bot_message": 1, "ts": 1},
    ).sort("ts", 1)
    return await cursor.to_list(length=200)


async def clear_today_complaints():
    now = datetime.now(MSK)
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start_utc = start_of_day - timedelta(hours=3)
    await _db()["complaints"].delete_many({"ts": {"$gte": start_utc}})


# ── full reset (admin) ──────────────────────────────────────────────

async def delete_user_data(user_id: int):
    db = _db()
    await db["messages"].delete_many({"user_id": user_id})
    await db["summaries"].delete_many({"user_id": user_id})
    await db["complaints"].delete_many({"user_id": user_id})
