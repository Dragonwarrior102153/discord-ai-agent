"""Persistent, per-channel conversation history backed by SQLite."""

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATABASE_PATH = Path(__file__).resolve().parent.parent / "data" / "memory.db"
DEFAULT_HISTORY_LIMIT = 20
MAX_STORED_MESSAGES = 20
MAX_PINNED_MESSAGES = 3
MAX_PINNED_CONTENT_LENGTH = 500
MEMORY_RETENTION_DAYS = 7


@dataclass(frozen=True)
class ConversationMessage:
    """One saved message in a channel conversation."""

    id: int
    channel_id: int
    role: str
    content: str
    timestamp: str


@dataclass(frozen=True)
class PinnedMessage:
    """One explicitly pinned context entry for a channel."""

    id: int
    channel_id: int
    content: str
    timestamp: str


def initialize_database() -> None:
    """Create the database directory, message table, and channel lookup index."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(DATABASE_PATH, timeout=5)) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel_id INTEGER NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'model')),
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS messages_channel_id_id
                ON messages (channel_id, id)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS channel_activity (
                    channel_id INTEGER PRIMARY KEY,
                    last_activity TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS pinned_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel_id INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO channel_activity (channel_id, last_activity)
                SELECT channel_id, MAX(timestamp)
                FROM messages
                GROUP BY channel_id
                """
            )
            connection.execute(
                """
                DELETE FROM messages
                WHERE id NOT IN (
                    SELECT id
                    FROM messages AS recent
                    WHERE recent.channel_id = messages.channel_id
                    ORDER BY recent.id DESC
                    LIMIT ?
                )
                """,
                (MAX_STORED_MESSAGES,),
            )


def _expire_channel_history(
    connection: sqlite3.Connection, channel_id: int, cutoff: str
) -> None:
    expired = connection.execute(
        """
        SELECT 1
        FROM channel_activity
        WHERE channel_id = ? AND last_activity < ?
        """,
        (channel_id, cutoff),
    ).fetchone()
    if expired:
        connection.execute("DELETE FROM messages WHERE channel_id = ?", (channel_id,))
        connection.execute(
            "DELETE FROM pinned_messages WHERE channel_id = ?",
            (channel_id,),
        )
        connection.execute(
            "DELETE FROM channel_activity WHERE channel_id = ?",
            (channel_id,),
        )


def save_message(channel_id: int, role: str, content: str) -> int:
    """Persist a message, refresh user activity, and enforce the storage cap."""
    if role not in {"user", "model"}:
        raise ValueError("role must be 'user' or 'model'")

    initialize_database()
    with closing(sqlite3.connect(DATABASE_PATH, timeout=5)) as connection:
        with connection:
            if role == "user":
                _expire_channel_history(
                    connection, channel_id, _retention_cutoff()
                )

            cursor = connection.execute(
                """
                INSERT INTO messages (channel_id, role, content)
                VALUES (?, ?, ?)
                """,
                (channel_id, role, content),
            )
            if role == "user":
                connection.execute(
                    """
                    INSERT INTO channel_activity (channel_id, last_activity)
                    VALUES (?, CURRENT_TIMESTAMP)
                    ON CONFLICT(channel_id) DO UPDATE SET
                        last_activity = CURRENT_TIMESTAMP
                    """,
                    (channel_id,),
                )
            connection.execute(
                """
                DELETE FROM messages
                WHERE channel_id = ?
                  AND id NOT IN (
                      SELECT id
                      FROM messages
                      WHERE channel_id = ?
                      ORDER BY id DESC
                      LIMIT ?
                  )
                """,
                (channel_id, channel_id, MAX_STORED_MESSAGES),
            )
            return int(cursor.lastrowid)


def get_history(
    channel_id: int, limit: int = DEFAULT_HISTORY_LIMIT
) -> list[ConversationMessage]:
    """Return up to the newest ``limit`` channel messages, oldest first."""
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")

    initialize_database()
    with closing(sqlite3.connect(DATABASE_PATH, timeout=5)) as connection:
        with connection:
            _expire_channel_history(connection, channel_id, _retention_cutoff())
            rows = connection.execute(
                """
                SELECT id, channel_id, role, content, timestamp
                FROM (
                    SELECT id, channel_id, role, content, timestamp
                    FROM messages
                    WHERE channel_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id ASC
                """,
                (channel_id, limit),
            ).fetchall()

    return [
        ConversationMessage(
            id=row[0],
            channel_id=row[1],
            role=row[2],
            content=row[3],
            timestamp=row[4],
        )
        for row in rows
    ]


def clear_history(channel_id: int) -> int:
    """Delete saved messages and activity for one channel only."""
    initialize_database()
    with closing(sqlite3.connect(DATABASE_PATH, timeout=5)) as connection:
        with connection:
            cursor = connection.execute(
                "DELETE FROM messages WHERE channel_id = ?",
                (channel_id,),
            )
            connection.execute(
                "DELETE FROM pinned_messages WHERE channel_id = ?",
                (channel_id,),
            )
            connection.execute(
                "DELETE FROM channel_activity WHERE channel_id = ?",
                (channel_id,),
            )
            return cursor.rowcount


def delete_expired_histories(
    retention_days: int = MEMORY_RETENTION_DAYS,
) -> int:
    """Delete all channel histories inactive beyond the retention window."""
    if not isinstance(retention_days, int) or retention_days < 1:
        raise ValueError("retention_days must be a positive integer")

    initialize_database()
    cutoff = _retention_cutoff(retention_days)
    with closing(sqlite3.connect(DATABASE_PATH, timeout=5)) as connection:
        with connection:
            connection.execute(
                """
                DELETE FROM messages
                WHERE channel_id IN (
                    SELECT channel_id
                    FROM channel_activity
                    WHERE last_activity < ?
                )
                """,
                (cutoff,),
            )
            connection.execute(
                """
                DELETE FROM pinned_messages
                WHERE channel_id IN (
                    SELECT channel_id
                    FROM channel_activity
                    WHERE last_activity < ?
                )
                """,
                (cutoff,),
            )
            cursor = connection.execute(
                "DELETE FROM channel_activity WHERE last_activity < ?",
                (cutoff,),
            )
            return cursor.rowcount


def pin_message(channel_id: int, content: str) -> int:
    """Store explicitly selected channel context within the configured limits."""
    content = content.strip()
    if not content:
        raise ValueError("Pinned content cannot be empty")
    if len(content) > MAX_PINNED_CONTENT_LENGTH:
        raise ValueError(
            f"Pinned content cannot exceed {MAX_PINNED_CONTENT_LENGTH} characters"
        )

    initialize_database()
    with closing(sqlite3.connect(DATABASE_PATH, timeout=5)) as connection:
        with connection:
            _expire_channel_history(connection, channel_id, _retention_cutoff())
            connection.execute(
                """
                INSERT INTO channel_activity (channel_id, last_activity)
                VALUES (?, CURRENT_TIMESTAMP)
                ON CONFLICT(channel_id) DO UPDATE SET
                    last_activity = CURRENT_TIMESTAMP
                """,
                (channel_id,),
            )
            count = connection.execute(
                "SELECT COUNT(*) FROM pinned_messages WHERE channel_id = ?",
                (channel_id,),
            ).fetchone()[0]
            if count >= MAX_PINNED_MESSAGES:
                raise ValueError(
                    f"A channel can have at most {MAX_PINNED_MESSAGES} pinned messages"
                )
            cursor = connection.execute(
                "INSERT INTO pinned_messages (channel_id, content) VALUES (?, ?)",
                (channel_id, content),
            )
            return int(cursor.lastrowid)


def get_pinned_messages(channel_id: int) -> list[PinnedMessage]:
    """Return a channel's pinned context in insertion order."""
    initialize_database()
    with closing(sqlite3.connect(DATABASE_PATH, timeout=5)) as connection:
        with connection:
            _expire_channel_history(connection, channel_id, _retention_cutoff())
            rows = connection.execute(
                """
                SELECT id, channel_id, content, timestamp
                FROM pinned_messages
                WHERE channel_id = ?
                ORDER BY id ASC
                """,
                (channel_id,),
            ).fetchall()

    return [
        PinnedMessage(
            id=row[0],
            channel_id=row[1],
            content=row[2],
            timestamp=row[3],
        )
        for row in rows
    ]


def unpin_message(channel_id: int, pin_id: int) -> bool:
    """Remove a pin only if it belongs to the supplied channel."""
    initialize_database()
    with closing(sqlite3.connect(DATABASE_PATH, timeout=5)) as connection:
        with connection:
            _expire_channel_history(connection, channel_id, _retention_cutoff())
            cursor = connection.execute(
                "DELETE FROM pinned_messages WHERE channel_id = ? AND id = ?",
                (channel_id, pin_id),
            )
            return cursor.rowcount > 0


def _retention_cutoff(retention_days: int = MEMORY_RETENTION_DAYS) -> str:
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    return cutoff.strftime("%Y-%m-%d %H:%M:%S")


def format_history_for_prompt(
    history: list[ConversationMessage],
    pinned_messages: list[PinnedMessage] | None = None,
) -> str:
    """Format pinned context and saved turns for Gemini."""
    pinned_context = ""
    if pinned_messages:
        pins = "\n".join(
            f"- {message.content}" for message in pinned_messages
        )
        pinned_context = (
            "User-pinned context for this channel (use only when relevant):\n"
            f"{pins}\n\n"
        )

    transcript = "\n".join(
        f"{'User' if message.role == 'user' else 'Gemini'}: {message.content}"
        for message in history
    )
    return (
        f"{pinned_context}"
        "Previous conversation history for this Discord channel, "
        "oldest message first. The latest message may be the current request:\n\n"
        f"{transcript}"
    )
