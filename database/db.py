from datetime import date
import os
from typing import Any

import psycopg2
from psycopg2.extras import Json, RealDictCursor


def get_database_url() -> str:
    return os.getenv("DATABASE_URL", "")


def _connect():
    database_url = get_database_url()
    if not database_url:
        return None
    try:
        return psycopg2.connect(database_url)
    except Exception:
        return None


def is_database_configured() -> bool:
    connection = _connect()
    if connection is None:
        return False
    connection.close()
    return True


def init_db() -> bool:
    connection = _connect()
    if connection is None:
        return False

    try:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        user_id TEXT PRIMARY KEY,
                        name TEXT,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS daily_logs (
                        id BIGSERIAL PRIMARY KEY,
                        user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                        log_date DATE NOT NULL DEFAULT CURRENT_DATE,
                        sleep_hours DOUBLE PRECISION,
                        steps INTEGER,
                        exercise_minutes INTEGER,
                        mood TEXT,
                        calories DOUBLE PRECISION,
                        screen_time DOUBLE PRECISION,
                        diet_notes TEXT,
                        water_liters DOUBLE PRECISION,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        UNIQUE(user_id, log_date)
                    );
                    """
                )
                # Keep schema backward-compatible with previous versions.
                cursor.execute("ALTER TABLE daily_logs ADD COLUMN IF NOT EXISTS exercise_minutes INTEGER;")
                cursor.execute("ALTER TABLE daily_logs ADD COLUMN IF NOT EXISTS mood TEXT;")
                cursor.execute("ALTER TABLE daily_logs ADD COLUMN IF NOT EXISTS calories DOUBLE PRECISION;")
                cursor.execute("ALTER TABLE daily_logs ADD COLUMN IF NOT EXISTS screen_time DOUBLE PRECISION;")
                cursor.execute("ALTER TABLE daily_logs ADD COLUMN IF NOT EXISTS diet_notes TEXT;")
                cursor.execute("ALTER TABLE daily_logs ADD COLUMN IF NOT EXISTS water_liters DOUBLE PRECISION;")
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS habits (
                        id BIGSERIAL PRIMARY KEY,
                        user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                        habit_key TEXT NOT NULL,
                        habit_value TEXT NOT NULL,
                        confidence DOUBLE PRECISION NOT NULL DEFAULT 0.5,
                        source TEXT NOT NULL DEFAULT 'inference',
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        UNIQUE(user_id, habit_key)
                    );
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS recommendations (
                        id BIGSERIAL PRIMARY KEY,
                        user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                        recommendation TEXT NOT NULL,
                        source_agent TEXT NOT NULL,
                        context JSONB NOT NULL DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )
        return True
    finally:
        connection.close()


def upsert_user(user_id: str, name: str | None = None) -> bool:
    connection = _connect()
    if connection is None:
        return False

    try:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO users (user_id, name)
                    VALUES (%s, %s)
                    ON CONFLICT (user_id)
                    DO UPDATE SET
                        name = COALESCE(EXCLUDED.name, users.name),
                        updated_at = NOW();
                    """,
                    (user_id, name),
                )
        return True
    finally:
        connection.close()


def upsert_habit(
    user_id: str,
    habit_key: str,
    habit_value: str,
    confidence: float = 0.5,
    source: str = "inference",
) -> dict[str, Any] | None:
    connection = _connect()
    if connection is None:
        return None

    upsert_user(user_id)
    try:
        with connection:
            with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    INSERT INTO habits (user_id, habit_key, habit_value, confidence, source)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (user_id, habit_key)
                    DO UPDATE SET
                        habit_value = EXCLUDED.habit_value,
                        confidence = EXCLUDED.confidence,
                        source = EXCLUDED.source,
                        updated_at = NOW()
                    RETURNING *;
                    """,
                    (user_id, habit_key, habit_value, confidence, source),
                )
                row = cursor.fetchone()
                return dict(row) if row else None
    finally:
        connection.close()


def get_user_habits(user_id: str, limit: int = 10) -> list[dict[str, Any]]:
    connection = _connect()
    if connection is None:
        return []

    try:
        with connection:
            with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT *
                    FROM habits
                    WHERE user_id = %s
                    ORDER BY updated_at DESC
                    LIMIT %s;
                    """,
                    (user_id, limit),
                )
                return [dict(row) for row in cursor.fetchall()]
    finally:
        connection.close()


def _infer_habit_updates(log_data: dict[str, Any]) -> list[tuple[str, str, float]]:
    updates: list[tuple[str, str, float]] = []
    sleep_hours = log_data.get("sleep_hours")
    steps = log_data.get("steps")
    exercise_minutes = log_data.get("exercise_minutes")
    mood = str(log_data.get("mood", "") or "").lower()
    calories = log_data.get("calories")
    screen_time = log_data.get("screen_time")

    if sleep_hours is not None:
        if float(sleep_hours) < 6:
            updates.append(("sleep_pattern", "short_sleep", 0.85))
        elif float(sleep_hours) >= 7:
            updates.append(("sleep_pattern", "healthy_sleep", 0.8))

    if steps is not None:
        if int(steps) < 5000:
            updates.append(("activity_pattern", "low_steps", 0.8))
        elif int(steps) >= 8000:
            updates.append(("activity_pattern", "active", 0.8))

    if exercise_minutes is not None and int(exercise_minutes) >= 30:
        updates.append(("exercise_pattern", "consistent_exercise", 0.75))

    if mood in {"stressed", "anxious", "sad", "low"}:
        updates.append(("mood_pattern", "low_mood", 0.8))
    elif mood in {"good", "great", "calm", "happy"}:
        updates.append(("mood_pattern", "stable_mood", 0.7))

    if calories is not None:
        if float(calories) < 1600:
            updates.append(("calorie_pattern", "low_calorie_intake", 0.65))
        elif float(calories) > 2800:
            updates.append(("calorie_pattern", "high_calorie_intake", 0.65))
        else:
            updates.append(("calorie_pattern", "balanced_calories", 0.6))

    if screen_time is not None:
        if float(screen_time) > 6:
            updates.append(("screen_time_pattern", "high_screen_time", 0.75))
        elif float(screen_time) < 3:
            updates.append(("screen_time_pattern", "controlled_screen_time", 0.65))

    return updates


def update_habits_from_daily_log(user_id: str, log_data: dict[str, Any]) -> list[dict[str, Any]]:
    updates = _infer_habit_updates(log_data)
    records: list[dict[str, Any]] = []
    for habit_key, habit_value, confidence in updates:
        record = upsert_habit(
            user_id=user_id,
            habit_key=habit_key,
            habit_value=habit_value,
            confidence=confidence,
            source="daily_log",
        )
        if record:
            records.append(record)
    return records


def upsert_daily_log(log_data: dict[str, Any]) -> dict[str, Any] | None:
    connection = _connect()
    if connection is None:
        return None

    user_id = str(log_data["user_id"])
    upsert_user(user_id)
    log_date = log_data.get("log_date") or date.today()

    try:
        with connection:
            with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    INSERT INTO daily_logs (
                        user_id,
                        log_date,
                        sleep_hours,
                        steps,
                        exercise_minutes,
                        mood,
                        calories,
                        screen_time,
                        diet_notes,
                        water_liters
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id, log_date)
                    DO UPDATE SET
                        sleep_hours = COALESCE(EXCLUDED.sleep_hours, daily_logs.sleep_hours),
                        steps = COALESCE(EXCLUDED.steps, daily_logs.steps),
                        exercise_minutes = COALESCE(EXCLUDED.exercise_minutes, daily_logs.exercise_minutes),
                        mood = COALESCE(EXCLUDED.mood, daily_logs.mood),
                        calories = COALESCE(EXCLUDED.calories, daily_logs.calories),
                        screen_time = COALESCE(EXCLUDED.screen_time, daily_logs.screen_time),
                        diet_notes = COALESCE(EXCLUDED.diet_notes, daily_logs.diet_notes),
                        water_liters = COALESCE(EXCLUDED.water_liters, daily_logs.water_liters)
                    RETURNING *;
                    """,
                    (
                        user_id,
                        log_date,
                        log_data.get("sleep_hours"),
                        log_data.get("steps"),
                        log_data.get("exercise_minutes"),
                        log_data.get("mood"),
                        log_data.get("calories"),
                        log_data.get("screen_time"),
                        log_data.get("diet_notes"),
                        log_data.get("water_liters"),
                    ),
                )
                row = cursor.fetchone()
                if row:
                    update_habits_from_daily_log(user_id=user_id, log_data=log_data)
                return dict(row) if row else None
    finally:
        connection.close()


def insert_recommendation(
    user_id: str,
    recommendation: str,
    source_agent: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    connection = _connect()
    if connection is None:
        return None

    upsert_user(user_id)
    try:
        with connection:
            with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    INSERT INTO recommendations (user_id, recommendation, source_agent, context)
                    VALUES (%s, %s, %s, %s)
                    RETURNING *;
                    """,
                    (user_id, recommendation, source_agent, Json(context or {})),
                )
                row = cursor.fetchone()
                return dict(row) if row else None
    finally:
        connection.close()


def get_recent_daily_logs(user_id: str, limit: int = 7) -> list[dict[str, Any]]:
    connection = _connect()
    if connection is None:
        return []

    try:
        with connection:
            with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT *
                    FROM daily_logs
                    WHERE user_id = %s
                    ORDER BY log_date DESC
                    LIMIT %s;
                    """,
                    (user_id, limit),
                )
                return [dict(row) for row in cursor.fetchall()]
    finally:
        connection.close()


def get_recent_recommendations(user_id: str, limit: int = 10) -> list[dict[str, Any]]:
    connection = _connect()
    if connection is None:
        return []

    try:
        with connection:
            with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT *
                    FROM recommendations
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s;
                    """,
                    (user_id, limit),
                )
                return [dict(row) for row in cursor.fetchall()]
    finally:
        connection.close()
