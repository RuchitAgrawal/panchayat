import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "panchayat.db")


def get_db_connection():
    """
    Return a SQLite connection with:
    - WAL journal mode  → concurrent reads while Spark is writing
    - NORMAL sync       → safe + faster than FULL
    - Row factory       → rows behave like dicts
    """
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    c = conn.cursor()

    # 1. posts_recent (Fast hot cache for the UI)
    c.execute('''
        CREATE TABLE IF NOT EXISTS posts_recent (
            id TEXT PRIMARY KEY,
            source TEXT,
            timestamp REAL,
            text TEXT,
            label TEXT,
            score REAL,
            created_at TEXT
        )
    ''')

    # 2. sentiment_stats (Distribution of sentiments — cumulative)
    c.execute('''
        CREATE TABLE IF NOT EXISTS sentiment_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            total_count INTEGER,
            positive_pct REAL,
            negative_pct REAL,
            neutral_pct REAL,
            updated_at TEXT
        )
    ''')

    # 3. hourly_trends (Time-series data for the chart)
    c.execute('''
        CREATE TABLE IF NOT EXISTS hourly_trends (
            time TEXT PRIMARY KEY,
            avg_score REAL,
            count INTEGER
        )
    ''')

    # 4. topics (Extracted MapReduce trending words)
    c.execute('''
        CREATE TABLE IF NOT EXISTS trending_topics (
            word TEXT PRIMARY KEY,
            count INTEGER,
            sentiment_association TEXT
        )
    ''')

    # ── Indexes (Fix #14) ─────────────────────────────────────────────────
    # Speed up the most common query patterns for the API
    c.execute('''
        CREATE INDEX IF NOT EXISTS idx_posts_timestamp
        ON posts_recent(timestamp DESC)
    ''')
    c.execute('''
        CREATE INDEX IF NOT EXISTS idx_hourly_trends_time
        ON hourly_trends(time ASC)
    ''')

    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")
