import os
import sys
import time
import logging
from datetime import datetime
import sqlite3

# Initialize PySpark
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, udf, explode, split, count, avg,
    from_unixtime, date_format, lower, regexp_replace, length,
)
from pyspark.sql.types import StructType, StructField, StringType, FloatType

# Add backend dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Fix #17 — import DB_PATH from the single source of truth
from db import DB_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_DIR      = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_DIR       = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Fix #4 — build stop-word set once at module level (not inside update_hot_storage)
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as _SKL_STOPS
_SOCIAL_STOPS = frozenset({
    'https', 'http', 'just', 'like', 'dont', 'people', 'know', 'think',
    'time', 'good', 'make', 'want', 'really', 'would', 'right', 'going',
    'even', 'much', 'about', 'some', 'what', 'they', 'when', 'how',
})
STOP_WORDS = list(_SKL_STOPS | _SOCIAL_STOPS)


def ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)


# ── Sentiment UDF ─────────────────────────────────────────────────────────────
# TextBlob runs in microseconds; BERT is reserved for the single-post /api/analyze endpoint
def _predict_sentiment(text):
    if not text:
        return {"label": "neutral", "score": 0.0}
    try:
        from textblob import TextBlob
        polarity = TextBlob(text).sentiment.polarity  # -1 to +1
        label = "positive" if polarity > 0.1 else ("negative" if polarity < -0.1 else "neutral")
        return {"label": label, "score": float(polarity)}
    except Exception:
        return {"label": "neutral", "score": 0.0}


sentiment_schema = StructType([
    StructField("label", StringType(), False),
    StructField("score", FloatType(), False),
])
sentiment_udf = udf(_predict_sentiment, sentiment_schema)


# ── Hot Storage Update ────────────────────────────────────────────────────────
def update_hot_storage(df):
    """
    Pushes processed Spark DataFrame summary metrics into SQLite Hot Storage.

    Fixes applied:
    #1  — cumulative sentiment percentages (back-calculate from previous row)
    #3  — single df.count() after cache() to avoid double full scan
    #5  — avg_score UPSERT uses weighted running average, not overwrite
    """
    # Fix #3 — cache before counting; reuse the cached count
    df.cache()
    batch_total = df.count()
    if batch_total == 0:
        df.unpersist()
        return

    logging.info(f"Updating Hot Storage (SQLite) with {batch_total} posts...")

    # ── Sentiment distribution for this batch ──────────────────────────────
    dist = df.groupBy("sentiment.label").count().collect()
    batch_pos = next((r["count"] for r in dist if r["label"] == "positive"), 0)
    batch_neg = next((r["count"] for r in dist if r["label"] == "negative"), 0)
    batch_neu = next((r["count"] for r in dist if r["label"] == "neutral"),  0)

    # ── Minutely trends in IST (UTC+5:30) ─────────────────────────────────
    IST_OFFSET = 19800  # seconds
    df_trends = df.withColumn(
        "minute",
        date_format(from_unixtime(col("timestamp") + IST_OFFSET), "yyyy-MM-dd HH:mm"),
    )
    hourly_trends = (
        df_trends.groupBy("minute")
        .agg(avg("sentiment.score").alias("avg_score"), count("id").alias("count"))
        .withColumnRenamed("minute", "hour")
        .collect()
    )

    # ── Recent posts (top 50) ──────────────────────────────────────────────
    recent_posts = df.orderBy(col("timestamp").desc()).limit(50).collect()

    # ── MapReduce: topic word frequencies ─────────────────────────────────
    words_df = df.withColumn(
        "word",
        explode(split(lower(regexp_replace(col("text"), r"[^a-zA-Z\s]", "")), r"\s+")),
    )
    words_df = words_df.filter(
        (length(col("word")) > 4) & (~col("word").isin(STOP_WORDS))   # Fix #4
    )
    topics = (
        words_df.groupBy("word", "sentiment.label")
        .count()
        .orderBy(col("count").desc())
        .limit(100)
        .collect()
    )

    # ── Write to SQLite ───────────────────────────────────────────────────
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Fix #1 — cumulative sentiment percentages
    # Back-calculate the previous absolute counts so percentages reflect ALL-TIME data
    prev = c.execute(
        "SELECT total_count, positive_pct, negative_pct, neutral_pct "
        "FROM sentiment_stats ORDER BY id DESC LIMIT 1"
    ).fetchone()

    if prev:
        prev_total, prev_pos_pct, prev_neg_pct, prev_neu_pct = prev
        prev_pos = (prev_pos_pct / 100) * prev_total
        prev_neg = (prev_neg_pct / 100) * prev_total
        prev_neu = (prev_neu_pct / 100) * prev_total
        running_total = prev_total + batch_total
        running_pos   = prev_pos + batch_pos
        running_neg   = prev_neg + batch_neg
        running_neu   = prev_neu + batch_neu
    else:
        running_total = batch_total
        running_pos   = batch_pos
        running_neg   = batch_neg
        running_neu   = batch_neu

    pos_pct = (running_pos / running_total) * 100 if running_total > 0 else 0
    neg_pct = (running_neg / running_total) * 100 if running_total > 0 else 0
    neu_pct = (running_neu / running_total) * 100 if running_total > 0 else 0

    c.execute("DELETE FROM sentiment_stats")
    c.execute(
        "INSERT INTO sentiment_stats (total_count, positive_pct, negative_pct, neutral_pct, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (running_total, pos_pct, neg_pct, neu_pct, datetime.utcnow().isoformat()),
    )

    # Fix #5 — weighted running average for avg_score on conflict
    for row in hourly_trends:
        c.execute(
            """
            INSERT INTO hourly_trends (time, avg_score, count)
            VALUES (?, ?, ?)
            ON CONFLICT(time) DO UPDATE SET
                avg_score = (avg_score * count + excluded.avg_score * excluded.count)
                            / (count + excluded.count),
                count     = count + excluded.count
            """,
            (row["hour"], row["avg_score"], row["count"]),
        )

    # Refresh recent posts
    c.execute("DELETE FROM posts_recent")
    for post in recent_posts:
        c.execute(
            "INSERT INTO posts_recent (id, source, timestamp, text, label, score, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                post["id"], post["source"], post["timestamp"], post["text"],
                post["sentiment"]["label"], post["sentiment"]["score"], post["created_at"],
            ),
        )

    # Refresh trending topics
    c.execute("DELETE FROM trending_topics")
    for t in topics:
        c.execute(
            "INSERT OR IGNORE INTO trending_topics (word, count, sentiment_association) "
            "VALUES (?, ?, ?)",
            (t["word"], t["count"], t["label"]),
        )

    conn.commit()
    conn.close()
    df.unpersist()
    logging.info("Hot Storage updated successfully.")


# ── Batch Runner ──────────────────────────────────────────────────────────────
def run_batch(spark):
    """
    Reads JSONL from the data lake, applies the sentiment UDF,
    writes cold storage (Parquet) and refreshes SQLite hot storage.
    """
    logging.info("Starting Spark Micro-batch...")
    raw_files = [
        os.path.join(RAW_DIR, f)
        for f in os.listdir(RAW_DIR)
        if f.endswith(".jsonl")
    ]
    if not raw_files:
        logging.info("No raw data to process.")
        return

    try:
        df = spark.read.json(raw_files)
        df_processed = df.withColumn("sentiment", sentiment_udf(col("text")))

        # Cold Storage — partitioned Parquet for historical queries
        df_processed = df_processed.withColumn(
            "date", date_format(from_unixtime(col("timestamp")), "yyyy-MM-dd")
        )
        df_processed.write.mode("append").partitionBy("date").parquet(PROCESSED_DIR)

        update_hot_storage(df_processed)

        # Archive raw files so they are not double-processed
        archive_dir = os.path.join(DATA_DIR, "lake_archive")
        os.makedirs(archive_dir, exist_ok=True)
        for f in raw_files:
            os.rename(f, os.path.join(archive_dir, os.path.basename(f)))

    except Exception as e:
        logging.error(f"Error processing batch: {e}")


if __name__ == "__main__":
    ensure_dirs()

    spark = (
        SparkSession.builder
        .appName("Panchayat_Analytics")
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    logging.info("Spark Worker started. Running continuously every 60 s...")
    while True:
        run_batch(spark)
        time.sleep(60)
