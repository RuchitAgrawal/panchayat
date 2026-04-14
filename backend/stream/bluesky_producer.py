import asyncio
import websockets
import json
import os
import random                                   # Fix #2 — moved to module top
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

JETSTREAM_URL = "wss://jetstream1.us-east.bsky.network/subscribe?wantedCollections=app.bsky.feed.post"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")

# Fix #11 — batch I/O constants
_WRITE_BUFFER: list = []
BUFFER_SIZE = 50  # flush every 50 posts (≈ one write per several seconds at 10% sample rate)

# Ensure Data Lake directory exists
os.makedirs(DATA_DIR, exist_ok=True)


def _flush_buffer(buffer: list, file_path: str) -> None:
    """Write all buffered records to the JSONL file in one shot."""
    if not buffer:
        return
    with open(file_path, "a", encoding="utf-8") as f:
        f.write("\n".join(buffer) + "\n")
    buffer.clear()


async def stream_firehose():
    """
    Connects to the Bluesky Jetstream WebSocket firehose, filters for English posts,
    and appends them directly to our Data Lake JSONL storage.

    Improvements (Fix #2, #11):
    - `import random` moved to module top (no repeated import-cache lookup per message)
    - Writes are batched in a 50-record buffer to minimise syscall overhead
    """
    global _WRITE_BUFFER
    logging.info(f"Connecting to Bluesky Jetstream Firehose: {JETSTREAM_URL}")

    while True:
        try:
            async with websockets.connect(JETSTREAM_URL) as websocket:
                logging.info("Connected successfully! Listening for posts...")

                while True:
                    message_str = await websocket.recv()
                    event = json.loads(message_str)

                    # Only process new-post commits
                    if event.get("kind") != "commit":
                        continue
                    commit = event.get("commit", {})
                    if commit.get("operation") != "create":
                        continue

                    record = commit.get("record", {})

                    # Language filter — English only
                    if "en" not in record.get("langs", []):
                        continue

                    # 10% sample rate
                    if random.random() > 0.10:
                        continue

                    text = record.get("text", "")
                    if not text:
                        continue

                    post_data = {
                        "id":         commit.get("rkey"),
                        "did":        event.get("did"),
                        "source":     "bluesky",
                        "timestamp":  event.get("time_us", 0) / 1_000_000,
                        "text":       text,
                        "created_at": record.get("createdAt"),
                    }

                    # Rotate JSONL files by day to prevent unbounded growth
                    date_str = datetime.utcnow().strftime("%Y-%m-%d")
                    file_path = os.path.join(DATA_DIR, f"bluesky_{date_str}.jsonl")

                    # Fix #11 — buffer records; only open the file every BUFFER_SIZE posts
                    _WRITE_BUFFER.append(json.dumps(post_data))
                    if len(_WRITE_BUFFER) >= BUFFER_SIZE:
                        _flush_buffer(_WRITE_BUFFER, file_path)

        except websockets.exceptions.ConnectionClosed:
            # Flush pending buffer so we don't lose data on reconnect
            if _WRITE_BUFFER:
                date_str = datetime.utcnow().strftime("%Y-%m-%d")
                _flush_buffer(_WRITE_BUFFER, os.path.join(DATA_DIR, f"bluesky_{date_str}.jsonl"))
            logging.warning("Connection closed. Reconnecting in 5 seconds...")
            await asyncio.sleep(5)
        except Exception as e:
            logging.error(f"Stream error: {e}")
            await asyncio.sleep(5)


if __name__ == "__main__":
    try:
        asyncio.run(stream_firehose())
    except KeyboardInterrupt:
        # Final flush before exit
        if _WRITE_BUFFER:
            date_str = datetime.utcnow().strftime("%Y-%m-%d")
            _flush_buffer(_WRITE_BUFFER, os.path.join(DATA_DIR, f"bluesky_{date_str}.jsonl"))
        logging.info("Stream stopped by user.")
