# Panchayat - Real-Time Sentiment Analysis Dashboard

A full-stack, distributed sentiment analysis platform that consumes the live **Bluesky Jetstream firehose** and processes it in real-time using distributed PySpark, storing results in a hot-storage SQLite database, and serving trends via a FastAPI backend to a modern React UI.

## 🔄 Project Evolution

**Before:** The project initially relied on static NLP datasets and a slower BERT-based batch inference system without real-time streaming capabilities.

**Updates & Improvements:** This major update transformed the architecture into a **Scalable, Real-Time Big Data Pipeline**:
- **Live Data Source:** Replaced static datasets with the Bluesky Jetstream firehose for continuous, real-time data ingestion.
- **Enhanced Performance:** Transitioned away from slow batch processing to high-performance TextBlob and PySpark micro-batch analytics, eliminating data staleness.
- **Distributed Architecture:** Integrated Dockerized Apache PySpark for robust and scalable multi-threaded event processing.
- **Hot-Storage Database:** Implemented WAL-mode SQLite for efficient concurrency handling and rapid read/write of live streams.
- **Premium Dashboard:** Completely revamped the React frontend with a professional, glassmorphic dark-theme UI that actively updates with live trends and metrics.
## ✨ Features

- **🌐 Live Social Firehose** - Consumes real-time events from Bluesky's Jetstream API.
- **⚡ Distributed Data Processing** - Utilizes **PySpark** structured streaming and micro-batch analytics to handle infinite data effectively.
- **🤖 High-Performance NLP** - Uses TextBlob and custom logic to compute accurate sentiment over thousands of events per minute.
- **📊 Real-time Dashboard** - Clean React UI with a dark glassmorphism theme, updating live every 10 seconds.
- **🐳 Dockerized Infrastructure** - 100% containerized deployment with `docker-compose`.
- **🛡️ Secure API** - Rate limited using `slowapi` to protect against spam.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
Ensure you have Docker and Docker Compose installed on your system.
You no longer need to worry about local Python/Node dependencies unless you intend to develop locally without Docker.

### 2. Clone the Repository
```bash
git clone https://github.com/RuchitAgrawal/panchayat.git
cd panchayat
```

### 3. Launch Services with Docker
To bring up the FastAPI backend, PySpark Analytics Worker, and Bluesky Producer simultaneously:

```bash
docker-compose up --build -d
```
*Note: This will download PyTorch and Spark, which may take some time during the first build.*

### 4. Run the React Frontend
Open a separate terminal and start the Vite development server:
```bash
cd frontend
npm install
npm run dev
```
*Frontend normally runs at: http://localhost:5173*
*Dashboard immediately populates as soon as Spark processes the first micro-batch (allow 10-30 seconds).*

---

## 🏗️ Architecture

The backend consists of three standalone Docker services:
1. **Producer (`panchayat_producer`)**: Connects to the Bluesky WSS Jetstream and pulls raw activity events, cleaning and appending them into localized `.jsonl` ingestion files.
2. **Spark Worker (`panchayat_spark`)**: Runs a continuous micro-batch loop using Apache PySpark to load JSONL partitions, run sentiment inference across threads, and aggregate metrics into the hot storage.
3. **API Server (`panchayat_api`)**: A standalone FastAPI instance rate-limited efficiently that serves REST endpoints. Connects to the synced SQLite hot-storage (`panchayat.db`) locally mounted via Docker volumes.

## 🔌 Core API Endpoints

| Endpoint           | Method | Description                                      |
|--------------------|--------|--------------------------------------------------|
| `/api/posts/stats` | GET    | Fetch total aggregated post counts and metrics   |
| `/api/trends`      | GET    | Fetch bucketed historical sentiment shifts       |
| `/api/posts`       | GET    | Fetch the most recent analyzed posts             |
| `/api/analyze`     | POST   | Manually analyze a custom string (Rate Limited)  |

## 🛠️ Tech Stack

**Pipeline & Backend:**
- Apache PySpark (Micro-Batch processing)
- Python (Producer + Data cleaning)
- FastAPI (REST endpoints)
- TextBlob (Sentiment mapping)

**Frontend:**
- React 18, Vite
- Recharts (Time-series data visualization)
- Vanilla CSS + Custom Layouts

## 📝 License

MIT License.
