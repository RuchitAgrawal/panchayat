# Datasets for Panchayat Sentiment Analyzer

This folder is for storing Kaggle datasets for sentiment analysis.

## Recommended Datasets

### 1. Sentiment140 (Twitter) ⭐ Recommended
- **Download:** https://www.kaggle.com/datasets/kazanova/sentiment140
- **Size:** 1.6 million tweets (~80MB compressed)
- **File:** `training.1600000.processed.noemoticon.csv`

**Columns:**
| Index | Column | Description |
|-------|--------|-------------|
| 0 | target | 0 = negative, 4 = positive |
| 1 | id | Tweet ID |
| 2 | date | Timestamp |
| 3 | flag | Query (usually NO_QUERY) |
| 4 | user | Username |
| 5 | text | Tweet text |

**Load command:**
```bash
# Quick load 100 tweets
curl -X POST http://localhost:8000/api/csv/load \
  -H "Content-Type: application/json" \
  -d '{"filepath": "datasets/sentiment140.csv", "text_column": "text", "limit": 100}'
```

---

### 2. IMDB Movie Reviews
- **Download:** https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
- **Columns:** `review`, `sentiment`

---

## How to Use

1. Download dataset from Kaggle (requires free account)
2. Extract CSV file to this `datasets/` folder
3. Rename to simple name (e.g., `sentiment140.csv`)
4. Use the API to load:

```powershell
# PowerShell
$body = @{filepath="datasets/sentiment140.csv"; text_column="text"; limit=500} | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/api/csv/load" -Method Post -Body $body -ContentType "application/json"
```

---

## Column Mapping

| Dataset | Text Column | Has Labels |
|---------|-------------|------------|
| Sentiment140 | `text` (index 5) | Yes (target) |
| IMDB Reviews | `review` | Yes (sentiment) |
| Amazon Reviews | `reviewText` | Yes (overall) |
