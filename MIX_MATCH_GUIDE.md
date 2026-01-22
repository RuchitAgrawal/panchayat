# Mix-Match Guide: GitHub Repos to Use

This guide tells you exactly which files to copy from which repos.

---

## 🧠 Phase 2: ML Pipeline (BERT + LSTM + RF)

### Option A: Use HuggingFace Pre-trained (Recommended)
No repo cloning needed! Just use this code directly:

```python
# backend/models/bert_sentiment.py
from transformers import pipeline

class BertSentiment:
    def __init__(self):
        # This model gives 1-5 star ratings
        self.classifier = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        )
    
    def predict(self, text: str) -> dict:
        result = self.classifier(text[:512])[0]  # BERT max 512 tokens
        # Convert 1-5 stars to sentiment
        score = int(result['label'].split()[0])
        return {
            "label": "positive" if score >= 4 else "negative" if score <= 2 else "neutral",
            "confidence": result['score'],
            "raw_score": score
        }
```

### Option B: Reference Repos for Custom Training
If you want to understand how models are trained:

| Model | GitHub Repo | File to Look At |
|-------|-------------|-----------------|
| BERT | `huggingface/transformers` | `examples/pytorch/text-classification/` |
| LSTM | `bentrevett/pytorch-sentiment-analysis` | Notebook 2 (LSTM) |
| RF | `scikit-learn/scikit-learn` | `examples/text/document_classification.py` |

---

## 📊 Phase 3: Topic Modeling (BERTopic)

### Use BERTopic Directly
```python
# backend/nlp/topic_modeling.py
from bertopic import BERTopic

class TopicModeler:
    def __init__(self):
        self.model = BERTopic(language="english", verbose=True)
    
    def extract_topics(self, documents: list[str]):
        topics, probs = self.model.fit_transform(documents)
        topic_info = self.model.get_topic_info()
        return topic_info.to_dict('records')
```

### Reference for LDA (Alternative)
| What | GitHub Repo | File |
|------|-------------|------|
| LDA | `RaRe-Technologies/gensim` | `docs/notebooks/lda_training.ipynb` |

---

## 🌐 Phase 4: Reddit Data (PRAW)

### Use PRAW Directly
```python
# backend/data/reddit_client.py
import praw
from config import settings

class RedditClient:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=settings.reddit_client_id,
            client_secret=settings.reddit_client_secret,
            user_agent=settings.reddit_user_agent
        )
    
    def get_posts(self, subreddit: str, limit: int = 100):
        sub = self.reddit.subreddit(subreddit)
        posts = []
        for post in sub.hot(limit=limit):
            posts.append({
                "id": post.id,
                "title": post.title,
                "text": post.selftext,
                "score": post.score,
                "created": post.created_utc,
                "num_comments": post.num_comments
            })
        return posts
```

---

## 🎨 Phase 5: React Dashboard

### Clone TailAdmin Template
```bash
# In your frontend folder
npx degit TailAdmin/free-react-tailwind-admin-dashboard .
npm install
```

### Files to Modify
| Original File | What to Change |
|--------------|----------------|
| `src/pages/Dashboard.tsx` | Replace sales chart with sentiment chart |
| `src/components/ChartOne.tsx` | Change to sentiment trend line chart |
| `src/components/ChartTwo.tsx` | Change to topic distribution |

### Or: Build From Scratch with Vite
```bash
npm create vite@latest . -- --template react
npm install tailwindcss postcss autoprefixer recharts axios react-wordcloud
npx tailwindcss init -p
```

---

## 🔗 Quick Links

- **HuggingFace Models**: https://huggingface.co/models
- **BERTopic Docs**: https://maartengr.github.io/BERTopic/
- **PRAW Docs**: https://praw.readthedocs.io/
- **TailAdmin**: https://github.com/TailAdmin/free-react-tailwind-admin-dashboard
- **Recharts**: https://recharts.org/
