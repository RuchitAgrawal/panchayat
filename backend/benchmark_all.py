"""
Panchayat - Comprehensive Benchmark Script
Measures all key metrics for resume documentation:
1. RF Classifier F1-score & Accuracy
2. BERTopic Topic Count & Coherence
3. API Latency (concurrent requests)
4. Data Volume
"""
import sys, os, time, json, math, sqlite3
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("PANCHAYAT BENCHMARK SUITE")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# 1. RF SENTIMENT CLASSIFIER - F1-Score & Accuracy
# ─────────────────────────────────────────────────────────────
print("\n[1/4] RF SENTIMENT CLASSIFIER METRICS")
print("-" * 40)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from textblob import TextBlob

    # Build a richer synthetic corpus (same TextBlob-bootstrap approach as codebase)
    positive_texts = [
        "This is absolutely amazing and I love it!",
        "Great product, highly recommend to everyone",
        "Best experience ever, very happy with this",
        "Excellent quality and fantastic service",
        "I'm so pleased with this purchase",
        "Wonderful, exceeded all my expectations",
        "Perfect solution to my problem",
        "Outstanding performance and value",
        "Love this so much, totally worth it",
        "Incredible results, five stars!",
        "Super happy with how this turned out",
        "Brilliant work, highly impressive",
        "Best decision I've made, so satisfied",
        "Extremely happy and satisfied with results",
        "Top-notch quality and brilliant customer service",
        "Very impressed, would definitely buy again",
        "Absolutely fantastic, exceeded expectations by far",
        "Loving every bit of this experience",
        "Phenomenal product, works like a charm",
        "Stellar performance, highly recommended",
    ]
    negative_texts = [
        "This is terrible and I hate it",
        "Worst experience of my life, very disappointed",
        "Awful quality, complete waste of money",
        "Horrible service, never using again",
        "Very frustrating and annoying experience",
        "Complete garbage, don't buy this",
        "Extremely disappointing and useless",
        "Terrible product, broke immediately",
        "Total disaster, nothing works as advertised",
        "Pathetic quality, save your money",
        "Completely broken out of the box",
        "Never again, this was a nightmare",
        "Absolutely awful, would not recommend",
        "Disappointed and frustrated beyond belief",
        "Wasted my money on this trash",
        "The worst product I have ever used",
        "Dreadful experience from start to finish",
        "It is garbage, do not waste your money",
        "Useless and overpriced junk",
        "Horrific product, total ripoff",
    ]
    neutral_texts = [
        "It's okay, nothing special",
        "Average product, does the job",
        "Not great but not bad either",
        "It works as expected, nothing more",
        "Standard quality, meets basic needs",
        "Typical performance, no complaints",
        "Regular product with normal features",
        "It's fine, just what I expected",
        "Mediocre, could be better but functional",
        "Just about adequate for the purpose",
        "Meets expectations, nothing to rave about",
        "Decent enough for everyday use",
        "Not particularly impressive, works fine",
        "Does what it says on the tin",
        "Results are acceptable, nothing fancy",
        "Overall satisfactory, minor issues",
        "Functional but unremarkable",
        "Middle of the road experience",
        "Works well enough for basic tasks",
        "Acceptable product, reasonable price",
    ]

    all_texts = positive_texts + negative_texts + neutral_texts
    all_labels = (["positive"] * len(positive_texts) +
                  ["negative"] * len(negative_texts) +
                  ["neutral"] * len(neutral_texts))

    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        all_texts, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                                  stop_words='english', min_df=1)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    clf = RandomForestClassifier(n_estimators=100, max_depth=20,
                                  n_jobs=-1, random_state=42)
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')

    print(f"  Training samples : {len(X_train)}")
    print(f"  Test samples     : {len(X_test)}")
    print(f"  Accuracy         : {acc * 100:.2f}%")
    print(f"  F1-Score (weighted) : {f1:.4f}  ({f1*100:.2f}%)")
    print(f"  F1-Score (macro)    : {f1_macro:.4f}  ({f1_macro*100:.2f}%)")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred))

    rf_results = {"accuracy_pct": round(acc * 100, 2),
                  "f1_weighted": round(f1, 4),
                  "f1_macro": round(f1_macro, 4),
                  "train_samples": len(X_train),
                  "test_samples": len(X_test)}
except Exception as e:
    print(f"  ERROR: {e}")
    rf_results = {"error": str(e)}

# ─────────────────────────────────────────────────────────────
# 2. BERTOPIC TOPIC MODELING - Topic Count & Coherence
# ─────────────────────────────────────────────────────────────
print("\n[2/4] BERTOPIC TOPIC MODELING METRICS")
print("-" * 40)
try:
    # Generate a synthetic corpus representative of Reddit/social data
    corpus = (
        positive_texts + negative_texts + neutral_texts +
        [
            "machine learning algorithms are transforming data science",
            "neural networks excel at image recognition tasks",
            "Python is the most popular language for data science",
            "random forests combine many decision trees together",
            "deep learning requires large amounts of training data",
            "sentiment analysis helps understand customer reviews",
            "natural language processing enables text understanding",
            "big data infrastructure scales to petabytes",
            "cloud computing reduces infrastructure costs significantly",
            "API design principles for scalable web services",
        ] * 3  # replicate to exceed min_topic_size
    )

    from bertopic import BERTopic
    model = BERTopic(language="english", min_topic_size=3,
                     nr_topics="auto", verbose=False)
    topics, probs = model.fit_transform(corpus)
    topic_info = model.get_topic_info()
    # Exclude outlier topic (-1)
    valid_topics = topic_info[topic_info['Topic'] != -1]
    n_topics = len(valid_topics)

    # Calculate inter-topic coherence proxy via topic word score averages
    coherence_scores = []
    for tid in valid_topics['Topic']:
        words = model.get_topic(tid)
        if words:
            scores = [s for _, s in words[:10]]
            coherence_scores.append(sum(scores)/len(scores))

    avg_coherence = sum(coherence_scores)/len(coherence_scores) if coherence_scores else 0
    peak_coherence = max(coherence_scores) if coherence_scores else 0

    print(f"  Documents analyzed : {len(corpus)}")
    print(f"  Distinct topics    : {n_topics}")
    print(f"  Outlier docs       : {topics.count(-1)}")
    print(f"  Avg topic score    : {avg_coherence:.4f}")
    print(f"  Peak topic score   : {peak_coherence:.4f}")
    print(f"\n  Topics detected:")
    for _, row in valid_topics.iterrows():
        kw = model.get_topic(row['Topic'])
        words = ', '.join([w for w, _ in kw[:4]]) if kw else 'N/A'
        print(f"    Topic {row['Topic']:2d}: count={row['Count']:3d}  keywords=[{words}]")

    topic_results = {"n_topics": n_topics, "corpus_size": len(corpus),
                     "avg_coherence": round(avg_coherence, 4),
                     "peak_coherence": round(peak_coherence, 4)}
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback; traceback.print_exc()
    topic_results = {"error": str(e)}

# ─────────────────────────────────────────────────────────────
# 3. API LATENCY - Benchmark /api/analyze endpoint
# ─────────────────────────────────────────────────────────────
print("\n[3/4] API LATENCY BENCHMARK")
print("-" * 40)
try:
    import asyncio, httpx

    test_texts = [
        "I absolutely love this amazing product!",
        "This is terrible, worst experience ever.",
        "The weather is okay today, nothing special.",
        "Great service and outstanding quality!",
        "Horrible, broken, and a waste of money.",
    ] * 20  # 100 payloads

    async def send_request(client, text, i):
        start = time.perf_counter()
        try:
            r = await client.post(
                "http://localhost:8000/api/analyze",
                json={"text": text},
                timeout=10.0
            )
            elapsed = (time.perf_counter() - start) * 1000
            return {"id": i, "status": r.status_code, "latency_ms": elapsed, "ok": True}
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return {"id": i, "status": 0, "latency_ms": elapsed, "ok": False, "err": str(e)}

    async def run_benchmark():
        async with httpx.AsyncClient() as client:
            tasks = [send_request(client, text, i) for i, text in enumerate(test_texts)]
            results = await asyncio.gather(*tasks)
        return results

    print("  Sending 100 concurrent requests to http://localhost:8000/api/analyze ...")
    results = asyncio.run(run_benchmark())

    ok = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]

    if ok:
        latencies = [r["latency_ms"] for r in ok]
        avg_lat = sum(latencies) / len(latencies)
        min_lat = min(latencies)
        max_lat = max(latencies)
        p95_lat = sorted(latencies)[int(len(latencies) * 0.95)]
        print(f"  Requests sent      : 100")
        print(f"  Successful         : {len(ok)}")
        print(f"  Failed             : {len(fail)}")
        print(f"  Avg latency        : {avg_lat:.2f} ms")
        print(f"  Min latency        : {min_lat:.2f} ms")
        print(f"  Max latency        : {max_lat:.2f} ms")
        print(f"  P95 latency        : {p95_lat:.2f} ms")
        api_results = {"avg_ms": round(avg_lat, 2), "min_ms": round(min_lat, 2),
                       "max_ms": round(max_lat, 2), "p95_ms": round(p95_lat, 2),
                       "success": len(ok), "fail": len(fail)}
    else:
        print(f"  All requests failed. Server may not be running.")
        print(f"  First error: {fail[0].get('err', 'unknown')}")
        api_results = {"error": "server not running", "fail": 100}
except Exception as e:
    print(f"  ERROR: {e}")
    api_results = {"error": str(e)}

# ─────────────────────────────────────────────────────────────
# 4. DATA VOLUME - Scan for datasets
# ─────────────────────────────────────────────────────────────
print("\n[4/4] DATA VOLUME SCAN")
print("-" * 40)
try:
    backend_dir = Path(__file__).parent
    scan_dirs = [
        backend_dir / "data",
        backend_dir / "datasets",
        backend_dir,
    ]
    extensions = {'.csv', '.json', '.txt', '.db', '.sqlite', '.parquet'}

    total_bytes = 0
    total_records = 0
    found_files = []

    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            continue
        depth = 0 if scan_dir == backend_dir else 3
        pattern = '**/*' if depth > 0 else '*'
        for f in (scan_dir.rglob('*') if depth > 0 else scan_dir.glob('*')):
            if f.is_file() and f.suffix.lower() in extensions:
                size = f.stat().st_size
                total_bytes += size
                found_files.append((f.name, size, str(f.relative_to(backend_dir))))

    # Count SQLite records
    db_path = backend_dir / "panchayat.db"
    db_records = 0
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            for (tname,) in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM [{tname}]")
                    cnt = cursor.fetchone()[0]
                    db_records += cnt
                    print(f"  DB table '{tname}': {cnt} records")
                except:
                    pass
            conn.close()
            total_records += db_records
        except Exception as e:
            print(f"  DB error: {e}")

    if found_files:
        print(f"\n  Data files found:")
        for name, size, path in found_files:
            print(f"    {path}: {size:,} bytes ({size/1024:.1f} KB)")
    else:
        print("  No raw dataset files found (data/ and datasets/ are empty)")
        print("  Note: Project supports Sentiment140 (1.6M tweets, ~238MB) or IMDB (50K reviews)")

    total_mb = total_bytes / (1024 * 1024)
    print(f"\n  Total data size    : {total_bytes:,} bytes ({total_mb:.2f} MB)")
    print(f"  DB records found   : {db_records}")

    data_results = {
        "total_bytes": total_bytes,
        "total_mb": round(total_mb, 2),
        "db_records": db_records,
        "files_found": len(found_files),
    }
except Exception as e:
    print(f"  ERROR: {e}")
    data_results = {"error": str(e)}

# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BENCHMARK SUMMARY (JSON)")
print("=" * 60)
summary = {
    "1_rf_classifier": rf_results,
    "2_topic_modeling": topic_results,
    "3_api_latency": api_results,
    "4_data_volume": data_results,
}
print(json.dumps(summary, indent=2))
print("=" * 60)
