import { useState, useEffect } from 'react';
import TrendChart from '../components/TrendChart';
import TrendingTopics from '../components/TrendingTopics';
import { fetchTrends, fetchTopics } from '../api/client';

export default function Trends() {
    const [trends, setTrends] = useState(null);
    const [topics, setTopics] = useState(null);
    const [lastUpdate, setLastUpdate] = useState(null);

    useEffect(() => {
        async function fetchData() {
            const [trendsData, topicsData] = await Promise.all([
                fetchTrends().catch(() => null),
                fetchTopics(15).catch(() => null),
            ]);
            setTrends(trendsData);
            setTopics(topicsData?.topics || []);
            setLastUpdate(new Date());
        }

        fetchData();
        const interval = setInterval(fetchData, 10000);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="dashboard">
            <div style={{ marginBottom: '24px' }}>
                <div className="page-title">Trends &amp; Analysis</div>
                <div className="page-subtitle">
                    Hourly sentiment trends and MapReduce-extracted topics from Bluesky firehose
                    {lastUpdate && (
                        <span style={{ marginLeft: '12px', opacity: 0.6 }}>
                            · Last refresh: {lastUpdate.toLocaleTimeString()}
                        </span>
                    )}
                </div>
            </div>

            {/* Charts side by side */}
            <div className="dashboard-grid--equal">
                {/* Pass as 'data' since TrendChart reads data?.trends */}
                <TrendChart data={trends} />
                <TrendingTopics topics={topics} />
            </div>

            {/* Info Box */}
            <div className="info-box">
                <div className="info-box-title">How this is calculated</div>
                <p>
                    <strong>Sentiment Trend:</strong> Every 60 seconds, PySpark collects all new Bluesky posts,
                    runs them through the BERT + LSTM ensemble model, and averages the resulting sentiment scores
                    into hourly buckets stored in SQLite. The chart shows the evolution over time.<br /><br />
                    <strong>Trending Topics:</strong> A MapReduce job tokenises each post, removes 300+ stop words
                    using scikit-learn's <code>ENGLISH_STOP_WORDS</code> plus a custom social media list, then
                    aggregates word frequencies grouped by their associated sentiment label. Only words longer
                    than 4 characters are included.
                </p>
            </div>
        </div>
    );
}
