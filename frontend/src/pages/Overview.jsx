import { useState, useEffect } from 'react';
import StatsCards from '../components/StatsCards';
import SentimentGauge from '../components/SentimentGauge';
import TrendChart from '../components/TrendChart';
import RecentPosts from '../components/RecentPosts';
import { fetchStats, fetchTrends, fetchPosts } from '../api/client';

export default function Overview() {
    const [stats, setStats]   = useState(null);
    const [trends, setTrends] = useState(null);
    const [posts, setPosts]   = useState(null);
    const [lastUpdate, setLastUpdate] = useState(null);

    useEffect(() => {
        async function fetchData() {
            const [statsData, trendsData, postsData] = await Promise.all([
                fetchStats().catch(() => null),
                fetchTrends().catch(() => null),
                fetchPosts(15).catch(() => null),
            ]);
            setStats(statsData);
            setTrends(trendsData);
            setPosts(postsData);
            setLastUpdate(new Date());
        }

        fetchData();
        const interval = setInterval(fetchData, 10000);
        return () => clearInterval(interval);
    }, []);

    const avgScore = trends?.summary?.avg_score || 0;

    return (
        <div className="dashboard">
            <div style={{ marginBottom: '24px' }}>
                <div className="page-title">Sentiment Dashboard</div>
                <div className="page-subtitle">
                    Real-time Bluesky firehose analysis · Updated every 10 seconds
                    {lastUpdate && (
                        <span style={{ marginLeft: '12px', opacity: 0.6 }}>
                            · Last refresh: {lastUpdate.toLocaleTimeString()}
                        </span>
                    )}
                </div>
            </div>

            {/* Row 1: 4 stat cards */}
            <StatsCards stats={stats} />

            {/* Row 2: Breakdown + Trend Chart */}
            <div className="dashboard-grid">
                <SentimentGauge
                    score={avgScore}
                    label={trends?.summary?.trend_direction}
                    stats={stats}
                />
                <TrendChart data={trends} />
            </div>

            {/* Row 3: Live Feed */}
            <RecentPosts posts={posts} />
        </div>
    );
}
