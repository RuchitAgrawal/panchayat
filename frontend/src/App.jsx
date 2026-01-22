// Panchayat Dashboard - Main App Component
import { useState, useEffect } from 'react';
import { ThemeProvider } from './hooks/useTheme';
import Header from './components/Header';
import StatsCards from './components/StatsCards';
import SentimentGauge from './components/SentimentGauge';
import TrendChart from './components/TrendChart';
import RecentPosts from './components/RecentPosts';
import { fetchStats, fetchTrends, fetchPosts, loadSampleData } from './api/client';
import './index.css';

function Dashboard() {
  const [stats, setStats] = useState(null);
  const [trends, setTrends] = useState(null);
  const [posts, setPosts] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch all data on mount
  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);
        const [statsData, trendsData, postsData] = await Promise.all([
          fetchStats().catch(() => null),
          fetchTrends().catch(() => null),
          fetchPosts().catch(() => null)
        ]);

        setStats(statsData);
        setTrends(trendsData);
        setPosts(postsData);
        setError(null);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }

    fetchData();
  }, []);

  // Load sample data button handler
  const handleLoadSample = async () => {
    try {
      setLoading(true);
      await loadSampleData(10);

      // Refresh data
      const [statsData, trendsData, postsData] = await Promise.all([
        fetchStats().catch(() => null),
        fetchTrends().catch(() => null),
        fetchPosts().catch(() => null)
      ]);

      setStats(statsData);
      setTrends(trendsData);
      setPosts(postsData);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Calculate average score for gauge
  const avgScore = trends?.summary?.avg_score || stats?.stats?.avg_score || 0;

  return (
    <div className="dashboard">
      {/* Stats Cards Row */}
      <StatsCards stats={stats} />

      {/* Main Grid: Gauge + Chart */}
      <div className="dashboard-grid">
        <SentimentGauge
          score={avgScore}
          label={trends?.summary?.trend_direction}
        />
        <TrendChart data={trends} />
      </div>

      {/* Recent Posts Table */}
      <RecentPosts posts={posts} />

      {/* Load Sample Data Button (for demo) */}
      {(!posts?.posts?.length || posts?.posts?.length < 5) && (
        <div style={{ textAlign: 'center', marginTop: '24px' }}>
          <button
            onClick={handleLoadSample}
            disabled={loading}
            style={{
              padding: '12px 24px',
              background: 'var(--accent)',
              color: 'white',
              border: 'none',
              borderRadius: 'var(--radius)',
              cursor: 'pointer',
              fontSize: '1rem',
              fontWeight: '500'
            }}
          >
            {loading ? '⏳ Loading...' : '📥 Load Sample Data'}
          </button>
        </div>
      )}
    </div>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <div className="app">
        <Header />
        <Dashboard />
      </div>
    </ThemeProvider>
  );
}
