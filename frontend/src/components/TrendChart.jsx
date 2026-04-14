import { useState } from 'react';
import {
    AreaChart, Area, XAxis, YAxis, Tooltip,
    ResponsiveContainer, CartesianGrid, ReferenceLine
} from 'recharts';

const PERIODS = ['1h', '6h', '24h', 'All'];

// Custom tooltip component
function CustomTooltip({ active, payload, label }) {
    if (!active || !payload || !payload.length) return null;
    const val = payload[0].value;
    const color = val >= 65 ? 'var(--positive)' : val <= 40 ? 'var(--negative)' : 'var(--neutral)';
    return (
        <div style={{
            background: '#1a2236',
            border: '1px solid rgba(255,255,255,0.08)',
            borderRadius: '10px',
            padding: '10px 14px',
            boxShadow: '0 8px 24px rgba(0,0,0,0.4)'
        }}>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginBottom: '4px' }}>{label}</div>
            <div style={{ fontSize: '1.1rem', fontWeight: 700, color }}>
                {val}% <span style={{ fontSize: '0.75rem', fontWeight: 400, color: 'var(--text-secondary)' }}>sentiment score</span>
            </div>
        </div>
    );
}

export default function TrendChart({ data }) {
    const [activePeriod, setActivePeriod] = useState('All');

    const allTrends = data?.trends || [];

    // Filter by period
    const now = Date.now();
    const periodMs = { '1h': 3600000, '6h': 21600000, '24h': 86400000, 'All': Infinity };
    const filtered = allTrends.filter(item => {
        const t = new Date(item.time).getTime();
        return (now - t) <= periodMs[activePeriod];
    });

    // Convert -1..1 avg_score to 0..100 for sanity
    const chartData = (filtered.length > 0 ? filtered : allTrends).map(item => ({
        time: (() => {
            try {
                // DB stores "YYYY-MM-DD HH:00" in IST — parse as local datetime
                const d = new Date(item.time.replace(' ', 'T'));
                return d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: true });
            } catch { return item.time; }
        })(),
        score: Math.round(((item.avg_score || 0) + 1) * 50),
        count: item.count || 0,
    }));

    // Determine trend over chart
    const trendDir = chartData.length >= 2
        ? chartData[chartData.length - 1].score - chartData[0].score
        : 0;
    const trendLabel = trendDir > 3 ? '↑ Rising' : trendDir < -3 ? '↓ Falling' : '→ Stable';
    const trendColor = trendDir > 3 ? 'var(--positive)' : trendDir < -3 ? 'var(--negative)' : 'var(--neutral)';

    return (
        <div className="card">
            <div className="card-header">
                <div>
                    <div className="card-title">Sentiment Trend</div>
                    {chartData.length > 0 && (
                        <div style={{ fontSize: '0.85rem', color: trendColor, fontWeight: 600, marginTop: '4px' }}>
                            {trendLabel}
                        </div>
                    )}
                </div>
                <div className="chart-controls">
                    {PERIODS.map(p => (
                        <button
                            key={p}
                            className={`chart-period-btn ${activePeriod === p ? 'active' : ''}`}
                            onClick={() => setActivePeriod(p)}
                        >
                            {p}
                        </button>
                    ))}
                </div>
            </div>

            <div className="chart-container">
                {chartData.length === 0 ? (
                    <div className="empty-state">
                        <div className="empty-state-icon">📈</div>
                        <div className="empty-state-text">
                            No trend data yet.<br />
                            Start the backend to begin collecting live sentiment data.
                        </div>
                    </div>
                ) : (
                    <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                            <defs>
                                <linearGradient id="sentimentGrad" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="0%"   stopColor="var(--accent)" stopOpacity={0.35} />
                                    <stop offset="100%" stopColor="var(--accent)" stopOpacity={0}    />
                                </linearGradient>
                            </defs>
                            <CartesianGrid
                                strokeDasharray="3 3"
                                stroke="rgba(255,255,255,0.04)"
                                vertical={false}
                            />
                            <XAxis
                                dataKey="time"
                                tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                                axisLine={false}
                                tickLine={false}
                                interval="preserveStartEnd"
                            />
                            <YAxis
                                domain={[0, 100]}
                                tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                                axisLine={false}
                                tickLine={false}
                                width={36}
                                tickFormatter={v => `${v}%`}
                            />
                            <Tooltip content={<CustomTooltip />} />
                            <ReferenceLine y={50} stroke="rgba(255,255,255,0.08)" strokeDasharray="4 4" />
                            <Area
                                type="monotone"
                                dataKey="score"
                                stroke="var(--accent)"
                                strokeWidth={2.5}
                                fill="url(#sentimentGrad)"
                                dot={false}
                                activeDot={{ r: 5, fill: 'var(--accent)', strokeWidth: 0 }}
                            />
                        </AreaChart>
                    </ResponsiveContainer>
                )}
            </div>
        </div>
    );
}
