// Trend chart component using Recharts
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

export default function TrendChart({ data }) {
    // Format data for chart
    const chartData = (data?.trends || []).map(item => ({
        time: new Date(item.time).toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit'
        }),
        score: Math.round(((item.avg_score || 0) + 1) * 50), // Convert -1..1 to 0..100
        count: item.count || 0
    }));

    // If no data, show placeholder
    if (chartData.length === 0) {
        return (
            <div className="card">
                <div className="card-title">Sentiment Trends</div>
                <div className="chart-container loading">
                    No trend data yet. Analyze some posts to see trends!
                </div>
            </div>
        );
    }

    return (
        <div className="card">
            <div className="card-title">Sentiment Trends</div>
            <div className="chart-container">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={chartData}>
                        <defs>
                            <linearGradient id="colorScore" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="var(--accent)" stopOpacity={0.3} />
                                <stop offset="95%" stopColor="var(--accent)" stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <XAxis
                            dataKey="time"
                            tick={{ fill: 'var(--text-secondary)', fontSize: 12 }}
                            axisLine={{ stroke: 'var(--border-color)' }}
                            tickLine={false}
                        />
                        <YAxis
                            domain={[0, 100]}
                            tick={{ fill: 'var(--text-secondary)', fontSize: 12 }}
                            axisLine={{ stroke: 'var(--border-color)' }}
                            tickLine={false}
                            width={40}
                        />
                        <Tooltip
                            contentStyle={{
                                background: 'var(--bg-card)',
                                border: '1px solid var(--border-color)',
                                borderRadius: '8px',
                                color: 'var(--text-primary)'
                            }}
                            formatter={(value) => [`${value}%`, 'Sentiment']}
                        />
                        <Area
                            type="monotone"
                            dataKey="score"
                            stroke="var(--accent)"
                            strokeWidth={2}
                            fillOpacity={1}
                            fill="url(#colorScore)"
                        />
                    </AreaChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
}
