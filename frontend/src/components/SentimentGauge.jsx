// Sentiment Breakdown component — replaces the old gauge with a clear visual breakdown
export default function SentimentGauge({ score, label, stats }) {
    const distribution = stats?.stats?.distribution || {};
    const total = stats?.stats?.total || 0;

    const positive = distribution.positive || 0;
    const negative = distribution.negative || 0;
    const neutral  = distribution.neutral  || 0;

    const positivePercent = total > 0 ? Math.round((positive / total) * 100) : 0;
    const negativePercent = total > 0 ? Math.round((negative / total) * 100) : 0;
    const neutralPercent  = total > 0 ? Math.round((neutral  / total) * 100) : 0;

    // Overall dominant mood
    let dominantLabel = 'Neutral';
    let dominantColor = 'var(--neutral)';
    if (positivePercent > negativePercent && positivePercent > neutralPercent) {
        dominantLabel = 'Mostly Positive';
        dominantColor = 'var(--positive)';
    } else if (negativePercent > positivePercent && negativePercent > neutralPercent) {
        dominantLabel = 'Mostly Negative';
        dominantColor = 'var(--negative)';
    }

    const rows = [
        { key: 'positive', label: 'Positive', pct: positivePercent, color: 'var(--positive)', dot: '#22c55e' },
        { key: 'neutral',  label: 'Neutral',  pct: neutralPercent,  color: 'var(--neutral)',  dot: '#94a3b8' },
        { key: 'negative', label: 'Negative', pct: negativePercent, color: 'var(--negative)', dot: '#ef4444' },
    ];

    return (
        <div className="card" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <div className="card-title" style={{ marginBottom: '20px' }}>Sentiment Breakdown</div>

            <div className="sentiment-breakdown">
                <div>
                    <div className="breakdown-header">
                        <span className="breakdown-score" style={{ color: dominantColor }}>
                            {positivePercent}%
                        </span>
                        <span className="breakdown-label">{dominantLabel}</span>
                    </div>

                    {/* Stacked bar */}
                    <div className="breakdown-stacked-bar">
                        <span style={{ width: `${positivePercent}%`, background: 'var(--positive)' }} />
                        <span style={{ width: `${neutralPercent}%`,  background: 'var(--neutral)'  }} />
                        <span style={{ width: `${negativePercent}%`, background: 'var(--negative)' }} />
                    </div>
                </div>

                <div className="breakdown-rows">
                    {rows.map(row => (
                        <div className="breakdown-row" key={row.key}>
                            <div className="breakdown-row-label">
                                <span className="breakdown-row-dot" style={{ background: row.dot }} />
                                {row.label}
                            </div>
                            <div className="breakdown-row-track">
                                <div
                                    className="breakdown-row-fill"
                                    style={{ width: `${row.pct}%`, background: row.color }}
                                />
                            </div>
                            <div className="breakdown-row-pct">{row.pct}%</div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
