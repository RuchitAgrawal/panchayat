// Stats cards — 4-across premium metric cards
export default function StatsCards({ stats }) {
    const total = stats?.stats?.total || 0;
    const distribution = stats?.stats?.distribution || {};

    const positive = distribution.positive || 0;
    const negative = distribution.negative || 0;
    const neutral  = distribution.neutral  || 0;

    const positivePercent = total > 0 ? Math.round((positive / total) * 100) : 0;
    const negativePercent = total > 0 ? Math.round((negative / total) * 100) : 0;
    const neutralPercent  = total > 0 ? Math.round((neutral  / total) * 100) : 0;

    const cards = [
        {
            key: 'total',
            icon: '📡',
            value: total.toLocaleString(),
            label: 'Posts Analyzed',
            className: 'total',
            barColor: 'var(--accent)',
            barWidth: '100%',
        },
        {
            key: 'positive',
            icon: '😊',
            value: `${positivePercent}%`,
            label: 'Positive',
            className: 'positive',
            barColor: 'var(--positive)',
            barWidth: `${positivePercent}%`,
        },
        {
            key: 'negative',
            icon: '😞',
            value: `${negativePercent}%`,
            label: 'Negative',
            className: 'negative',
            barColor: 'var(--negative)',
            barWidth: `${negativePercent}%`,
        },
        {
            key: 'neutral',
            icon: '😐',
            value: `${neutralPercent}%`,
            label: 'Neutral',
            className: 'neutral',
            barColor: 'var(--neutral)',
            barWidth: `${neutralPercent}%`,
        },
    ];

    return (
        <div className="stats-row">
            {cards.map(card => (
                <div key={card.key} className="card stat-card">
                    <div className="stat-card-top">
                        <div className={`stat-icon-wrap ${card.className}`}>
                            {card.icon}
                        </div>
                        <span className={`stat-trend flat`}>Bluesky</span>
                    </div>
                    <div className="stat-value">{card.value}</div>
                    <div className="stat-label">{card.label}</div>
                    <div className="stat-bar-track">
                        <div
                            className="stat-bar-fill"
                            style={{ width: card.barWidth, background: card.barColor }}
                        />
                    </div>
                </div>
            ))}
        </div>
    );
}
