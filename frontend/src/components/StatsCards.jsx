// Stats cards component
export default function StatsCards({ stats }) {
    const total = stats?.stats?.total || 0;
    const distribution = stats?.stats?.distribution || {};

    const positive = distribution.positive || 0;
    const negative = distribution.negative || 0;
    const neutral = distribution.neutral || 0;

    const positivePercent = total > 0 ? Math.round((positive / total) * 100) : 0;
    const negativePercent = total > 0 ? Math.round((negative / total) * 100) : 0;

    return (
        <div className="dashboard-row">
            <div className="card stat-card neutral">
                <div className="stat-icon">📊</div>
                <div className="stat-value">{total}</div>
                <div className="stat-label">Total Posts</div>
            </div>

            <div className="card stat-card positive">
                <div className="stat-icon">😊</div>
                <div className="stat-value">{positivePercent}%</div>
                <div className="stat-label">Positive</div>
            </div>

            <div className="card stat-card negative">
                <div className="stat-icon">😞</div>
                <div className="stat-value">{negativePercent}%</div>
                <div className="stat-label">Negative</div>
            </div>
        </div>
    );
}
