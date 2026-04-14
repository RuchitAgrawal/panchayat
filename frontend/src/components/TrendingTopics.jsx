// Modern horizontal topic bar chart with sentiment coloring
export default function TrendingTopics({ topics }) {
    if (!topics || topics.length === 0) {
        return (
            <div className="card">
                <div className="card-header">
                    <div className="card-title">Trending Topics</div>
                    <div className="card-badge">MapReduce</div>
                </div>
                <div className="empty-state">
                    <div className="empty-state-icon">🔍</div>
                    <div className="empty-state-text">No topics extracted yet.<br/>Processing incoming posts…</div>
                </div>
            </div>
        );
    }

    const top = topics.slice(0, 12);
    const maxCount = Math.max(...top.map(t => t.count), 1);

    const getColor = (sentiment) => {
        if (sentiment === 'positive') return 'var(--positive)';
        if (sentiment === 'negative') return 'var(--negative)';
        return 'var(--neutral)';
    };

    return (
        <div className="card">
            <div className="card-header">
                <div className="card-title">Trending Topics</div>
                <div className="card-badge">MapReduce</div>
            </div>

            <div className="topic-list">
                {top.map((t, i) => (
                    <div className="topic-row" key={t.word || i}>
                        <div className="topic-name" title={t.word}>{t.word}</div>
                        <div className="topic-bar-track">
                            <div
                                className="topic-bar-fill"
                                style={{
                                    width: `${Math.round((t.count / maxCount) * 100)}%`,
                                    background: getColor(t.sentiment_association)
                                }}
                            />
                        </div>
                        <div className="topic-count">{t.count}</div>
                    </div>
                ))}
            </div>
        </div>
    );
}
