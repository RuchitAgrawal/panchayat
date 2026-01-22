// Sentiment gauge component - animated dial
export default function SentimentGauge({ score, label }) {
    // Score ranges from -1 to 1, convert to 0-100 for display
    const displayValue = Math.round(((score || 0) + 1) * 50);

    // Calculate rotation for gauge needle (-90 to +90 degrees)
    const rotation = (displayValue / 100) * 180 - 90;

    // Determine color based on score
    const getColor = () => {
        if (score >= 0.3) return 'var(--positive)';
        if (score <= -0.3) return 'var(--negative)';
        return 'var(--neutral)';
    };

    return (
        <div className="card gauge-container">
            <div className="card-title">Overall Sentiment</div>

            <svg viewBox="0 0 200 120" width="200" height="120">
                {/* Background arc */}
                <path
                    d="M 20 100 A 80 80 0 0 1 180 100"
                    fill="none"
                    stroke="var(--border-color)"
                    strokeWidth="12"
                    strokeLinecap="round"
                />

                {/* Colored segments */}
                <path
                    d="M 20 100 A 80 80 0 0 1 60 35"
                    fill="none"
                    stroke="var(--negative)"
                    strokeWidth="12"
                    strokeLinecap="round"
                    opacity="0.6"
                />
                <path
                    d="M 60 35 A 80 80 0 0 1 140 35"
                    fill="none"
                    stroke="var(--gauge-neutral)"
                    strokeWidth="12"
                    opacity="0.6"
                />
                <path
                    d="M 140 35 A 80 80 0 0 1 180 100"
                    fill="none"
                    stroke="var(--positive)"
                    strokeWidth="12"
                    strokeLinecap="round"
                    opacity="0.6"
                />

                {/* Needle */}
                <line
                    x1="100"
                    y1="100"
                    x2="100"
                    y2="30"
                    stroke={getColor()}
                    strokeWidth="4"
                    strokeLinecap="round"
                    transform={`rotate(${rotation}, 100, 100)`}
                    style={{ transition: 'transform 0.5s ease-out' }}
                />

                {/* Center dot */}
                <circle cx="100" cy="100" r="8" fill={getColor()} />
            </svg>

            <div className="gauge-value" style={{ color: getColor() }}>
                {displayValue}%
            </div>
            <div className="gauge-label">
                {label || (score >= 0.3 ? 'Positive' : score <= -0.3 ? 'Negative' : 'Neutral')}
            </div>
        </div>
    );
}
