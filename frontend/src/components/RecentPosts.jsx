// Premium live post feed
export default function RecentPosts({ posts }) {
    const postList = posts?.posts || [];

    return (
        <div className="card">
            <div className="card-header">
                <div className="card-title">Live Feed</div>
                <div className="card-badge">{postList.length} posts</div>
            </div>

            {postList.length === 0 ? (
                <div className="empty-state">
                    <div className="empty-state-icon">📭</div>
                    <div className="empty-state-text">
                        No posts yet.<br />Start the backend to see live Bluesky posts here.
                    </div>
                </div>
            ) : (
                <div className="posts-feed">
                    {postList.map((post, index) => {
                        const label = post.sentiment?.label || 'neutral';
                        return (
                            <div
                                key={post.id || index}
                                className={`post-feed-card border-${label}`}
                            >
                                <div className="post-feed-header">
                                    <span className="post-source">
                                        {post.subreddit || 'bluesky'}
                                    </span>
                                    <span className={`sentiment-badge ${label}`}>
                                        {label === 'positive' ? '😊 ' : label === 'negative' ? '😞 ' : '😐 '}
                                        {label}
                                    </span>
                                </div>
                                <div className="post-feed-content">
                                    {post.title || 'No content'}
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
}
