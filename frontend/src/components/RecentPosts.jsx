// Recent posts table component
export default function RecentPosts({ posts }) {
    const postList = posts?.posts || [];

    if (postList.length === 0) {
        return (
            <div className="card">
                <div className="card-title">Recent Posts</div>
                <div className="loading">No posts yet. Load some data to get started!</div>
            </div>
        );
    }

    return (
        <div className="card">
            <div className="card-title">Recent Posts</div>
            <table className="posts-table">
                <thead>
                    <tr>
                        <th>Title</th>
                        <th>Source</th>
                        <th>Sentiment</th>
                    </tr>
                </thead>
                <tbody>
                    {postList.slice(0, 10).map((post, index) => (
                        <tr key={post.id || index}>
                            <td className="post-title" title={post.title}>
                                {post.title?.substring(0, 60) || 'No title'}
                                {post.title?.length > 60 ? '...' : ''}
                            </td>
                            <td>{post.subreddit || 'Unknown'}</td>
                            <td>
                                <span className={`sentiment-badge ${post.sentiment?.label || 'neutral'}`}>
                                    {post.sentiment?.label || 'N/A'}
                                </span>
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}
