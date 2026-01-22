// API client for backend communication
const API_BASE = 'http://localhost:8000';

export async function fetchStats() {
    const res = await fetch(`${API_BASE}/api/posts/stats`);
    if (!res.ok) throw new Error('Failed to fetch stats');
    return res.json();
}

export async function fetchTrends(period = '1h') {
    const res = await fetch(`${API_BASE}/api/trends?period=${period}`);
    if (!res.ok) throw new Error('Failed to fetch trends');
    return res.json();
}

export async function fetchPosts(limit = 20) {
    const res = await fetch(`${API_BASE}/api/posts?limit=${limit}`);
    if (!res.ok) throw new Error('Failed to fetch posts');
    return res.json();
}

export async function analyzeSentiment(text) {
    const res = await fetch(`${API_BASE}/api/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
    });
    if (!res.ok) throw new Error('Failed to analyze');
    return res.json();
}

export async function loadSampleData(count = 10) {
    const res = await fetch(`${API_BASE}/api/sample/quick?count=${count}`);
    if (!res.ok) throw new Error('Failed to load sample data');
    return res.json();
}
