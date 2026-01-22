import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const apiClient = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const api = {
    // Analyze a single text
    analyzeText: async (text) => {
        const response = await apiClient.post('/api/analyze', null, {
            params: { text }
        });
        return response.data;
    },

    // Get sentiment trends
    getTrends: async () => {
        const response = await apiClient.get('/api/trends');
        return response.data;
    },

    // Get trending topics
    getTopics: async () => {
        const response = await apiClient.get('/api/topics');
        return response.data;
    },

    // Health check
    healthCheck: async () => {
        const response = await apiClient.get('/api/health');
        return response.data;
    },
};

export default api;
