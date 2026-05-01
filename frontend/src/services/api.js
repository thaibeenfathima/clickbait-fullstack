import axios from 'axios';

// ML Server base URL - lightweight inference server
const ML_SERVER_URL = import.meta.env.VITE_ML_SERVER_URL || 'http://localhost:5000/api';

const mlApi = axios.create({
  baseURL: ML_SERVER_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000,
});

// Add response interceptor for error handling
mlApi.interceptors.response.use(
  response => response,
  error => {
    console.error('ML Server Error:', error.message);
    return Promise.reject(error);
  }
);

export const analyzeHeadline = async (headline) => {
  try {
    const response = await mlApi.post('/analyze', { headline });
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.error || error.message || 'Failed to analyze headline');
  }
};

export const processBatch = async (formData) => {
  try {
    const response = await mlApi.post('/batch', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.error || error.message || 'Failed to process batch');
  }
};

export const checkServerStatus = async () => {
  try {
    const response = await mlApi.get('/status');
    return response.data;
  } catch (error) {
    console.warn('ML Server not available');
    return { status: 'unavailable', models_available: false };
  }
};

export default mlApi;
