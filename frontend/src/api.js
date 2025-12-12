import axios from "axios";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export const getDaily = () =>
  axios.get(`${API_BASE}/api/stats/daily`).then(r => r.data);

export const getHourly = () =>
  axios.get(`${API_BASE}/api/stats/hourly`).then(r => r.data);

export const getHistory = (limit = 100) =>
  axios.get(`${API_BASE}/api/history?limit=${limit}`).then(r => r.data);
