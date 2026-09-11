/**
 * FaceSense Frontend API Client Abstraction Layer
 * Configured for future integration with FastAPI backend.
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

export class ApiError extends Error {
  constructor(message, status = 500, details = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.details = details;
  }
}

export async function request(endpoint, options = {}) {
  const token = localStorage.getItem('facesense_token');
  
  const headers = {
    Accept: 'application/json',
    ...options.headers,
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  // Handle JSON vs FormData
  if (options.body && !(options.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json';
    options.body = JSON.stringify(options.body);
  }

  const url = `${API_BASE_URL}${endpoint.startsWith('/') ? endpoint : `/${endpoint}`}`;

  try {
    const response = await fetch(url, {
      ...options,
      headers,
    });

    if (!response.ok) {
      let errorData = {};
      try {
        errorData = await response.json();
      } catch {
        // Fallback for non-JSON error
      }
      throw new ApiError(
        errorData.detail || errorData.message || `API request failed with status ${response.status}`,
        response.status,
        errorData
      );
    }

    return await response.json();
  } catch (err) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(err.message || 'Network communication error', 0);
  }
}

export { API_BASE_URL };
