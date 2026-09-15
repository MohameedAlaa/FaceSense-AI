/**
 * FaceSense Frontend API Client Layer
 * Centralized HTTP communication with the FastAPI backend.
 */

const RAW_BASE_URL = import.meta.env?.VITE_API_BASE_URL || 'http://127.0.0.1:8000/api/v1';
export const API_BASE_URL = RAW_BASE_URL.replace(/\/+$/, '');

export const TOKEN_STORAGE_KEY = 'facesense_token';

export class ApiError extends Error {
  constructor(message, status = 500, details = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.details = details;
  }
}

export function getAuthToken() {
  try {
    return localStorage.getItem(TOKEN_STORAGE_KEY);
  } catch {
    return null;
  }
}

export function setAuthToken(token) {
  try {
    if (token) {
      localStorage.setItem(TOKEN_STORAGE_KEY, token);
    } else {
      localStorage.removeItem(TOKEN_STORAGE_KEY);
    }
  } catch {
    // localStorage might be unavailable in restricted environments
  }
}

export function clearAuthToken() {
  try {
    localStorage.removeItem(TOKEN_STORAGE_KEY);
  } catch {
    // ignore
  }
}

/**
 * Dispatches an event when a 401 is encountered, allowing AuthContext to reset state.
 */
function notifyUnauthorized() {
  clearAuthToken();
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent('facesense:unauthorized'));
  }
}

/**
 * Primary HTTP request wrapper.
 * Handles JSON, FormData, and URLSearchParams gracefully without header conflicts.
 */
export async function request(endpoint, options = {}) {
  const token = getAuthToken();

  const headers = {
    Accept: 'application/json',
    ...(options.headers || {}),
  };

  if (token && !headers['Authorization']) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  let body = options.body;

  // Header and serialization handling depending on payload type
  if (body instanceof FormData) {
    // Browser automatically sets multipart/form-data and boundary
    delete headers['Content-Type'];
  } else if (body instanceof URLSearchParams) {
    headers['Content-Type'] = 'application/x-www-form-urlencoded';
    body = body.toString();
  } else if (body && typeof body === 'object' && !(body instanceof Blob)) {
    headers['Content-Type'] = 'application/json';
    body = JSON.stringify(body);
  }

  let cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
  if (cleanEndpoint.startsWith('/api/v1/')) {
    cleanEndpoint = cleanEndpoint.slice(7);
  }
  const url = `${API_BASE_URL}${cleanEndpoint}`;

  try {
    const response = await fetch(url, {
      ...options,
      headers,
      body,
    });

    if (!response.ok) {
      let errorData = null;
      try {
        errorData = await response.json();
      } catch {
        // Response was not JSON
      }

      // Handle 401 Unauthorized
      if (response.status === 401) {
        notifyUnauthorized();
      }

      let message = `Request failed with status ${response.status}`;
      if (errorData) {
        if (typeof errorData.detail === 'string') {
          message = errorData.detail;
        } else if (Array.isArray(errorData.detail)) {
          // Pydantic validation errors
          message = errorData.detail.map((err) => `${err.loc?.join('.') || 'field'}: ${err.msg}`).join(', ');
        } else if (errorData.message) {
          message = errorData.message;
        } else if (errorData.error) {
          message = typeof errorData.error === 'string' ? errorData.error : JSON.stringify(errorData.error);
        }
      }

      throw new ApiError(message, response.status, errorData);
    }

    // Return empty object on 204 No Content
    if (response.status === 204) {
      return {};
    }

    return await response.json();
  } catch (err) {
    if (err instanceof ApiError) {
      throw err;
    }
    // Network failure, CORS blockage, or offline
    throw new ApiError(
      err.message || 'Unable to connect to FaceSense backend service. Please verify the server is running.',
      0,
      err
    );
  }
}

/**
 * Authenticated binary/blob HTTP request wrapper.
 * Useful for authenticated images, downloads, and media resources.
 */
export async function requestBlob(endpoint, options = {}) {
  const token = getAuthToken();

  const headers = {
    ...(options.headers || {}),
  };

  if (token && !headers['Authorization']) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  let cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
  if (cleanEndpoint.startsWith('/api/v1/')) {
    cleanEndpoint = cleanEndpoint.slice(7);
  }
  const url = `${API_BASE_URL}${cleanEndpoint}`;

  try {
    const response = await fetch(url, {
      ...options,
      headers,
    });

    if (!response.ok) {
      let errorData = null;
      try {
        errorData = await response.json();
      } catch {
        // Response was not JSON
      }

      if (response.status === 401) {
        notifyUnauthorized();
      }

      let message = `Request failed with status ${response.status}`;
      if (errorData?.detail) {
        message = errorData.detail;
      }
      throw new ApiError(message, response.status, errorData);
    }

    return await response.blob();
  } catch (err) {
    if (err instanceof ApiError) {
      throw err;
    }
    throw new ApiError(
      err.message || 'Unable to load resource from FaceSense backend service.',
      0,
      err
    );
  }
}
