import { request } from './client';

/**
 * FaceSense Authentication API
 * Communicates with FastAPI auth endpoints:
 * - POST /api/v1/auth/login (OAuth2 form-urlencoded with username & password)
 * - POST /api/v1/auth/register (JSON with email & password)
 * - GET  /api/v1/auth/me (Bearer authenticated)
 */
export const authApi = {
  /**
   * Logs in a user using OAuth2PasswordRequestForm convention (x-www-form-urlencoded).
   * @param {string} email
   * @param {string} password
   * @returns {Promise<{access_token: string, token_type: string, expires_in_seconds: number, role: string}>}
   */
  login: async (email, password) => {
    const params = new URLSearchParams();
    params.append('username', email.trim());
    params.append('password', password);

    return request('/auth/login', {
      method: 'POST',
      body: params,
    });
  },

  /**
   * Registers a new user.
   * Strictly sends only email and password as defined by FastAPI UserRegisterRequest schema.
   * @param {string} email
   * @param {string} password
   * @returns {Promise<{id: number, email: string, role: string, is_active: boolean, created_at: string}>}
   */
  register: async (email, password) => {
    return request('/auth/register', {
      method: 'POST',
      body: {
        email: email.trim(),
        password: password,
      },
    });
  },

  /**
   * Retrieves profile details for the currently authenticated user.
   * @returns {Promise<{id: number, email: string, role: string, is_active: boolean, created_at: string}>}
   */
  me: async () => {
    return request('/auth/me', {
      method: 'GET',
    });
  },
};
