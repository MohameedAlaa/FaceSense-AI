import { request } from './client';

/**
 * Authentication endpoint contracts (prepared for Phase 2 integration)
 */
export const authApi = {
  login: async (email, password) => {
    return request('/auth/login', {
      method: 'POST',
      body: { email, password },
    });
  },

  register: async (email, password) => {
    return request('/auth/register', {
      method: 'POST',
      body: { email, password },
    });
  },
};
