import { request } from './client';

/**
 * Feedback and Stats endpoint contracts (prepared for Phase 2 integration)
 */
export const feedbackApi = {
  submitFeedback: async (feedbackPayload) => {
    return request('/feedback', {
      method: 'POST',
      body: feedbackPayload,
    });
  },

  getStats: async () => {
    return request('/feedback/stats');
  },
};
