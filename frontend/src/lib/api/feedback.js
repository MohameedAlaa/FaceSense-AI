import { request, requestBlob } from './client.js';

/**
 * FaceSense Feedback API
 * Communicates with FastAPI feedback endpoints:
 * - POST /api/v1/feedback (Submits human evaluation feedback)
 * - GET  /api/v1/feedback/stats (Admin-only aggregate statistics)
 */
export const feedbackApi = {
  /**
   * Submits user evaluation feedback (correct, incorrect, or uncertain) for model predictions.
   * Conforms strictly to FastAPI FeedbackCreateRequest schema.
   * @param {Object} payload
   * @param {'correct'|'incorrect'|'uncertain'} payload.feedback_type
   * @param {string} payload.predicted_emotion
   * @param {number} payload.confidence
   * @param {string} [payload.corrected_emotion] Ground-truth label if feedback_type is 'incorrect'
   * @param {string} [payload.image_base64] Optional base64 crop image
   * @param {string} [payload.notes] Optional notes
   * @returns {Promise<{status: string, record_id: string, message: string}>}
   */
  submitFeedback: async (payload) => {
    const cleanPayload = {
      feedback_type: payload.feedback_type,
      predicted_emotion: payload.predicted_emotion,
      confidence: payload.confidence,
    };

    if (payload.corrected_emotion) {
      cleanPayload.corrected_emotion = payload.corrected_emotion;
    }
    if (typeof payload.face_index === 'number') {
      cleanPayload.face_index = payload.face_index;
    }
    if (payload.bounding_box) {
      cleanPayload.bounding_box = payload.bounding_box;
    }
    if (payload.model_version) {
      cleanPayload.model_version = payload.model_version;
    }
    if (payload.image_base64) {
      cleanPayload.image_base64 = payload.image_base64;
    }
    if (payload.notes) {
      cleanPayload.notes = payload.notes;
    }

    return request('/feedback', {
      method: 'POST',
      body: cleanPayload,
    });
  },

  /**
   * Retrieves aggregate feedback statistics. Strictly protected by admin authorization boundary.
   * @returns {Promise<{total_records: number, correct: number, incorrect: number, uncertain: number, emotions: Object, accuracy_rate: number|null}>}
   */
  getStats: async () => {
    return request('/feedback/stats', {
      method: 'GET',
    });
  },

  /**
   * Admin-only: Retrieves feedback review counts grouped by 7 FER2013 emotions with pending/approved/rejected counts.
   * @returns {Promise<{emotions: Object, total_records: number}>}
   */
  getReviewOverview: async () => {
    return request('/feedback/review/overview', {
      method: 'GET',
    });
  },

  /**
   * Admin-only: Retrieves feedback records for a specific emotion.
   * @param {string} emotion
   * @param {string} [statusFilter] 'pending' | 'approved' | 'rejected' | 'all'
   * @param {number} [limit=100]
   * @param {number} [offset=0]
   * @returns {Promise<{emotion: string, status_filter: string|null, total: number, items: Array}>}
   */
  getReviewByEmotion: async (emotion, statusFilter = null, limit = 100, offset = 0) => {
    const params = new URLSearchParams();
    if (statusFilter && statusFilter !== 'all') {
      params.append('status', statusFilter);
    }
    params.append('limit', String(limit));
    params.append('offset', String(offset));

    const queryString = params.toString();
    return request(`/feedback/review/emotion/${encodeURIComponent(emotion)}${queryString ? `?${queryString}` : ''}`, {
      method: 'GET',
    });
  },

  /**
   * Admin-only: Submits authoritative decision (approve or reject) for a feedback record.
   * @param {string} feedbackId
   * @param {'approve'|'reject'} decision
   * @param {string} [finalLabel]
   * @returns {Promise<{status: string, feedback_id: string, review_status: string, final_label: string|null, message: string}>}
   */
  updateReviewDecision: async (feedbackId, decision, finalLabel = null) => {
    const body = { decision };
    if (finalLabel) {
      body.final_label = finalLabel;
    }
    return request(`/feedback/review/${encodeURIComponent(feedbackId)}/decision`, {
      method: 'PATCH',
      body,
    });
  },

  /**
   * Admin-only: Fetches a stored feedback image as a Blob using the authenticated API client.
   * Accepts either an image filename (e.g. "crop_123.jpg") or an image URL (e.g. "/api/v1/feedback/images/crop_123.jpg").
   * @param {string} imageNameOrUrl
   * @returns {Promise<Blob>}
   */
  getFeedbackImageBlob: async (imageNameOrUrl) => {
    if (!imageNameOrUrl || typeof imageNameOrUrl !== 'string') {
      throw new Error('Valid image name or URL is required');
    }
    let endpoint = imageNameOrUrl.trim();
    if (endpoint.includes('/feedback/images/')) {
      endpoint = `/feedback/images/${endpoint.split('/feedback/images/')[1]}`;
    } else if (!endpoint.startsWith('/feedback/images/')) {
      endpoint = `/feedback/images/${encodeURIComponent(endpoint)}`;
    }
    return requestBlob(endpoint);
  },
};


