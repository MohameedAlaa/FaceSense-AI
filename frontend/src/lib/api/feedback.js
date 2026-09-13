import { request } from './client.js';

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
};
