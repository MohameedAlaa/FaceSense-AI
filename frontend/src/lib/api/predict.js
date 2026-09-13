import { request } from './client';

/**
 * FaceSense Vision Prediction API
 * Communicates with FastAPI prediction and model endpoints:
 * - POST /api/v1/predict/image
 * - GET  /api/v1/model/info
 */
export const predictApi = {
  /**
   * Performs face detection and emotion classification on an uploaded image.
   * @param {File|Blob} file Image file to analyze
   * @param {Object} options Optional query options
   * @param {number} [options.confidenceThreshold] Minimum confidence threshold (0.0 to 1.0)
   * @param {boolean} [options.allowFallback] If true, treats entire image as face when face detector detects no faces
   * @returns {Promise<{faces_detected: number, predictions: Array<{box: {x: number, y: number, w: number, h: number}, emotion: string, confidence: number, all_probabilities: Object}>, processing_time_ms: number}>}
   */
  predictImage: async (file, options = {}) => {
    const formData = new FormData();
    formData.append('file', file);

    const queryParams = new URLSearchParams();
    if (typeof options.confidenceThreshold === 'number') {
      queryParams.append('confidence_threshold', options.confidenceThreshold.toString());
    }
    if (typeof options.allowFallback === 'boolean') {
      queryParams.append('allow_fallback', options.allowFallback.toString());
    }

    const queryString = queryParams.toString() ? `?${queryParams.toString()}` : '';

    return request(`/predict/image${queryString}`, {
      method: 'POST',
      body: formData,
    });
  },

  /**
   * Retrieves current active model metadata, architecture details, and supported classes.
   * @returns {Promise<{model_name: string, architecture: string, num_classes: number, classes: string[], device: string, input_shape: number[], status: string, metrics: Object|null}>}
   */
  getModelInfo: async () => {
    return request('/model/info', {
      method: 'GET',
    });
  },
};
