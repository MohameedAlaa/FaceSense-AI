import { request } from './client';

/**
 * Vision Prediction endpoint contracts (prepared for Phase 2 integration)
 */
export const predictApi = {
  predictImage: async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    return request('/predict/image', {
      method: 'POST',
      body: formData,
    });
  },

  getModelInfo: async () => {
    return request('/model/info');
  },
};
