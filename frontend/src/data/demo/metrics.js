/**
 * Isolated presentation demo metrics for FaceSense overview, insights, and admin UI.
 */

export const DEMO_METRICS = {
  totalAnalyses: 1248,
  facesAnalyzed: 3184,
  averageConfidence: 89.2,
  feedbackAccuracy: 94.6,
  processingLatencyMs: 183,
  activeEngine: "ResidualEmotionCNN-Candidate-epoch39",
  
  emotionDistribution: [
    { emotion: "happy", count: 1240, percentage: 38.9, color: "#22D3EE" },
    { emotion: "neutral", count: 864, percentage: 27.1, color: "#6C63FF" },
    { emotion: "surprise", count: 412, percentage: 12.9, color: "#8B5CF6" },
    { emotion: "sad", count: 288, percentage: 9.0, color: "#94A3B8" },
    { emotion: "fear", count: 160, percentage: 5.0, color: "#F59E0B" },
    { emotion: "angry", count: 132, percentage: 4.1, color: "#EF4444" },
    { emotion: "disgust", count: 88, percentage: 3.0, color: "#10B981" },
  ],

  recentActivities: [
    { id: 'act_1', type: 'analysis', title: 'Batch Editorial Session', time: '12m ago', result: '2 faces (Happy, Neutral)' },
    { id: 'act_2', type: 'feedback', title: 'Ground truth verified', time: '45m ago', result: 'Subject 01 labeled Happy' },
    { id: 'act_3', type: 'analysis', title: 'Portrait photo evaluation', time: '2h ago', result: '1 face (Surprise 91%)' },
    { id: 'act_4', type: 'system', title: 'Model checkpoint loaded', time: '5h ago', result: 'ResidualEmotionCNN v2.4' },
  ],

  adminOverview: {
    modelVersion: "ResidualEmotionCNN-Candidate-epoch39",
    macroF1: 0.742,
    weightedF1: 0.758,
    validationAccuracy: 74.8,
    totalFeedbackRecords: 19,
    correctFeedback: 16,
    incorrectFeedback: 1,
    uncertainFeedback: 2,
  },
};
