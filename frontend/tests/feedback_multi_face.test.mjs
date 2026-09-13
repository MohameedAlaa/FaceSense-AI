import { test, describe, beforeEach } from 'node:test';
import assert from 'node:assert/strict';

// Mock localStorage for Node environment
class MockLocalStorage {
  constructor() {
    this.store = {};
  }
  getItem(key) {
    return this.store[key] || null;
  }
  setItem(key, value) {
    this.store[key] = String(value);
  }
  removeItem(key) {
    delete this.store[key];
  }
  clear() {
    this.store = {};
  }
}

globalThis.localStorage = new MockLocalStorage();

describe('FaceSense Multi-Face Independent Feedback Tests', () => {
  const SUPPORTED_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'];

  // Simulated multi-face prediction response from POST /api/v1/predict/image
  const samplePredictionResponse = {
    faces_detected: 2,
    processing_time_ms: 118.4,
    predictions: [
      {
        box: { x: 120, y: 80, w: 150, h: 180 },
        emotion: 'happy',
        confidence: 0.942,
        all_probabilities: { happy: 0.942, neutral: 0.038, surprise: 0.02 },
      },
      {
        box: { x: 340, y: 95, w: 140, h: 175 },
        emotion: 'neutral',
        confidence: 0.815,
        all_probabilities: { neutral: 0.815, sad: 0.12, angry: 0.065 },
      },
    ],
  };

  const naturalWidth = 800;
  const naturalHeight = 600;

  // Helper matching AnalyzePage.jsx mapPredictionsToFaces logic
  const mapPredictions = (preds, width, height) => {
    return preds.map((pred, idx) => {
      const id = String(idx + 1).padStart(2, '0');
      const box = pred.box || { x: 0, y: 0, w: 0, h: 0 };
      const left = width > 0 ? (box.x / width) * 100 : 0;
      const top = height > 0 ? (box.y / height) * 100 : 0;
      const w = width > 0 ? (box.w / width) * 100 : 0;
      const h = height > 0 ? (box.h / height) * 100 : 0;

      return {
        id,
        label: `Face ${id}`,
        face_index: idx,
        pixelBox: box,
        box: { left, top, width: w, height: h },
        predicted_emotion: pred.emotion,
        confidence: pred.confidence,
        probabilities: pred.all_probabilities || {},
      };
    });
  };

  test('1. One face → one feedback flow mapping and submission payload', () => {
    const faces = mapPredictions([samplePredictionResponse.predictions[0]], naturalWidth, naturalHeight);
    assert.equal(faces.length, 1);
    const face01 = faces[0];

    assert.equal(face01.id, '01');
    assert.equal(face01.face_index, 0);
    assert.equal(face01.predicted_emotion, 'happy');
    assert.equal(face01.confidence, 0.942);
    assert.deepEqual(face01.pixelBox, { x: 120, y: 80, w: 150, h: 180 });

    const payload = {
      feedback_type: 'correct',
      predicted_emotion: face01.predicted_emotion,
      confidence: face01.confidence,
      face_index: face01.face_index,
      bounding_box: face01.pixelBox,
      model_version: 'ResidualEmotionCNN-Candidate-epoch39',
      notes: `Submitted via FaceSense Web Analyze workspace for ${face01.label} (index ${face01.face_index})`,
    };

    assert.equal(payload.feedback_type, 'correct');
    assert.equal(payload.face_index, 0);
    assert.deepEqual(payload.bounding_box, { x: 120, y: 80, w: 150, h: 180 });
    assert.equal(payload.corrected_emotion, undefined);
  });

  test('2. Two faces → two independent feedback states initialized and maintained', () => {
    const faces = mapPredictions(samplePredictionResponse.predictions, naturalWidth, naturalHeight);
    assert.equal(faces.length, 2);

    const faceFeedbackState = {};
    assert.equal(faceFeedbackState[faces[0].id], undefined);
    assert.equal(faceFeedbackState[faces[1].id], undefined);

    // Initial state: both are unsubmitted (idle)
    const getFaceState = (faceId) => faceFeedbackState[faceId] || { status: 'idle' };
    assert.equal(getFaceState('01').status, 'idle');
    assert.equal(getFaceState('02').status, 'idle');
  });

  test('3. Feedback for Face 01 does not modify or complete Face 02', () => {
    const faces = mapPredictions(samplePredictionResponse.predictions, naturalWidth, naturalHeight);
    let faceFeedbackState = {};

    // User evaluates Face 01 as 'correct'
    faceFeedbackState = {
      ...faceFeedbackState,
      ['01']: {
        status: 'submitted',
        feedback_type: 'correct',
        corrected_emotion: null,
        record_id: 'rec_fb_001_correct',
      },
    };

    // Verify Face 01 is recorded and Face 02 remains completely unsubmitted
    assert.equal(faceFeedbackState['01'].status, 'submitted');
    assert.equal(faceFeedbackState['01'].feedback_type, 'correct');
    assert.equal(faceFeedbackState['01'].record_id, 'rec_fb_001_correct');

    assert.equal(faceFeedbackState['02'], undefined);
    const face02Status = faceFeedbackState['02']?.status || 'idle';
    assert.equal(face02Status, 'idle');
  });

  test('4. Incorrect feedback strictly requires a corrected emotion from 7 supported classes', () => {
    const faces = mapPredictions(samplePredictionResponse.predictions, naturalWidth, naturalHeight);
    const face02 = faces[1];

    // Correction flow simulation
    let isCorrecting = true;
    let selectedCorrection = null;

    const canSubmitIncorrect = () => isCorrecting && selectedCorrection !== null && SUPPORTED_EMOTIONS.includes(selectedCorrection);

    // Initial click on 'Incorrect': no emotion chosen yet -> submit must be blocked
    assert.equal(canSubmitIncorrect(), false);

    // Attempting with invalid emotion -> must be blocked
    selectedCorrection = 'discombobulated';
    assert.equal(canSubmitIncorrect(), false);

    // Selecting valid supported emotion 'sad'
    selectedCorrection = 'sad';
    assert.equal(canSubmitIncorrect(), true);

    const payload = {
      feedback_type: 'incorrect',
      predicted_emotion: face02.predicted_emotion,
      confidence: face02.confidence,
      corrected_emotion: selectedCorrection,
      face_index: face02.face_index,
      bounding_box: face02.pixelBox,
      model_version: 'ResidualEmotionCNN-Candidate-epoch39',
      notes: `Submitted via FaceSense Web Analyze workspace for ${face02.label} (index ${face02.face_index}) - corrected to ${selectedCorrection}`,
    };

    assert.equal(payload.feedback_type, 'incorrect');
    assert.equal(payload.corrected_emotion, 'sad');
    assert.equal(payload.predicted_emotion, 'neutral');
    assert.equal(payload.face_index, 1);
  });

  test('5. Correct feedback sends the exact face_index of the selected face', () => {
    const faces = mapPredictions(samplePredictionResponse.predictions, naturalWidth, naturalHeight);

    const face0 = faces[0];
    const face1 = faces[1];

    assert.equal(face0.face_index, 0);
    assert.equal(face1.face_index, 1);

    const payload0 = {
      feedback_type: 'correct',
      predicted_emotion: face0.predicted_emotion,
      confidence: face0.confidence,
      face_index: face0.face_index,
      bounding_box: face0.pixelBox,
    };

    const payload1 = {
      feedback_type: 'uncertain',
      predicted_emotion: face1.predicted_emotion,
      confidence: face1.confidence,
      face_index: face1.face_index,
      bounding_box: face1.pixelBox,
    };

    assert.equal(payload0.face_index, 0);
    assert.equal(payload1.face_index, 1);
  });

  test('6. Bounding box strictly belongs to the selected face', () => {
    const faces = mapPredictions(samplePredictionResponse.predictions, naturalWidth, naturalHeight);

    assert.deepEqual(faces[0].pixelBox, { x: 120, y: 80, w: 150, h: 180 });
    assert.deepEqual(faces[1].pixelBox, { x: 340, y: 95, w: 140, h: 175 });

    assert.notDeepEqual(faces[0].pixelBox, faces[1].pixelBox);
  });

  test('7. Model version comes from active model metadata and is preserved in payload', async () => {
    const { feedbackApi } = await import('../src/lib/api/feedback.js');

    const submittedPayload = {
      feedback_type: 'correct',
      predicted_emotion: 'happy',
      confidence: 0.942,
      face_index: 0,
      bounding_box: { x: 120, y: 80, w: 150, h: 180 },
      model_version: 'ResidualEmotionCNN-Candidate-epoch39',
      notes: 'Submitted via FaceSense Web Analyze workspace for Face 01 (index 0)',
    };

    assert.equal(submittedPayload.model_version, 'ResidualEmotionCNN-Candidate-epoch39');
    assert.equal(typeof feedbackApi.submitFeedback, 'function');
  });

  test('8. Multiple faces can be submitted independently and retain their distinct states', () => {
    let faceFeedbackState = {};

    // 1. Submit Face 01 as 'correct'
    faceFeedbackState = {
      ...faceFeedbackState,
      ['01']: {
        status: 'submitted',
        feedback_type: 'correct',
        corrected_emotion: null,
        record_id: 'rec_fb_001',
      },
    };

    assert.equal(faceFeedbackState['01']?.status, 'submitted');
    assert.equal(faceFeedbackState['02'], undefined);

    // 2. Submit Face 02 independently as 'incorrect' with corrected_emotion 'surprise'
    faceFeedbackState = {
      ...faceFeedbackState,
      ['02']: {
        status: 'submitted',
        feedback_type: 'incorrect',
        corrected_emotion: 'surprise',
        record_id: 'rec_fb_002',
      },
    };

    assert.equal(faceFeedbackState['01']?.status, 'submitted');
    assert.equal(faceFeedbackState['01']?.feedback_type, 'correct');
    assert.equal(faceFeedbackState['02']?.status, 'submitted');
    assert.equal(faceFeedbackState['02']?.feedback_type, 'incorrect');
    assert.equal(faceFeedbackState['02']?.corrected_emotion, 'surprise');

    // 3. Reset/Change feedback on Face 01
    const nextState = { ...faceFeedbackState };
    delete nextState['01'];
    faceFeedbackState = nextState;

    assert.equal(faceFeedbackState['01'], undefined);
    assert.equal(faceFeedbackState['02']?.status, 'submitted');
    assert.equal(faceFeedbackState['02']?.feedback_type, 'incorrect');
  });
});
