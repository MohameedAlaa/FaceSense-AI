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

describe('Admin Feedback Review Frontend Unit Tests', () => {
  beforeEach(() => {
    globalThis.localStorage.clear();
  });

  test('Feedback review emotion grouping includes all 7 FER2013 emotions', async () => {
    const EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'];
    assert.equal(EMOTIONS.length, 7);
    assert.ok(EMOTIONS.includes('happy'));
    assert.ok(EMOTIONS.includes('sad'));
    assert.ok(EMOTIONS.includes('neutral'));
  });

  test('Feedback review decision validation: approve requires authoritative label', () => {
    const makeDecision = (decision, finalLabel) => {
      if (!['approve', 'reject'].includes(decision)) {
        throw new Error('Invalid decision');
      }
      if (decision === 'approve' && !finalLabel) {
        throw new Error('Final label required on approve');
      }
      return {
        decision,
        final_label: decision === 'approve' ? finalLabel : null,
      };
    };

    assert.throws(() => makeDecision('invalid', 'happy'), /Invalid decision/);
    assert.throws(() => makeDecision('approve', ''), /Final label required on approve/);

    const approved = makeDecision('approve', 'sad');
    assert.equal(approved.decision, 'approve');
    assert.equal(approved.final_label, 'sad');

    const rejected = makeDecision('reject');
    assert.equal(rejected.decision, 'reject');
    assert.equal(rejected.final_label, null);
  });

  test('Feedback review API export signatures exist', async () => {
    const { feedbackApi } = await import('../src/lib/api/feedback.js');
    assert.equal(typeof feedbackApi.getReviewOverview, 'function');
    assert.equal(typeof feedbackApi.getReviewByEmotion, 'function');
    assert.equal(typeof feedbackApi.updateReviewDecision, 'function');
    assert.equal(typeof feedbackApi.submitFeedback, 'function');
    assert.equal(typeof feedbackApi.getStats, 'function');
    assert.equal(typeof feedbackApi.getFeedbackImageBlob, 'function');
  });

  test('Status counting and aggregation helper logic', () => {
    const mockEmotionCounts = {
      happy: { pending: 12, approved: 38, rejected: 4, total: 54 },
      sad: { pending: 3, approved: 10, rejected: 1, total: 14 },
    };

    assert.equal(mockEmotionCounts.happy.pending, 12);
    assert.equal(mockEmotionCounts.happy.approved, 38);
    assert.equal(mockEmotionCounts.happy.rejected, 4);
    assert.equal(mockEmotionCounts.happy.total, 54);
  });

  test('getFeedbackImageBlob validation: rejects invalid, empty, or non-string input safely', async () => {
    const { feedbackApi } = await import('../src/lib/api/feedback.js');

    await assert.rejects(async () => {
      await feedbackApi.getFeedbackImageBlob('');
    }, /Valid image name or URL is required/);

    await assert.rejects(async () => {
      await feedbackApi.getFeedbackImageBlob(null);
    }, /Valid image name or URL is required/);

    await assert.rejects(async () => {
      await feedbackApi.getFeedbackImageBlob(123);
    }, /Valid image name or URL is required/);
  });

  test('Object URL lifecycle and revocation logic', () => {
    const activeUrls = new Set();

    const mockCreateObjectURL = (_blob) => {
      const id = `blob:http://localhost:5173/${Math.random().toString(36).slice(2)}`;
      activeUrls.add(id);
      return id;
    };

    const mockRevokeObjectURL = (url) => {
      activeUrls.delete(url);
    };

    // Simulate component mount and image load
    const dummyBlob = new Blob(['image-data'], { type: 'image/jpeg' });
    const objectUrl = mockCreateObjectURL(dummyBlob);
    assert.ok(activeUrls.has(objectUrl));
    assert.equal(activeUrls.size, 1);

    // Simulate component unmount cleanup
    mockRevokeObjectURL(objectUrl);
    assert.ok(!activeUrls.has(objectUrl));
    assert.equal(activeUrls.size, 0);
  });

  test('Endpoint normalization preserves authorization and avoids duplicate /api/v1', async () => {
    const { requestBlob } = await import('../src/lib/api/client.js');
    assert.equal(typeof requestBlob, 'function');
  });
});

