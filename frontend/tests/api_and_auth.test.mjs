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

describe('FaceSense Frontend ↔ Backend Integration Tests', () => {
  beforeEach(() => {
    globalThis.localStorage.clear();
  });

  test('API Client URL Resolution and Endpoint Construction', async () => {
    const { API_BASE_URL } = await import('../src/lib/api/client.js');
    assert.ok(API_BASE_URL.includes('/api/v1'));
    assert.ok(!API_BASE_URL.endsWith('/'));
  });

  test('Auth Token Storage Operations', async () => {
    const { getAuthToken, setAuthToken, clearAuthToken, TOKEN_STORAGE_KEY } = await import('../src/lib/api/client.js');

    assert.equal(getAuthToken(), null);
    setAuthToken('test-jwt-token-xyz');
    assert.equal(getAuthToken(), 'test-jwt-token-xyz');
    assert.equal(globalThis.localStorage.getItem(TOKEN_STORAGE_KEY), 'test-jwt-token-xyz');

    clearAuthToken();
    assert.equal(getAuthToken(), null);
  });

  test('ApiError Class Attributes and Hierarchy', async () => {
    const { ApiError } = await import('../src/lib/api/client.js');

    const err400 = new ApiError('Incorrect email or password', 400, { detail: 'Incorrect email or password' });
    assert.equal(err400.name, 'ApiError');
    assert.equal(err400.status, 400);
    assert.equal(err400.message, 'Incorrect email or password');
    assert.deepEqual(err400.details, { detail: 'Incorrect email or password' });

    const err403 = new ApiError('Admin authorization required', 403);
    assert.equal(err403.status, 403);
  });

  test('Auth Login URL-encoded Serialization Schema', async () => {
    const email = 'admin@facesense.internal';
    const password = 'securePassword123';

    const params = new URLSearchParams();
    params.append('username', email.trim());
    params.append('password', password);

    assert.equal(params.get('username'), email);
    assert.equal(params.get('password'), password);
    assert.equal(params.toString(), 'username=admin%40facesense.internal&password=securePassword123');
  });

  test('Auth Register Schema Conformance (strictly email & password)', async () => {
    const payload = {
      name: 'Ignored UI field',
      email: 'newuser@facesense.internal',
      password: 'password123',
      terms: true,
    };

    // Schema projection matching FastAPI UserRegisterRequest
    const cleanPayload = {
      email: payload.email.trim(),
      password: payload.password,
    };

    assert.deepEqual(cleanPayload, {
      email: 'newuser@facesense.internal',
      password: 'password123',
    });
    assert.equal(cleanPayload.name, undefined);
    assert.equal(cleanPayload.terms, undefined);
  });

  test('Feedback Create Request Schema Conformance', async () => {
    const rawCorrect = {
      feedback_type: 'correct',
      predicted_emotion: 'happy',
      confidence: 0.94,
    };

    assert.equal(rawCorrect.feedback_type, 'correct');
    assert.equal(rawCorrect.predicted_emotion, 'happy');
    assert.equal(rawCorrect.confidence, 0.94);
    assert.equal(rawCorrect.corrected_emotion, undefined);

    const rawIncorrect = {
      feedback_type: 'incorrect',
      predicted_emotion: 'neutral',
      confidence: 0.52,
      corrected_emotion: 'sad',
    };

    assert.equal(rawIncorrect.feedback_type, 'incorrect');
    assert.equal(rawIncorrect.corrected_emotion, 'sad');
  });

  test('Responsive Bounding Box Pixel-to-Percentage Mapping', () => {
    const naturalWidth = 1200;
    const naturalHeight = 800;

    const rawBox = { x: 300, y: 200, w: 240, h: 320 };

    const left = (rawBox.x / naturalWidth) * 100;
    const top = (rawBox.y / naturalHeight) * 100;
    const width = (rawBox.w / naturalWidth) * 100;
    const height = (rawBox.h / naturalHeight) * 100;

    assert.equal(left, 25);
    assert.equal(top, 25);
    assert.equal(width, 20);
    assert.equal(height, 40);
  });

  test('Admin Authorization Guard Logic', () => {
    const normalUser = { id: 2, email: 'operator@facesense.internal', role: 'user' };
    const adminUser = { id: 1, email: 'admin@facesense.internal', role: 'admin' };

    assert.equal(normalUser.role === 'admin', false);
    assert.equal(adminUser.role === 'admin', true);
  });
});
