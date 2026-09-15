import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { normalizeBoundingBox, cropFaceAsBase64 } from '../src/utils/faceCrop.js';

describe('Face Crop & Multi-Face Feedback Tests', () => {
  test('normalizeBoundingBox handles object with x, y, w, h', () => {
    const box = { x: 120, y: 85, w: 200, h: 220 };
    const normalized = normalizeBoundingBox(box);
    assert.deepEqual(normalized, [120, 85, 200, 220]);
  });

  test('normalizeBoundingBox handles array format [x, y, w, h]', () => {
    const box = [50, 60, 100, 100];
    const normalized = normalizeBoundingBox(box);
    assert.deepEqual(normalized, [50, 60, 100, 100]);
  });

  test('normalizeBoundingBox handles object with left, top, width, height fallbacks', () => {
    const box = { left: 45, top: 30, width: 80, height: 95 };
    const normalized = normalizeBoundingBox(box);
    assert.deepEqual(normalized, [45, 30, 80, 95]);
  });

  test('normalizeBoundingBox rejects invalid or zero dimensions', () => {
    assert.equal(normalizeBoundingBox(null), null);
    assert.equal(normalizeBoundingBox(undefined), null);
    assert.equal(normalizeBoundingBox({ x: 10, y: 20, w: 0, h: 50 }), null);
    assert.equal(normalizeBoundingBox({ x: 10, y: 20, w: 50, h: -5 }), null);
    assert.equal(normalizeBoundingBox([10, 20]), null);
  });

  test('cropFaceAsBase64 returns null safely when image source is null or empty', async () => {
    const crop = await cropFaceAsBase64(null, { x: 10, y: 10, w: 50, h: 50 });
    assert.equal(crop, null);

    const cropEmpty = await cropFaceAsBase64('', { x: 10, y: 10, w: 50, h: 50 });
    assert.equal(cropEmpty, null);
  });

  test('cropFaceAsBase64 returns null safely when box is invalid or missing', async () => {
    const crop = await cropFaceAsBase64('data:image/jpeg;base64,...', null);
    assert.equal(crop, null);

    const cropZero = await cropFaceAsBase64('data:image/jpeg;base64,...', { x: 0, y: 0, w: 0, h: 0 });
    assert.equal(cropZero, null);
  });

  test('Multi-face independence: distinct faces have distinct bounding boxes and indexes', () => {
    const face1 = {
      id: '01',
      label: 'Face 01',
      face_index: 0,
      pixelBox: { x: 50, y: 100, w: 150, h: 150 },
    };
    const face2 = {
      id: '02',
      label: 'Face 02',
      face_index: 1,
      pixelBox: { x: 300, y: 120, w: 180, h: 180 },
    };

    const norm1 = normalizeBoundingBox(face1.pixelBox);
    const norm2 = normalizeBoundingBox(face2.pixelBox);

    assert.deepEqual(norm1, [50, 100, 150, 150]);
    assert.deepEqual(norm2, [300, 120, 180, 180]);
    assert.notDeepEqual(norm1, norm2);
    assert.equal(face1.face_index, 0);
    assert.equal(face2.face_index, 1);
  });
});
