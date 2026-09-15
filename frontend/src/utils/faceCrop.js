/**
 * Face Crop Utility
 * Extracts a facial bounding box from a source image and encodes it as a JPEG base64 Data URL.
 * Works seamlessly with Camera captures and Uploaded images.
 */

/**
 * Normalizes bounding box coordinates into [x, y, w, h] integers.
 * @param {Object|Array} box - Bounding box in {x, y, w, h} or [x, y, w, h] format.
 * @returns {[number, number, number, number]|null}
 */
export function normalizeBoundingBox(box) {
  if (!box || typeof box !== 'object') return null;

  let x = 0, y = 0, w = 0, h = 0;
  if (Array.isArray(box)) {
    if (box.length < 4) return null;
    [x, y, w, h] = box;
  } else {
    x = box.x ?? box.left ?? 0;
    y = box.y ?? box.top ?? 0;
    w = box.w ?? box.width ?? 0;
    h = box.h ?? box.height ?? 0;
  }

  const ix = Math.round(Number(x) || 0);
  const iy = Math.round(Number(y) || 0);
  const iw = Math.round(Number(w) || 0);
  const ih = Math.round(Number(h) || 0);

  if (iw <= 0 || ih <= 0 || ix < 0 || iy < 0) return null;
  return [ix, iy, iw, ih];
}

/**
 * Loads an image from a Data URL, Object URL, or File.
 * @param {string|File|Blob|HTMLImageElement} source
 * @returns {Promise<HTMLImageElement>}
 */
function loadImageElement(source) {
  return new Promise((resolve, reject) => {
    if (!source) {
      reject(new Error('No image source provided'));
      return;
    }

    if (typeof HTMLImageElement !== 'undefined' && source instanceof HTMLImageElement) {
      if (source.complete && source.naturalWidth > 0) {
        resolve(source);
        return;
      }
    }

    let srcUrl = source;
    let revokeNeeded = false;

    if (typeof Blob !== 'undefined' && source instanceof Blob) {
      srcUrl = URL.createObjectURL(source);
      revokeNeeded = true;
    }

    const img = new Image();
    img.crossOrigin = 'anonymous';

    img.onload = () => {
      if (revokeNeeded && typeof URL !== 'undefined' && URL.revokeObjectURL) {
        URL.revokeObjectURL(srcUrl);
      }
      resolve(img);
    };

    img.onerror = (err) => {
      if (revokeNeeded && typeof URL !== 'undefined' && URL.revokeObjectURL) {
        URL.revokeObjectURL(srcUrl);
      }
      reject(err || new Error('Failed to load image element'));
    };

    img.src = srcUrl;
  });
}

/**
 * Crops a face from an image source using integer pixel coordinates.
 * Returns the cropped face as a JPEG Base64 Data URL.
 * 
 * @param {string|File|Blob|HTMLImageElement} imageSource - Source image (Data URL, Object URL, File)
 * @param {Object|Array} box - Bounding box {x, y, w, h} or [x, y, w, h] in pixels
 * @param {number} [quality=0.90] - JPEG compression quality (0.0 to 1.0)
 * @returns {Promise<string|null>} - Base64 Data URL (e.g. "data:image/jpeg;base64,...") or null if failed
 */
export async function cropFaceAsBase64(imageSource, box, quality = 0.90) {
  const normBox = normalizeBoundingBox(box);
  if (!normBox || !imageSource) return null;

  const [x, y, w, h] = normBox;

  try {
    const img = await loadImageElement(imageSource);
    const naturalWidth = img.naturalWidth || img.width || 0;
    const naturalHeight = img.naturalHeight || img.height || 0;

    if (naturalWidth <= 0 || naturalHeight <= 0) return null;

    // Clamp crop rectangle strictly within source image dimensions
    const sx = Math.max(0, Math.min(x, naturalWidth - 1));
    const sy = Math.max(0, Math.min(y, naturalHeight - 1));
    const sw = Math.min(w, naturalWidth - sx);
    const sh = Math.min(h, naturalHeight - sy);

    if (sw <= 0 || sh <= 0) return null;

    if (typeof document === 'undefined') return null;

    const canvas = document.createElement('canvas');
    canvas.width = sw;
    canvas.height = sh;
    const ctx = canvas.getContext('2d');

    if (!ctx) return null;

    ctx.drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh);
    return canvas.toDataURL('image/jpeg', quality);
  } catch {
    return null;
  }
}
