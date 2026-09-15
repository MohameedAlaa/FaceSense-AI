import cv2
import numpy as np
from fastapi import UploadFile, HTTPException

from backend.app.core.config import settings


async def validate_image_file(file: UploadFile, max_bytes: int = None) -> bytes:
    from backend.app.services.prediction_service import ImageDecodeError
    if max_bytes is None:
        max_bytes = settings.MAX_IMAGE_UPLOAD_MB * 1024 * 1024

    if file.content_type not in settings.ALLOWED_IMAGE_MIMES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    file_bytes = await file.read()
    
    if len(file_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File too large. Max size is {settings.MAX_IMAGE_UPLOAD_MB}MB")

    # Basic magic byte check for common image formats
    if not (
        file_bytes.startswith(b'\xff\xd8\xff') or  # JPEG
        file_bytes.startswith(b'\x89PNG\r\n\x1a\n') or  # PNG
        file_bytes[0:4] == b'RIFF' and file_bytes[8:12] == b'WEBP'  # WEBP
    ):
        raise HTTPException(status_code=400, detail="Invalid image content (magic bytes mismatch)")

    # Attempt to decode it using OpenCV to ensure it's a valid, non-corrupted image
    np_img = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img_bgr is None or img_bgr.size == 0:
        raise ImageDecodeError("Failed to decode uploaded image. File may be corrupted.")

    return file_bytes


def validate_image_bytes(image_bytes: bytes, max_bytes: int = None) -> np.ndarray:
    """
    Validates raw image bytes:
    1. Checks that length does not exceed max_bytes (if provided).
    2. Verifies magic bytes for supported formats (JPEG, PNG, WebP).
    3. Decodes using OpenCV to ensure data is a valid, non-corrupted image.
    Returns decoded BGR image as a numpy ndarray.
    Raises ValueError on validation failure.
    """
    if max_bytes is not None and len(image_bytes) > max_bytes:
        raise ValueError(f"Image payload too large. Maximum allowed size is {max_bytes} bytes.")

    if not (
        image_bytes.startswith(b'\xff\xd8\xff') or  # JPEG
        image_bytes.startswith(b'\x89PNG\r\n\x1a\n') or  # PNG
        (len(image_bytes) >= 12 and image_bytes[0:4] == b'RIFF' and image_bytes[8:12] == b'WEBP')  # WEBP
    ):
        raise ValueError("Invalid image content: not a supported JPEG, PNG, or WebP format.")

    np_img = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Failed to decode image data: file may be corrupted or truncated.")

    return img_bgr
