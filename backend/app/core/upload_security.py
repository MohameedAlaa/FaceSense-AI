import cv2
import numpy as np
from fastapi import UploadFile, HTTPException

from backend.app.core.config import settings
from backend.app.services.prediction_service import ImageDecodeError

async def validate_image_file(file: UploadFile, max_bytes: int = None) -> bytes:
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
