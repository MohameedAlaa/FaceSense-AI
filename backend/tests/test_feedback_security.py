"""Feedback payload size and validation security regression tests.

Verifies:
1. notes within allowed limit (<= 500 chars) is accepted.
2. notes over limit (> 500 chars) is rejected with 422.
3. empty notes is accepted.
4. valid image_base64 within size limit is accepted.
5. oversized image_base64 is rejected.
6. invalid Base64 format is rejected with 422.
7. Base64 encoding of plain text / non-image is rejected with 422.
8. corrupted / truncated image bytes is rejected with 422.
9. valid JPEG is accepted.
10. valid PNG is accepted.
11. extremely large HTTP payload is safely rejected (413).
12. validation failures never write to the database or JSONL feedback log.
13. existing valid feedback requests continue working.
"""
import base64
import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from backend.app.schemas.feedback import (
    FeedbackCreateRequest,
    MAX_NOTES_LENGTH,
    MAX_FEEDBACK_IMAGE_BYTES,
    MAX_FEEDBACK_BASE64_LEN,
)
from backend.app.models.feedback import FeedbackRecord
from sqlalchemy import select


# ==============================================================================
# Helper functions for generating test assets
# ==============================================================================

def _create_test_jpeg_b64(width: int = 48, height: int = 48) -> str:
    """Create a minimal valid JPEG image encoded as Base64."""
    img = np.full((height, width, 3), 120, dtype=np.uint8)
    cv2.circle(img, (width // 2, height // 2), min(width, height) // 4, (255, 255, 255), -1)
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _create_test_png_b64(width: int = 48, height: int = 48) -> str:
    """Create a minimal valid PNG image encoded as Base64."""
    img = np.full((height, width, 3), 180, dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _create_corrupted_jpeg_b64() -> str:
    """Create a base64 string that has JPEG magic bytes but corrupted/truncated image data."""
    corrupted_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00corrupted_payload_data"
    return base64.b64encode(corrupted_bytes).decode("utf-8")


# ==============================================================================
# 1. Notes Validation Tests
# ==============================================================================

class TestNotesValidation:
    """Regression tests for notes length enforcement."""

    def test_notes_within_limit_accepted(self, client, isolated_feedback_service):
        """Notes up to 500 characters are accepted."""
        notes = "A" * MAX_NOTES_LENGTH
        response = client.post(
            "/api/v1/feedback",
            json={
                "feedback_type": "correct",
                "predicted_emotion": "happy",
                "confidence": 0.90,
                "notes": notes,
            },
        )
        assert response.status_code == 201
        assert response.json()["status"] == "success"

    def test_notes_over_limit_rejected(self, client, isolated_feedback_service):
        """Notes exceeding 500 characters are rejected with 422 Unprocessable Entity."""
        notes = "A" * (MAX_NOTES_LENGTH + 1)
        response = client.post(
            "/api/v1/feedback",
            json={
                "feedback_type": "correct",
                "predicted_emotion": "happy",
                "confidence": 0.90,
                "notes": notes,
            },
        )
        assert response.status_code == 422
        errors = response.json().get("detail", [])
        assert any("notes" in str(err) for err in errors)

    def test_notes_empty_string_accepted(self, client, isolated_feedback_service):
        """Empty string notes are accepted without error."""
        response = client.post(
            "/api/v1/feedback",
            json={
                "feedback_type": "correct",
                "predicted_emotion": "happy",
                "confidence": 0.90,
                "notes": "",
            },
        )
        assert response.status_code == 201

    def test_notes_none_accepted(self, client, isolated_feedback_service):
        """None/omitted notes are accepted without error."""
        response = client.post(
            "/api/v1/feedback",
            json={
                "feedback_type": "correct",
                "predicted_emotion": "happy",
                "confidence": 0.90,
            },
        )
        assert response.status_code == 201


# ==============================================================================
# 2. Image Base64 Validation & Format Tests
# ==============================================================================

class TestImageBase64Validation:
    """Regression tests for image_base64 size, encoding, and integrity."""

    def test_valid_jpeg_accepted(self, client, isolated_feedback_service):
        """Valid JPEG image submitted as Base64 is accepted."""
        b64 = _create_test_jpeg_b64(48, 48)
        response = client.post(
            "/api/v1/feedback",
            json={
                "feedback_type": "correct",
                "predicted_emotion": "happy",
                "confidence": 0.95,
                "image_base64": b64,
            },
        )
        assert response.status_code == 201
        assert response.json()["status"] == "success"

    def test_valid_png_accepted(self, client, isolated_feedback_service):
        """Valid PNG image submitted as Base64 is accepted."""
        b64 = _create_test_png_b64(48, 48)
        response = client.post(
            "/api/v1/feedback",
            json={
                "feedback_type": "correct",
                "predicted_emotion": "neutral",
                "confidence": 0.85,
                "image_base64": b64,
            },
        )
        assert response.status_code == 201

    def test_valid_data_uri_jpeg_accepted(self, client, isolated_feedback_service):
        """Valid data URI formatted image is parsed and accepted."""
        b64 = _create_test_jpeg_b64(48, 48)
        data_uri = f"data:image/jpeg;base64,{b64}"
        response = client.post(
            "/api/v1/feedback",
            json={
                "feedback_type": "correct",
                "predicted_emotion": "happy",
                "confidence": 0.92,
                "image_base64": data_uri,
            },
        )
        assert response.status_code == 201

    def test_invalid_base64_string_rejected(self, client, isolated_feedback_service):
        """Malformed Base64 syntax is rejected with 422."""
        response = client.post(
            "/api/v1/feedback",
            json={
                "feedback_type": "correct",
                "predicted_emotion": "happy",
                "confidence": 0.90,
                "image_base64": "not_valid_base64!@#$%",
            },
        )
        assert response.status_code == 422

    def test_plaintext_in_base64_rejected(self, client, isolated_feedback_service):
        """Valid Base64 containing non-image plain text is rejected with 422."""
        text_b64 = base64.b64encode(b"This is plain text and not a JPEG or PNG image.").decode("utf-8")
        response = client.post(
            "/api/v1/feedback",
            json={
                "feedback_type": "correct",
                "predicted_emotion": "happy",
                "confidence": 0.90,
                "image_base64": text_b64,
            },
        )
        assert response.status_code == 422

    def test_corrupted_image_rejected(self, client, isolated_feedback_service):
        """Image with JPEG header but corrupted/truncated payload is rejected with 422."""
        corrupted_b64 = _create_corrupted_jpeg_b64()
        response = client.post(
            "/api/v1/feedback",
            json={
                "feedback_type": "correct",
                "predicted_emotion": "happy",
                "confidence": 0.90,
                "image_base64": corrupted_b64,
            },
        )
        assert response.status_code == 422

    def test_invalid_data_uri_mimetype_rejected(self, client, isolated_feedback_service):
        """Data URI declaring a non-image MIME type is rejected with 422."""
        b64 = _create_test_jpeg_b64(48, 48)
        bad_uri = f"data:text/plain;base64,{b64}"
        response = client.post(
            "/api/v1/feedback",
            json={
                "feedback_type": "correct",
                "predicted_emotion": "happy",
                "confidence": 0.90,
                "image_base64": bad_uri,
            },
        )
        assert response.status_code == 422

    def test_oversized_base64_string_rejected(self, client, isolated_feedback_service):
        """Base64 string exceeding MAX_FEEDBACK_BASE64_LEN is rejected with 422."""
        oversized_b64 = "A" * (MAX_FEEDBACK_BASE64_LEN + 10)
        response = client.post(
            "/api/v1/feedback",
            json={
                "feedback_type": "correct",
                "predicted_emotion": "happy",
                "confidence": 0.90,
                "image_base64": oversized_b64,
            },
        )
        assert response.status_code == 422

    def test_empty_string_image_base64_accepted(self, client, isolated_feedback_service):
        """Empty string image_base64 is treated as absent and succeeds."""
        response = client.post(
            "/api/v1/feedback",
            json={
                "feedback_type": "correct",
                "predicted_emotion": "happy",
                "confidence": 0.90,
                "image_base64": "",
            },
        )
        assert response.status_code == 201


# ==============================================================================
# 3. Request Body Size Protection
# ==============================================================================

class TestRequestBodySizeProtection:
    """Tests the endpoint-level request body guard."""

    def test_extremely_large_payload_rejected(self, client, isolated_feedback_service):
        """Requests declaring Content-Length > 4MB are rejected with 413 Payload Too Large."""
        response = client.post(
            "/api/v1/feedback",
            headers={"Content-Length": "10485760"},  # 10 MB
            content=b"oversized_payload",
        )
        assert response.status_code == 413


# ==============================================================================
# 4. Storage & State Integrity Tests
# ==============================================================================

class TestStorageIntegrityOnValidationFailure:
    """Verifies that failed feedback requests never pollute storage or database."""

    def test_invalid_feedback_does_not_write_db_or_jsonl(
        self, client_with_db, test_db_session, isolated_feedback_service
    ):
        """When validation fails, no record is added to the database or JSONL file."""
        initial_jsonl_count = len(isolated_feedback_service.collector.load_feedback_records())
        initial_db_count = len(test_db_session.scalars(select(FeedbackRecord)).all())

        # Attempt 1: Notes over limit
        client_with_db.post(
            "/api/v1/feedback",
            json={
                "feedback_type": "correct",
                "predicted_emotion": "happy",
                "confidence": 0.90,
                "notes": "X" * 600,
            },
        )

        # Attempt 2: Malformed image
        client_with_db.post(
            "/api/v1/feedback",
            json={
                "feedback_type": "correct",
                "predicted_emotion": "happy",
                "confidence": 0.90,
                "image_base64": "invalid_base64_data",
            },
        )

        # Attempt 3: Non-image Base64
        client_with_db.post(
            "/api/v1/feedback",
            json={
                "feedback_type": "correct",
                "predicted_emotion": "happy",
                "confidence": 0.90,
                "image_base64": base64.b64encode(b"not an image").decode("utf-8"),
            },
        )

        # Assert no writes occurred
        after_jsonl_count = len(isolated_feedback_service.collector.load_feedback_records())
        after_db_count = len(test_db_session.scalars(select(FeedbackRecord)).all())

        assert after_jsonl_count == initial_jsonl_count
        assert after_db_count == initial_db_count


# ==============================================================================
# 5. Schema Direct Unit Tests
# ==============================================================================

class TestFeedbackSchemaUnit:
    """Direct Pydantic unit validation tests."""

    def test_schema_notes_max_length(self):
        """Direct Pydantic instantiation fails on notes > 500 chars."""
        with pytest.raises(ValidationError):
            FeedbackCreateRequest(
                feedback_type="correct",
                predicted_emotion="happy",
                confidence=0.9,
                notes="X" * 501,
            )

    def test_schema_image_format_validation(self):
        """Direct Pydantic instantiation fails on plain text base64."""
        with pytest.raises(ValidationError):
            FeedbackCreateRequest(
                feedback_type="correct",
                predicted_emotion="happy",
                confidence=0.9,
                image_base64=base64.b64encode(b"plain text").decode(),
            )
