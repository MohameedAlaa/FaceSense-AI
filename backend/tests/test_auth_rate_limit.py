"""Targeted security tests for authentication rate limiting.

Verifies:
1. POST /api/v1/auth/login is throttled at RATE_LIMIT_AUTH (10/min) with Retry-After header.
2. POST /api/v1/auth/register is throttled at RATE_LIMIT_AUTH (10/min) with Retry-After header.
3. POST /api/v1/auth/admin-bootstrap is throttled at RATE_LIMIT_AUTH (10/min) with Retry-After header.
4. Endpoint isolation: exhausting auth limit does not exhaust predict or feedback limits.
5. IP isolation: independent client IPs receive independent rate-limit quotas.
6. RATE_LIMIT_ENABLED=False disables rate limiting across auth endpoints.
7. Existing authentication, registration, and bootstrap semantics remain fully intact.
"""
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.core.config import settings
from backend.app.core.rate_limit import parse_rate_limit, rate_limit_auth, rate_limit_predict, rate_limit_feedback


LOGIN_URL = "/api/v1/auth/login"
REGISTER_URL = "/api/v1/auth/register"
BOOTSTRAP_URL = "/api/v1/auth/admin-bootstrap"
PREDICT_URL = "/api/v1/predict/image"
FEEDBACK_URL = "/api/v1/feedback"


def _make_client(ip: str = "127.0.0.1") -> TestClient:
    return TestClient(app, client=(ip, 50000))


class TestAuthRateLimiting:
    """Security verification of authentication rate limiting."""

    def test_login_rate_limiting(self, client_with_db):
        """Login requests under limit are allowed; request exceeding limit returns HTTP 429."""
        limit_times, limit_seconds = parse_rate_limit(settings.RATE_LIMIT_AUTH)
        assert limit_times == 10

        # Send limit_times (10) failed login attempts -> all return 400 (not 429)
        for i in range(limit_times):
            resp = client_with_db.post(
                LOGIN_URL,
                data={"username": f"test_user_{i}@example.com", "password": "WrongPassword123!"},
            )
            assert resp.status_code == 400
            assert resp.json()["detail"] == "Incorrect email or password"

        # The (limit_times + 1)th attempt must be throttled with 429
        blocked_resp = client_with_db.post(
            LOGIN_URL,
            data={"username": "test_user_exceeded@example.com", "password": "WrongPassword123!"},
        )
        assert blocked_resp.status_code == 429
        assert blocked_resp.json()["detail"] == "Too Many Requests"
        assert "Retry-After" in blocked_resp.headers
        assert blocked_resp.headers["Retry-After"] == str(limit_seconds)

    def test_register_rate_limiting(self, client_with_db):
        """Registration requests under limit succeed; request exceeding limit returns HTTP 429."""
        limit_times, limit_seconds = parse_rate_limit(settings.RATE_LIMIT_AUTH)

        # Send limit_times (10) registration attempts -> all return 200
        for i in range(limit_times):
            resp = client_with_db.post(
                REGISTER_URL,
                json={"email": f"ratelimited_reg_{i}@example.com", "password": "SecurePassword123!"},
            )
            assert resp.status_code == 200
            assert resp.json()["role"] == "user"

        # The 11th registration attempt from the same client must be rejected with 429
        blocked_resp = client_with_db.post(
            REGISTER_URL,
            json={"email": "ratelimited_reg_overflow@example.com", "password": "SecurePassword123!"},
        )
        assert blocked_resp.status_code == 429
        assert blocked_resp.json()["detail"] == "Too Many Requests"
        assert "Retry-After" in blocked_resp.headers
        assert blocked_resp.headers["Retry-After"] == str(limit_seconds)

    def test_admin_bootstrap_rate_limiting(self, client_with_db, monkeypatch):
        """Admin bootstrap requests are throttled at RATE_LIMIT_AUTH."""
        limit_times, limit_seconds = parse_rate_limit(settings.RATE_LIMIT_AUTH)
        monkeypatch.setattr(settings, "ADMIN_BOOTSTRAP_KEY", "test-key-12345")

        # 10 attempts with invalid key -> return 403 Forbidden
        for i in range(limit_times):
            resp = client_with_db.post(
                BOOTSTRAP_URL,
                json={
                    "email": f"admin_probe_{i}@example.com",
                    "password": "AdminPassword123!",
                    "bootstrap_key": "wrong-key",
                },
            )
            assert resp.status_code == 403

        # 11th attempt must return 429 Too Many Requests
        blocked_resp = client_with_db.post(
            BOOTSTRAP_URL,
            json={
                "email": "admin_probe_overflow@example.com",
                "password": "AdminPassword123!",
                "bootstrap_key": "wrong-key",
            },
        )
        assert blocked_resp.status_code == 429
        assert blocked_resp.json()["detail"] == "Too Many Requests"
        assert "Retry-After" in blocked_resp.headers

    def test_endpoint_isolation(self, client_with_db):
        """Exhausting auth rate limit does not block predict or feedback endpoints."""
        limit_times, _ = parse_rate_limit(settings.RATE_LIMIT_AUTH)

        # Exhaust auth limit on /api/v1/auth/login
        for _ in range(limit_times):
            client_with_db.post(
                LOGIN_URL,
                data={"username": "user@example.com", "password": "bad"},
            )
        assert client_with_db.post(LOGIN_URL, data={"username": "u@e.com", "password": "p"}).status_code == 429

        # Feedback endpoint must still accept requests
        fb_resp = client_with_db.post(
            FEEDBACK_URL,
            json={"feedback_type": "correct", "predicted_emotion": "happy", "confidence": 0.9},
        )
        assert fb_resp.status_code != 429

        # Predict endpoint must still accept requests (e.g. 422 for missing image, not 429)
        pred_resp = client_with_db.post(PREDICT_URL)
        assert pred_resp.status_code != 429

    def test_ip_isolation(self, client_with_db):
        """Exhausting auth rate limit for IP A does not affect IP B."""
        limit_times, _ = parse_rate_limit(settings.RATE_LIMIT_AUTH)
        client_a = _make_client(ip="192.168.1.101")
        client_b = _make_client(ip="192.168.1.102")

        # Exhaust limit for Client A
        for _ in range(limit_times):
            client_a.post(LOGIN_URL, data={"username": "a@example.com", "password": "wrong"})
        assert client_a.post(LOGIN_URL, data={"username": "a@example.com", "password": "wrong"}).status_code == 429

        # Client B must NOT be blocked
        client_b_resp = client_b.post(LOGIN_URL, data={"username": "b@example.com", "password": "wrong"})
        assert client_b_resp.status_code == 400
        assert client_b_resp.json()["detail"] == "Incorrect email or password"

    def test_rate_limit_disabled_flag(self, client_with_db, monkeypatch):
        """When RATE_LIMIT_ENABLED is False, authentication requests are not throttled."""
        monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", False)
        limit_times, _ = parse_rate_limit(settings.RATE_LIMIT_AUTH)

        # Send limit_times + 5 requests
        for _ in range(limit_times + 5):
            resp = client_with_db.post(
                LOGIN_URL,
                data={"username": "bypass@example.com", "password": "wrong"},
            )
            assert resp.status_code == 400
            assert resp.status_code != 429

    def test_existing_authentication_behavior_preserved(self, client_with_db):
        """Existing login and registration semantics remain completely intact under limit."""
        email = "legit_user@example.com"
        password = "LegitPassword123!"

        # 1. Normal registration succeeds
        reg_resp = client_with_db.post(
            REGISTER_URL,
            json={"email": email, "password": password},
        )
        assert reg_resp.status_code == 200
        assert reg_resp.json()["email"] == email

        # 2. Duplicate registration returns generic 400
        dup_resp = client_with_db.post(
            REGISTER_URL,
            json={"email": email, "password": password},
        )
        assert dup_resp.status_code == 400
        assert dup_resp.json()["detail"] == "Registration failed. Please check the provided information and try again."

        # 3. Valid login succeeds
        login_resp = client_with_db.post(
            LOGIN_URL,
            data={"username": email, "password": password},
        )
        assert login_resp.status_code == 200
        token_data = login_resp.json()
        assert "access_token" in token_data
        assert token_data["role"] == "user"
