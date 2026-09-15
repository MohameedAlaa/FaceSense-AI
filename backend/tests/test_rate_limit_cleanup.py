"""Tests for in-memory rate limiter stale-key cleanup and memory bounding.

Verifies:
1. Key is created after a request.
2. Key remains while timestamps are still inside the window.
3. Key is removed once all timestamps expire.
4. Multiple expired keys are removed correctly.
5. Active keys are not accidentally removed during cleanup of expired keys.
6. Requests after expiration create a clean new bucket.
7. Rate limit behavior is unchanged (HTTP 429 at threshold).
8. Retry-After header is preserved.
9. Thread safety under concurrent requests.
10. Maximum key capacity bound (max_keys) and deterministic oldest-entry eviction.
11. Existing authentication, prediction, and feedback rate-limiting behavior remains intact.
"""
import time
import threading
import pytest
from starlette.requests import Request
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.core.config import settings
from backend.app.core.rate_limit import RateLimiter, rate_limit_auth, rate_limit_predict, rate_limit_feedback


def _dummy_request(client_ip: str = "127.0.0.1", path: str = "/api/v1/test") -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": [],
        "client": (client_ip, 50000),
        "scheme": "http",
        "server": ("testserver", 80),
    }
    return Request(scope)


class TestRateLimiterMemoryCleanup:
    """Verification of memory cleanup and key lifecycle in RateLimiter."""

    def test_1_key_created_after_request(self):
        """1. Key is created in history after a request."""
        limiter = RateLimiter(times=5, seconds=10)
        req = _dummy_request("10.0.0.1", "/api/v1/predict/image")
        limiter(req)

        assert "10.0.0.1:/api/v1/predict/image" in limiter.history
        assert len(limiter.history["10.0.0.1:/api/v1/predict/image"]) == 1

    def test_2_key_remains_while_inside_window(self):
        """2. Key remains in history while its timestamps are still within the time window."""
        limiter = RateLimiter(times=5, seconds=10)
        req = _dummy_request("10.0.0.2", "/api/v1/feedback")
        limiter(req)

        # Trigger cleanup at t + 2 seconds (well within 10-second window)
        now = limiter.history["10.0.0.2:/api/v1/feedback"][0] + 2.0
        cleaned = limiter.cleanup(now=now)

        assert cleaned == 0
        assert "10.0.0.2:/api/v1/feedback" in limiter.history

    def test_3_key_removed_once_all_timestamps_expire(self):
        """3. Key is completely removed from history once all its timestamps expire."""
        limiter = RateLimiter(times=5, seconds=2)
        req = _dummy_request("10.0.0.3", "/api/v1/auth/login")
        limiter(req)
        assert "10.0.0.3:/api/v1/auth/login" in limiter.history

        # Trigger cleanup after expiration (> 2.0 seconds)
        ts = limiter.history["10.0.0.3:/api/v1/auth/login"][0]
        cleaned = limiter.cleanup(now=ts + 2.1)

        assert cleaned == 1
        assert "10.0.0.3:/api/v1/auth/login" not in limiter.history

    def test_4_multiple_expired_keys_removed_correctly(self):
        """4. Multiple expired keys are pruned in a single cleanup cycle."""
        limiter = RateLimiter(times=5, seconds=2)
        for i in range(5):
            limiter(_dummy_request(f"10.0.1.{i}", "/test"))
        assert len(limiter.history) == 5

        # Expire all keys
        latest_ts = max(ts_list[-1] for ts_list in limiter.history.values())
        cleaned = limiter.cleanup(now=latest_ts + 2.5)

        assert cleaned == 5
        assert len(limiter.history) == 0

    def test_5_active_keys_not_accidentally_removed(self):
        """5. Active keys remain while expired keys are pruned."""
        limiter = RateLimiter(times=5, seconds=5)
        # Old key created at t=100
        limiter.history["expired_ip:/test"] = [100.0]
        # Active key created at t=104
        limiter.history["active_ip:/test"] = [104.0]

        # At t=106: expired_ip (age 6s > 5s) must be pruned, active_ip (age 2s < 5s) must remain
        cleaned = limiter.cleanup(now=106.0)

        assert cleaned == 1
        assert "expired_ip:/test" not in limiter.history
        assert "active_ip:/test" in limiter.history

    def test_6_request_after_expiration_creates_clean_bucket(self):
        """6. A client request after previous window expiration creates a fresh, clean bucket."""
        limiter = RateLimiter(times=2, seconds=1)
        req = _dummy_request("10.0.0.6", "/test")

        limiter(req)
        limiter(req)
        # Should now be throttled
        with pytest.raises(Exception) as exc_info:
            limiter(req)
        assert getattr(exc_info.value, "status_code", None) == 429

        # Sleep past window
        time.sleep(1.1)

        # Next request must succeed and create a fresh bucket with count=1
        limiter(req)
        assert len(limiter.history["10.0.0.6:/test"]) == 1

    def test_7_rate_limit_and_429_behavior_unchanged(self):
        """7 & 8. Rate limit threshold and HTTP 429 status code are preserved."""
        limiter = RateLimiter(times=3, seconds=10)
        req = _dummy_request("10.0.0.7", "/test")

        limiter(req)
        limiter(req)
        limiter(req)

        with pytest.raises(Exception) as exc_info:
            limiter(req)
        exc = exc_info.value
        assert getattr(exc, "status_code", None) == 429
        assert getattr(exc, "detail", None) == "Too Many Requests"

    def test_8_retry_after_header_preserved(self):
        """9. Retry-After response header is preserved with correct window value."""
        limiter = RateLimiter(times=1, seconds=45)
        req = _dummy_request("10.0.0.8", "/test")
        limiter(req)

        with pytest.raises(Exception) as exc_info:
            limiter(req)
        exc = exc_info.value
        assert getattr(exc, "status_code", None) == 429
        assert "Retry-After" in exc.headers
        assert exc.headers["Retry-After"] == "45"

    def test_9_concurrency_thread_safety(self):
        """10. Thread safety is preserved under concurrent requests during cleanup."""
        limiter = RateLimiter(times=20, seconds=5)
        errors = []

        def worker(idx: int):
            try:
                ip = f"172.16.0.{idx % 4}"
                req = _dummy_request(ip, "/concurrent")
                limiter(req)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All 16 threads should complete without unhandled concurrency errors
        assert len(errors) == 0
        assert len(limiter.history) <= 4

    def test_10_max_keys_bounding_and_eviction(self):
        """11. Maximum key capacity bound (max_keys) evicts oldest entries under active flood."""
        # Limiter with hard cap of 3 keys
        limiter = RateLimiter(times=5, seconds=60, max_keys=3)

        # Add 3 distinct active keys
        limiter(_dummy_request("10.0.0.1", "/test"))
        limiter(_dummy_request("10.0.0.2", "/test"))
        limiter(_dummy_request("10.0.0.3", "/test"))
        assert len(limiter.history) == 3

        # Add a 4th key: oldest (10.0.0.1) must be evicted to respect max_keys=3
        limiter(_dummy_request("10.0.0.4", "/test"))
        assert len(limiter.history) == 3
        assert "10.0.0.1:/test" not in limiter.history
        assert "10.0.0.4:/test" in limiter.history

    def test_11_controlled_memory_test(self):
        """Controlled before-and-after demonstration of stale key cleanup."""
        limiter = RateLimiter(times=5, seconds=1)

        # Step 1: Create 4 keys
        for i in range(4):
            limiter(_dummy_request(f"192.168.10.{i}", "/endpoint"))

        # BEFORE cleanup: history contains those keys
        assert len(limiter.history) == 4
        for i in range(4):
            assert f"192.168.10.{i}:/endpoint" in limiter.history

        # Wait for expiration
        time.sleep(1.1)

        # Trigger cleanup
        cleaned_count = limiter.cleanup()

        # AFTER cleanup: expired keys are gone
        assert cleaned_count == 4
        assert len(limiter.history) == 0


class TestEndpointIntegrationWithCleanup:
    """Integration verification that auth/predict/feedback endpoints function normally."""

    def test_auth_and_feedback_endpoints_function_with_cleanup(self, client_with_db):
        """End-to-end endpoint requests correctly execute inline cleanup and succeed."""
        client = TestClient(app)

        # Auth register
        reg = client.post(
            "/api/v1/auth/register",
            json={"email": "cleanup_test_user@example.com", "password": "Password123!"},
        )
        assert reg.status_code == 200

        # Feedback
        fb = client.post(
            "/api/v1/feedback",
            json={"feedback_type": "correct", "predicted_emotion": "happy", "confidence": 0.9},
        )
        assert fb.status_code == 201

        # Check rate_limit_auth and rate_limit_feedback instances have keys tracked
        assert len(rate_limit_auth.history) >= 1
        assert len(rate_limit_feedback.history) >= 1
