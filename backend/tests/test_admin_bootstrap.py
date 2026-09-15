"""Admin bootstrap security regression tests.

Tests the secure admin bootstrap mechanism implemented in POST /api/v1/auth/admin-bootstrap.

Security requirements verified:
- Normal registration always produces role='user' (no automatic first-user admin)
- Missing ADMIN_BOOTSTRAP_KEY -> bootstrap endpoint disabled (403)
- Incorrect bootstrap key -> rejected (403)
- Correct key + empty table -> exactly one admin created (201)
- Second bootstrap attempt (after admin exists) -> rejected (409)
- Normal registration after bootstrap -> role='user'
- Concurrent/duplicate bootstrap cannot create multiple admins
- Admin endpoint authorization still enforced after bootstrap
- The bootstrap key value never appears in error message text
"""
import threading
from typing import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from backend.app.main import app
from backend.app.core.config import settings
from backend.app.db.base import Base
from backend.app.db.session import get_db
from backend.app.models.user import User
from backend.app.core.security import create_access_token
from backend.app.services.auth_service import auth_service, _bootstrap_lock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BOOTSTRAP_URL = "/api/v1/auth/admin-bootstrap"
REGISTER_URL = "/api/v1/auth/register"
LOGIN_URL = "/api/v1/auth/login"

_VALID_BOOTSTRAP_KEY = "test-bootstrap-key-for-unit-tests-only"
_WRONG_BOOTSTRAP_KEY = "definitely-wrong-key"


@pytest.fixture
def bootstrap_db_engine():
    """In-memory SQLite engine for bootstrap tests (isolated per test)."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture
def bootstrap_db_session(bootstrap_db_engine) -> Generator[Session, None, None]:
    """Yield a session bound to the isolated in-memory engine."""
    TestingSession = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=bootstrap_db_engine,
        expire_on_commit=False,
    )
    session = TestingSession()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def bootstrap_client(bootstrap_db_session):
    """TestClient with DB dependency overridden to isolated in-memory DB."""
    def override_get_db():
        try:
            yield bootstrap_db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture(autouse=True)
def set_bootstrap_key(monkeypatch):
    """Ensure tests run with a known bootstrap key unless overridden."""
    monkeypatch.setattr(settings, "ADMIN_BOOTSTRAP_KEY", _VALID_BOOTSTRAP_KEY)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _bootstrap_payload(
    email: str = "admin@example.com",
    password: str = "securepassword123",
    key: str = _VALID_BOOTSTRAP_KEY,
) -> dict:
    return {"email": email, "password": password, "bootstrap_key": key}


def _register_payload(
    email: str = "user@example.com",
    password: str = "securepassword123",
) -> dict:
    return {"email": email, "password": password}


# ---------------------------------------------------------------------------
# 1. Normal registration always produces role='user'
# ---------------------------------------------------------------------------

class TestNormalRegistration:
    def test_first_registration_is_always_user_role(self, bootstrap_client):
        """First registered user must have role='user', never 'admin'."""
        resp = bootstrap_client.post(REGISTER_URL, json=_register_payload())
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["role"] == "user"

    def test_multiple_registrations_are_always_user_role(self, bootstrap_client):
        """All registrations via /register produce role='user' regardless of order."""
        for i in range(3):
            resp = bootstrap_client.post(
                REGISTER_URL, json=_register_payload(email=f"user{i}@example.com")
            )
            assert resp.status_code == 200
            assert resp.json()["role"] == "user"

    def test_duplicate_email_registration_rejected(self, bootstrap_client):
        """Registering the same email twice returns 400."""
        bootstrap_client.post(REGISTER_URL, json=_register_payload())
        resp = bootstrap_client.post(REGISTER_URL, json=_register_payload())
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# 2 & 3. Missing / incorrect bootstrap key -> rejected
# ---------------------------------------------------------------------------

class TestBootstrapKeyRejection:
    def test_missing_bootstrap_key_disabled(self, bootstrap_client, monkeypatch):
        """When ADMIN_BOOTSTRAP_KEY is not set the endpoint returns 403."""
        monkeypatch.setattr(settings, "ADMIN_BOOTSTRAP_KEY", None)
        resp = bootstrap_client.post(BOOTSTRAP_URL, json=_bootstrap_payload())
        assert resp.status_code == 403
        # Must not reveal that the key is 'None' or any internal details
        body = resp.json()["detail"]
        assert "bootstrap" in body.lower() or "not enabled" in body.lower()

    def test_empty_bootstrap_key_disabled(self, bootstrap_client, monkeypatch):
        """Empty string ADMIN_BOOTSTRAP_KEY is treated as not set -> 403."""
        monkeypatch.setattr(settings, "ADMIN_BOOTSTRAP_KEY", "")
        resp = bootstrap_client.post(BOOTSTRAP_URL, json=_bootstrap_payload())
        assert resp.status_code == 403

    def test_wrong_bootstrap_key_rejected(self, bootstrap_client):
        """A request with a wrong key returns 403."""
        resp = bootstrap_client.post(
            BOOTSTRAP_URL, json=_bootstrap_payload(key=_WRONG_BOOTSTRAP_KEY)
        )
        assert resp.status_code == 403

    def test_wrong_key_error_does_not_echo_key(self, bootstrap_client):
        """The 403 response body must not repeat the submitted incorrect key."""
        resp = bootstrap_client.post(
            BOOTSTRAP_URL, json=_bootstrap_payload(key=_WRONG_BOOTSTRAP_KEY)
        )
        assert resp.status_code == 403
        detail = resp.json().get("detail", "")
        assert _WRONG_BOOTSTRAP_KEY not in detail

    def test_correct_key_error_does_not_appear_in_response(self, bootstrap_client, monkeypatch):
        """The configured ADMIN_BOOTSTRAP_KEY must never appear in any HTTP response body."""
        # Trigger the 'key not set' path
        monkeypatch.setattr(settings, "ADMIN_BOOTSTRAP_KEY", None)
        resp = bootstrap_client.post(BOOTSTRAP_URL, json=_bootstrap_payload())
        assert _VALID_BOOTSTRAP_KEY not in resp.text

    def test_partial_key_rejected(self, bootstrap_client):
        """A key that is a prefix of the correct key must be rejected."""
        partial = _VALID_BOOTSTRAP_KEY[: len(_VALID_BOOTSTRAP_KEY) // 2]
        resp = bootstrap_client.post(BOOTSTRAP_URL, json=_bootstrap_payload(key=partial))
        assert resp.status_code == 403

    def test_empty_string_key_in_request_rejected(self, bootstrap_client):
        """Submitting an empty bootstrap_key in the request body must be rejected at schema level."""
        payload = {"email": "admin@example.com", "password": "securepassword123", "bootstrap_key": ""}
        resp = bootstrap_client.post(BOOTSTRAP_URL, json=payload)
        # Schema min_length or endpoint logic must reject this
        assert resp.status_code in (400, 403, 422)


# ---------------------------------------------------------------------------
# 4. Correct key + empty table -> exactly one admin created
# ---------------------------------------------------------------------------

class TestSuccessfulBootstrap:
    def test_correct_key_empty_table_creates_admin(self, bootstrap_client):
        """Correct key with empty users table returns 201 and role='admin'."""
        resp = bootstrap_client.post(BOOTSTRAP_URL, json=_bootstrap_payload())
        assert resp.status_code == 201, resp.text
        data = resp.json()
        assert data["role"] == "admin"
        assert data["email"] == "admin@example.com"
        assert data["is_active"] is True

    def test_bootstrap_response_does_not_contain_key(self, bootstrap_client):
        """The success response must not echo back the bootstrap key."""
        resp = bootstrap_client.post(BOOTSTRAP_URL, json=_bootstrap_payload())
        assert resp.status_code == 201
        assert _VALID_BOOTSTRAP_KEY not in resp.text

    def test_bootstrapped_admin_can_login(self, bootstrap_client):
        """Admin created via bootstrap must be able to log in."""
        bootstrap_client.post(BOOTSTRAP_URL, json=_bootstrap_payload())
        login_resp = bootstrap_client.post(
            LOGIN_URL,
            data={"username": "admin@example.com", "password": "securepassword123"},
        )
        assert login_resp.status_code == 200
        token_data = login_resp.json()
        assert token_data["role"] == "admin"
        assert "access_token" in token_data


# ---------------------------------------------------------------------------
# 5. Second bootstrap attempt -> rejected (409)
# ---------------------------------------------------------------------------

class TestBootstrapIdempotency:
    def test_second_bootstrap_attempt_rejected(self, bootstrap_client):
        """After first successful bootstrap, any further bootstrap attempt returns 409."""
        # First bootstrap succeeds
        r1 = bootstrap_client.post(BOOTSTRAP_URL, json=_bootstrap_payload())
        assert r1.status_code == 201
        # Second attempt (different email) must be rejected
        r2 = bootstrap_client.post(
            BOOTSTRAP_URL,
            json=_bootstrap_payload(email="admin2@example.com"),
        )
        assert r2.status_code == 409

    def test_bootstrap_rejected_when_normal_users_exist(self, bootstrap_client):
        """If a normal user was registered first, bootstrap must be rejected (409)."""
        bootstrap_client.post(REGISTER_URL, json=_register_payload())
        resp = bootstrap_client.post(BOOTSTRAP_URL, json=_bootstrap_payload())
        assert resp.status_code == 409

    def test_same_email_bootstrap_twice_rejected(self, bootstrap_client):
        """Even the exact same email cannot be bootstrapped twice."""
        r1 = bootstrap_client.post(BOOTSTRAP_URL, json=_bootstrap_payload())
        assert r1.status_code == 201
        r2 = bootstrap_client.post(BOOTSTRAP_URL, json=_bootstrap_payload())
        assert r2.status_code == 409


# ---------------------------------------------------------------------------
# 6. Normal registration after bootstrap -> role='user'
# ---------------------------------------------------------------------------

class TestRegistrationAfterBootstrap:
    def test_registration_after_bootstrap_is_user(self, bootstrap_client):
        """After admin bootstrap, normal /register still creates role='user'."""
        bootstrap_client.post(BOOTSTRAP_URL, json=_bootstrap_payload())
        resp = bootstrap_client.post(REGISTER_URL, json=_register_payload())
        assert resp.status_code == 200
        assert resp.json()["role"] == "user"

    def test_registration_cannot_elevate_to_admin_after_bootstrap(self, bootstrap_client):
        """Normal registration must never produce an admin even after bootstrap."""
        bootstrap_client.post(BOOTSTRAP_URL, json=_bootstrap_payload())
        for i in range(5):
            r = bootstrap_client.post(
                REGISTER_URL, json=_register_payload(email=f"attacker{i}@example.com")
            )
            assert r.status_code == 200
            assert r.json()["role"] == "user"


# ---------------------------------------------------------------------------
# 7. Concurrent bootstrap cannot create multiple admins
# ---------------------------------------------------------------------------

class TestConcurrentBootstrap:
    def test_concurrent_bootstrap_only_one_succeeds(
        self, bootstrap_db_engine, monkeypatch
    ):
        """Simulate concurrent bootstrap requests: only one must create an admin.

        We spin up N threads that each call bootstrap_admin() simultaneously.
        At most one should succeed (return a User); all others must return None.
        """
        monkeypatch.setattr(settings, "ADMIN_BOOTSTRAP_KEY", _VALID_BOOTSTRAP_KEY)

        TestingSession = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=bootstrap_db_engine,
            expire_on_commit=False,
        )

        results = []
        errors = []

        def try_bootstrap(thread_id: int):
            session = TestingSession()
            try:
                result = auth_service.bootstrap_admin(
                    session,
                    email=f"admin{thread_id}@example.com",
                    password="securepassword123",
                )
                results.append(result)
            except Exception as exc:
                errors.append(exc)
            finally:
                session.close()

        threads = [threading.Thread(target=try_bootstrap, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Unexpected errors during concurrent bootstrap: {errors}"
        successes = [r for r in results if r is not None]
        assert len(successes) == 1, (
            f"Expected exactly 1 successful bootstrap, got {len(successes)}"
        )
        assert successes[0].role == "admin"

    def test_concurrent_registration_does_not_elevate_any_to_admin(
        self, bootstrap_db_engine, monkeypatch
    ):
        """Concurrent normal registrations must never produce an admin user."""
        monkeypatch.setattr(settings, "ADMIN_BOOTSTRAP_KEY", _VALID_BOOTSTRAP_KEY)

        TestingSession = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=bootstrap_db_engine,
            expire_on_commit=False,
        )

        def register_user(i: int):
            session = TestingSession()
            try:
                from backend.app.schemas.auth import UserRegisterRequest
                req = UserRegisterRequest(
                    email=f"concurrent{i}@example.com",
                    password="securepassword123",
                )
                auth_service.register_user(session, req, role="user")
            except Exception:
                pass
            finally:
                session.close()

        threads = [threading.Thread(target=register_user, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        check_session = TestingSession()
        try:
            admins = check_session.query(User).filter(User.role == "admin").all()
            assert len(admins) == 0, (
                f"Concurrent registration created {len(admins)} admin accounts"
            )
        finally:
            check_session.close()


# ---------------------------------------------------------------------------
# 8 & 9. Existing auth and admin authorization tests still work
# ---------------------------------------------------------------------------

class TestExistingAuthBehavior:
    def test_login_works_after_bootstrap(self, bootstrap_client):
        """Login endpoint still works correctly after bootstrap."""
        bootstrap_client.post(BOOTSTRAP_URL, json=_bootstrap_payload())
        resp = bootstrap_client.post(
            LOGIN_URL,
            data={"username": "admin@example.com", "password": "securepassword123"},
        )
        assert resp.status_code == 200
        assert resp.json()["token_type"] == "bearer"

    def test_admin_can_access_protected_endpoint(
        self, bootstrap_client, bootstrap_db_session
    ):
        """A bootstrapped admin can access admin-only endpoints."""
        bootstrap_client.post(BOOTSTRAP_URL, json=_bootstrap_payload())
        admin = bootstrap_db_session.query(User).filter(User.role == "admin").first()
        assert admin is not None
        token = create_access_token(admin.id, admin.role)
        # The /feedback/stats endpoint requires admin
        resp = bootstrap_client.get(
            "/api/v1/feedback/stats",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

    def test_regular_user_cannot_access_admin_endpoint(
        self, bootstrap_client, bootstrap_db_session
    ):
        """A regular user created after bootstrap cannot access admin-only endpoints."""
        bootstrap_client.post(BOOTSTRAP_URL, json=_bootstrap_payload())
        bootstrap_client.post(REGISTER_URL, json=_register_payload())
        user = bootstrap_db_session.query(User).filter(User.role == "user").first()
        assert user is not None
        token = create_access_token(user.id, user.role)
        resp = bootstrap_client.get(
            "/api/v1/feedback/stats",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 403

    def test_unauthenticated_access_to_protected_endpoint_rejected(
        self, bootstrap_client
    ):
        """No token -> 401 on admin-protected endpoints."""
        resp = bootstrap_client.get("/api/v1/feedback/stats")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# 10. Bootstrap key value never appears in error output
# ---------------------------------------------------------------------------

class TestBootstrapKeyNotLeaked:
    def test_configured_key_not_in_wrong_key_response(self, bootstrap_client):
        """The configured ADMIN_BOOTSTRAP_KEY must not appear in the wrong-key 403 response."""
        resp = bootstrap_client.post(
            BOOTSTRAP_URL, json=_bootstrap_payload(key=_WRONG_BOOTSTRAP_KEY)
        )
        assert resp.status_code == 403
        assert _VALID_BOOTSTRAP_KEY not in resp.text
        assert _WRONG_BOOTSTRAP_KEY not in resp.json().get("detail", "")

    def test_configured_key_not_in_disabled_response(self, bootstrap_client, monkeypatch):
        """When bootstrap is disabled the response must not hint at any key value."""
        monkeypatch.setattr(settings, "ADMIN_BOOTSTRAP_KEY", None)
        resp = bootstrap_client.post(BOOTSTRAP_URL, json=_bootstrap_payload())
        assert resp.status_code == 403
        assert _VALID_BOOTSTRAP_KEY not in resp.text

    def test_success_response_does_not_leak_key(self, bootstrap_client):
        """The 201 success response must not contain the bootstrap key."""
        resp = bootstrap_client.post(BOOTSTRAP_URL, json=_bootstrap_payload())
        assert resp.status_code == 201
        assert _VALID_BOOTSTRAP_KEY not in resp.text


# ---------------------------------------------------------------------------
# Privilege escalation security tests
# ---------------------------------------------------------------------------

class TestPrivilegeEscalation:
    def test_cannot_register_with_admin_role_via_normal_registration(
        self, bootstrap_client
    ):
        """The /register endpoint must not accept a 'role' field to escalate privileges.

        UserRegisterRequest does not include a 'role' field, so any 'role' in the
        request body is silently ignored by Pydantic — the result must always be 'user'.
        """
        payload = {
            "email": "attacker@example.com",
            "password": "securepassword123",
            "role": "admin",  # Extra field — should be ignored
        }
        resp = bootstrap_client.post(REGISTER_URL, json=payload)
        assert resp.status_code == 200
        assert resp.json()["role"] == "user"

    def test_bootstrap_key_tampering_cannot_bypass_check(self, bootstrap_client):
        """Various tampered keys must all be rejected."""
        tampered_keys = [
            _VALID_BOOTSTRAP_KEY + " ",
            " " + _VALID_BOOTSTRAP_KEY,
            _VALID_BOOTSTRAP_KEY.upper(),
            _VALID_BOOTSTRAP_KEY.replace("-", "_"),
            _VALID_BOOTSTRAP_KEY[:-1],   # one char short
            _VALID_BOOTSTRAP_KEY + "x",  # one char long
        ]
        for bad_key in tampered_keys:
            resp = bootstrap_client.post(
                BOOTSTRAP_URL, json=_bootstrap_payload(key=bad_key)
            )
            assert resp.status_code == 403, (
                f"Expected 403 for tampered key '{bad_key}', got {resp.status_code}"
            )

    def test_repeated_bootstrap_attempts_cannot_accumulate_admins(
        self, bootstrap_client
    ):
        """Even many repeated correct-key bootstrap attempts can only produce 1 admin."""
        from backend.app.core.rate_limit import rate_limit_auth

        first = bootstrap_client.post(BOOTSTRAP_URL, json=_bootstrap_payload())
        assert first.status_code == 201

        for i in range(10):
            rate_limit_auth.history.clear()
            resp = bootstrap_client.post(
                BOOTSTRAP_URL,
                json=_bootstrap_payload(email=f"admin{i}@example.com"),
            )
            assert resp.status_code == 409
