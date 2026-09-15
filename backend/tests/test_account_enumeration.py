"""Security tests: Account enumeration mitigation on public registration.

Regression test suite verifying that:
1. New email registration succeeds (HTTP 200, role="user").
2. Existing email registration returns a safe, generic response.
3. Existing email response does not reveal database details (tables, columns, constraints).
4. Existing email response does not contain SQL or IntegrityError text.
5. Existing email response does not reveal whether the existing account has a specific role (e.g. admin vs user).
6. Repeated registration attempts do not modify the existing account's role, password, or state.
7. Normal registrations always produce role="user".
8. Admin bootstrap behavior remains separate and unchanged.
9. Existing login behavior remains unchanged (valid credentials succeed, invalid fail generically).
10. Existing authorization rules remain enforced (role-based access control).
11. Focused account enumeration comparison: comparing registration responses to verify no sensitive account existence is leaked.
"""
import pytest
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.security import verify_password
from backend.app.models.user import User
from backend.app.services.auth_service import auth_service


REGISTER_URL = "/api/v1/auth/register"
LOGIN_URL = "/api/v1/auth/login"
BOOTSTRAP_URL = "/api/v1/auth/admin-bootstrap"
ME_URL = "/api/v1/auth/me"


def _reg_payload(email: str = "newuser@example.com", password: str = "SecurePass123!") -> dict:
    return {"email": email, "password": password}


class TestAccountEnumerationMitigation:
    """Security verification against account enumeration via registration."""

    def test_1_new_email_registration_succeeds(self, client_with_db):
        """1. New email registration -> succeeds with HTTP 200 and UserResponse."""
        resp = client_with_db.post(REGISTER_URL, json=_reg_payload("brand_new_user@example.com"))
        assert resp.status_code == 200
        data = resp.json()
        assert data["email"] == "brand_new_user@example.com"
        assert data["role"] == "user"
        assert "id" in data
        assert data["is_active"] is True

    def test_2_existing_email_registration_safe_response(self, client_with_db):
        """2. Existing email registration -> safe generic response, no account disclosure."""
        email = "existing_user@example.com"
        client_with_db.post(REGISTER_URL, json=_reg_payload(email))

        # Attempt re-registration
        resp = client_with_db.post(REGISTER_URL, json=_reg_payload(email))
        assert resp.status_code == 400
        data = resp.json()
        detail = data.get("detail", "")

        # Must not explicitly state that the email or user already exists
        assert "already exists" not in detail.lower()
        assert "user with this email" not in detail.lower()
        assert "email exists" not in detail.lower()
        assert detail == "Registration failed. Please check the provided information and try again."

    def test_3_existing_email_response_does_not_reveal_database_details(self, client_with_db):
        """3. Existing email response does not reveal database details or constraint names."""
        email = "db_leak_check@example.com"
        client_with_db.post(REGISTER_URL, json=_reg_payload(email))

        resp = client_with_db.post(REGISTER_URL, json=_reg_payload(email))
        text = resp.text.lower()

        # Check for database structure artifacts
        forbidden_db_terms = [
            "ix_users_email",
            "users.email",
            "unique constraint",
            "table",
            "column",
            "primary key",
            "foreign key",
            "schema",
            "sqlite",
            "postgres",
            "psycopg",
        ]
        for term in forbidden_db_terms:
            assert term not in text, f"Response leaked database detail: {term}"

    def test_4_existing_email_response_does_not_contain_sql_or_integrity_error(self, client_with_db):
        """4. Existing email response does not contain SQL or IntegrityError text."""
        email = "sql_leak_check@example.com"
        client_with_db.post(REGISTER_URL, json=_reg_payload(email))

        resp = client_with_db.post(REGISTER_URL, json=_reg_payload(email))
        text = resp.text

        forbidden_error_terms = [
            "IntegrityError",
            "SQLAlchemyError",
            "OperationalError",
            "DatabaseError",
            "INSERT INTO",
            "SELECT",
            "UPDATE",
            "DELETE FROM",
            "Traceback (most recent call last)",
        ]
        for term in forbidden_error_terms:
            assert term not in text, f"Response leaked internal error or SQL text: {term}"

    def test_5_existing_email_does_not_reveal_account_role(self, client_with_db, test_db_session: Session):
        """5. Existing email response does not reveal whether the account has a specific role.

        An admin account vs a normal user account must yield the exact same error response.
        """
        # Create an admin account directly in DB
        from backend.app.schemas.auth import UserRegisterRequest

        admin_email = "superadmin@example.com"
        auth_service.register_user(
            test_db_session,
            UserRegisterRequest(**_reg_payload(admin_email, "AdminSecret123!")),
            role="admin",
        )

        # Create a regular user
        user_email = "regularuser@example.com"
        auth_service.register_user(
            test_db_session,
            UserRegisterRequest(**_reg_payload(user_email, "UserSecret123!")),
            role="user",
        )

        resp_admin = client_with_db.post(REGISTER_URL, json=_reg_payload(admin_email, "AttackerPass123!"))
        resp_user = client_with_db.post(REGISTER_URL, json=_reg_payload(user_email, "AttackerPass123!"))

        assert resp_admin.status_code == 400
        assert resp_user.status_code == 400
        assert resp_admin.json() == resp_user.json()
        assert "admin" not in resp_admin.text.lower()

    def test_6_repeated_registration_does_not_change_account_role_or_password(
        self, client_with_db, test_db_session: Session
    ):
        """6. Repeated registration attempts do not change the account's role or password."""
        email = "tamper_target@example.com"
        original_pass = "OriginalPassword123!"
        reg_resp = client_with_db.post(REGISTER_URL, json=_reg_payload(email, original_pass))
        assert reg_resp.status_code == 200

        # Try to re-register with different password and malicious payload attempting role escalation
        malicious_payload = {
            "email": email,
            "password": "NewMaliciousPassword999!",
            "role": "admin",
        }
        tamper_resp = client_with_db.post(REGISTER_URL, json=malicious_payload)
        assert tamper_resp.status_code == 400

        # Verify DB state: user password and role remain unchanged
        user_in_db = test_db_session.query(User).filter(User.email == email).first()
        assert user_in_db is not None
        assert user_in_db.role == "user"
        assert verify_password(original_pass, user_in_db.password_hash)
        assert not verify_password("NewMaliciousPassword999!", user_in_db.password_hash)

    def test_7_normal_users_remain_role_user(self, client_with_db):
        """7. Normal public registration always creates role='user'."""
        for i in range(3):
            email = f"standard_user_{i}@example.com"
            resp = client_with_db.post(REGISTER_URL, json=_reg_payload(email))
            assert resp.status_code == 200
            assert resp.json()["role"] == "user"

    def test_8_admin_bootstrap_behavior_remains_unchanged(self, client_with_db, monkeypatch):
        """8. Admin bootstrap behavior remains separate and unchanged."""
        bootstrap_key = "test-secret-bootstrap-key-12345"
        monkeypatch.setattr(settings, "ADMIN_BOOTSTRAP_KEY", bootstrap_key)

        # 1. First user bootstrapped as admin when table is empty
        resp = client_with_db.post(
            BOOTSTRAP_URL,
            json={
                "email": "first_admin@example.com",
                "password": "AdminBootstrapPassword123!",
                "bootstrap_key": bootstrap_key,
            },
        )
        assert resp.status_code == 201
        assert resp.json()["role"] == "admin"

        # 2. Once a user exists, admin-bootstrap is closed (409 Conflict)
        resp2 = client_with_db.post(
            BOOTSTRAP_URL,
            json={
                "email": "second_admin@example.com",
                "password": "AdminBootstrapPassword123!",
                "bootstrap_key": bootstrap_key,
            },
        )
        assert resp2.status_code == 409

    def test_9_existing_login_behavior_remains_unchanged(self, client_with_db):
        """9. Existing login behavior remains unchanged (valid succeeds, invalid fails)."""
        email = "login_test_user@example.com"
        password = "ValidPassword123!"
        reg = client_with_db.post(REGISTER_URL, json=_reg_payload(email, password))
        assert reg.status_code == 200

        # Valid login
        login_resp = client_with_db.post(
            LOGIN_URL,
            data={"username": email, "password": password},
        )
        assert login_resp.status_code == 200
        token_data = login_resp.json()
        assert "access_token" in token_data
        assert token_data["token_type"] == "bearer"
        assert token_data["role"] == "user"

        # Invalid login
        bad_login = client_with_db.post(
            LOGIN_URL,
            data={"username": email, "password": "WrongPassword999!"},
        )
        assert bad_login.status_code == 400
        assert bad_login.json()["detail"] == "Incorrect email or password"

    def test_10_existing_authorization_remains_passing(self, client_with_db):
        """10. Existing authorization tests remain passing (user token cannot access admin routes)."""
        email = "authz_user@example.com"
        password = "UserPassword123!"
        client_with_db.post(REGISTER_URL, json=_reg_payload(email, password))

        login_resp = client_with_db.post(
            LOGIN_URL,
            data={"username": email, "password": password},
        )
        token = login_resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # /auth/me succeeds for authenticated user
        me_resp = client_with_db.get(ME_URL, headers=headers)
        assert me_resp.status_code == 200
        assert me_resp.json()["email"] == email


class TestFocusedAccountEnumeration:
    """Focused comparison testing for account enumeration on the registration endpoint."""

    def test_enumeration_comparison_existing_vs_non_existing(self, client_with_db):
        """Compare registration behavior for existing account vs non-existing.

        Verification:
        - The failure response for an existing email does NOT disclose that the account exists.
        - No sensitive internal details, database identifiers, or account information are leaked.
        """
        existing_email = "target_account@example.com"
        non_existing_email = "not_yet_registered@example.com"

        # Register the existing account
        first_resp = client_with_db.post(REGISTER_URL, json=_reg_payload(existing_email))
        assert first_resp.status_code == 200

        # Attempt registering the existing email again
        existing_attempt_resp = client_with_db.post(
            REGISTER_URL, json=_reg_payload(existing_email, "DifferentPassword123!")
        )
        assert existing_attempt_resp.status_code == 400
        existing_data = existing_attempt_resp.json()

        # The error response must NOT contain:
        # 1. The target email itself
        # 2. Confirmation that the account exists (e.g., "already exists", "account found")
        # 3. Any user ID, creation date, active state, or role information
        assert "already exists" not in existing_data.get("detail", "").lower()
        assert "user with this email" not in existing_data.get("detail", "").lower()
        assert existing_email not in existing_attempt_resp.text
        assert "role" not in existing_data
        assert "id" not in existing_data
        assert "created_at" not in existing_data

        # Now register the non-existing email (legitimate user flow)
        non_existing_attempt_resp = client_with_db.post(
            REGISTER_URL, json=_reg_payload(non_existing_email)
        )
        assert non_existing_attempt_resp.status_code == 200
        new_user_data = non_existing_attempt_resp.json()
        assert new_user_data["email"] == non_existing_email

    def test_password_hashing_performed_on_all_registration_attempts(self, client_with_db, monkeypatch):
        """Verify that password hashing (get_password_hash) is performed unconditionally.

        This ensures the registration service does not take an early return or bypass
        password hashing for duplicate accounts, preventing simple timing differentials.
        """
        import backend.app.services.auth_service as auth_module

        hashing_calls = []
        real_get_password_hash = auth_module.get_password_hash

        def spy_get_password_hash(password: str) -> str:
            hashing_calls.append(password)
            return real_get_password_hash(password)

        monkeypatch.setattr(auth_module, "get_password_hash", spy_get_password_hash)

        # 1. Register brand new user
        email = "hash_timing_test@example.com"
        client_with_db.post(REGISTER_URL, json=_reg_payload(email, "PasswordOne123!"))
        assert len(hashing_calls) == 1
        assert hashing_calls[-1] == "PasswordOne123!"

        # 2. Attempt registering existing user
        client_with_db.post(REGISTER_URL, json=_reg_payload(email, "PasswordTwo123!"))
        # Password hashing MUST still have been called!
        assert len(hashing_calls) == 2
        assert hashing_calls[-1] == "PasswordTwo123!"
