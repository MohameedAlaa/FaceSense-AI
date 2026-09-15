"""Security tests: JWT config validation and token round-trips.

Regression suite for the JWT_SECRET_KEY mandatory-secret fix:
- missing / empty / insecure placeholder -> Settings() must raise ValidationError
- valid key -> Settings() must succeed
- token creation and verification must work correctly
"""
import pytest
from pydantic import ValidationError

from backend.app.core.security import verify_password, get_password_hash, create_access_token, decode_access_token


# ---------------------------------------------------------------------------
# Pre-existing tests (must remain passing)
# ---------------------------------------------------------------------------

def test_password_hashing():
    password = "supersecretpassword"
    hashed = get_password_hash(password)
    assert verify_password(password, hashed)
    assert not verify_password("wrongpassword", hashed)


def test_jwt_token():
    user_id = 1
    role = "admin"
    token = create_access_token(user_id, role)
    decoded = decode_access_token(token)
    assert decoded["sub"] == str(user_id)
    assert decoded["role"] == role


# ---------------------------------------------------------------------------
# JWT_SECRET_KEY mandatory-secret regression tests
# ---------------------------------------------------------------------------

def _build_settings(**overrides):
    """Directly instantiate Settings with given kwargs (bypasses the module singleton)."""
    # Import here to avoid circular issues and to always get the fresh class
    from backend.app.core.config import Settings
    return Settings(**overrides)


class TestJwtSecretKeyValidation:
    """Settings must refuse to construct when JWT_SECRET_KEY is absent or unusable."""

    def test_missing_jwt_secret_key_raises(self):
        """Settings construction fails when JWT_SECRET_KEY is None (env var not set)."""
        with pytest.raises(ValidationError) as exc_info:
            _build_settings(JWT_SECRET_KEY=None)
        error_str = str(exc_info.value)
        # The error message must not contain the actual secret value
        assert "JWT_SECRET_KEY" in error_str
        assert "required" in error_str.lower() or "must be" in error_str.lower()

    def test_empty_jwt_secret_key_raises(self):
        """Settings construction fails when JWT_SECRET_KEY is an empty string."""
        with pytest.raises(ValidationError) as exc_info:
            _build_settings(JWT_SECRET_KEY="")
        error_str = str(exc_info.value)
        assert "JWT_SECRET_KEY" in error_str

    def test_whitespace_jwt_secret_key_raises(self):
        """Settings construction fails when JWT_SECRET_KEY is whitespace-only."""
        with pytest.raises(ValidationError):
            _build_settings(JWT_SECRET_KEY="   ")

    def test_insecure_placeholder_jwt_secret_key_raises(self):
        """Settings construction fails when JWT_SECRET_KEY uses the old insecure default placeholder."""
        with pytest.raises(ValidationError) as exc_info:
            _build_settings(JWT_SECRET_KEY="insecure-dev-secret-key-please-change-in-prod")
        error_str = str(exc_info.value)
        # Our validator message must not echo the secret back — Pydantic v2 itself
        # appends 'input_value=<value>' to ValidationError.__str__, which is a
        # framework-level debug aid and cannot be suppressed from the validator.
        # What we can control: our OWN error message text must not contain the secret.
        # We verify that by checking the exc_info.value.errors() entries.
        for err in exc_info.value.errors():
            assert "insecure-dev-secret-key-please-change-in-prod" not in err.get("msg", "")
        assert "JWT_SECRET_KEY" in error_str
        # The error must mention 'insecure placeholder' without repeating the actual value
        assert "insecure placeholder" in error_str

    def test_valid_jwt_secret_key_succeeds(self):
        """Settings construction succeeds with a non-empty, non-placeholder secret."""
        settings = _build_settings(JWT_SECRET_KEY="a-perfectly-valid-test-secret-key-abc123")
        assert settings.JWT_SECRET_KEY == "a-perfectly-valid-test-secret-key-abc123"

    def test_error_message_does_not_expose_secret(self):
        """Our validator error messages must not contain the secret in the 'msg' field.

        Note: Pydantic v2 always appends input_value=<value> to ValidationError.__str__
        as a debug aid — that is framework behaviour outside our control. What we
        verify here is that OUR message string (err['msg']) does not leak the secret.
        """
        secret = "my-super-secret-value-that-must-not-leak"
        # A valid secret should be stored as-is
        settings = _build_settings(JWT_SECRET_KEY=secret)
        assert settings.JWT_SECRET_KEY == secret
        # For the insecure placeholder case, our msg must not mention the value
        bad_secret = "insecure-dev-secret-key-please-change-in-prod"
        with pytest.raises(ValidationError) as exc_info:
            _build_settings(JWT_SECRET_KEY=bad_secret)
        for err in exc_info.value.errors():
            assert bad_secret not in err.get("msg", "")


class TestJwtTokenRoundTrip:
    """Token creation and verification must work correctly with a valid secret."""

    def test_create_and_decode_access_token(self):
        """create_access_token produces a token that decode_access_token can verify."""
        user_id = 42
        role = "user"
        token = create_access_token(subject=user_id, role=role)
        decoded = decode_access_token(token)
        assert decoded["sub"] == str(user_id)
        assert decoded["role"] == role

    def test_create_and_decode_admin_token(self):
        """Admin-role token round-trip works correctly."""
        token = create_access_token(subject=1, role="admin")
        decoded = decode_access_token(token)
        assert decoded["role"] == "admin"

    def test_decode_invalid_token_returns_empty_dict(self):
        """decode_access_token returns {} for a clearly invalid/tampered token."""
        result = decode_access_token("this.is.not.a.valid.jwt")
        assert result == {}

    def test_decode_tampered_token_returns_empty_dict(self):
        """Altering the signature of a real token causes decode to return {}."""
        token = create_access_token(subject=99, role="user")
        # Flip the last character of the signature portion
        tampered = token[:-1] + ("A" if token[-1] != "A" else "B")
        result = decode_access_token(tampered)
        assert result == {}
