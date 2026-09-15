"""CORS Security & Configuration Regression Test Suite.

Verifies:
1. Explicit localhost and 127.0.0.1 origins are accepted.
2. Multiple comma-separated explicit origins are parsed and trimmed correctly.
3. Wildcard "*" with credentials enabled is strictly rejected with a clear configuration error.
4. Wildcard "*" is only permitted when credentials are explicitly disabled.
5. Empty / whitespace-only CORS origin configurations are handled safely.
6. Trailing slashes in configured origins are normalized.
7. Preflight OPTIONS requests for allowed origins succeed with proper CORS headers.
8. Requests from disallowed/malicious origins are NOT granted CORS access.
9. Credentialed cross-origin behavior fails safely for untrusted origins.
"""
import pytest
from pydantic import ValidationError
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from backend.app.core.config import Settings


def _build_settings(**overrides) -> Settings:
    """Build a Settings instance with default test-safe overrides."""
    kwargs = {
        "JWT_SECRET_KEY": "test-secret-key-for-cors-validation-suite",
        **overrides,
    }
    return Settings(**kwargs)


# ==============================================================================
# 1. Configuration & Validator Unit Tests
# ==============================================================================

class TestCorsConfiguration:
    """Settings-level unit tests for CORS parsing and validation."""

    def test_default_origins_explicit_localhost_and_127(self):
        """Default CORS configuration must use explicit localhost origins, not wildcard."""
        settings = _build_settings()
        origins = settings.get_cors_origins
        assert "*" not in origins
        assert "http://localhost:5173" in origins
        assert "http://127.0.0.1:5173" in origins

    def test_explicit_localhost_origin_accepted(self):
        """Explicit localhost origin is accepted and parsed."""
        settings = _build_settings(CORS_ALLOWED_ORIGINS="http://localhost:5173")
        assert settings.get_cors_origins == ["http://localhost:5173"]

    def test_explicit_127_origin_accepted(self):
        """Explicit 127.0.0.1 origin is accepted and parsed."""
        settings = _build_settings(CORS_ALLOWED_ORIGINS="http://127.0.0.1:5173")
        assert settings.get_cors_origins == ["http://127.0.0.1:5173"]

    def test_multiple_comma_separated_origins_parsed_correctly(self):
        """Multiple comma-separated origins with whitespace are cleanly trimmed."""
        settings = _build_settings(
            CORS_ALLOWED_ORIGINS="http://localhost:5173, http://127.0.0.1:5173, https://app.facesense.ai"
        )
        assert settings.get_cors_origins == [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "https://app.facesense.ai",
        ]

    def test_wildcard_with_credentials_enabled_raises_validation_error(self):
        """Wildcard '*' origin MUST be rejected when credentials are enabled."""
        with pytest.raises(ValidationError) as exc_info:
            _build_settings(CORS_ALLOWED_ORIGINS="*", CORS_ALLOW_CREDENTIALS=True)
        error_msg = str(exc_info.value)
        assert "wildcard" in error_msg.lower() or "*" in error_msg
        assert "credentials" in error_msg.lower()

    def test_wildcard_surrounded_by_whitespace_rejected_with_credentials(self):
        """Wildcard with spaces '  *  ' is still recognized and rejected."""
        with pytest.raises(ValidationError):
            _build_settings(CORS_ALLOWED_ORIGINS="  *  ", CORS_ALLOW_CREDENTIALS=True)

    def test_wildcard_inside_multiple_origins_rejected_with_credentials(self):
        """Wildcard combined with other origins is also rejected when credentials are on."""
        with pytest.raises(ValidationError):
            _build_settings(
                CORS_ALLOWED_ORIGINS="http://localhost:5173, *",
                CORS_ALLOW_CREDENTIALS=True,
            )

    def test_wildcard_without_credentials_is_accepted(self):
        """Wildcard '*' is accepted ONLY if credentials are explicitly disabled."""
        settings = _build_settings(CORS_ALLOWED_ORIGINS="*", CORS_ALLOW_CREDENTIALS=False)
        assert settings.get_cors_origins == ["*"]

    def test_empty_cors_origin_handled_safely(self):
        """Empty string origin config is handled safely without crashing, returning empty list."""
        settings = _build_settings(CORS_ALLOWED_ORIGINS="")
        assert settings.get_cors_origins == []

    def test_whitespace_and_commas_handled_safely(self):
        """Whitespace and stray commas are handled safely and filtered out."""
        settings = _build_settings(CORS_ALLOWED_ORIGINS="  ,   ,  ")
        assert settings.get_cors_origins == []

    def test_sparse_commas_between_valid_origins_filtered(self):
        """Stray empty entries between valid origins are cleanly ignored."""
        settings = _build_settings(
            CORS_ALLOWED_ORIGINS="http://localhost:5173, , http://127.0.0.1:5173"
        )
        assert settings.get_cors_origins == [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]

    def test_trailing_slashes_normalized(self):
        """Origins configured with a trailing slash are normalized to avoid browser mismatches."""
        settings = _build_settings(
            CORS_ALLOWED_ORIGINS="http://localhost:5173/,https://facesense.ai/"
        )
        assert settings.get_cors_origins == [
            "http://localhost:5173",
            "https://facesense.ai",
        ]


# ==============================================================================
# 2. HTTP & Middleware Integration Tests
# ==============================================================================

class TestCorsMiddlewareBehavior:
    """Integration tests verifying Starlette CORSMiddleware enforcement with the live app."""

    def test_preflight_allowed_localhost_origin_succeeds(self, client):
        """Preflight OPTIONS request from localhost frontend origin receives CORS approval."""
        response = client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
        assert response.headers.get("access-control-allow-credentials") == "true"

    def test_preflight_allowed_127_origin_succeeds(self, client):
        """Preflight OPTIONS request from 127.0.0.1 frontend origin receives CORS approval."""
        response = client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://127.0.0.1:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://127.0.0.1:5173"
        assert response.headers.get("access-control-allow-credentials") == "true"

    def test_actual_request_allowed_origin_returns_cors_header(self, client):
        """GET request with an allowed Origin receives the matching Access-Control-Allow-Origin header."""
        response = client.get(
            "/api/v1/health",
            headers={"Origin": "http://localhost:5173"},
        )
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
        assert response.headers.get("access-control-allow-credentials") == "true"

    def test_disallowed_origin_does_not_receive_cors_headers(self, client):
        """Request from an untrusted origin must NOT receive Access-Control-Allow-Origin."""
        response = client.get(
            "/api/v1/health",
            headers={"Origin": "https://untrusted-third-party.com"},
        )
        assert response.status_code == 200
        assert "access-control-allow-origin" not in response.headers

    def test_malicious_origin_preflight_rejected(self, client):
        """Preflight from a malicious origin is not granted CORS access."""
        response = client.options(
            "/api/v1/health",
            headers={
                "Origin": "https://attacker.site",
                "Access-Control-Request-Method": "POST",
            },
        )
        # CORSMiddleware does not grant ACAO for disallowed origins
        assert "access-control-allow-origin" not in response.headers

    def test_origin_reflection_attack_prevented(self, client):
        """Origin reflection attack is prevented: random origins must never be echoed back."""
        for malicious_origin in [
            "https://evil.com",
            "http://localhost:5173.evil.com",
            "http://evil-localhost:5173",
            "null",
        ]:
            response = client.get(
                "/api/v1/health",
                headers={"Origin": malicious_origin},
            )
            assert response.headers.get("access-control-allow-origin") != malicious_origin
            assert "access-control-allow-origin" not in response.headers


# ==============================================================================
# 3. Dynamic App & Production Configuration Tests
# ==============================================================================

class TestCustomCorsConfigurationApps:
    """Verifies that custom application configurations behave as expected."""

    def test_custom_production_origins_app(self):
        """FastAPI app configured with specific production origins grants only those origins."""
        custom_settings = _build_settings(
            CORS_ALLOWED_ORIGINS="https://facesense.example.com,https://app.facesense.example.com"
        )
        test_app = FastAPI()
        test_app.add_middleware(
            CORSMiddleware,
            allow_origins=custom_settings.get_cors_origins,
            allow_credentials=custom_settings.CORS_ALLOW_CREDENTIALS,
            allow_methods=["GET"],
            allow_headers=["*"],
        )

        @test_app.get("/test")
        def endpoint():
            return {"status": "ok"}

        test_client = TestClient(test_app)

        # Configured production origin is allowed
        resp = test_client.get("/test", headers={"Origin": "https://facesense.example.com"})
        assert resp.headers.get("access-control-allow-origin") == "https://facesense.example.com"

        # Localhost is NOT allowed when only production origins are configured
        resp_local = test_client.get("/test", headers={"Origin": "http://localhost:5173"})
        assert "access-control-allow-origin" not in resp_local.headers

    def test_empty_origins_app_allows_no_cors(self):
        """FastAPI app configured with empty CORS origins permits no cross-origin requests."""
        custom_settings = _build_settings(CORS_ALLOWED_ORIGINS="")
        test_app = FastAPI()
        test_app.add_middleware(
            CORSMiddleware,
            allow_origins=custom_settings.get_cors_origins,
            allow_credentials=custom_settings.CORS_ALLOW_CREDENTIALS,
            allow_methods=["GET"],
            allow_headers=["*"],
        )

        @test_app.get("/test")
        def endpoint():
            return {"status": "ok"}

        test_client = TestClient(test_app)
        resp = test_client.get("/test", headers={"Origin": "http://localhost:5173"})
        assert "access-control-allow-origin" not in resp.headers
