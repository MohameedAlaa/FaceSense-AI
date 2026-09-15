import hmac
import logging
from datetime import timedelta
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.security import create_access_token
from backend.app.db.session import get_db
from backend.app.schemas.auth import (
    AdminBootstrapRequest,
    UserRegisterRequest,
    TokenResponse,
    UserResponse,
)
from backend.app.services.auth_service import auth_service
from backend.app.core.dependencies import get_current_active_user
from backend.app.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/register", response_model=UserResponse)
def register(
    *,
    db: Session = Depends(get_db),
    request: UserRegisterRequest,
) -> Any:
    """Register a new user account with role='user'.

    Normal public registration always creates a regular user.
    To create the initial admin account use POST /auth/admin-bootstrap.
    """
    user = auth_service.register_user(db, request=request, role="user")
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Registration failed. Please check the provided information and try again.",
        )
    return user


@router.post("/admin-bootstrap", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def admin_bootstrap(
    *,
    db: Session = Depends(get_db),
    request: AdminBootstrapRequest,
) -> Any:
    """Create the initial admin account (one-time operation).

    Requirements that must ALL be satisfied for this to succeed:
    - ADMIN_BOOTSTRAP_KEY must be configured in the environment.
    - The provided bootstrap_key must match ADMIN_BOOTSTRAP_KEY exactly.
    - The users table must be completely empty (no prior registrations).

    Once any user exists this endpoint is permanently disabled.
    The bootstrap_key value is never logged.
    """
    # 1. Verify the bootstrap feature is enabled at all
    configured_key = settings.ADMIN_BOOTSTRAP_KEY
    if not configured_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin bootstrap is not enabled on this server.",
        )

    # 2. Constant-time comparison to prevent timing-based key oracle attacks
    provided_key = request.bootstrap_key
    keys_match = hmac.compare_digest(configured_key, provided_key)
    if not keys_match:
        # Log a warning without ever recording the submitted key value
        logger.warning("Admin bootstrap attempt rejected: incorrect bootstrap key supplied.")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid bootstrap key.",
        )

    # 3. Attempt to create the admin (service enforces empty-table invariant + locking)
    admin = auth_service.bootstrap_admin(db, email=request.email, password=request.password)
    if admin is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Admin bootstrap is no longer available: users already exist.",
        )

    logger.info("Admin account successfully bootstrapped for email: %s", request.email)
    return admin


@router.post("/login", response_model=TokenResponse)
def login(
    db: Session = Depends(get_db), form_data: OAuth2PasswordRequestForm = Depends()
) -> Any:
    """OAuth2 compatible token login, get an access token for future requests."""
    user = auth_service.authenticate_user(
        db, email=form_data.username, password=form_data.password
    )
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    elif not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return {
        "access_token": create_access_token(
            user.id, role=user.role, expires_delta=access_token_expires
        ),
        "token_type": "bearer",
        "expires_in_seconds": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "role": user.role,
    }


@router.get("/me", response_model=UserResponse)
def get_current_user_info(
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get current user info."""
    return current_user
