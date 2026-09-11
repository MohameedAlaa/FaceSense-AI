from datetime import timedelta
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.security import create_access_token
from backend.app.db.session import get_db
from backend.app.schemas.auth import UserRegisterRequest, TokenResponse, UserResponse
from backend.app.services.auth_service import auth_service
from backend.app.core.dependencies import get_current_active_user, require_admin
from backend.app.models.user import User

router = APIRouter()


@router.post("/register", response_model=UserResponse)
def register(
    *,
    db: Session = Depends(get_db),
    request: UserRegisterRequest,
) -> Any:
    """Register a new user."""
    # Temporarily hardcode the first user as admin if no users exist
    # In production, this should be handled by a secure setup process
    role = "user"
    user_count = db.query(User).count()
    if user_count == 0:
        role = "admin"

    user = auth_service.register_user(db, request=request, role=role)
    if not user:
        raise HTTPException(
            status_code=400,
            detail="The user with this email already exists in the system.",
        )
    return user


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
