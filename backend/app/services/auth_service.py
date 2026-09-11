from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from typing import Optional

from backend.app.models.user import User
from backend.app.schemas.auth import UserRegisterRequest
from backend.app.core.security import get_password_hash, verify_password

class AuthService:
    def register_user(self, db: Session, request: UserRegisterRequest, role: str = "user") -> Optional[User]:
        db_user = User(
            email=request.email,
            password_hash=get_password_hash(request.password),
            role=role,
        )
        try:
            db.add(db_user)
            db.commit()
            db.refresh(db_user)
            return db_user
        except IntegrityError:
            db.rollback()
            return None

    def authenticate_user(self, db: Session, email: str, password: str) -> Optional[User]:
        user = self.get_user_by_email(db, email)
        if not user:
            return None
        if not verify_password(password, user.password_hash):
            return None
        return user

    def get_user_by_email(self, db: Session, email: str) -> Optional[User]:
        return db.query(User).filter(User.email == email).first()

    def get_user_by_id(self, db: Session, user_id: int) -> Optional[User]:
        return db.query(User).filter(User.id == user_id).first()

auth_service = AuthService()
