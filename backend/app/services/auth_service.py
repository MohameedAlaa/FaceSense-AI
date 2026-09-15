import logging
import threading
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from typing import Optional

from backend.app.models.user import User
from backend.app.schemas.auth import UserRegisterRequest
from backend.app.core.security import get_password_hash, verify_password

logger = logging.getLogger(__name__)

# Module-level lock to serialize concurrent admin bootstrap attempts.
# This prevents a TOCTOU race where two simultaneous requests both pass
# the "no admin exists" check and both succeed in creating an admin.
_bootstrap_lock = threading.Lock()


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
            logger.warning("User registration failed: database integrity constraint violation.")
            return None
        except SQLAlchemyError as exc:
            db.rollback()
            logger.error("User registration failed due to database error: %s", type(exc).__name__)
            return None
        except Exception as exc:
            db.rollback()
            logger.error("Unexpected error during user registration: %s", type(exc).__name__)
            return None

    def bootstrap_admin(self, db: Session, email: str, password: str) -> Optional[User]:
        """Create the very first admin account.

        Returns the new User on success, or None if bootstrap conditions are not met:
        - the users table must be completely empty (no existing users of any role)
        - only one caller can succeed even under concurrent requests

        The threading lock + database-level uniqueness constraint together prevent
        race conditions where two simultaneous requests both observe an empty table.
        """
        with _bootstrap_lock:
            # Re-check inside the lock: abort if any user already exists
            existing_count = db.query(User).count()
            if existing_count > 0:
                return None

            admin = User(
                email=email,
                password_hash=get_password_hash(password),
                role="admin",
            )
            try:
                db.add(admin)
                db.commit()
                db.refresh(admin)
                return admin
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
