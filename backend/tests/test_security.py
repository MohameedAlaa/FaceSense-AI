import pytest
from backend.app.core.security import verify_password, get_password_hash, create_access_token, decode_access_token

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
