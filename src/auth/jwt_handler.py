"""JWT token handling for authentication."""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from ..utils.logger import get_logger
from ..utils.config import settings

logger = get_logger(__name__)

# JWT Configuration
SECRET_KEY = settings.secret_key if hasattr(settings, 'secret_key') else "your-secret-key-change-in-production-min-32-characters"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # 1 hour
REFRESH_TOKEN_EXPIRE_DAYS = 7  # 7 days


class JWTHandler:
    """Handle JWT token creation and verification."""

    @staticmethod
    def create_token(
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a JWT token."""
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)

        to_encode.update({"exp": expire, "iat": datetime.utcnow()})

        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    @staticmethod
    def verify_token(token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except JWTError as e:
            logger.warning(f"JWT verification failed: {e}")
            return None


def create_access_token(user_id: str, phone_number: str) -> str:
    """Create an access token for a user."""
    data = {
        "sub": user_id,
        "phone": phone_number,
        "type": "access"
    }
    expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    token = JWTHandler.create_token(data, expires)

    logger.info(f"Access token created for user {user_id}")
    return token


def create_refresh_token(user_id: str, phone_number: str) -> str:
    """Create a refresh token for a user."""
    data = {
        "sub": user_id,
        "phone": phone_number,
        "type": "refresh"
    }
    expires = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    token = JWTHandler.create_token(data, expires)

    logger.info(f"Refresh token created for user {user_id}")
    return token


def verify_token(token: str) -> Optional[str]:
    """Verify token and return user_id if valid."""
    payload = JWTHandler.verify_token(token)

    if payload is None:
        return None

    user_id: str = payload.get("sub")
    if user_id is None:
        logger.warning("Token payload missing 'sub' field")
        return None

    return user_id
