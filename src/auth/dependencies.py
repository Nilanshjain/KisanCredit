"""FastAPI dependencies for authentication."""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
from ..database.connection import get_db
from ..database.repositories import UserRepository
from ..database.models import User
from .jwt_handler import verify_token
from ..utils.logger import get_logger
from sqlalchemy.ext.asyncio import AsyncSession

logger = get_logger(__name__)

# HTTP Bearer token scheme
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    session: AsyncSession = Depends(get_db)
) -> User:
    """Get current authenticated user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # Get token from authorization header
    token = credentials.credentials

    # Verify token and extract user_id
    user_id = verify_token(token)
    if user_id is None:
        logger.warning("Invalid or expired token")
        raise credentials_exception

    # Fetch user from database
    user = await UserRepository.get_by_id(session, user_id)
    if user is None:
        logger.warning(f"User not found for user_id: {user_id}")
        raise credentials_exception

    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current active user (not deactivated)."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )

    return current_user


# Optional authentication - doesn't fail if no token provided
async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    session: AsyncSession = Depends(get_db)
) -> Optional[User]:
    """Get current user if token provided, otherwise return None."""
    if credentials is None:
        return None

    try:
        token = credentials.credentials
        user_id = verify_token(token)
        if user_id:
            user = await UserRepository.get_by_id(session, user_id)
            return user
    except Exception as e:
        logger.warning(f"Optional auth failed: {e}")

    return None
