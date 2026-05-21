"""Authentication API endpoints — email-OTP passwordless flow.

Flow: POST /send-otp (email) -> user receives 6-digit code -> POST /verify-otp
(email + code) -> JWT access + refresh tokens. New users are created on first
successful verification. No passwords — matches the rural-India / thin-file
product thesis and the 2026 passwordless trend.
"""

import re
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession

from ..database.connection import get_db
from ..database.repositories import UserRepository
from ..database.models import User
from ..auth.otp_manager import OTPManager
from ..auth.jwt_handler import create_access_token, create_refresh_token, verify_token
from ..auth.dependencies import get_current_active_user
from ..utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])

_EMAIL_RE = re.compile(r'^[\w.+-]+@[\w-]+\.[\w.-]+$')


# ─── Request / response models ───


class SendOTPRequest(BaseModel):
    """Request a one-time code be emailed."""
    email: str = Field(..., description="Email address to send the OTP to")

    @validator('email')
    def _validate_email(cls, v):
        v = v.strip().lower()
        if not _EMAIL_RE.match(v):
            raise ValueError('Invalid email address')
        return v


class VerifyOTPRequest(BaseModel):
    """Verify a one-time code and issue tokens."""
    email: str = Field(..., description="Email the OTP was sent to")
    otp: str = Field(..., min_length=6, max_length=6, description="6-digit OTP")
    full_name: Optional[str] = Field(None, description="Full name (used when creating a new user)")

    @validator('email')
    def _validate_email(cls, v):
        return v.strip().lower()


class TokenResponse(BaseModel):
    """JWT token pair + minimal user identity."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user_id: str
    email: str
    is_new_user: bool


class RefreshTokenRequest(BaseModel):
    refresh_token: str


# ─── Endpoints ───


@router.post("/send-otp", status_code=status.HTTP_200_OK)
async def send_otp(request: SendOTPRequest):
    """Generate an OTP and email it to the address."""
    otp = OTPManager.generate_otp()
    OTPManager.store_otp(request.email, otp)
    sent = await OTPManager.send_otp_email(request.email, otp)

    if not sent:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send OTP. Please try again.",
        )

    logger.info(f"OTP dispatched to {request.email}")
    return {
        "success": True,
        "message": "OTP sent to your email",
        "email": request.email,
        "expires_in_minutes": 10,
    }


@router.post("/verify-otp", response_model=TokenResponse)
async def verify_otp(
    request: VerifyOTPRequest,
    session: AsyncSession = Depends(get_db),
):
    """Verify the emailed OTP; create the user on first login; return tokens."""
    if not OTPManager.verify_otp(request.email, request.otp):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired OTP",
        )

    user = await UserRepository.get_by_email(session, request.email)
    is_new_user = user is None

    if is_new_user:
        user = await UserRepository.create(session, {
            "user_id": f"USER_{uuid.uuid4().hex[:12].upper()}",
            "email": request.email,
            "full_name": request.full_name or request.email.split("@")[0],
            "is_active": True,
            "email_verified": True,
        })
        await session.commit()
        logger.info(f"New user created via email OTP: {user.user_id}")
    else:
        logger.info(f"Existing user logged in: {user.user_id}")

    access_token = create_access_token(user.user_id, request.email)
    refresh_token = create_refresh_token(user.user_id, request.email)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user_id=user.user_id,
        email=request.email,
        is_new_user=is_new_user,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: RefreshTokenRequest,
    session: AsyncSession = Depends(get_db),
):
    """Exchange a valid refresh token for a fresh token pair."""
    user_id = verify_token(request.refresh_token)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    user = await UserRepository.get_by_id(session, user_id)
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )

    email = user.email or ""
    return TokenResponse(
        access_token=create_access_token(user.user_id, email),
        refresh_token=create_refresh_token(user.user_id, email),
        user_id=user.user_id,
        email=email,
        is_new_user=False,
    )


@router.post("/logout")
async def logout(current_user: User = Depends(get_current_active_user)):
    """Logout — the client discards its tokens. (Token blacklisting -> future work.)"""
    logger.info(f"User logged out: {current_user.user_id}")
    return {"success": True, "message": "Logged out successfully"}


@router.get("/validate")
async def validate_token(current_user: User = Depends(get_current_active_user)):
    """Validate the access token and echo back minimal user info."""
    return {
        "valid": True,
        "user_id": current_user.user_id,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "role": current_user.role,
    }
