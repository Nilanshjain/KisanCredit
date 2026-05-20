"""Authentication API endpoints."""

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field, validator
from typing import Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from ..database.connection import get_db
from ..database.repositories import UserRepository
from ..database.models import User
from ..auth.otp_manager import OTPManager
from ..auth.jwt_handler import create_access_token, create_refresh_token, verify_token
from ..auth.dependencies import get_current_active_user
from ..auth.password_utils import hash_password, verify_password
from ..utils.logger import get_logger
from .schemas import SignupRequest, LoginRequest
import re
import uuid

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])


# Request/Response Models
class SendOTPRequest(BaseModel):
    """Request to send OTP to phone number."""
    phone_number: str = Field(..., description="10-digit Indian mobile number")

    @validator('phone_number')
    def validate_phone(cls, v):
        # Remove +91 prefix if present
        v = v.replace("+91", "").replace(" ", "").replace("-", "")

        if not re.match(r'^[6-9]\d{9}$', v):
            raise ValueError('Invalid Indian mobile number. Must be 10 digits starting with 6-9')
        return v


class VerifyOTPRequest(BaseModel):
    """Request to verify OTP and login/signup."""
    phone_number: str = Field(..., description="10-digit mobile number")
    otp: str = Field(..., min_length=6, max_length=6, description="6-digit OTP")
    full_name: Optional[str] = Field(None, description="Full name (required for new users)")


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user_id: str
    phone_number: str
    is_new_user: bool


class RefreshTokenRequest(BaseModel):
    """Request to refresh access token."""
    refresh_token: str


# API Endpoints
@router.post("/send-otp", status_code=status.HTTP_200_OK)
async def send_otp(
    request: SendOTPRequest,
    session: AsyncSession = Depends(get_db)
):
    """Send OTP to phone number for login/signup."""
    try:
        phone_number = request.phone_number

        # Generate OTP
        otp = OTPManager.generate_otp()

        # Store OTP
        OTPManager.store_otp(phone_number, otp)

        # Send OTP via SMS
        sms_sent = await OTPManager.send_otp_sms(phone_number, otp)

        if not sms_sent:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to send OTP. Please try again."
            )

        logger.info(f"OTP sent successfully to {phone_number}")

        return {
            "success": True,
            "message": "OTP sent successfully",
            "phone_number": phone_number,
            "expires_in_minutes": 10
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error sending OTP: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/verify-otp", response_model=TokenResponse)
async def verify_otp(
    request: VerifyOTPRequest,
    session: AsyncSession = Depends(get_db)
):
    """Verify OTP and return JWT tokens (login or signup)."""
    try:
        phone_number = request.phone_number
        otp = request.otp

        # Verify OTP
        is_valid = OTPManager.verify_otp(phone_number, otp)

        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired OTP"
            )

        # Check if user exists
        user = await UserRepository.get_by_phone(session, phone_number)
        is_new_user = user is None

        if is_new_user:
            # Create new user
            if not request.full_name:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Full name is required for new users"
                )

            user_data = {
                "user_id": f"USER_{uuid.uuid4().hex[:12].upper()}",
                "phone_number": phone_number,
                "full_name": request.full_name,
                "is_active": True
                # created_at and updated_at are set by database defaults
            }

            user = await UserRepository.create(session, user_data)
            await session.commit()

            logger.info(f"New user created: {user.user_id}")
        else:
            logger.info(f"Existing user logged in: {user.user_id}")

        # Generate JWT tokens
        access_token = create_access_token(user.user_id, phone_number)
        refresh_token = create_refresh_token(user.user_id, phone_number)

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            user_id=user.user_id,
            phone_number=phone_number,
            is_new_user=is_new_user
        )

    except HTTPException:
        raise
    except Exception as e:
        # Avoid Unicode encoding errors in logging
        error_msg = str(e).encode('ascii', 'replace').decode('ascii')
        logger.error(f"Error verifying OTP: {error_msg}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: RefreshTokenRequest,
    session: AsyncSession = Depends(get_db)
):
    """Refresh access token using refresh token."""
    try:
        # Verify refresh token
        user_id = verify_token(request.refresh_token)

        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired refresh token"
            )

        # Fetch user
        user = await UserRepository.get_by_id(session, user_id)

        if user is None or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )

        # Generate new tokens
        access_token = create_access_token(user.user_id, user.phone_number)
        refresh_token_new = create_refresh_token(user.user_id, user.phone_number)

        logger.info(f"Tokens refreshed for user: {user.user_id}")

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token_new,
            user_id=user.user_id,
            phone_number=user.phone_number,
            is_new_user=False
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refreshing token: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def signup(
    request: SignupRequest,
    session: AsyncSession = Depends(get_db)
):
    """Sign up a new user with email and password. Sends OTP for verification."""
    try:
        # Check if user already exists with this email
        existing_user = await UserRepository.get_by_email(session, request.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists"
            )

        # Check if phone number is already registered
        existing_phone = await UserRepository.get_by_phone(session, request.phone_number)
        if existing_phone:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this phone number already exists"
            )

        # Hash the password
        password_hash = hash_password(request.password)

        # Create new user
        user_data = {
            "user_id": f"USER_{uuid.uuid4().hex[:12].upper()}",
            "email": request.email,
            "phone_number": request.phone_number,
            "full_name": request.full_name,
            "password_hash": password_hash,
            "is_active": True,
            "email_verified": False
        }

        user = await UserRepository.create(session, user_data)
        await session.commit()

        logger.info(f"New user signed up: {user.user_id}")

        # Generate OTP for phone verification
        otp = OTPManager.generate_otp()
        OTPManager.store_otp(request.phone_number, otp)
        await OTPManager.send_otp_sms(request.phone_number, otp)

        # Generate JWT tokens
        access_token = create_access_token(user.user_id, user.phone_number)
        refresh_token = create_refresh_token(user.user_id, user.phone_number)

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            user_id=user.user_id,
            phone_number=user.phone_number,
            is_new_user=True
        )

    except HTTPException:
        raise
    except Exception as e:
        await session.rollback()
        error_msg = str(e).encode('ascii', 'replace').decode('ascii')
        logger.error(f"Error during signup: {error_msg}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/login", response_model=TokenResponse)
async def login(
    request: LoginRequest,
    session: AsyncSession = Depends(get_db)
):
    """Login with email and password."""
    try:
        # Get user by email
        user = await UserRepository.get_by_email(session, request.email)

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )

        # Check if user has password set
        if not user.password_hash:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Please use OTP login or reset your password"
            )

        # Verify password
        if not verify_password(request.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )

        # Check if user is active
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is inactive"
            )

        logger.info(f"User logged in: {user.user_id}")

        # Generate JWT tokens
        access_token = create_access_token(user.user_id, user.phone_number)
        refresh_token = create_refresh_token(user.user_id, user.phone_number)

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            user_id=user.user_id,
            phone_number=user.phone_number,
            is_new_user=False
        )

    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e).encode('ascii', 'replace').decode('ascii')
        logger.error(f"Error during login: {error_msg}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.post("/logout")
async def logout(current_user: User = Depends(get_current_active_user)):
    """Logout user (client should discard tokens)."""
    # In a production app, you might want to blacklist the token in Redis
    # For now, we rely on client-side token deletion

    logger.info(f"User logged out: {current_user.user_id}")

    return {
        "success": True,
        "message": "Logged out successfully"
    }


@router.get("/validate")
async def validate_token(current_user: User = Depends(get_current_active_user)):
    """Validate access token and return user info."""
    return {
        "valid": True,
        "user_id": current_user.user_id,
        "phone_number": current_user.phone_number,
        "full_name": current_user.full_name
    }
