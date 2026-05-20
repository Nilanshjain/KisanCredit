"""User management API endpoints."""

from fastapi import APIRouter, HTTPException, status, Depends, Query
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from ..database.connection import get_db
from ..database.repositories import UserRepository, ApplicationRepository
from ..database.models import User, Application
from ..auth.dependencies import get_current_active_user
from ..utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/users", tags=["Users"])


# Response Models
class UserProfileResponse(BaseModel):
    """User profile response."""
    user_id: str
    phone_number: Optional[str]
    email: Optional[str]
    full_name: Optional[str]
    date_of_birth: Optional[datetime]
    address: Optional[str]
    city: Optional[str]
    state: Optional[str]
    pincode: Optional[str]
    employment_type: Optional[str]
    monthly_income: Optional[float]
    kyc_verified: bool
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class UpdateProfileRequest(BaseModel):
    """Request to update user profile."""
    full_name: Optional[str] = None
    email: Optional[str] = None
    date_of_birth: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    pincode: Optional[str] = None
    employment_type: Optional[str] = None
    monthly_income: Optional[float] = None


class ApplicationSummary(BaseModel):
    """Summary of a loan application."""
    id: str
    application_id: str
    loan_amount: float
    loan_purpose: str
    status: str
    submitted_at: datetime
    processed_at: Optional[datetime]

    class Config:
        from_attributes = True


class UserApplicationsResponse(BaseModel):
    """List of user's applications."""
    total: int
    applications: List[ApplicationSummary]


# API Endpoints
@router.get("/me", response_model=UserProfileResponse)
async def get_current_user_profile(
    current_user: User = Depends(get_current_active_user)
):
    """Get current user's profile."""
    return current_user


@router.put("/me", response_model=UserProfileResponse)
async def update_current_user_profile(
    update_data: UpdateProfileRequest,
    current_user: User = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_db)
):
    """Update current user's profile."""
    try:
        # Prepare update data (only non-None fields)
        update_fields = {
            k: v for k, v in update_data.dict().items()
            if v is not None
        }

        if not update_fields:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No fields to update"
            )

        # Update user
        updated_user = await UserRepository.update(
            session,
            current_user.user_id,
            update_fields
        )

        await session.commit()

        logger.info(f"Profile updated for user: {current_user.user_id}")

        return updated_user

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating profile: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile"
        )


@router.get("/me/applications", response_model=UserApplicationsResponse)
async def get_current_user_applications(
    current_user: User = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_db),
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(20, ge=1, le=100, description="Number of applications to return")
):
    """Get current user's loan applications."""
    try:
        # Get user's applications
        applications = await ApplicationRepository.get_by_user(
            session,
            current_user.id,  # Use internal id, not user_id
            limit=limit
        )

        # Filter by status if provided
        if status_filter:
            applications = [app for app in applications if app.status == status_filter]

        # Convert to response format
        application_summaries = [
            ApplicationSummary(
                id=app.id,
                application_id=app.application_id,
                loan_amount=app.loan_amount,
                loan_purpose=app.loan_purpose,
                status=app.status,
                submitted_at=app.submitted_at,
                processed_at=app.processed_at
            )
            for app in applications
        ]

        return UserApplicationsResponse(
            total=len(application_summaries),
            applications=application_summaries
        )

    except Exception as e:
        logger.error(f"Error fetching applications: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch applications"
        )


@router.get("/{user_id}", response_model=UserProfileResponse)
async def get_user_by_id(
    user_id: str,
    session: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get user profile by ID (admin/self only)."""
    # Users can only view their own profile (unless admin role is added)
    if current_user.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view this profile"
        )

    user = await UserRepository.get_by_id(session, user_id)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    return user
