"""OTP management for phone-based authentication."""

import random
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict
from ..utils.logger import get_logger
from ..utils.config import settings

logger = get_logger(__name__)


class OTPManager:
    """Manage OTP generation, storage, and validation."""

    # In-memory storage for development (use Redis in production)
    _otp_store: Dict[str, Dict] = {}

    @staticmethod
    def generate_otp(length: int = 6) -> str:
        """Generate a random numeric OTP."""
        return ''.join([str(random.randint(0, 9)) for _ in range(length)])

    @staticmethod
    def _hash_otp(otp: str) -> str:
        """Hash OTP for secure storage."""
        return hashlib.sha256(otp.encode()).hexdigest()

    @classmethod
    def store_otp(
        cls,
        phone_number: str,
        otp: str,
        expiry_minutes: int = 10
    ) -> None:
        """Store OTP with expiry time."""
        hashed_otp = cls._hash_otp(otp)
        expiry_time = datetime.utcnow() + timedelta(minutes=expiry_minutes)

        cls._otp_store[phone_number] = {
            'otp_hash': hashed_otp,
            'expiry': expiry_time,
            'attempts': 0,
            'created_at': datetime.utcnow()
        }

        logger.info(f"OTP stored for {phone_number}, expires at {expiry_time}")

    @classmethod
    def verify_otp(cls, phone_number: str, otp: str, max_attempts: int = 3) -> bool:
        """Verify OTP for a phone number."""
        if phone_number not in cls._otp_store:
            logger.warning(f"No OTP found for {phone_number}")
            return False

        stored_data = cls._otp_store[phone_number]

        # Check if expired
        if datetime.utcnow() > stored_data['expiry']:
            logger.warning(f"OTP expired for {phone_number}")
            cls.delete_otp(phone_number)
            return False

        # Check attempts
        if stored_data['attempts'] >= max_attempts:
            logger.warning(f"Max OTP attempts exceeded for {phone_number}")
            cls.delete_otp(phone_number)
            return False

        # Increment attempts
        stored_data['attempts'] += 1

        # Verify OTP
        hashed_input = cls._hash_otp(otp)
        is_valid = hashed_input == stored_data['otp_hash']

        if is_valid:
            logger.info(f"OTP verified successfully for {phone_number}")
            cls.delete_otp(phone_number)
        else:
            logger.warning(f"Invalid OTP attempt for {phone_number} (attempt {stored_data['attempts']})")

        return is_valid

    @classmethod
    def delete_otp(cls, phone_number: str) -> None:
        """Delete OTP from store."""
        if phone_number in cls._otp_store:
            del cls._otp_store[phone_number]
            logger.info(f"OTP deleted for {phone_number}")

    @classmethod
    async def send_otp_sms(cls, phone_number: str, otp: str) -> bool:
        """Send OTP via SMS (placeholder for actual SMS gateway integration)."""
        # TODO: Integrate with SMS gateway (Twilio, MSG91, etc.)
        # For development, just log the OTP
        logger.info(f"[DEV MODE] OTP for {phone_number}: {otp}")

        # In production, uncomment and configure:
        # try:
        #     from twilio.rest import Client
        #     client = Client(settings.twilio_sid, settings.twilio_token)
        #     message = client.messages.create(
        #         body=f"Your KisanCredit OTP is: {otp}. Valid for 10 minutes.",
        #         from_=settings.twilio_phone,
        #         to=phone_number
        #     )
        #     logger.info(f"SMS sent successfully: {message.sid}")
        #     return True
        # except Exception as e:
        #     logger.error(f"Failed to send SMS: {e}")
        #     return False

        # Development mode - always return True
        return True

    @classmethod
    def cleanup_expired_otps(cls) -> int:
        """Remove all expired OTPs from store."""
        now = datetime.utcnow()
        expired_phones = [
            phone for phone, data in cls._otp_store.items()
            if now > data['expiry']
        ]

        for phone in expired_phones:
            cls.delete_otp(phone)

        logger.info(f"Cleaned up {len(expired_phones)} expired OTPs")
        return len(expired_phones)
