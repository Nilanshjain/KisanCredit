"""OTP management for email-based passwordless authentication.

The OTP store is keyed by a generic *identifier* string — email address in the
current email-OTP flow. Delivery goes through Resend; when RESEND_API_KEY is
unset (local dev) it falls back to console-logging the OTP so the flow still
works without an email provider.
"""

import random
import hashlib
from datetime import datetime, timedelta
from typing import Dict

from ..utils.logger import get_logger
from ..utils.config import settings

logger = get_logger(__name__)


class OTPManager:
    """Generate, store, validate, and deliver one-time passcodes."""

    # In-memory store keyed by identifier (email). Fine for single-instance
    # free-tier Render; move to Redis when scaling past one worker.
    _otp_store: Dict[str, Dict] = {}

    @staticmethod
    def generate_otp(length: int = 6) -> str:
        """Generate a random numeric OTP."""
        return ''.join([str(random.randint(0, 9)) for _ in range(length)])

    @staticmethod
    def _hash_otp(otp: str) -> str:
        """Hash OTP for secure storage — we never store the plaintext code."""
        return hashlib.sha256(otp.encode()).hexdigest()

    @classmethod
    def store_otp(cls, identifier: str, otp: str, expiry_minutes: int = 10) -> None:
        """Store a hashed OTP for `identifier` with an expiry."""
        cls._otp_store[identifier] = {
            'otp_hash': cls._hash_otp(otp),
            'expiry': datetime.utcnow() + timedelta(minutes=expiry_minutes),
            'attempts': 0,
            'created_at': datetime.utcnow(),
        }
        logger.info(f"OTP stored for {identifier}")

    @classmethod
    def verify_otp(cls, identifier: str, otp: str, max_attempts: int = 3) -> bool:
        """Verify an OTP. Consumes it on success; enforces expiry + attempt cap."""
        record = cls._otp_store.get(identifier)
        if record is None:
            logger.warning(f"No OTP found for {identifier}")
            return False

        if datetime.utcnow() > record['expiry']:
            logger.warning(f"OTP expired for {identifier}")
            cls.delete_otp(identifier)
            return False

        if record['attempts'] >= max_attempts:
            logger.warning(f"Max OTP attempts exceeded for {identifier}")
            cls.delete_otp(identifier)
            return False

        record['attempts'] += 1
        is_valid = cls._hash_otp(otp) == record['otp_hash']

        if is_valid:
            logger.info(f"OTP verified for {identifier}")
            cls.delete_otp(identifier)
        else:
            logger.warning(f"Invalid OTP attempt for {identifier} (attempt {record['attempts']})")
        return is_valid

    @classmethod
    def delete_otp(cls, identifier: str) -> None:
        """Remove an OTP from the store."""
        cls._otp_store.pop(identifier, None)

    @classmethod
    async def send_otp_email(cls, email: str, otp: str) -> bool:
        """Deliver the OTP to `email` via Resend.

        When RESEND_API_KEY is unset (local dev), logs the OTP to the console
        and returns True — the auth flow stays usable without an email provider.
        Any Resend error also degrades to console-log so a transient provider
        outage never blocks login during a demo.
        """
        if not settings.resend_api_key:
            logger.info(f"[DEV MODE — no RESEND_API_KEY] OTP for {email}: {otp}")
            return True

        try:
            import resend
            resend.api_key = settings.resend_api_key
            resend.Emails.send({
                "from": settings.resend_from_email,
                "to": [email],
                "subject": "Your KisanCredit verification code",
                "html": (
                    f"<div style='font-family:system-ui,sans-serif;max-width:420px'>"
                    f"<h2 style='color:#1C1917'>KisanCredit verification</h2>"
                    f"<p>Your one-time code is:</p>"
                    f"<p style='font-size:32px;font-weight:700;letter-spacing:6px;"
                    f"color:#16a34a'>{otp}</p>"
                    f"<p style='color:#78716c;font-size:14px'>Valid for 10 minutes. "
                    f"If you didn't request this, ignore this email.</p>"
                    f"</div>"
                ),
            })
            logger.info(f"OTP email sent to {email} via Resend")
            return True
        except Exception as e:
            # Degrade gracefully — log the OTP so the demo flow survives a
            # provider outage or quota exhaustion.
            logger.warning(f"Resend send failed ({e}); falling back to console OTP for {email}: {otp}")
            return True

    @classmethod
    def cleanup_expired_otps(cls) -> int:
        """Drop all expired OTPs from the store. Returns the count removed."""
        now = datetime.utcnow()
        expired = [k for k, v in cls._otp_store.items() if now > v['expiry']]
        for k in expired:
            cls.delete_otp(k)
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired OTPs")
        return len(expired)
