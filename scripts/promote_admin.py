"""Promote a user to admin role.

Usage:
    python scripts/promote_admin.py <phone_number>

Examples:
    python scripts/promote_admin.py 9876543210
    python scripts/promote_admin.py +91-9876543210      (normalised to 9876543210)

The target user must already exist in the `users` table (i.e. they have
completed at least one OTP login). Run this once for your demo account; the
user-facing OTP flow doesn't expose role-elevation.
"""
import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.connection import db_manager
from src.database.repositories import UserRepository
from sqlalchemy import update
from src.database.models import User


def _normalise_phone(raw: str) -> str:
    """Strip +91, dashes, spaces. Keep only the trailing 10 digits."""
    digits = re.sub(r'\D', '', raw)
    return digits[-10:]


async def promote(phone_raw: str) -> int:
    phone = _normalise_phone(phone_raw)
    if not re.fullmatch(r'[6-9]\d{9}', phone):
        print(f"ERROR: '{phone_raw}' (normalised to '{phone}') is not a valid 10-digit Indian mobile number.", file=sys.stderr)
        return 2

    await db_manager.connect()
    try:
        async with db_manager.get_session() as session:
            user = await UserRepository.get_by_phone(session, phone)
            if user is None:
                print(f"ERROR: No user found with phone {phone}. They must complete an OTP login first.", file=sys.stderr)
                return 1

            if user.role == "admin":
                print(f"User {user.user_id} (phone {phone}) is already an admin. Nothing to do.")
                return 0

            await session.execute(
                update(User).where(User.id == user.id).values(role="admin")
            )
            await session.commit()
            print(f"Promoted user {user.user_id} (phone {phone}) to admin role.")
            return 0
    finally:
        await db_manager.disconnect()


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(asyncio.run(promote(sys.argv[1])))


if __name__ == "__main__":
    main()
