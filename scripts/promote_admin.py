"""Promote a user to admin role.

Usage:
    python scripts/promote_admin.py <email>

Example:
    python scripts/promote_admin.py operator@example.com

The target user must already exist (i.e. they've completed at least one
email-OTP login). Run this once for your demo operator account; the
user-facing OTP flow doesn't expose role-elevation.
"""
import asyncio
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import update

from src.database.connection import db_manager
from src.database.repositories import UserRepository
from src.database.models import User

_EMAIL_RE = re.compile(r'^[\w.+-]+@[\w-]+\.[\w.-]+$')


async def promote(email_raw: str) -> int:
    email = email_raw.strip().lower()
    if not _EMAIL_RE.match(email):
        print(f"ERROR: '{email_raw}' is not a valid email address.", file=sys.stderr)
        return 2

    await db_manager.connect()
    try:
        async with db_manager.get_session() as session:
            user = await UserRepository.get_by_email(session, email)
            if user is None:
                print(
                    f"ERROR: No user found with email {email}. "
                    "They must complete an email-OTP login first.",
                    file=sys.stderr,
                )
                return 1

            if user.role == "admin":
                print(f"User {user.user_id} ({email}) is already an admin. Nothing to do.")
                return 0

            await session.execute(
                update(User).where(User.id == user.id).values(role="admin")
            )
            await session.commit()
            print(f"Promoted user {user.user_id} ({email}) to admin role.")
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
